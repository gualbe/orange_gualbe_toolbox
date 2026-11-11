"""Motor de Test & Score Pro (sin Qt): evaluación, puntuaciones y comparaciones.

Se apoya en lo mismo que el Test & Score nativo (Orange.evaluation y sus scorers) para
que las puntuaciones y las salidas sean idénticas. Lo propio son las puntuaciones por
partición, los tests y su interpretación.
"""
import copy
import os
import threading
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

import Orange.evaluation
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, StringVariable, Table
from Orange.widgets.evaluate.owtestandscore import results_merge, results_one_vs_rest
from Orange.widgets.evaluate.utils import scorer_caller

from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_stats as st
from orangecontrib.orange_gualbe_toolbox.widgets import _tsp_window as tw

KFOLD, FEATURE, SHUFFLE, LOO, ON_TRAIN, ON_TEST, WINDOW = range(7)
FOLD_KINDS = (KFOLD, FEATURE, SHUFFLE, WINDOW)
ROWS_ONCE_KINDS = (KFOLD, FEATURE, LOO, ON_TRAIN, ON_TEST)

LOWER_IS_BETTER = frozenset(("LogLoss", "MSE", "RMSE", "MAE", "MAPE", "SMAPE", "CVRMSE"))
BOUNDED_METRICS = frozenset(("AUC", "CA", "F1", "Precision", "Recall", "Specificity",
                             "MatthewsCorrCoefficient", "R2"))
PERCENT_METRICS = frozenset(("MAPE", "SMAPE", "CVRMSE"))
TARGET_SCALE_METRICS = frozenset(("RMSE", "MAE"))

N_BOOTSTRAP = 200
N_BOOTSTRAP_LARGE = 50
LARGE_SAMPLE = 50000
MAX_WORKERS = 4
# Por debajo de este tamaño, repartir en hilos cuesta más de lo que ahorra (medido).
MIN_ROWS_PARALLEL = 5000


class Cancelled(Exception):
    """El usuario ha cancelado o han cambiado las entradas."""


@dataclass
class Sampling:
    kind: int = KFOLD
    k: int = 5
    stratified: bool = True
    n_repeats: int = 10
    train_fraction: float = 0.66
    feature: object = None
    seed: int = 42
    window: object = None   # tw.WindowSpec
    plan: object = None     # tw.WindowPlan, lo rellena check_sampling

    def describe(self):
        if self.kind == WINDOW:
            return self.window.describe() if self.window is not None else "Ventana deslizante"
        strat = " estratificada" if self.stratified else ""
        if self.kind == KFOLD:
            return f"Validación cruzada{strat} de {self.k} particiones"
        if self.kind == FEATURE:
            return f"Validación cruzada por el atributo «{getattr(self.feature, 'name', '?')}»"
        if self.kind == SHUFFLE:
            return (f"Muestreo aleatorio{strat}: {self.n_repeats} repeticiones con "
                    f"{self.train_fraction:.0%} para entrenar")
        if self.kind == LOO:
            return "Dejar uno fuera"
        if self.kind == ON_TRAIN:
            return "Evaluación sobre los datos de entrenamiento"
        return "Evaluación sobre datos de test separados"


@dataclass
class Outcome:
    results: Optional[object] = None
    error: Optional[Exception] = None
    per_target: Optional[list] = None  # ventana deslizante: un Outcome por variable objetivo

    def part(self, target=0):
        if self.per_target is None:
            return self
        return self.per_target[min(max(target, 0), len(self.per_target) - 1)]


def check_sampling(data, test_data, sampling):
    """Devuelve (muestreo ajustado, avisos, error). Nada lanza excepciones."""
    notes = []
    n_rows = len(data)
    if len(data.domain.class_vars) > 1 and sampling.kind != WINDOW:
        return sampling, notes, ("Hay varias variables objetivo: elige «Ventana deslizante» para "
                                 "evaluarlas como horizontes, o deja solo una con Select Columns.")
    if sampling.kind == WINDOW:
        return _check_window(data, sampling, notes)
    class_var = data.domain.class_var
    discrete = class_var is not None and class_var.is_discrete
    least, n_classes = None, 0
    if discrete:
        counts = np.bincount(data.Y.astype(int))
        counts = counts[counts > 0]
        least, n_classes = int(counts.min()), len(counts)
    if sampling.kind == KFOLD:
        if n_rows < sampling.k:
            return sampling, notes, (f"Hay {n_rows} filas y se piden {sampling.k} particiones: "
                                     "elige menos particiones.")
        if sampling.stratified and not discrete:
            notes.append("La estratificación solo se aplica a clases categóricas: se ignora.")
            sampling.stratified = False
        elif sampling.stratified and least < sampling.k:
            notes.append(f"No se puede estratificar en {sampling.k} particiones: la clase menos "
                         f"frecuente tiene {least} filas. Se hace sin estratificar.")
            sampling.stratified = False
    elif sampling.kind == SHUFFLE:
        n_train = int(round(n_rows * sampling.train_fraction))
        n_test = n_rows - n_train
        if n_train < 1 or n_test < 1:
            return sampling, notes, (f"Con {n_rows} filas no se puede repartir un "
                                     f"{sampling.train_fraction:.0%} para entrenar y el resto "
                                     "para test.")
        if sampling.stratified and not discrete:
            notes.append("La estratificación solo se aplica a clases categóricas: se ignora.")
            sampling.stratified = False
        elif sampling.stratified and (least < 2 or min(n_train, n_test) < n_classes):
            notes.append("No se puede estratificar: alguna clase tiene muy pocas filas. Se hace "
                         "sin estratificar.")
            sampling.stratified = False
    elif sampling.kind == FEATURE:
        if sampling.feature is None:
            return sampling, notes, ("Elige un atributo categórico (entre las metas) para "
                                     "definir las particiones.")
        try:
            column = data.transform(Domain([sampling.feature])).X.ravel()
        except Exception as error:  # pylint: disable=broad-except
            return sampling, notes, f"No se puede usar ese atributo: {error}"
        present = np.unique(column[~np.isnan(column)])
        if len(present) < 2:
            return sampling, notes, (f"El atributo «{sampling.feature.name}» necesita al menos dos "
                                     "valores distintos.")
        missing = int(np.isnan(column).sum())
        if missing:
            notes.append(f"{missing} filas sin valor en «{sampling.feature.name}» no se evalúan.")
        notes.append("Las particiones por atributo tienen tamaños distintos: la corrección de "
                     "Nadeau-Bengio se aplica con el tamaño medio y es aproximada.")
    elif sampling.kind == LOO:
        notes.append("Dejar uno fuera: una fila por partición, así que los tests se hacen por "
                     "filas y varias métricas no se pueden calcular partición a partición.")
    elif sampling.kind == ON_TRAIN:
        notes.append("Se evalúa sobre los mismos datos con los que se entrena: las puntuaciones "
                     "son optimistas y no estiman el rendimiento en datos nuevos.")
    elif sampling.kind == ON_TEST:
        if test_data is None:
            return sampling, notes, None
        if test_data.domain.class_var != data.domain.class_var:
            return sampling, notes, ("Los datos de entrenamiento y de test tienen variables "
                                     "objetivo distintas.")
    return sampling, notes, None


def _check_window(data, sampling, notes):
    problem = tw.check_targets(data)
    if problem:
        return sampling, notes, problem
    plan = tw.plan_windows(data, sampling.window)
    sampling.plan = plan
    notes.extend(plan.notes)
    if plan.error:
        return sampling, notes, plan.error
    for table in tw.target_tables(data, plan):
        target_notes, problem = tw.target_notes(plan, table)
        notes.extend(target_notes)
        if problem:
            return sampling, notes, problem
    if sampling.window.advance == tw.EXPANDING:
        notes.append("Diebold-Mariano es el test principal; los tests sobre iteraciones son "
                     "orientativos, porque las iteraciones comparten entrenamiento y son "
                     "consecutivas en el tiempo.")
    else:
        notes.append("Diebold-Mariano es el test principal (con ventana de tamaño fijo vale también "
                     "teniendo en cuenta la estimación de los modelos); los tests sobre iteraciones "
                     "son orientativos.")
    notes.append(NOT_FOR_WINDOW)
    return sampling, notes, None


def make_sampler(sampling):
    if sampling.kind == KFOLD:
        return Orange.evaluation.CrossValidation(
            k=sampling.k, random_state=sampling.seed, stratified=sampling.stratified,
            store_data=True)
    if sampling.kind == FEATURE:
        return Orange.evaluation.CrossValidationFeature(feature=sampling.feature, store_data=True)
    if sampling.kind == SHUFFLE:
        return Orange.evaluation.ShuffleSplit(
            n_resamples=sampling.n_repeats, train_size=sampling.train_fraction, test_size=None,
            stratified=sampling.stratified, random_state=sampling.seed, store_data=True)
    if sampling.kind == LOO:
        return Orange.evaluation.LeaveOneOut(store_data=True)
    if sampling.kind == ON_TRAIN:
        return Orange.evaluation.TestOnTrainingData(store_data=True, store_models=True)
    return Orange.evaluation.TestOnTestData(store_data=True, store_models=True)


def _safe_copy(learner):
    try:
        return copy.deepcopy(learner)
    except Exception:  # pylint: disable=broad-except
        return learner  # algunos modelos de add-ons no se dejan copiar


def default_workers(n_learners, preprocessor=None, n_rows=MIN_ROWS_PARALLEL):
    """Hilos para evaluar a la vez; cada modelo solo lo usa un hilo."""
    if preprocessor is not None or n_learners < 2 or n_rows < MIN_ROWS_PARALLEL:
        return 1  # con preprocesador, se comparte y se hace una vez por partición
    return max(1, min(n_learners, (os.cpu_count() or 2) - 1, MAX_WORKERS))


def evaluate(data, test_data, learners, sampling, preprocessor=None, workers=1,
             report_progress=None, is_cancelled=None):
    """Evalúa `learners` (lista de (clave, learner)) y devuelve {clave: Outcome}.

    Un modelo que falla no afecta a los demás. Con varios hilos, cada uno evalúa un
    modelo completo: todos usan la misma semilla y por tanto las mismas particiones.
    """
    is_cancelled = is_cancelled or (lambda: False)
    report_progress = report_progress or (lambda value: None)
    groups = ([[item] for item in learners] if workers > 1 else [list(learners)])
    fractions = [0.0] * len(groups)
    stop = threading.Event()

    def run_group(position):
        group = groups[position]

        def callback(fraction):
            if stop.is_set():
                raise Cancelled()
            fractions[position] = float(fraction)

        copies = [_safe_copy(learner) for _, learner in group]
        if sampling.kind == WINDOW:
            return _run_window_group(group, copies, callback)
        sampler = make_sampler(sampling)
        try:
            if sampling.kind == ON_TEST:
                joint = sampler(data, test_data, copies, preprocessor, callback=callback)
            else:
                joint = sampler(data, copies, preprocessor, callback=callback)
        except Cancelled:
            raise
        except Exception as error:  # pylint: disable=broad-except
            return {key: Outcome(error=error) for key, _ in group}
        out = {}
        for (key, learner), result in zip(group, joint.split_by_model()):
            if result.failed[0]:
                out[key] = Outcome(error=result.failed[0])
            else:
                result.learners = [learner]
                out[key] = Outcome(result)
        return out

    def _run_window_group(group, copies, callback):
        try:
            per_target = tw.run_window(sampling.plan, data, copies, preprocessor, callback)
        except Cancelled:
            raise
        except Exception as error:  # pylint: disable=broad-except
            return {key: Outcome(error=error) for key, _ in group}
        parts = {key: [] for key, _ in group}
        for joint in per_target:
            if isinstance(joint, Exception):
                for key, _ in group:
                    parts[key].append(Outcome(error=joint))
                continue
            for (key, learner), result in zip(group, joint.split_by_model()):
                if result.failed[0]:
                    parts[key].append(Outcome(error=result.failed[0]))
                else:
                    result.learners = [learner]
                    parts[key].append(Outcome(result))
        out = {}
        for key, _ in group:
            first = parts[key][0]
            out[key] = Outcome(first.results, first.error, parts[key])
        return out

    outcomes = {}
    pool = ThreadPoolExecutor(max_workers=max(workers, 1), thread_name_prefix="tsp")
    pending = {pool.submit(run_group, position) for position in range(len(groups))}
    try:
        while pending:
            done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            for future in done:
                outcomes.update(future.result())
            if is_cancelled():
                stop.set()
                raise Cancelled()
            report_progress(90.0 * sum(fractions) / len(groups))
    finally:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
    return outcomes


@dataclass
class ScoreSet:
    keys: list
    names: List[str]
    scorers: list
    target: Optional[int]
    overall: Dict[str, np.ndarray]
    folds: Dict[str, Optional[np.ndarray]]
    errors: Dict[str, str]
    undefined: Dict[str, int]
    train_time: np.ndarray
    test_time: np.ndarray
    combined: object
    has_folds: bool
    test_train_ratio: float = 0.0
    rho: float = 0.0
    fold_sizes: Optional[np.ndarray] = None


def is_lower_better(scorer):
    return scorer.__name__ in LOWER_IS_BETTER


def metric_label(scorer):
    return scorer.name or scorer.__name__


def unique_labels(names):
    used = {}
    out = []
    for name in names:
        used[name] = used.get(name, 0) + 1
        out.append(name if used[name] == 1 else f"{name} ({used[name]})")
    return out


def _log_loss_by_hand(fold):
    """Pérdida logarítmica para particiones de una clase, que sklearn rechaza."""
    probabilities = fold.probabilities
    probabilities = probabilities / probabilities.sum(axis=2, keepdims=True)
    actual = np.asarray(fold.actual).astype(int)
    chosen = np.take_along_axis(probabilities, actual[None, :, None].repeat(len(probabilities), axis=0), axis=2)
    return -np.mean(np.log(np.clip(chosen[:, :, 0], np.finfo(float).eps, 1)), axis=1)


def _specificity_of(actual, predicted, positive):
    negatives = actual != positive
    if not negatives.any():
        return np.full(len(predicted), np.nan)  # sin negativos no está definida
    return np.sum((predicted != positive) & negatives, axis=1) / negatives.sum()


def _specificity_by_hand(fold, target):
    actual, predicted = np.asarray(fold.actual, dtype=float), np.asarray(fold.predicted, dtype=float)
    if target is not None:
        return _specificity_of(actual, predicted, target)
    classes, counts = np.unique(actual, return_counts=True)
    return np.sum([_specificity_of(actual, predicted, value) * count / len(actual)
                   for value, count in zip(classes, counts)], axis=0)


def _score_fold(instance, scorer, fold, keywords):
    """Puntuación de una partición; NaN solo si de verdad no está definida."""
    try:
        return np.ravel(instance.compute_score(fold, **keywords))
    except Exception:  # pylint: disable=broad-except
        if scorer.__name__ == "LogLoss" and fold.probabilities is not None:
            return _log_loss_by_hand(fold)
        if scorer.__name__ == "Specificity":
            return _specificity_by_hand(fold, keywords.get("target"))
        return np.nan  # p. ej. AUC con una sola clase en la partición


def compute_scores(keys, names, results, scorers, sampling, n_rows, target=None, progress=None,
                   raw_names=None):
    """Puntuaciones globales (las del nativo) y por partición."""
    progress = progress or (lambda: None)
    combined = results_merge(results)
    combined.learner_names = list(raw_names if raw_names is not None else names)
    overall, errors = {}, {}
    per_model = [results_one_vs_rest(res, target) if target is not None else res for res in results]
    for scorer in scorers:
        progress()
        values = np.full(len(results), np.nan)
        for position, result in enumerate(per_model):
            try:
                value = scorer_caller(scorer, result, target=1 if target is not None else None)()
                values[position] = float(np.ravel(value)[0])
            except Exception as error:  # pylint: disable=broad-except
                errors[scorer.__name__] = f"{type(error).__name__}: {error}"
        overall[scorer.__name__] = values
    folds, undefined = {}, {}
    base = results_one_vs_rest(combined, target) if target is not None else combined
    has_folds = sampling.kind in FOLD_KINDS and base.folds is not None and len(base.folds) >= 2
    ratio = rho = 0.0
    sizes = None
    if has_folds:
        fold_results = [base.get_fold(index) for index in range(len(base.folds))]
        sizes = np.array([len(fold.actual) for fold in fold_results], dtype=float)
        n_test = float(sizes.mean())
        # Las filas de verdad: por característica, las que tienen valor en ese atributo.
        n_used = float(sizes.sum()) if sampling.kind == FEATURE else float(n_rows)
        n_train = max(n_used - n_test, 1.0)
        ratio = n_test / n_train
        rho = n_test / (n_train + n_test)
        if sampling.kind == KFOLD:
            rho = 1.0 / len(fold_results)
            ratio = 1.0 / (len(fold_results) - 1)
        elif sampling.kind == WINDOW:
            ratio = rho = 0.0  # Nadeau-Bengio no tiene sentido con ventanas en el tiempo
        for scorer in scorers:
            progress()
            instance = scorer()
            keywords = {}
            if scorer.is_binary:
                keywords = dict(target=1 if target is not None else None, average="weighted")
            array = np.full((len(results), len(fold_results)), np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for position, fold in enumerate(fold_results):
                    array[:, position] = _score_fold(instance, scorer, fold, keywords)
            array[~np.isfinite(array)] = np.nan
            folds[scorer.__name__] = array
            undefined[scorer.__name__] = int(np.isnan(array).any(axis=0).sum())
    return ScoreSet(list(keys), list(names), list(scorers), target, overall, folds, errors, undefined,
                    np.array([float(res.train_time) for res in results]),
                    np.array([float(res.test_time) for res in results]),
                    combined, has_folds, ratio, rho, sizes)


@dataclass
class TestOptions:
    metric: str = ""
    alpha: float = 0.05
    correction: str = "holm"
    rope_auto: bool = True
    rope: float = 0.01
    horizon: int = 1  # h de Diebold-Mariano


@dataclass
class TestReport:
    names: List[str]
    metric: str
    metric_label: str
    lower_better: bool
    values: np.ndarray
    order: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    mean_ranks: np.ndarray
    pairs: List[st.PairTest]
    friedman: Optional[st.FriedmanResult]
    primary: str
    available: List[str]
    alpha: float
    correction: str
    rope: float
    notes: List[str] = field(default_factory=list)
    recommendation: List[str] = field(default_factory=list)
    best: int = -1
    blocks: str = "particiones"  # cómo se llaman las unidades de remuestreo: particiones o iteraciones

    def pairs_of(self, test):
        return [pair for pair in self.pairs if pair.test == test]


def _target_scale(actual):
    actual = np.asarray(actual, dtype=float)
    finite = actual[np.isfinite(actual)]
    return float(np.abs(finite).max()) if finite.size else 0.0


def noise_floor(scorer, actual, values=None):
    """Por debajo de esto, una diferencia de la métrica es ruido numérico.

    Hace falta cuando la métrica vale casi cero (un ajuste perfecto, RMSE de 1e-15): ahí la
    escala de los propios valores es también casi cero y no protege de nada. Se expresa en la
    escala natural de la métrica; para una métrica desconocida no se aplica ningún suelo.

    Sea δ = tol·máx|y| el error de redondeo de un residuo. Entonces RMSE y MAE arrastran un
    error del orden de δ, sea cual sea su valor, pero MSE arrastra 2·δ·RMSE + δ². Por eso en
    MSE el suelo depende de cada valor (`values`): con y muy desplazada, δ² o máx|y|² serían
    enormes frente al error real y ocultarían diferencias de verdad.
    """
    name = scorer.__name__
    tolerance = st.RELATIVE_TOLERANCE
    if name == "MSE":
        delta = tolerance * _target_scale(actual)
        rmse = 0.0 if values is None else np.sqrt(np.abs(np.asarray(values, dtype=float)))
        return 2 * delta * rmse + delta ** 2
    if name in BOUNDED_METRICS or name == "LogLoss":
        return tolerance
    if name in PERCENT_METRICS:
        return 100 * tolerance
    if name in TARGET_SCALE_METRICS:
        return tolerance * _target_scale(actual)
    return 0.0


def effective_rope(options, scorer, values):
    if not options.rope_auto:
        return max(options.rope, 0.0)
    if scorer.__name__ in BOUNDED_METRICS:
        return 0.01
    finite = values[np.isfinite(values)]
    return 0.01 * float(np.mean(np.abs(finite))) if len(finite) else 0.0


def _bootstrap_intervals(scoreset, scorer, target, alpha, progress):
    """Intervalos por remuestreo de filas para cuando no hay particiones."""
    base = results_one_vs_rest(scoreset.combined, target) if target is not None else scoreset.combined
    n_rows = len(base.actual)
    rounds = N_BOOTSTRAP_LARGE if n_rows > LARGE_SAMPLE else N_BOOTSTRAP
    rng = np.random.RandomState(0)
    instance = scorer()
    keywords = {}
    if scorer.is_binary:
        keywords = dict(target=1 if target is not None else None, average="weighted")
    draws = np.full((len(scoreset.names), rounds), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for round_index in range(rounds):
            progress()
            sample = Orange.evaluation.Results()
            indices = rng.randint(0, n_rows, n_rows)
            sample.actual = base.actual[indices]
            sample.predicted = base.predicted[:, indices]
            sample.row_indices = base.row_indices[indices]
            sample.domain = base.domain
            if base.probabilities is not None:
                sample.probabilities = base.probabilities[:, indices]
            try:
                draws[:, round_index] = np.ravel(instance.compute_score(sample, **keywords))
            except Exception:  # pylint: disable=broad-except
                pass
    low = np.nanpercentile(draws, 100 * alpha / 2, axis=1)
    high = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=1)
    return low, high


def _adjust(pairs, method, alpha):
    if not pairs:
        return
    adjusted = st.adjust_pvalues([pair.p_value for pair in pairs], method)
    for pair, value in zip(pairs, adjusted):
        pair.p_adjusted = float(value)
        st.decide_frequentist(pair, alpha)


FLIP_STATISTIC = (st.TTEST_CORRECTED, st.TTEST_PAIRED, st.ROW_TTEST, st.HOLM_POSTHOC, st.DIEBOLD_MARIANO)
RATIO_EFFECT = (st.MCNEMAR, st.DIEBOLD_MARIANO)  # el efecto es una razón A/B: no cambia de signo
ITERATION_ASSUMPTION = ("Iteraciones consecutivas en el tiempo y con entrenamientos solapados: no "
                        "son independientes. Orientativo.")
BAYES_WINDOW_ASSUMPTION = ("Posterior t de Student sobre las iteraciones sin correlación (ρ = 0): con una "
                           "ventana deslizante no hay una correlación conocida que corregir, así que no "
                           "tiene en cuenta el solapamiento de los entrenamientos. Orientativo.")
NOT_FOR_WINDOW = ("Con ventana deslizante no se aplican el t corregido de Nadeau-Bengio, que supone "
                  "particiones al azar, ni McNemar, que supone filas independientes: el test principal "
                  "es Diebold-Mariano.")
PAIRED_WINDOW_WARNING = ("Sin corregir: las iteraciones no son independientes y este test da demasiados "
                         "falsos positivos. Usa Diebold-Mariano, el test principal de la ventana deslizante.")


def _describe(pairs, sign, units, evidence):
    """Pasa las diferencias a la orientación que ve el usuario: valor(A) − valor(B).

    Por dentro «positivo» significa «A es mejor»; en métricas de error y en rangos eso
    es lo contrario de A − B. La decisión ya está tomada y no cambia.
    """
    for pair in pairs:
        pair.units, pair.evidence = units, evidence
        if sign >= 0:
            continue
        pair.mean_diff = -pair.mean_diff
        pair.ci_low, pair.ci_high = -pair.ci_high, -pair.ci_low
        if pair.test not in RATIO_EFFECT:
            pair.effect = -pair.effect
        if pair.test in FLIP_STATISTIC:
            pair.statistic = -pair.statistic


def run_tests(scoreset, sampling, options, progress=None):
    """Calcula todas las comparaciones aplicables sobre `options.metric`."""
    progress = progress or (lambda: None)
    scorer = next(s for s in scoreset.scorers if s.__name__ == options.metric)
    sign = -1.0 if is_lower_better(scorer) else 1.0
    values = scoreset.overall[options.metric]
    oriented = sign * values
    count = len(scoreset.names)
    order = np.argsort(np.where(np.isnan(oriented), -np.inf, oriented))[::-1]
    alpha = options.alpha
    notes = []
    rope = effective_rope(options, scorer, values)
    ci_low = ci_high = np.full(count, np.nan)
    mean_ranks = np.full(count, np.nan)
    friedman = None
    pairs_by_test = {}

    fold_matrix = scoreset.folds.get(options.metric) if scoreset.has_folds else None
    index_pairs = [(a, b) for a in range(count) for b in range(a + 1, count)]
    primary = ""
    if fold_matrix is not None:
        adjusted_folds = sign * fold_matrix
        floors = np.broadcast_to(noise_floor(scorer, scoreset.combined.actual, fold_matrix), fold_matrix.shape)
        ratio, rho = scoreset.test_train_ratio, scoreset.rho
        intervals = [st.corrected_interval(row, ratio, alpha, limit) for row, limit in zip(fold_matrix, floors)]
        ci_low = np.array([low for low, _ in intervals])
        ci_high = np.array([high for _, high in intervals])
        window = sampling.kind == WINDOW
        primary = st.DIEBOLD_MARIANO if window else st.TTEST_CORRECTED
        undefined = scoreset.undefined.get(options.metric, 0)
        if undefined:
            where = "iteraciones" if window else "particiones"
            notes.append(f"{metric_label(scorer)} no está definida en {undefined} de "
                         f"{fold_matrix.shape[1]} {where} (se descartan al comparar).")
        if count > 1:
            nominal = {st.TTEST_PAIRED: [], st.WILCOXON: [], st.BAYES: []}
            if not window:
                nominal = {st.TTEST_CORRECTED: [], **nominal}
            for a, b in index_pairs:
                progress()
                x, y = adjusted_folds[a], adjusted_folds[b]
                floor = np.maximum(floors[a], floors[b])
                paired_warning = PAIRED_WINDOW_WARNING if window else st.PAIRED_WARNING
                results = [st.paired_ttest(x, y, alpha, floor, paired_warning),
                           st.wilcoxon_test(x, y, alpha=alpha, floor=floor),
                           st.bayes_correlated(x, y, rope, rho, floor)]
                if not window:
                    results.insert(0, st.corrected_ttest(x, y, ratio, alpha, floor))
                for result in results:
                    result.a, result.b = a, b
                    if window:
                        result.assumption = (BAYES_WINDOW_ASSUMPTION if result.test == st.BAYES
                                             else ITERATION_ASSUMPTION)
                    nominal[result.test].append(result)
            for key in (st.TTEST_CORRECTED, st.TTEST_PAIRED, st.WILCOXON):
                if key in nominal:
                    _adjust(nominal[key], options.correction, alpha)
            for group in nominal.values():
                _describe(group, sign, metric_label(scorer), metric_label(scorer))
            pairs_by_test.update(nominal)
        friedman = st.friedman_test(adjusted_folds, alpha, on_folds=True, floor=floors)
        if friedman.n_blocks:
            mean_ranks = friedman.mean_ranks
        if count >= 3 and friedman.n_blocks >= 2:
            nemenyi = st.nemenyi_pairs(friedman.mean_ranks, friedman.n_blocks, alpha)
            ranks_of = f"los rangos de {metric_label(scorer)}"
            _describe(nemenyi, -1.0, "rango medio", ranks_of)
            pairs_by_test[st.NEMENYI] = nemenyi
            holm = st.holm_pairs(friedman.mean_ranks, friedman.n_blocks)
            _adjust(holm, "holm", alpha)
            _describe(holm, -1.0, "rango medio", ranks_of)
            pairs_by_test[st.HOLM_POSTHOC] = holm
    else:
        ci_low, ci_high = _bootstrap_intervals(scoreset, scorer, scoreset.target, alpha, progress)
        finite = np.where(np.isnan(oriented), -np.inf, oriented)
        mean_ranks = 1.0 + np.argsort(np.argsort(-finite, kind="stable"), kind="stable").astype(float)

    combined = scoreset.combined
    classification = combined.domain.has_discrete_class
    if sampling.kind == WINDOW and count > 1:
        primary = st.DIEBOLD_MARIANO
        base = results_one_vs_rest(combined, scoreset.target) if scoreset.target is not None else combined
        kind = tw.loss_kind(options.metric, classification)
        losses, floor = tw.row_losses(base, kind, st.RELATIVE_TOLERANCE)
        tests = []
        for a, b in index_pairs:
            progress()
            result = st.diebold_mariano(losses[a], losses[b], options.horizon, alpha, floor)
            result.a, result.b = a, b
            tests.append(result)
        _adjust(tests, options.correction, alpha)
        _describe(tests, -1.0, tw.LOSS_UNITS[kind], tw.LOSS_EVIDENCE[kind])
        pairs_by_test = {st.DIEBOLD_MARIANO: tests, **pairs_by_test}
    if classification and count > 1 and sampling.kind in ROWS_ONCE_KINDS:
        correct = combined.predicted == combined.actual
        tests = []
        assumption = None
        if sampling.kind == ON_TRAIN:
            assumption = ("Filas evaluadas con el mismo conjunto de entrenamiento: el resultado "
                          "no se generaliza a datos nuevos.")
        for a, b in index_pairs:
            progress()
            result = st.mcnemar_test(correct[a], correct[b], assumption, alpha)
            result.a, result.b = a, b
            tests.append(result)
        _adjust(tests, options.correction, alpha)
        _describe(tests, 1.0, "proporción de aciertos", "aciertos por fila")
        pairs_by_test[st.MCNEMAR] = tests
        if not primary:
            primary = st.MCNEMAR
    if not classification and count > 1 and fold_matrix is None and sampling.kind != WINDOW:
        errors = -np.abs(combined.predicted - combined.actual)
        row_floor = st.RELATIVE_TOLERANCE * float(np.nanmax(np.abs(combined.actual)))
        row_t, row_w = [], []
        assumption = {
            ON_TEST: "Filas de test independientes; se compara el error absoluto por fila.",
            LOO: "Los conjuntos de entrenamiento casi coinciden: filas no del todo independientes.",
            ON_TRAIN: "Filas usadas también para entrenar: no generaliza a datos nuevos.",
        }.get(sampling.kind, "")
        for a, b in index_pairs:
            progress()
            result_t = st.corrected_ttest(errors[a], errors[b], 0.0, alpha, row_floor)
            result_t.test, result_t.assumption = st.ROW_TTEST, assumption
            result_w = st.wilcoxon_test(errors[a], errors[b], st.ROW_WILCOXON, assumption, alpha, row_floor)
            for result in (result_t, result_w):
                result.a, result.b = a, b
            row_t.append(result_t)
            row_w.append(result_w)
        _adjust(row_t, options.correction, alpha)
        _adjust(row_w, options.correction, alpha)
        _describe(row_t, -1.0, "error absoluto", "error absoluto por fila")
        _describe(row_w, -1.0, "error absoluto", "error absoluto por fila")
        pairs_by_test[st.ROW_TTEST] = row_t
        pairs_by_test[st.ROW_WILCOXON] = row_w
        primary = primary or st.ROW_WILCOXON

    pairs = [pair for group in pairs_by_test.values() for pair in group]
    available = list(pairs_by_test)
    report = TestReport(list(scoreset.names), options.metric, metric_label(scorer), bool(sign < 0),
                        values, order, ci_low, ci_high, mean_ranks, pairs, friedman, primary,
                        available, alpha, options.correction, rope, notes)
    if sampling.kind == WINDOW:
        report.blocks = "iteraciones"
    if count and np.isfinite(oriented).any():
        report.best = int(order[0])
    report.recommendation = _recommend(report, scoreset)
    return report


def _pair_between(pairs, a, b):
    """Busca el resultado de una pareja y lo devuelve con A = `a`."""
    for pair in pairs:
        if (pair.a, pair.b) == (a, b):
            return pair, False
        if (pair.a, pair.b) == (b, a):
            return pair, True
    return None, False


def is_clearly_better(report, a, b):
    """¿Hay evidencia (según el test principal) de que `a` supera a `b`?"""
    pair, flipped = _pair_between(report.pairs_of(report.primary), a, b)
    if pair is None:
        return False
    return pair.decision == (st.B_BETTER if flipped else st.A_BETTER)


METRIC_OF_EVIDENCE = {"aciertos por fila": {"CA"}, "error absoluto por fila": {"MAE"},
                      **{tw.LOSS_EVIDENCE[kind]: set(metrics) for kind, metrics in tw.LOSS_METRICS.items()}}


def _shown(names, indices):
    return ", ".join(names[i] for i in indices[:5]) + ("…" if len(indices) > 5 else "")


def _recommend(report, scoreset):
    if report.best < 0:
        return ["No hay puntuaciones definidas con las que recomendar un modelo."]
    names, best = report.names, report.best
    lines = [f"Mejor modelo por {report.metric_label}: {names[best]} "
             f"({report.values[best]:.4g})."]
    if len(names) < 2:
        return lines
    rivals = [index for index in range(len(names)) if index != best
              and np.isfinite(report.values[index])]
    primary = report.pairs_of(report.primary)
    clearly, close, unknown = [], [], []
    for rival in rivals:
        pair, _ = _pair_between(primary, best, rival)
        if pair is None or pair.decision == st.NOT_AVAILABLE:
            unknown.append(rival)
        elif is_clearly_better(report, best, rival):
            clearly.append(rival)
        else:
            close.append(rival)
    label = st.TEST_NAMES.get(report.primary, "")
    evidence = primary[0].evidence if primary else ""
    # McNemar mide aciertos y los tests por filas, error absoluto: solo aportan
    # evidencia sobre la métrica elegida si coincide con lo que miden.
    measured = METRIC_OF_EVIDENCE.get(evidence)
    about_metric = measured is None or report.metric in measured
    if clearly:
        if about_metric:
            lines.append(f"Supera con evidencia ({label}, α = {report.alpha:g}) a: "
                         f"{_shown(names, clearly)}.")
        else:
            lines.append(f"En {evidence} ({label}, α = {report.alpha:g}) supera a: "
                         f"{_shown(names, clearly)}. Ese test no mide {report.metric_label}, así que "
                         "no es evidencia sobre esa métrica.")
    if close:
        where = "" if about_metric else f" en {evidence} (no en {report.metric_label})"
        lines.append(f"No se puede distinguir{where} con estos datos de: {_shown(names, close)}.")
        total = scoreset.train_time + scoreset.test_time
        faster = [i for i in close if total[i] * 2 < total[best]]
        if faster:
            quickest = min(faster, key=lambda i: total[i])
            lines.append(f"Si importa el coste, {names[quickest]} tarda "
                         f"{total[best] / max(total[quickest], 1e-9):.0f} veces menos y no es "
                         f"distinguible del mejor{where}.")
    if unknown:
        lines.append(f"No se ha podido evaluar la comparación con: {_shown(names, unknown)}.")
    bayes = report.pairs_of(st.BAYES)
    if bayes and close:
        rival = close[0]
        pair, flipped = _pair_between(bayes, best, rival)
        if pair is not None and np.isfinite(pair.p_a):
            better, worse = (pair.p_b, pair.p_a) if flipped else (pair.p_a, pair.p_b)
            lines.append(f"Frente a {names[rival]}, la probabilidad bayesiana de que "
                         f"{names[best]} sea mejor es {better:.0%}, de que sean equivalentes "
                         f"{pair.p_rope:.0%} y de que sea peor {worse:.0%}.")
    return lines


# ---------------------------------------------------------------- salidas ----

def _table(attributes, X, metas_vars, metas, name, class_vars=(), Y=None):
    domain = Domain(attributes, class_vars, metas_vars)
    X = np.asarray(X, dtype=float) if len(attributes) else np.empty((len(metas), 0))
    table = Table.from_numpy(domain, X, Y, metas=np.array(metas, dtype=object))
    table.name = name
    return table


def scores_table(scoreset, report):
    labels = [metric_label(scorer) for scorer in scoreset.scorers]
    unique = unique_labels(labels + ["IC inferior", "IC superior", "Rango medio",
                                     "Entrenamiento (s)", "Predicción (s)"])
    columns = [scoreset.overall[scorer.__name__] for scorer in scoreset.scorers]
    columns += [report.ci_low, report.ci_high, report.mean_ranks, scoreset.train_time,
                scoreset.test_time]
    attributes = [ContinuousVariable(name) for name in unique]
    return _table(attributes, np.column_stack(columns), [StringVariable("Modelo")],
                  [[name] for name in scoreset.names], "Puntuaciones por modelo")


def folds_table(scoreset):
    if not scoreset.has_folds:
        return None
    n_models, n_folds = len(scoreset.names), len(scoreset.combined.folds)
    labels = unique_labels([metric_label(scorer) for scorer in scoreset.scorers])
    attributes = [ContinuousVariable(name) for name in labels]
    X = np.column_stack([scoreset.folds[scorer.__name__].T.ravel() for scorer in scoreset.scorers])
    # Columna a columna: una fila por partición y modelo.
    model = np.tile(np.arange(n_models), n_folds).astype(float)
    fold = np.repeat(np.arange(n_folds), n_models).astype(float)
    model_var = DiscreteVariable("Modelo", values=tuple(scoreset.names))
    fold_var = DiscreteVariable("Partición", values=tuple(str(i + 1) for i in range(n_folds)))
    domain = Domain(attributes, model_var, [fold_var])
    table = Table.from_numpy(domain, X, model, metas=fold[:, None].astype(object))
    table.name = "Puntuaciones por partición"
    return table


def pairs_table(report):
    if not report.pairs:
        return None
    numeric = ("Estadístico", "p", "p ajustado", "Diferencia (A − B)", "IC inferior", "IC superior",
               "Efecto", "P(A mejor)", "P(equivalentes)", "P(B mejor)")
    X = [[pair.statistic, pair.p_value, pair.p_adjusted, pair.mean_diff, pair.ci_low,
          pair.ci_high, pair.effect, pair.p_a, pair.p_rope, pair.p_b] for pair in report.pairs]
    tests = sorted({pair.test for pair in report.pairs})
    labels = [st.TEST_NAMES[test] for test in tests]
    decisions = sorted({pair.decision for pair in report.pairs})
    metas_vars = [
        StringVariable("Modelo A"), StringVariable("Modelo B"),
        DiscreteVariable("Test", values=tuple(labels)),
        DiscreteVariable("Decisión", values=tuple(decisions)),
        StringVariable("Métrica"), StringVariable("Unidad de la diferencia"),
        StringVariable("Evidencia sobre"), StringVariable("Tipo de efecto"),
        StringVariable("Interpretación"), StringVariable("Supuesto"), StringVariable("Avisos"),
    ]
    metas = [[report.names[pair.a], report.names[pair.b], float(tests.index(pair.test)),
              float(decisions.index(pair.decision)), report.metric_label, pair.units, pair.evidence,
              pair.effect_name, st.interpret(pair, report.names, report.metric_label, report.alpha),
              pair.assumption, " ".join(pair.warnings)] for pair in report.pairs]
    return _table([ContinuousVariable(name) for name in numeric], X, metas_vars, metas,
                  "Comparaciones por parejas")


def ranks_table(report):
    position = np.empty(len(report.names))
    position[report.order] = np.arange(1, len(report.names) + 1)
    X = np.column_stack((report.mean_ranks, position, report.values))
    attributes = [ContinuousVariable("Rango medio"), ContinuousVariable("Posición"),
                  ContinuousVariable(report.metric_label)]
    return _table(attributes, X, [StringVariable("Modelo")], [[name] for name in report.names],
                  "Rangos")


# ------------------------------------------------------- ventana deslizante ----

def _window_time_vars(plan, names):
    """Variables para las fechas de las ventanas: de tiempo, o números de fila."""
    from Orange.data import TimeVariable
    var = plan.spec.time_var
    if var is None:
        return [ContinuousVariable(name, number_of_decimals=0) for name in names]
    return [TimeVariable(name, have_date=var.have_date, have_time=var.have_time) for name in names]


WINDOW_COLUMNS = ("Inicio entrenamiento", "Fin entrenamiento", "Inicio test", "Fin test")


def _window_bounds(plan, iteration):
    return [plan.value_at(iteration.train_lo), plan.value_at(iteration.train_hi - 1),
            plan.value_at(iteration.test_lo), plan.value_at(iteration.test_hi - 1)]


def window_folds_table(scoresets, plan, target_names):
    """Una fila por modelo, iteración y objetivo, con las fechas de cada ventana."""
    scoresets = [s for s in scoresets if s is not None and s.has_folds]
    if not scoresets:
        return None
    first = scoresets[0]
    labels = unique_labels([metric_label(scorer) for scorer in first.scorers])
    attributes = [ContinuousVariable(name) for name in labels]
    model_var = DiscreteVariable("Modelo", values=tuple(first.names))
    n_iterations = len(first.combined.folds)
    iteration_var = DiscreteVariable("Iteración", values=tuple(str(i + 1) for i in range(n_iterations)))
    target_var = DiscreteVariable("Objetivo", values=tuple(target_names))
    time_vars = _window_time_vars(plan, WINDOW_COLUMNS)
    counts = [ContinuousVariable("Filas de entrenamiento", number_of_decimals=0),
              ContinuousVariable("Filas de test", number_of_decimals=0)]
    X, Y, metas = [], [], []
    for target, scoreset in enumerate(scoresets):
        n_models = len(scoreset.names)
        sizes = scoreset.fold_sizes if scoreset.fold_sizes is not None else np.full(n_iterations, np.nan)
        for fold in range(n_iterations):
            iteration = plan.iterations[fold]
            for model in range(n_models):
                X.append([scoreset.folds[s.__name__][model, fold] for s in scoreset.scorers])
                Y.append(model)
                metas.append([fold, target] + _window_bounds(plan, iteration) + [iteration.n_train, sizes[fold]])
    domain = Domain(attributes, model_var, [iteration_var, target_var] + time_vars + counts)
    table = Table.from_numpy(domain, np.array(X, dtype=float), np.array(Y, dtype=float),
                             metas=np.array(metas, dtype=float).astype(object))
    table.name = "Puntuaciones por iteración"
    return table


def horizon_table(scoresets, target_names):
    """Una fila por modelo y objetivo con todas las métricas globales."""
    scoresets = [s for s in scoresets if s is not None]
    if not scoresets:
        return None
    first = scoresets[0]
    labels = unique_labels([metric_label(scorer) for scorer in first.scorers])
    attributes = [ContinuousVariable(name) for name in labels]
    X, metas = [], []
    for target, scoreset in enumerate(scoresets):
        for model, name in enumerate(scoreset.names):
            X.append([scoreset.overall[s.__name__][model] for s in scoreset.scorers])
            metas.append([name, target_names[target], float(target + 1)])
    metas_vars = [StringVariable("Modelo"), StringVariable("Objetivo"),
                  ContinuousVariable("Horizonte", number_of_decimals=0)]
    domain = Domain(attributes, metas=metas_vars)
    table = Table.from_numpy(domain, np.array(X, dtype=float), metas=np.array(metas, dtype=object))
    table.name = "Puntuaciones por objetivo"
    return table


def add_window_columns(table, results, plan, target_name):
    """Añade a Predictions la iteración, las fechas de su ventana y el objetivo."""
    from Orange.data.util import get_unique_names
    if table is None or plan is None:
        return table
    iteration = np.empty(len(results.row_indices), dtype=int)
    for fold, part in enumerate(results.folds):
        iteration[part] = fold
    proposed = ["Iteración"] + list(WINDOW_COLUMNS) + ["Objetivo"]
    names = get_unique_names(table.domain, proposed)
    new_vars = ([ContinuousVariable(names[0], number_of_decimals=0)] + _window_time_vars(plan, names[1:5])
                + [StringVariable(names[5])])
    bounds = np.array([_window_bounds(plan, plan.iterations[i]) for i in range(len(plan.iterations))])
    values = np.empty((len(iteration), len(new_vars)), dtype=object)
    values[:, 0] = iteration + 1.0
    values[:, 1:5] = bounds[iteration]
    values[:, 5] = target_name
    domain = Domain(table.domain.attributes, table.domain.class_vars, table.domain.metas + tuple(new_vars))
    metas = np.hstack((np.asarray(table.metas, dtype=object).reshape(len(table), -1), values))
    out = Table.from_numpy(domain, table.X, table.Y, metas=metas, W=table.W if table.has_weights() else None)
    out.name = table.name
    return out
