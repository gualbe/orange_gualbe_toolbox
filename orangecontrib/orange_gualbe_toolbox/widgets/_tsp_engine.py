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

KFOLD, FEATURE, SHUFFLE, LOO, ON_TRAIN, ON_TEST = range(6)
FOLD_KINDS = (KFOLD, FEATURE, SHUFFLE)
ROWS_ONCE_KINDS = (KFOLD, FEATURE, LOO, ON_TRAIN, ON_TEST)

LOWER_IS_BETTER = frozenset(("LogLoss", "MSE", "RMSE", "MAE", "MAPE", "SMAPE", "CVRMSE"))
BOUNDED_METRICS = frozenset(("AUC", "CA", "F1", "Precision", "Recall", "Specificity",
                             "MatthewsCorrCoefficient", "R2"))

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

    def describe(self):
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


def check_sampling(data, test_data, sampling):
    """Devuelve (muestreo ajustado, avisos, error). Nada lanza excepciones."""
    notes = []
    n_rows = len(data)
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

    def pairs_of(self, test):
        return [pair for pair in self.pairs if pair.test == test]


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


FLIP_STATISTIC = (st.TTEST_CORRECTED, st.TTEST_PAIRED, st.ROW_TTEST, st.HOLM_POSTHOC)


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
        ratio, rho = scoreset.test_train_ratio, scoreset.rho
        intervals = [st.corrected_interval(row, ratio, alpha) for row in fold_matrix]
        ci_low = np.array([low for low, _ in intervals])
        ci_high = np.array([high for _, high in intervals])
        primary = st.TTEST_CORRECTED
        undefined = scoreset.undefined.get(options.metric, 0)
        if undefined:
            notes.append(f"{metric_label(scorer)} no está definida en {undefined} de "
                         f"{fold_matrix.shape[1]} particiones (se descartan al comparar).")
        if count > 1:
            nominal = {st.TTEST_CORRECTED: [], st.TTEST_PAIRED: [], st.WILCOXON: [], st.BAYES: []}
            for a, b in index_pairs:
                progress()
                x, y = adjusted_folds[a], adjusted_folds[b]
                results = (st.corrected_ttest(x, y, ratio, alpha), st.paired_ttest(x, y, alpha),
                           st.wilcoxon_test(x, y, alpha=alpha), st.bayes_correlated(x, y, rope, rho))
                for result in results:
                    result.a, result.b = a, b
                    nominal[result.test].append(result)
            for key in (st.TTEST_CORRECTED, st.TTEST_PAIRED, st.WILCOXON):
                _adjust(nominal[key], options.correction, alpha)
            for group in nominal.values():
                _describe(group, sign, metric_label(scorer), metric_label(scorer))
            pairs_by_test.update(nominal)
        friedman = st.friedman_test(adjusted_folds, alpha, on_folds=True)
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
    if not classification and count > 1 and fold_matrix is None:
        errors = -np.abs(combined.predicted - combined.actual)
        row_t, row_w = [], []
        assumption = {
            ON_TEST: "Filas de test independientes; se compara el error absoluto por fila.",
            LOO: "Los conjuntos de entrenamiento casi coinciden: filas no del todo independientes.",
            ON_TRAIN: "Filas usadas también para entrenar: no generaliza a datos nuevos.",
        }.get(sampling.kind, "")
        for a, b in index_pairs:
            progress()
            result_t = st.corrected_ttest(errors[a], errors[b], 0.0, alpha)
            result_t.test, result_t.assumption = st.ROW_TTEST, assumption
            result_w = st.wilcoxon_test(errors[a], errors[b], st.ROW_WILCOXON, assumption, alpha)
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


METRIC_OF_EVIDENCE = {"aciertos por fila": "CA", "error absoluto por fila": "MAE"}


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
    about_metric = measured is None or measured == report.metric
    if clearly:
        if about_metric:
            lines.append(f"Supera con evidencia ({label}, α = {report.alpha:g}) a: "
                         f"{_shown(names, clearly)}.")
        else:
            lines.append(f"Aparte de {report.metric_label}, comete menos errores en {evidence} "
                         f"({label}, α = {report.alpha:g}) que: {_shown(names, clearly)}. Ese test "
                         f"no mide {report.metric_label}, así que no es evidencia sobre esa métrica.")
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
               "Efecto (A − B)", "P(A mejor)", "P(equivalentes)", "P(B mejor)")
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
