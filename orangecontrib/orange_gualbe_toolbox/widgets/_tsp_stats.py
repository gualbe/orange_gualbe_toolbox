"""Tests estadísticos de Test & Score Pro (sin Qt ni Orange).

Convención en todo el módulo: las puntuaciones llegan ya orientadas, de modo que
cuanto más alto, mejor (las métricas de error se pasan cambiadas de signo). Por tanto
`a - b > 0` significa que A es mejor que B.

Cada resultado lleva el supuesto en el que se apoya y los avisos concretos de ese
cálculo: la interfaz y el informe los muestran siempre junto al número.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import stats

ALPHA_DEFAULT = 0.05
BAYES_DECISION = 0.95
MIN_FRIEDMAN_BLOCKS = 10
# Dos valores que difieren menos que esto, en relación con su propia escala, son iguales:
# 14/15 - 13/15 no da exactamente lo mismo que 13/15 - 12/15. Siempre se aplica valor a
# valor: una escala global haría que un solo valor enorme borrase las diferencias del resto.
RELATIVE_TOLERANCE = 1e-13
# Para declarar empates en los rangos se exige más: una diferencia constante necesita margen
# para el ruido, pero un empate falso cambia el valor p. El redondeo real queda muy por debajo.
TIE_FACTOR = 0.1

TTEST_CORRECTED = "corrected_t"
TTEST_PAIRED = "paired_t"
WILCOXON = "wilcoxon"
MCNEMAR = "mcnemar"
BAYES = "bayes"
NEMENYI = "nemenyi"
HOLM_POSTHOC = "friedman_holm"
ROW_TTEST = "row_t"
ROW_WILCOXON = "row_wilcoxon"
DIEBOLD_MARIANO = "diebold_mariano"

TEST_NAMES = {
    TTEST_CORRECTED: "t corregido (Nadeau-Bengio)",
    TTEST_PAIRED: "t emparejado sin corregir",
    WILCOXON: "Wilcoxon",
    MCNEMAR: "McNemar",
    BAYES: "t bayesiano correlacionado",
    NEMENYI: "Nemenyi (tras Friedman)",
    HOLM_POSTHOC: "Holm (tras Friedman)",
    ROW_TTEST: "t emparejado por filas (error absoluto)",
    ROW_WILCOXON: "Wilcoxon por filas (error absoluto)",
    DIEBOLD_MARIANO: "Diebold-Mariano (HLN)",
}

CORRECTIONS = {
    "holm": "Holm",
    "bonferroni": "Bonferroni",
    "fdr_bh": "Benjamini-Hochberg (FDR)",
    "none": "Sin corrección",
}

SAME, A_BETTER, B_BETTER, UNDECIDED, NOT_AVAILABLE = (
    "sin diferencia", "A mejor", "B mejor", "indeciso", "no evaluable")


@dataclass
class PairTest:
    test: str
    a: int = -1
    b: int = -1
    n: int = 0
    statistic: float = np.nan
    p_value: float = np.nan
    p_adjusted: float = np.nan
    mean_diff: float = np.nan   # tras run_tests: métrica(A) - métrica(B), en sus unidades
    ci_low: float = np.nan
    ci_high: float = np.nan
    effect: float = np.nan
    effect_name: str = ""
    effect_label: str = ""
    p_a: float = np.nan
    p_rope: float = np.nan
    p_b: float = np.nan
    rope: float = 0.0
    decision: str = NOT_AVAILABLE
    assumption: str = ""
    warnings: List[str] = field(default_factory=list)
    direction: int = 0          # según el estadístico del propio test: +1 si A es mejor
    numeric_tie: bool = False   # iguales solo dentro de la precisión numérica
    units: str = ""             # en qué se mide la diferencia
    evidence: str = ""          # sobre qué aporta evidencia el test


@dataclass
class FriedmanResult:
    n_models: int
    n_blocks: int
    mean_ranks: np.ndarray
    chi2: float = np.nan
    chi2_uncorrected: float = np.nan
    p_value: float = np.nan
    iman_davenport: float = np.nan
    p_iman_davenport: float = np.nan
    critical_difference: float = np.nan
    assumption: str = ""
    warnings: List[str] = field(default_factory=list)
    on_folds: bool = False


def paired_values(a, b):
    """Filtra las posiciones en que alguna de las dos puntuaciones no está definida."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep], int((~keep).sum())


def _paired_with_floor(a, b, floor):
    """Como paired_values, manteniendo alineado un suelo que varíe por partición."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    floor = np.asarray(floor, dtype=float)
    return a[keep], b[keep], int((~keep).sum()), floor[keep] if floor.ndim else floor


def _same_scores_warning(a, b):
    if np.array_equal(a, b):
        return "Los dos modelos dan puntuaciones idénticas en todas las particiones."
    return "Los dos modelos dan puntuaciones iguales, dentro de la precisión numérica, en todas las particiones."


def _tolerances(values, others=None, floor=0.0):
    """Error de redondeo admisible de cada valor: el suyo propio, con un suelo opcional."""
    scale = np.abs(values) if others is None else np.maximum(np.abs(values), np.abs(others))
    return np.maximum(RELATIVE_TOLERANCE * scale, floor)


def _shared_range(values, tolerances):
    """Parte común de los intervalos valor ± tolerancia; vacía si inferior > superior."""
    return float(np.max(values - tolerances)), float(np.min(values + tolerances))


def _group_starts(ordered, tolerances):
    """Marca dónde empieza cada grupo de valores ordenados indistinguibles entre sí.

    Un grupo son valores cuyos intervalos tienen un punto común: cada uno lleva su propio
    error de redondeo y uno grande no se come a los demás.
    """
    lower, upper = ordered - tolerances, ordered + tolerances
    starts = np.r_[True, lower[1:] > upper[:-1]]
    highest_lower = smallest_upper = 0.0
    for position in np.flatnonzero(~starts):
        if starts[position - 1]:
            highest_lower, smallest_upper = lower[position - 1], upper[position - 1]
        highest_lower = max(highest_lower, lower[position])
        smallest_upper = min(smallest_upper, upper[position])
        if highest_lower > smallest_upper:
            starts[position] = True
    return starts


def clean_differences(a, b, floor=0.0):
    """Diferencias sin ruido de coma flotante.

    Si todas las diferencias caben en un mismo valor, la diferencia es constante. Si no, son
    cero las que no superan el redondeo de sus propios valores, y las de magnitud
    indistinguible pasan a ser exactamente iguales, para que los rangos las traten como
    empates. `floor` es el suelo absoluto del error, en unidades de la métrica.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    differences = a - b
    if not len(differences):
        return differences
    tolerances = _tolerances(a, b, floor)
    low, high = _shared_range(differences, tolerances)
    if low <= high:
        return np.full(len(differences), 0.0 if low <= 0 <= high else (low + high) / 2)
    differences[np.abs(differences) <= tolerances] = 0.0
    nonzero = np.flatnonzero(differences)
    if len(nonzero) < 2:
        return differences
    magnitudes = np.abs(differences[nonzero])
    order = np.argsort(magnitudes, kind="stable")
    ordered = magnitudes[order]
    starts = _group_starts(ordered, TIE_FACTOR * tolerances[nonzero][order])
    representative = ordered[np.flatnonzero(starts)][np.cumsum(starts) - 1]
    merged = magnitudes.copy()
    merged[order] = representative
    differences[nonzero] = np.sign(differences[nonzero]) * merged
    return differences


def _nearly_constant(values, floor=0.0):
    low, high = _shared_range(values, _tolerances(values, floor=floor))
    return low <= high


def _rank_block(values, floor=0.0):
    """Rangos de un bloque (1 el mejor); los valores indistinguibles empatan."""
    order = np.argsort(values, kind="stable")[::-1]
    ordered = values[order][::-1]  # de menor a mayor, como espera _group_starts
    floor = np.broadcast_to(floor, values.shape)[order][::-1]
    starts = _group_starts(ordered, TIE_FACTOR * _tolerances(ordered, floor=floor))[::-1]
    # Al recorrer de mayor a menor, un grupo acaba donde el orden ascendente lo empezó.
    groups = np.r_[0, np.cumsum(starts[:-1])]
    ranks = np.empty(len(values))
    ranks[order] = stats.rankdata(groups)
    return ranks


def _undefined_warning(n_removed):
    if not n_removed:
        return []
    return [f"Se han descartado {n_removed} particiones con la métrica sin definir."]


def effect_label_cohen(value):
    size = abs(value)
    if size < 0.2:
        return "insignificante"
    if size < 0.5:
        return "pequeño"
    if size < 0.8:
        return "medio"
    return "grande"


def effect_label_rank(value):
    size = abs(value)
    if size < 0.1:
        return "insignificante"
    if size < 0.3:
        return "pequeño"
    if size < 0.5:
        return "medio"
    return "grande"


def hedges_g(differences):
    """g de Hedges para diferencias emparejadas (d_z con corrección de sesgo)."""
    n = len(differences)
    deviation = np.std(differences, ddof=1)
    if n < 2 or deviation == 0:
        return np.nan
    return float(np.mean(differences) / deviation * (1 - 3 / (4 * (n - 1) - 1)))


def _t_family(a, b, ratio, test, assumption, alpha, floor=0.0):
    """Base común del t emparejado: `ratio=0` es el t clásico."""
    a, b, removed, floor = _paired_with_floor(a, b, floor)
    result = PairTest(test, n=len(a), assumption=assumption,
                      warnings=_undefined_warning(removed))
    n = len(a)
    if n < 2:
        result.warnings.append("Hacen falta al menos 2 particiones comparables.")
        return result
    # Siempre la diferencia limpia: el ruido de una partición enorme no debe pesar.
    differences = clean_differences(a, b, floor)
    mean = float(np.mean(differences))
    variance = float(np.var(differences, ddof=1)) if np.ptp(differences) else 0.0
    result.mean_diff = mean
    result.direction = int(np.sign(mean))
    if variance == 0:
        if mean == 0:
            result.statistic, result.p_value = 0.0, 1.0
            result.ci_low = result.ci_high = 0.0
            result.decision = SAME
            result.numeric_tie = not np.array_equal(a, b)
            result.warnings.append(_same_scores_warning(a, b))
        else:
            result.ci_low = result.ci_high = mean
            result.warnings.append("Varianza nula: la diferencia es constante en todas "
                                   "las particiones y el test no es aplicable.")
        return result
    scale = np.sqrt(variance * (1 / n + ratio))
    df = n - 1
    t_value = mean / scale
    half = stats.t.ppf(1 - alpha / 2, df) * scale
    result.statistic = float(t_value)
    result.p_value = float(2 * stats.t.sf(abs(t_value), df))
    result.ci_low, result.ci_high = mean - half, mean + half
    result.effect = hedges_g(differences)
    result.effect_name = "g de Hedges, A − B"
    result.effect_label = effect_label_cohen(result.effect)
    if n < 5:
        result.warnings.append(f"Solo {n} particiones: la potencia del test es muy baja.")
    return result


def corrected_ttest(a, b, test_train_ratio, alpha=ALPHA_DEFAULT, floor=0.0):
    """t remuestreado corregido de Nadeau y Bengio (2003).

    Sustituye la varianza del t clásico, s²/J, por s²·(1/J + n_test/n_train) para tener
    en cuenta que las particiones comparten datos de entrenamiento.
    """
    return _t_family(
        a, b, test_train_ratio, TTEST_CORRECTED,
        "Particiones dependientes con la corrección de Nadeau-Bengio de la varianza; "
        "diferencias aproximadamente normales.", alpha, floor)


def paired_ttest(a, b, alpha=ALPHA_DEFAULT, floor=0.0):
    """t clásico. Con particiones de validación cruzada da demasiados positivos."""
    result = _t_family(
        a, b, 0.0, TTEST_PAIRED,
        "Observaciones independientes, que las particiones no cumplen; solo como referencia.",
        alpha, floor)
    result.warnings.append("Sin corregir: las particiones no son independientes y este "
                           "test da demasiados falsos positivos. Usa el t corregido.")
    return result


def signed_ranks(differences, zero_method="wilcox"):
    """Sumas de rangos positivos y negativos de Wilcoxon, calculadas a mano."""
    differences = np.asarray(differences, dtype=float)
    if zero_method == "wilcox":
        differences = differences[differences != 0]
    if not len(differences):
        return 0.0, 0.0
    ranks = stats.rankdata(np.abs(differences))
    positive = float(ranks[differences > 0].sum())
    negative = float(ranks[differences < 0].sum())
    if zero_method == "zsplit":
        zero = float(ranks[differences == 0].sum())
        positive += zero / 2
        negative += zero / 2
    return positive, negative


def wilcoxon_test(a, b, test=WILCOXON, assumption=None, alpha=ALPHA_DEFAULT, floor=0.0):
    a, b, removed, floor = _paired_with_floor(a, b, floor)
    result = PairTest(test, n=len(a), warnings=_undefined_warning(removed), assumption=(
        assumption or "Diferencias simétricas e independientes; las particiones de una "
        "validación cruzada no son independientes, así que el valor p es orientativo."))
    if not len(a):
        result.warnings.append("No hay particiones comparables.")
        return result
    differences = clean_differences(a, b, floor)
    result.mean_diff = float(np.mean(differences))
    nonzero = differences[differences != 0]
    n_zero = len(differences) - len(nonzero)
    if not len(nonzero):
        result.statistic, result.p_value = 0.0, 1.0
        result.decision = SAME
        result.numeric_tie = not np.array_equal(a, b)
        result.warnings.append(_same_scores_warning(a, b))
        return result
    if n_zero:
        exact = int(np.sum(a == b)) == n_zero
        what = "diferencia cero" if exact else "diferencia nula dentro de la precisión numérica"
        result.warnings.append(f"{n_zero} empates ({what}) descartados.")
    ties = len(np.unique(np.abs(nonzero))) < len(nonzero)
    if ties:
        result.warnings.append("Hay diferencias empatadas: se usa la aproximación normal.")
    exact = len(nonzero) <= 50 and not ties
    outcome = stats.wilcoxon(nonzero, method="exact" if exact else "approx",
                             correction=not exact)
    result.statistic = float(outcome.statistic)
    result.p_value = float(outcome.pvalue)
    positive, negative = signed_ranks(nonzero)
    result.direction = int(np.sign(positive - negative))
    result.effect = (positive - negative) / (positive + negative)
    result.effect_name = "r biserial de rangos, A − B"
    result.effect_label = effect_label_rank(result.effect)
    smallest = 2 / 2 ** len(nonzero)
    if smallest > alpha:
        result.warnings.append(
            f"Con {len(nonzero)} diferencias no nulas el menor valor p posible es "
            f"{smallest:.3f}: nunca podrá ser significativo con α = {alpha:g}.")
    return result


def mcnemar_test(correct_a, correct_b, assumption=None, alpha=ALPHA_DEFAULT):
    """McNemar sobre aciertos por fila. Binomial exacto si hay menos de 25 discordancias."""
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    n = len(correct_a)
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    result = PairTest(MCNEMAR, n=n, assumption=(
        assumption or "Filas independientes, cada una predicha una sola vez; compara "
        "aciertos, no probabilidades."))
    discordant = only_a + only_b
    if not n:
        result.warnings.append("No hay filas que comparar.")
        return result
    result.mean_diff = (only_a - only_b) / n
    result.direction = int(np.sign(only_a - only_b))
    if not discordant:
        result.statistic, result.p_value = 0.0, 1.0
        result.decision = SAME
        result.warnings.append("Los dos modelos aciertan y fallan en las mismas filas.")
        return result
    if discordant < 25:
        result.statistic = float(min(only_a, only_b))
        result.p_value = float(stats.binomtest(min(only_a, only_b), discordant, 0.5).pvalue)
        result.warnings.append(f"Solo {discordant} filas discordantes: binomial exacto.")
    else:
        result.statistic = (abs(only_a - only_b) - 1) ** 2 / discordant
        result.p_value = float(stats.chi2.sf(result.statistic, 1))
    result.effect = (only_a + 0.5) / (only_b + 0.5)
    result.effect_name = "razón de discordancias (A/B)"
    ratio = max(result.effect, 1 / result.effect)
    result.effect_label = ("pequeño" if ratio < 1.5 else "medio" if ratio < 3 else "grande")
    spread = np.sqrt(max(discordant - (only_a - only_b) ** 2 / n, 0)) / n
    critical = stats.norm.ppf(1 - alpha / 2)
    result.ci_low = result.mean_diff - critical * spread
    result.ci_high = result.mean_diff + critical * spread
    return result


def bayes_correlated(a, b, rope, rho, floor=0.0):
    """t bayesiano correlacionado (Corani y Benavoli, 2015).

    En validación cruzada de k particiones ρ = 1/k y el cálculo lo hace baycomp. Con otro
    muestreo se aplica la misma posterior con ρ = n_test / n_total, que baycomp no permite.
    """
    a, b, removed, floor = _paired_with_floor(a, b, floor)
    result = PairTest(BAYES, n=len(a), rope=rope, warnings=_undefined_warning(removed), assumption=(
        "Posterior t de Student con correlación entre particiones "
        f"ρ = {rho:.3g}; ROPE = ±{rope:g} en unidades de la métrica."))
    n = len(a)
    if n < 2:
        result.warnings.append("Hacen falta al menos 2 particiones comparables.")
        return result
    differences = clean_differences(a, b, floor)
    degenerate = np.ptp(differences) == 0
    result.mean_diff = float(np.mean(differences))
    variance = 0.0 if degenerate else float(np.var(differences, ddof=1))
    result.direction = int(np.sign(result.mean_diff))
    if degenerate:
        result.warnings.append("Varianza nula: la posterior es un punto.")
    if not degenerate and abs(rho - 1 / n) < 1e-9:
        import baycomp  # lazy: tarda en importarse y Orange carga todos los widgets al arrancar
        probabilities = baycomp.two_on_single(differences, np.zeros(n), rope=rope)
        if rope > 0:
            p_a, p_rope, p_b = probabilities
        else:
            (p_a, p_b), p_rope = probabilities, 0.0
    else:
        p_a, p_rope, p_b = _student_posterior(result.mean_diff, variance, n, rope, rho)
    result.p_a, result.p_rope, result.p_b = float(p_a), float(p_rope), float(p_b)
    return _decide_bayes(result)


def _student_posterior(mean, variance, n, rope, rho):
    scale = np.sqrt(variance * (1 / n + rho / (1 - rho)))
    if scale == 0:
        p_a = float(mean > rope) if rope > 0 else float(mean > 0) + 0.5 * (mean == 0)
        p_b = float(mean < -rope) if rope > 0 else 1 - p_a
        return p_a, 1 - p_a - p_b, p_b
    df = n - 1
    p_b = stats.t.cdf(-rope, df, mean, scale)
    p_a = 1 - stats.t.cdf(rope, df, mean, scale)
    return p_a, 1 - p_a - p_b, p_b


def _decide_bayes(result):
    if not np.isfinite(result.p_a):
        result.decision = NOT_AVAILABLE
    elif result.p_a >= BAYES_DECISION:
        result.decision = A_BETTER
    elif result.p_b >= BAYES_DECISION:
        result.decision = B_BETTER
    elif result.p_rope >= BAYES_DECISION:
        result.decision = "equivalentes"
    else:
        result.decision = UNDECIDED
    return result


def long_run_variance(differences, lags):
    """Varianza a largo plazo de una serie: γ₀ + 2·Σ γₖ, k = 1…lags (núcleo rectangular).

    Las autocovarianzas se centran en la media y se dividen por T, como en Diebold y Mariano
    (1995) y en `forecast::dm.test` de R.
    """
    d = np.asarray(differences, dtype=float)
    n = len(d)
    centred = d - d.mean()
    gammas = [float(centred @ centred) / n]
    for lag in range(1, min(lags, n - 1) + 1):
        gammas.append(float(centred[lag:] @ centred[:-lag]) / n)
    return gammas[0] + 2 * sum(gammas[1:]), gammas


def diebold_mariano(loss_a, loss_b, horizon=1, alpha=ALPHA_DEFAULT, floor=0.0):
    """Diebold-Mariano con la corrección de Harvey, Leybourne y Newbold (1997).

    `loss_a` y `loss_b` son las pérdidas por instancia de test, en orden temporal (menos es
    mejor). Por dentro la diferencia se orienta como el resto del módulo: d = pérdida(B) −
    pérdida(A), positiva si A es mejor. La varianza de la media usa las autocovarianzas hasta
    el retardo h − 1, porque los errores de un pronóstico a h pasos son un MA(h − 1). Si esa
    varianza sale negativa, se repite con h = 1, como hace `forecast::dm.test`.
    """
    a, b = np.asarray(loss_a, dtype=float), np.asarray(loss_b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    floor = np.asarray(floor, dtype=float)
    floor = floor[keep] if floor.ndim else floor
    a, b = a[keep], b[keep]
    horizon = max(int(horizon), 1)
    result = PairTest(DIEBOLD_MARIANO, n=len(a), assumption=(
        "Diferencia de pérdidas estacionaria, con autocorrelación hasta el retardo h − 1 "
        f"(h = {horizon}); corrección de Harvey, Leybourne y Newbold para muestras pequeñas."))
    removed = int((~keep).sum())
    if removed:
        result.warnings.append(f"Se han descartado {removed} filas con la pérdida sin definir.")
    n = len(a)
    if n < 3:
        result.warnings.append("Hacen falta al menos 3 instancias de test comparables.")
        return result
    differences = clean_differences(b, a, floor)
    mean = float(np.mean(differences))
    result.mean_diff = mean
    result.direction = int(np.sign(mean))
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    if mean_b > 0:
        result.effect = mean_a / mean_b
        result.effect_name = "razón de pérdidas medias (A/B)"
        ratio = max(result.effect, 1 / result.effect) if result.effect > 0 else np.inf
        result.effect_label = ("insignificante" if ratio < 1.05 else "pequeño" if ratio < 1.25
                               else "medio" if ratio < 2 else "grande")
    if not np.ptp(differences):
        if mean == 0:
            result.statistic, result.p_value = 0.0, 1.0
            result.ci_low = result.ci_high = 0.0
            result.decision = SAME
            result.numeric_tie = not np.array_equal(a, b)
            result.warnings.append("Los dos modelos tienen la misma pérdida en todas las instancias"
                                   + ("." if np.array_equal(a, b) else ", dentro de la precisión numérica."))
        else:
            result.ci_low = result.ci_high = mean
            result.warnings.append("La diferencia de pérdidas es constante: el test no es aplicable.")
        return result
    used = horizon
    variance, _ = long_run_variance(differences, used - 1)
    if variance <= 0 and used > 1:
        result.warnings.append(f"La varianza a largo plazo con h = {used} sale negativa: se usa h = 1.")
        used = 1
        variance, _ = long_run_variance(differences, 0)
    correction = (n + 1 - 2 * used + used * (used - 1) / n) / n
    if correction <= 0:
        result.warnings.append(f"Con {n} instancias no se puede corregir para h = {used}.")
        return result
    scale = np.sqrt(variance / n)
    statistic = mean / scale * np.sqrt(correction)
    df = n - 1
    result.statistic = float(statistic)
    result.p_value = float(2 * stats.t.sf(abs(statistic), df))
    half = stats.t.ppf(1 - alpha / 2, df) * scale / np.sqrt(correction)
    result.ci_low, result.ci_high = mean - half, mean + half
    if n < 30:
        result.warnings.append(f"Solo {n} instancias de test: la potencia del test es baja.")
    return result


def adjust_pvalues(values, method):
    """Corrección por comparaciones múltiples; los NaN no cuentan como comparaciones."""
    values = np.asarray(values, dtype=float)
    adjusted = values.copy()
    valid = np.flatnonzero(np.isfinite(values))
    m = len(valid)
    if not m or method == "none":
        return adjusted
    ordered = valid[np.argsort(values[valid], kind="stable")]
    sorted_values = values[ordered]
    if method == "bonferroni":
        adjusted[valid] = np.minimum(values[valid] * m, 1)
    elif method == "holm":
        scaled = sorted_values * (m - np.arange(m))
        adjusted[ordered] = np.minimum(np.maximum.accumulate(scaled), 1)
    elif method == "fdr_bh":
        scaled = sorted_values * m / np.arange(1, m + 1)
        adjusted[ordered] = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1)
    else:
        raise ValueError(f"Corrección desconocida: {method}")
    return adjusted


def decide_frequentist(result, alpha):
    """Decisión a partir del valor p ajustado."""
    if result.decision == SAME:
        return result
    if not np.isfinite(result.p_adjusted):
        result.decision = NOT_AVAILABLE
    elif result.p_adjusted >= alpha:
        result.decision = "sin diferencia significativa"
    elif result.direction == 0:
        result.decision = "sin diferencia significativa"
    else:
        # Cada test decide con su propio estadístico: en Wilcoxon, los rangos.
        result.decision = A_BETTER if result.direction > 0 else B_BETTER
    return result


def rank_blocks(scores, floor=0.0):
    """Rangos por bloque (1 es el mejor) con empates promediados.

    `scores` tiene forma (modelos, bloques). Solo se usan bloques sin valores perdidos.
    """
    scores = np.asarray(scores, dtype=float)
    complete = np.all(np.isfinite(scores), axis=0)
    blocks = scores[:, complete]
    if not blocks.size:
        return blocks, int((~complete).sum())
    floors = np.broadcast_to(floor, scores.shape)[:, complete]
    return (np.column_stack([_rank_block(column, limit) for column, limit in zip(blocks.T, floors.T)]),
            int((~complete).sum()))


def friedman_from_mean_ranks(mean_ranks, n):
    """χ² de Friedman sin corregir por empates, la fórmula de Demšar (2006)."""
    mean_ranks = np.asarray(mean_ranks, dtype=float)
    k = len(mean_ranks)
    return 12 * n / (k * (k + 1)) * (np.sum(mean_ranks ** 2) - k * (k + 1) ** 2 / 4)


def friedman_statistics(ranks):
    return friedman_from_mean_ranks(ranks.mean(axis=1), ranks.shape[1])


def iman_davenport(chi2, n, k):
    denominator = n * (k - 1) - chi2
    if denominator <= RELATIVE_TOLERANCE * n * (k - 1):
        return np.nan  # concordancia perfecta: el estadístico no está definido
    return (n - 1) * chi2 / denominator


def nemenyi_q(k, alpha=ALPHA_DEFAULT):
    """Valor crítico q_α de Nemenyi: rango estudentizado con gl infinitos dividido por √2."""
    return float(stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))


def critical_difference(k, n, alpha=ALPHA_DEFAULT):
    return nemenyi_q(k, alpha) * np.sqrt(k * (k + 1) / (6 * n))


FRIEDMAN_ON_FOLDS = (
    "Friedman y Nemenyi se concibieron para comparar modelos en varios conjuntos de datos. "
    "Aquí los bloques son particiones de un solo conjunto, que no son independientes: "
    "úsalos como descripción, no como prueba de generalización.")


def friedman_test(scores, alpha=ALPHA_DEFAULT, on_folds=True, floor=0.0):
    """Friedman con la corrección por empates (la de scipy) e Iman-Davenport."""
    ranks, removed = rank_blocks(scores, floor)
    k, n = ranks.shape
    result = FriedmanResult(n_models=k, n_blocks=n, mean_ranks=ranks.mean(axis=1) if n else
                            np.full(k, np.nan), assumption=(
        "Bloques independientes; el estadístico se aproxima por χ²."))
    result.on_folds = bool(on_folds)
    if on_folds:
        result.warnings.append(FRIEDMAN_ON_FOLDS)
    if removed:
        result.warnings.append(f"Se han descartado {removed} particiones con la métrica "
                               "sin definir en algún modelo.")
    if k < 3:
        result.warnings.append("Friedman necesita al menos 3 modelos.")
        return result
    if n < 2:
        result.warnings.append("Friedman necesita al menos 2 particiones comparables.")
        return result
    if n < MIN_FRIEDMAN_BLOCKS:
        result.warnings.append(f"Solo {n} bloques (se recomiendan al menos "
                               f"{MIN_FRIEDMAN_BLOCKS}): la aproximación es imprecisa.")
    result.critical_difference = float(critical_difference(k, n, alpha))
    result.chi2_uncorrected = float(friedman_statistics(ranks))
    if np.all(np.ptp(ranks, axis=0) == 0):
        result.chi2, result.p_value = 0.0, 1.0
        result.iman_davenport, result.p_iman_davenport = 0.0, 1.0
        result.warnings.append("Todos los modelos empatan en todas las particiones.")
        return result
    outcome = stats.friedmanchisquare(*ranks)
    result.chi2, result.p_value = float(outcome.statistic), float(outcome.pvalue)
    result.iman_davenport = float(iman_davenport(result.chi2, n, k))
    if np.isnan(result.iman_davenport):
        result.warnings.append("Iman-Davenport no está definido porque los rangos son casi "
                               "idénticos en todos los bloques: usa el χ² de Friedman.")
    else:
        result.p_iman_davenport = float(stats.f.sf(result.iman_davenport, k - 1, (k - 1) * (n - 1)))
    return result


def nemenyi_pairs(mean_ranks, n, alpha=ALPHA_DEFAULT, on_folds=True):
    """Comparaciones por parejas de Nemenyi a partir de los rangos medios."""
    k = len(mean_ranks)
    scale = np.sqrt(k * (k + 1) / (12 * n))
    difference = critical_difference(k, n, alpha)
    out = []
    for a in range(k):
        for b in range(a + 1, k):
            delta = mean_ranks[b] - mean_ranks[a]  # el rango menor es el mejor
            q_value = abs(delta) / scale
            result = PairTest(NEMENYI, a, b, n=n, statistic=float(q_value), mean_diff=float(delta),
                              direction=int(np.sign(delta)),
                              assumption=f"Bloques independientes; CD = {difference:.3f}.",
                              warnings=[FRIEDMAN_ON_FOLDS] if on_folds else [])
            result.p_value = float(stats.studentized_range.sf(q_value, k, np.inf))
            result.p_adjusted = result.p_value  # Nemenyi ya controla el error de la familia
            if abs(delta) > difference:
                result.decision = A_BETTER if delta > 0 else B_BETTER
            else:
                result.decision = "sin diferencia significativa"
            out.append(result)
    return out


def holm_pairs(mean_ranks, n, on_folds=True):
    """Post-hoc de Holm sobre los z de Friedman (sin ajustar: lo ajusta quien llama)."""
    k = len(mean_ranks)
    scale = np.sqrt(k * (k + 1) / (6 * n))
    out = []
    for a in range(k):
        for b in range(a + 1, k):
            delta = mean_ranks[b] - mean_ranks[a]
            z_value = delta / scale
            result = PairTest(HOLM_POSTHOC, a, b, n=n, statistic=float(z_value),
                              p_value=float(2 * stats.norm.sf(abs(z_value))),
                              mean_diff=float(delta), direction=int(np.sign(delta)),
                              assumption="Bloques independientes; z de los rangos medios.",
                              warnings=[FRIEDMAN_ON_FOLDS] if on_folds else [])
            out.append(result)
    return out


def corrected_interval(values, test_train_ratio, alpha=ALPHA_DEFAULT, floor=0.0):
    """Intervalo de confianza de la media por particiones, con Nadeau-Bengio."""
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    floor = np.asarray(floor, dtype=float)
    floor = floor[keep] if floor.ndim else floor
    values = values[keep]
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    mean = float(np.mean(values))
    if _nearly_constant(values, floor):
        return mean, mean
    scale = np.sqrt(np.var(values, ddof=1) * (1 / n + test_train_ratio))
    half = stats.t.ppf(1 - alpha / 2, n - 1) * scale
    return mean - half, mean + half


def interpret(result, names, metric, alpha):
    """Una frase en lenguaje llano con la lectura del resultado."""
    a, b = names[result.a], names[result.b]
    target = result.evidence or metric
    if result.test == BAYES:
        if not np.isfinite(result.p_a):
            return "No se ha podido calcular."
        return (f"Probabilidad de que {a} sea mejor en {target}: {result.p_a:.1%}; de que sean "
                f"equivalentes: {result.p_rope:.1%}; de que lo sea {b}: {result.p_b:.1%}.")
    if result.decision == SAME:
        if result.numeric_tie:
            return f"{a} y {b} dan los mismos resultados, dentro de la precisión numérica."
        return f"{a} y {b} dan exactamente los mismos resultados."
    if result.decision == NOT_AVAILABLE:
        return "El test no se ha podido evaluar con estos datos."
    if result.decision in (A_BETTER, B_BETTER):
        winner = a if result.decision == A_BETTER else b
        return f"Hay evidencia (α = {alpha:g}) de que {winner} es mejor en {target}."
    return (f"Sin evidencia suficiente (α = {alpha:g}) de que {a} y {b} difieran en {target}; "
            "esto no demuestra que sean iguales.")
