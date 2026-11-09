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
# Dos valores que difieren menos que esto, en relación con la escala de los datos, son
# iguales: 14/15 - 13/15 no da exactamente lo mismo que 13/15 - 12/15.
RELATIVE_TOLERANCE = 1e-12

TTEST_CORRECTED = "corrected_t"
TTEST_PAIRED = "paired_t"
WILCOXON = "wilcoxon"
MCNEMAR = "mcnemar"
BAYES = "bayes"
NEMENYI = "nemenyi"
HOLM_POSTHOC = "friedman_holm"
ROW_TTEST = "row_t"
ROW_WILCOXON = "row_wilcoxon"

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


def _tolerance(*arrays):
    """Tolerancia absoluta, proporcional al mayor valor presente en los datos."""
    scale = 0.0
    for array in arrays:
        array = np.asarray(array, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size:
            scale = max(scale, float(np.abs(finite).max()))
    return RELATIVE_TOLERANCE * scale


def clean_differences(a, b):
    """Diferencias sin ruido de coma flotante: lo que la tolerancia no separa, es igual."""
    tolerance = _tolerance(a, b)
    if tolerance == 0:
        return np.zeros_like(a, dtype=float)
    return np.rint((a - b) / tolerance) * tolerance


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


def _t_family(a, b, ratio, test, assumption, alpha):
    """Base común del t emparejado: `ratio=0` es el t clásico."""
    a, b, removed = paired_values(a, b)
    result = PairTest(test, n=len(a), assumption=assumption,
                      warnings=_undefined_warning(removed))
    n = len(a)
    if n < 2:
        result.warnings.append("Hacen falta al menos 2 particiones comparables.")
        return result
    differences = a - b
    cleaned = clean_differences(a, b)
    if np.ptp(cleaned) == 0:
        differences = cleaned
    mean = float(np.mean(differences))
    variance = float(np.var(differences, ddof=1)) if np.ptp(cleaned) else 0.0
    result.mean_diff = mean
    result.direction = int(np.sign(mean))
    if variance == 0:
        if mean == 0:
            result.statistic, result.p_value = 0.0, 1.0
            result.ci_low = result.ci_high = 0.0
            result.decision = SAME
            result.warnings.append("Los dos modelos dan puntuaciones idénticas en todas "
                                   "las particiones.")
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
    result.effect_name = "g de Hedges"
    result.effect_label = effect_label_cohen(result.effect)
    if n < 5:
        result.warnings.append(f"Solo {n} particiones: la potencia del test es muy baja.")
    return result


def corrected_ttest(a, b, test_train_ratio, alpha=ALPHA_DEFAULT):
    """t remuestreado corregido de Nadeau y Bengio (2003).

    Sustituye la varianza del t clásico, s²/J, por s²·(1/J + n_test/n_train) para tener
    en cuenta que las particiones comparten datos de entrenamiento.
    """
    return _t_family(
        a, b, test_train_ratio, TTEST_CORRECTED,
        "Particiones dependientes con la corrección de Nadeau-Bengio de la varianza; "
        "diferencias aproximadamente normales.", alpha)


def paired_ttest(a, b, alpha=ALPHA_DEFAULT):
    """t clásico. Con particiones de validación cruzada da demasiados positivos."""
    result = _t_family(
        a, b, 0.0, TTEST_PAIRED,
        "Observaciones independientes, que las particiones no cumplen; solo como referencia.",
        alpha)
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


def wilcoxon_test(a, b, test=WILCOXON, assumption=None, alpha=ALPHA_DEFAULT):
    a, b, removed = paired_values(a, b)
    result = PairTest(test, n=len(a), warnings=_undefined_warning(removed), assumption=(
        assumption or "Diferencias simétricas e independientes; las particiones de una "
        "validación cruzada no son independientes, así que el valor p es orientativo."))
    if not len(a):
        result.warnings.append("No hay particiones comparables.")
        return result
    differences = clean_differences(a, b)
    result.mean_diff = float(np.mean(differences))
    nonzero = differences[differences != 0]
    n_zero = len(differences) - len(nonzero)
    if not len(nonzero):
        result.statistic, result.p_value = 0.0, 1.0
        result.decision = SAME
        result.warnings.append("Los dos modelos dan puntuaciones idénticas en todas las "
                               "particiones.")
        return result
    if n_zero:
        result.warnings.append(f"{n_zero} empates (diferencia cero) descartados.")
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
    result.effect_name = "r biserial de rangos"
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


def bayes_correlated(a, b, rope, rho):
    """t bayesiano correlacionado (Corani y Benavoli, 2015).

    En validación cruzada de k particiones ρ = 1/k y el cálculo lo hace baycomp. Con otro
    muestreo se aplica la misma posterior con ρ = n_test / n_total, que baycomp no permite.
    """
    a, b, removed = paired_values(a, b)
    result = PairTest(BAYES, n=len(a), rope=rope, warnings=_undefined_warning(removed), assumption=(
        "Posterior t de Student con correlación entre particiones "
        f"ρ = {rho:.3g}; ROPE = ±{rope:g} en unidades de la métrica."))
    n = len(a)
    if n < 2:
        result.warnings.append("Hacen falta al menos 2 particiones comparables.")
        return result
    cleaned = clean_differences(a, b)
    degenerate = np.ptp(cleaned) == 0
    differences = cleaned if degenerate else a - b
    result.mean_diff = float(np.mean(differences))
    variance = 0.0 if degenerate else float(np.var(differences, ddof=1))
    result.direction = int(np.sign(result.mean_diff))
    if degenerate:
        result.warnings.append("Varianza nula: la posterior es un punto.")
    if not degenerate and abs(rho - 1 / n) < 1e-9:
        import baycomp  # lazy: tarda en importarse y Orange carga todos los widgets al arrancar
        probabilities = baycomp.two_on_single(a, b, rope=rope)
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


def rank_blocks(scores):
    """Rangos por bloque (1 es el mejor) con empates promediados.

    `scores` tiene forma (modelos, bloques). Solo se usan bloques sin valores perdidos.
    """
    scores = np.asarray(scores, dtype=float)
    complete = np.all(np.isfinite(scores), axis=0)
    blocks = scores[:, complete]
    tolerance = _tolerance(blocks)
    if tolerance > 0:
        blocks = np.rint(blocks / tolerance)
    return stats.rankdata(-blocks, axis=0), int((~complete).sum())


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


def friedman_test(scores, alpha=ALPHA_DEFAULT, on_folds=True):
    """Friedman con la corrección por empates (la de scipy) e Iman-Davenport."""
    ranks, removed = rank_blocks(scores)
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


def corrected_interval(values, test_train_ratio, alpha=ALPHA_DEFAULT):
    """Intervalo de confianza de la media por particiones, con Nadeau-Bengio."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    mean = float(np.mean(values))
    tolerance = _tolerance(values)
    if tolerance == 0 or np.ptp(np.rint(values / tolerance)) == 0:
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
        return f"{a} y {b} dan exactamente los mismos resultados."
    if result.decision == NOT_AVAILABLE:
        return "El test no se ha podido evaluar con estos datos."
    if result.decision in (A_BETTER, B_BETTER):
        winner = a if result.decision == A_BETTER else b
        return f"Hay evidencia (α = {alpha:g}) de que {winner} es mejor en {target}."
    return (f"Sin evidencia suficiente (α = {alpha:g}) de que {a} y {b} difieran en {target}; "
            "esto no demuestra que sean iguales.")
