"""Modelagem: métricas, validação temporal e definição dos modelos (Fases 4 e 5).

Estratégia da regressão
-----------------------
O alvo de treino é o **crescimento do PIB em log**, ``g_t = log(PIB_t / PIB_t-1)``.
O PIB previsto é reconstruído a partir do PIB conhecido do ano anterior::

    PIB_previsto_t = PIB_t-1 × exp(g_previsto_t)

Por que modelar o crescimento e não o nível:

* o nível é dominado pelo tamanho do município (Curitiba é 10.000× maior que o
  menor município), e qualquer modelo acertaria a ordem de grandeza só copiando
  o PIB do ano anterior;
* o erro percentual do nível é quase igual ao erro absoluto do crescimento em log,
  ``|PIB_prev/PIB - 1| = |exp(g_prev - g) - 1| ≈ |g_prev - g|``, então errar pouco
  no crescimento é exatamente o que reduz o MAPE.

Validação
---------
Sempre **temporal**, nunca aleatória: para cada ano de validação v, o modelo
treina com os alvos de ``inicio`` a ``v-1`` e é avaliado em ``v``. O ano de
**2023 fica fora de toda escolha de modelo** e só é usado no teste final (Etapa A).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.features import colunas_modelo, preprocessador

SEMENTE = 42
ANOS_VALIDACAO = [2019, 2020, 2021, 2022]
ANO_TESTE = 2023


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def mape(y: np.ndarray, y_prev: np.ndarray) -> float:
    """Erro percentual absoluto médio (%)."""
    y, y_prev = np.asarray(y, float), np.asarray(y_prev, float)
    return float(np.mean(np.abs(y_prev - y) / np.abs(y)) * 100)


def metricas_regressao(pib: np.ndarray, pib_prev: np.ndarray, pib_l1: np.ndarray) -> dict:
    """Métricas do enunciado (R² e MAPE do PIB) e complementares.

    * ``R2_pib``: R² no nível do PIB (exigido). É dominado pelos maiores municípios.
    * ``MAPE_%``: erro percentual absoluto médio do PIB (exigido). Cada município pesa igual.
    * ``MdAPE_%``: mediana do erro percentual, menos sensível a poucos casos extremos.
    * ``R2_log_pib``: R² no log do PIB, que equilibra municípios grandes e pequenos.
    * ``R2_crescimento``: R² do crescimento em log; mede o que o modelo realmente aprende
      além de copiar o PIB do ano anterior.
    """
    pib, pib_prev, pib_l1 = (np.asarray(a, float) for a in (pib, pib_prev, pib_l1))
    erro_pct = np.abs(pib_prev - pib) / pib * 100
    return {
        "R2_pib": r2_score(pib, pib_prev),
        "MAPE_%": float(erro_pct.mean()),
        "MdAPE_%": float(np.median(erro_pct)),
        "R2_log_pib": r2_score(np.log(pib), np.log(pib_prev)),
        "R2_crescimento": r2_score(np.log(pib / pib_l1), np.log(pib_prev / pib_l1)),
    }


# ---------------------------------------------------------------------------
# Modelos de regressão
# ---------------------------------------------------------------------------

def _pipeline(modelo, numericas, categoricas):
    from sklearn.pipeline import Pipeline
    return Pipeline([("prep", preprocessador(numericas, categoricas)), ("modelo", modelo)])


def fabrica_ridge(numericas, categoricas, alpha=1.0):
    from sklearn.linear_model import Ridge
    return _pipeline(Ridge(alpha=alpha), numericas, categoricas)


def fabrica_huber(numericas, categoricas, alpha=1e-3, epsilon=1.35):
    """Regressão linear robusta: perda de Huber, que é quadrática para erros pequenos e
    linear para erros grandes, de modo que os saltos extremos pesam menos."""
    from sklearn.linear_model import HuberRegressor
    return _pipeline(HuberRegressor(alpha=alpha, epsilon=epsilon, max_iter=2000), numericas, categoricas)


def fabrica_random_forest(numericas, categoricas, **params):
    from sklearn.ensemble import RandomForestRegressor
    base = dict(n_estimators=300, n_jobs=-1, random_state=SEMENTE)
    return _pipeline(RandomForestRegressor(**{**base, **params}), numericas, categoricas)


def fabrica_lightgbm(numericas, categoricas, **params):
    from lightgbm import LGBMRegressor
    base = dict(n_estimators=600, learning_rate=0.03, random_state=SEMENTE, n_jobs=-1, verbose=-1,
                subsample_freq=1)
    return _pipeline(LGBMRegressor(**{**base, **params}), numericas, categoricas)


FABRICAS_REGRESSAO: dict[str, Callable] = {
    "Ridge": fabrica_ridge,
    "Huber": fabrica_huber,
    "Random Forest": fabrica_random_forest,
    "LightGBM": fabrica_lightgbm,
}

GRADES_REGRESSAO: dict[str, dict[str, list]] = {
    "Ridge": {"alpha": [0.1, 1, 10, 100, 1000]},
    "Huber": {"alpha": [1e-4, 1e-2, 1], "epsilon": [1.1, 1.35, 2.0]},
    "Random Forest": {"min_samples_leaf": [5, 20, 50], "max_features": [0.33, 0.6]},
    "LightGBM": {
        "objective": ["l2", "huber", "l1"],
        "num_leaves": [15, 31],
        "min_child_samples": [20, 60],
        "colsample_bytree": [0.6],
        "subsample": [0.8],
        "reg_lambda": [0.0, 5.0],
    },
}


def combinacoes(grade: dict[str, list]) -> list[dict]:
    chaves = list(grade)
    return [dict(zip(chaves, valores)) for valores in product(*(grade[c] for c in chaves))]


# ---------------------------------------------------------------------------
# Baselines (sem aprendizado)
# ---------------------------------------------------------------------------

def baseline_persistencia(df: pd.DataFrame) -> np.ndarray:
    """PIB_t = PIB_t-1 (crescimento zero)."""
    return np.zeros(len(df))


def baseline_media_3a(df: pd.DataFrame) -> np.ndarray:
    """PIB_t = PIB_t-1 × crescimento médio anual do município nos 3 anos anteriores."""
    return df["cresc_pib_media_3a"].fillna(df["cresc_pib_media_3a"].median()).to_numpy()


def baseline_uf_ano_anterior(df: pd.DataFrame) -> np.ndarray:
    """PIB_t = PIB_t-1 × crescimento do PIB da UF no ano anterior."""
    return df["cresc_pib_uf_l1"].to_numpy()


BASELINES = {
    "Persistência (PIB_t = PIB_t-1)": baseline_persistencia,
    "Média 3 anos do município": baseline_media_3a,
    "Crescimento da UF em t-1": baseline_uf_ano_anterior,
}


# ---------------------------------------------------------------------------
# Treino, validação temporal e teste
# ---------------------------------------------------------------------------

def amostra_modelagem(dataset: pd.DataFrame, inicio: int, fim: int) -> pd.DataFrame:
    """Linhas com alvo conhecido e anos-alvo em [inicio, fim]."""
    m = dataset["ano"].between(inicio, fim) & dataset["cresc_pib_log"].notna() & dataset["pib_l1"].notna()
    return dataset[m]


def treina_preve(fabrica: Callable, params: dict, treino: pd.DataFrame, alvo: pd.DataFrame,
                 numericas: list[str], categoricas: list[str], clip_treino: tuple | None = None):
    """Treina no crescimento em log e devolve (crescimento previsto, modelo ajustado).

    ``clip_treino`` limita o alvo **só no treino** (winsorização), para reduzir a influência
    dos saltos extremos. O teste é sempre avaliado contra o valor real.
    """
    modelo = fabrica(numericas, categoricas, **params)
    y = treino["cresc_pib_log"]
    if clip_treino is not None:
        y = y.clip(*clip_treino)
    modelo.fit(treino[numericas + categoricas], y)
    return modelo.predict(alvo[numericas + categoricas]), modelo


def avalia_previsao(alvo: pd.DataFrame, cresc_prev: np.ndarray) -> dict:
    pib_prev = alvo["pib_l1"].to_numpy() * np.exp(cresc_prev)
    return metricas_regressao(alvo["pib"], pib_prev, alvo["pib_l1"])


@dataclass
class ResultadoValidacao:
    modelo: str
    params: dict
    por_ano: pd.DataFrame = field(repr=False)

    @property
    def media(self) -> pd.Series:
        return self.por_ano.mean(numeric_only=True)


class EnsembleMedia:
    """Média simples das previsões de vários modelos já ajustados (mesmas entradas)."""

    def __init__(self, modelos: dict):
        self.modelos = modelos

    def fit(self, X, y=None):
        """Os componentes já chegam ajustados. Existe só para a interface do scikit-learn
        (ex.: ``permutation_importance``, que exige ``fit``)."""
        return self

    def predict(self, X):
        return np.mean([mod.predict(X) for mod in self.modelos.values()], axis=0)


# Ensemble avaliado como mais um candidato: componentes e seus hiperparâmetros vêm da busca.
COMPONENTES_ENSEMBLE = ["Ridge", "Random Forest", "LightGBM"]
NOME_ENSEMBLE = "Ensemble (Ridge + RF + LightGBM)"


def treina_preve_modelo(nome: str, params: dict, treino: pd.DataFrame, alvo: pd.DataFrame,
                        numericas: list[str], categoricas: list[str]):
    """Como :func:`treina_preve`, mas aceita também o ensemble.

    Para o ensemble, ``params`` é ``{componente: hiperparâmetros}``.
    """
    if nome == NOME_ENSEMBLE:
        ajustados = {}
        for comp in COMPONENTES_ENSEMBLE:
            _, ajustados[comp] = treina_preve(FABRICAS_REGRESSAO[comp], params[comp], treino, alvo, numericas, categoricas)
        ens = EnsembleMedia(ajustados)
        return ens.predict(alvo[numericas + categoricas]), ens
    return treina_preve(FABRICAS_REGRESSAO[nome], params, treino, alvo, numericas, categoricas)


def valida_temporal(nome: str, params: dict, dataset: pd.DataFrame, inicio: int,
                    anos_validacao: list[int] = ANOS_VALIDACAO, clip_treino: tuple | None = None,
                    incluir_brasil: bool = False) -> ResultadoValidacao:
    """Janela expansiva: para cada ano v, treina com [inicio, v-1] e avalia em v."""
    numericas, categoricas = colunas_modelo(dataset, incluir_brasil=incluir_brasil)
    linhas = []
    for v in anos_validacao:
        treino = amostra_modelagem(dataset, inicio, v - 1)
        val = amostra_modelagem(dataset, v, v)
        if nome in BASELINES:
            cresc = BASELINES[nome](val)
        elif nome == NOME_ENSEMBLE:
            cresc, _ = treina_preve_modelo(nome, params, treino, val, numericas, categoricas)
        else:
            cresc, _ = treina_preve(FABRICAS_REGRESSAO[nome], params, treino, val, numericas, categoricas, clip_treino)
        linhas.append({"ano_validacao": v, "n_treino": len(treino), **avalia_previsao(val, cresc)})
    return ResultadoValidacao(nome, params, pd.DataFrame(linhas).set_index("ano_validacao"))


def busca_hiperparametros(nome: str, dataset: pd.DataFrame, inicio: int, grade: dict | None = None,
                          criterio: str = "MAPE_%", clip_treino: tuple | None = None, verbose: bool = True
                          ) -> pd.DataFrame:
    """Avalia todas as combinações da grade na validação temporal e ordena pelo critério (média dos anos)."""
    grade = GRADES_REGRESSAO[nome] if grade is None else grade
    linhas = []
    for params in combinacoes(grade):
        t0 = time.time()
        res = valida_temporal(nome, params, dataset, inicio, clip_treino=clip_treino)
        linhas.append({"modelo": nome, "params": params, **res.media.drop("n_treino").to_dict(),
                       "MAPE_pior_ano_%": res.por_ano["MAPE_%"].max(), "segundos": time.time() - t0})
        if verbose:
            print(f"  {nome} {params} -> MAPE médio {linhas[-1]['MAPE_%']:.2f}%  ({linhas[-1]['segundos']:.0f}s)")
    return pd.DataFrame(linhas).sort_values(criterio).reset_index(drop=True)


# ===========================================================================
# Classificação (Fase 5): categoria de crescimento do PIB
# ===========================================================================
#
# Duas abordagens, avaliadas com a mesma validação temporal da regressão:
#
# 1. **Regressão → categoria**: o crescimento previsto pelo modelo de regressão
#    escolhido na Fase 4 é convertido nas 5 faixas do enunciado. A categoria fica
#    coerente, por construção, com o PIB previsto.
# 2. **Classificadores diretos**: aprendem a categoria diretamente (Regressão
#    Logística multinomial, Random Forest e LightGBM), com a opção de pesar as
#    classes pelo inverso da frequência (``class_weight="balanced"``), já que
#    "Forte crescimento" domina e "Estabilidade" é rara.
#
# Critério de escolha: **F1 macro** médio em 2019–2022 (média simples do F1 das
# 5 classes, para que as classes raras contem tanto quanto as frequentes).
#
# Etapa B: ao prever 2024, o ano de 2023 já é conhecido. A escolha é refeita com o
# mesmo protocolo, mas validando em 2019–2023 (``ANOS_VALIDACAO_ETAPA_B``).

from src.features import CATEGORIAS, categoriza_crescimento  # noqa: E402

N_CLASSES = len(CATEGORIAS)
ANOS_VALIDACAO_ETAPA_B = ANOS_VALIDACAO + [ANO_TESTE]
NOME_REG_CATEGORIA = "Regressão → categoria (Ensemble)"


def codigos_categoria(cat) -> np.ndarray:
    """Categoria ordenada -> inteiro 0 (Forte retração) ... 4 (Forte crescimento)."""
    return pd.Categorical(cat, categories=CATEGORIAS, ordered=True).codes.astype(int)


def categoria_de_log(cresc_log) -> np.ndarray:
    """Crescimento em log -> código da categoria (pelas faixas em % do enunciado)."""
    pct = (np.exp(np.asarray(cresc_log, float)) - 1) * 100
    return codigos_categoria(categoriza_crescimento(pct))


def metricas_classificacao(y, y_prev) -> dict:
    """Métricas da classificação, com y e y_prev em códigos 0–4.

    * ``F1_macro``: média simples do F1 das 5 classes (exigido; critério de escolha).
    * ``F1_ponderado``: média do F1 ponderada pela frequência de cada classe.
    * ``Acuracia``: fração de acertos exatos.
    * ``Acuracia_adjacente``: acerto exato ou erro de uma só categoria vizinha.
      As classes são ordenadas, e confundir "Forte crescimento" com "Crescimento
      moderado" é bem menos grave do que com "Forte retração".
    """
    from sklearn.metrics import accuracy_score, f1_score
    y, y_prev = np.asarray(y, int), np.asarray(y_prev, int)
    rotulos = list(range(N_CLASSES))
    return {
        "F1_macro": f1_score(y, y_prev, labels=rotulos, average="macro", zero_division=0),
        "F1_ponderado": f1_score(y, y_prev, labels=rotulos, average="weighted", zero_division=0),
        "Acuracia": accuracy_score(y, y_prev),
        "Acuracia_adjacente": float(np.mean(np.abs(y - y_prev) <= 1)),
    }


def fabrica_logistica(numericas, categoricas, C=1.0, class_weight=None):
    from sklearn.linear_model import LogisticRegression
    return _pipeline(LogisticRegression(C=C, class_weight=class_weight, max_iter=3000), numericas, categoricas)


def fabrica_rf_classificador(numericas, categoricas, **params):
    from sklearn.ensemble import RandomForestClassifier
    base = dict(n_estimators=300, n_jobs=-1, random_state=SEMENTE)
    return _pipeline(RandomForestClassifier(**{**base, **params}), numericas, categoricas)


def fabrica_lgbm_classificador(numericas, categoricas, **params):
    from lightgbm import LGBMClassifier
    base = dict(n_estimators=300, learning_rate=0.05, random_state=SEMENTE, n_jobs=-1, verbose=-1,
                subsample_freq=1, subsample=0.8, colsample_bytree=0.6)
    return _pipeline(LGBMClassifier(**{**base, **params}), numericas, categoricas)


FABRICAS_CLASSIFICACAO: dict[str, Callable] = {
    "Regressão Logística": fabrica_logistica,
    "Random Forest (clf)": fabrica_rf_classificador,
    "LightGBM (clf)": fabrica_lgbm_classificador,
}

GRADES_CLASSIFICACAO: dict[str, dict[str, list]] = {
    "Regressão Logística": {"C": [0.01, 0.1, 1.0], "class_weight": [None, "balanced"]},
    "Random Forest (clf)": {"min_samples_leaf": [5, 20], "max_features": [0.33, 0.6],
                            "class_weight": [None, "balanced_subsample"]},
    "LightGBM (clf)": {"num_leaves": [15, 31], "min_child_samples": [20, 60],
                       "class_weight": [None, "balanced"]},
}


def baseline_classe_majoritaria(treino: pd.DataFrame, alvo: pd.DataFrame) -> np.ndarray:
    """Prevê sempre a classe mais frequente no treino."""
    return np.full(len(alvo), np.bincount(codigos_categoria(treino["categoria_crescimento"])).argmax())


def baseline_categoria_anterior(treino: pd.DataFrame, alvo: pd.DataFrame) -> np.ndarray:
    """Prevê a categoria do próprio município no ano anterior."""
    l1 = alvo["cresc_pib_l1"].fillna(alvo["cresc_pib_l1"].median())
    return categoria_de_log(l1)


BASELINES_CLASSIFICACAO = {
    "Classe majoritária do treino": baseline_classe_majoritaria,
    "Categoria do ano anterior": baseline_categoria_anterior,
}


def treina_preve_classificador(nome: str, params: dict, treino: pd.DataFrame, alvo: pd.DataFrame,
                               numericas: list[str], categoricas: list[str]):
    """Treina um classificador direto e devolve (códigos previstos, probabilidades, modelo)."""
    modelo = FABRICAS_CLASSIFICACAO[nome](numericas, categoricas, **params)
    modelo.fit(treino[numericas + categoricas], codigos_categoria(treino["categoria_crescimento"]))
    X = alvo[numericas + categoricas]
    return modelo.predict(X).astype(int), modelo.predict_proba(X), modelo


def preve_categoria(nome: str, params: dict, treino: pd.DataFrame, alvo: pd.DataFrame,
                    numericas: list[str], categoricas: list[str]) -> np.ndarray:
    """Códigos de categoria previstos por qualquer abordagem (baseline, regressão ou classificador)."""
    if nome in BASELINES_CLASSIFICACAO:
        return BASELINES_CLASSIFICACAO[nome](treino, alvo)
    if nome == NOME_REG_CATEGORIA:
        cresc, _ = treina_preve_modelo(NOME_ENSEMBLE, params, treino, alvo, numericas, categoricas)
        return categoria_de_log(cresc)
    return treina_preve_classificador(nome, params, treino, alvo, numericas, categoricas)[0]


def valida_temporal_classificacao(nome: str, params: dict, dataset: pd.DataFrame, inicio: int,
                                  anos_validacao: list[int] = ANOS_VALIDACAO) -> ResultadoValidacao:
    """Janela expansiva, como na regressão: treina com [inicio, v-1] e avalia em v."""
    numericas, categoricas = colunas_modelo(dataset)
    linhas = []
    for v in anos_validacao:
        treino = amostra_modelagem(dataset, inicio, v - 1)
        val = amostra_modelagem(dataset, v, v)
        prev = preve_categoria(nome, params, treino, val, numericas, categoricas)
        linhas.append({"ano_validacao": v, "n_treino": len(treino),
                       **metricas_classificacao(codigos_categoria(val["categoria_crescimento"]), prev)})
    return ResultadoValidacao(nome, params, pd.DataFrame(linhas).set_index("ano_validacao"))


def busca_classificacao(nome: str, dataset: pd.DataFrame, inicio: int, grade: dict | None = None,
                        anos_validacao: list[int] = ANOS_VALIDACAO, verbose: bool = True) -> pd.DataFrame:
    """Avalia a grade do classificador na validação temporal e ordena pelo F1 macro médio (maior primeiro)."""
    grade = GRADES_CLASSIFICACAO[nome] if grade is None else grade
    linhas = []
    for params in combinacoes(grade):
        t0 = time.time()
        res = valida_temporal_classificacao(nome, params, dataset, inicio, anos_validacao)
        linhas.append({"modelo": nome, "params": params, **res.media.drop("n_treino").to_dict(),
                       "F1_macro_pior_ano": res.por_ano["F1_macro"].min(), "segundos": time.time() - t0})
        if verbose:
            print(f"  {nome} {params} -> F1 macro {linhas[-1]['F1_macro']:.3f}  ({linhas[-1]['segundos']:.0f}s)")
    return pd.DataFrame(linhas).sort_values("F1_macro", ascending=False).reset_index(drop=True)
