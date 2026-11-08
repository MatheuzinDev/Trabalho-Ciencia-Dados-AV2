"""Pré-processamento e engenharia de variáveis (Fase 3).

Parte do painel integrado (``data/processed/painel_municipios.csv``) e gera o
dataset de modelagem (``data/processed/dataset_tratado.csv``): **uma linha por
município e ano-alvo t**, com:

* as variáveis-alvo do ano t (PIB, crescimento e categoria de crescimento);
* variáveis explicativas que **estariam disponíveis ao prever o ano t**, seguindo
  o calendário real de divulgação do IBGE:

  - PIB do município: até t-1 (defasagens t-1 e t-2);
  - estrutura setorial (VAB por setor e impostos): só até t-3, porque o IBGE
    publicou só o PIB total para 2022–2023 e, ao prever 2024, o dado setorial
    mais recente é o de 2021;
  - PAM, PPM, população e PIA/PAS (UF): disponíveis no próprio ano t.

Uso pela linha de comando (na raiz do projeto)::

    python -m src.features
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.integracao import ARQ_PAINEL, DIR_PROCESSED, ANOS_QUEBRA_POPULACAO

ARQ_DATASET = DIR_PROCESSED / "dataset_tratado.csv"
ARQ_DICIONARIO_DATASET = DIR_PROCESSED / "dicionario_dataset.csv"

# Primeiro ano-alvo gravado no dataset. Usamos alvos a partir de 2013 na modelagem
# padrão, mas guardamos desde 2008 para poder testar janelas alternativas na Fase 4.
PRIMEIRO_ANO_ALVO = 2008
PRIMEIRO_ANO_AMOSTRA_PADRAO = 2013
ANO_PREVISAO = 2024

DEFASAGEM_SETORIAL = 3

# ---------------------------------------------------------------------------
# Categorias de crescimento (definição do enunciado)
# ---------------------------------------------------------------------------

CATEGORIAS = ["Forte retração", "Retração", "Estabilidade", "Crescimento moderado", "Forte crescimento"]


def categoriza_crescimento(cresc_pct: pd.Series | np.ndarray) -> pd.Series:
    """Classifica a taxa de crescimento percentual do PIB nas 5 categorias do enunciado.

    O enunciado dá as faixas sem dizer a que lado ficam os limites. Adotamos:

    ============================  ====================
    Categoria                     Faixa (crescimento %)
    ============================  ====================
    Forte retração                x < -5
    Retração                      -5 <= x < -1
    Estabilidade                  -1 <= x <= 1
    Crescimento moderado          1 < x <= 5
    Forte crescimento             x > 5
    ============================  ====================

    Os limites -5 e -1 vão para a classe de cima, e 1 e 5 para a de baixo, de modo
    que a faixa de "Estabilidade" fique simétrica em torno de zero. Valores ausentes
    continuam ausentes.
    """
    x = pd.Series(cresc_pct, dtype=float)
    condicoes = [x < -5, x < -1, x <= 1, x <= 5, x > 5]
    rotulos = np.select(condicoes, CATEGORIAS, default="")
    cat = pd.Series(rotulos, index=x.index).replace("", np.nan)
    return pd.Categorical(cat, categories=CATEGORIAS, ordered=True)


# ---------------------------------------------------------------------------
# Limpeza
# ---------------------------------------------------------------------------

def _prefixos(painel: pd.DataFrame, *prefixos: str) -> list[str]:
    return [c for c in painel.columns if c.startswith(prefixos)]


def trata_ausentes_agro(painel: pd.DataFrame) -> pd.DataFrame:
    """Imputa as lacunas de PAM e PPM dentro de cada município.

    1. Lacunas **internas**, com dado antes e depois, são preenchidas por interpolação
       linear no tempo. Ex.: Governador Celso Ramos (2020–2022) e Imbé (2020).
    2. O que sobra, ou seja, lacunas no fim ou no início da série e municípios que
       nunca têm dado, vira **zero**. São municípios urbanos ou litorâneos, onde o
       IBGE não publica produção (Balneário Camboriú, Bombinhas, Cachoeirinha…), e
       o zero é a melhor descrição da sua produção agrícola.

    Os 3 municípios instalados em 2013 não têm dado antes disso porque não existiam,
    e lá as lacunas ficam como estão.
    """
    df = painel.sort_values(["cod_ibge", "ano"]).copy()
    cols = _prefixos(df, "pam_", "ppm_")
    existe = df["pib"].notna() | (df["ano"] >= 2013)  # anos em que o município existia
    existe_2d = np.broadcast_to(existe.to_numpy()[:, None], (len(df), len(cols)))

    df["agro_imputado"] = (df[cols].isna().to_numpy() & existe_2d).any(axis=1).astype(int)

    interp = df.groupby("cod_ibge")[cols].transform(lambda s: s.interpolate(limit_area="inside"))
    # Zera o que sobrou, mas só nos anos em que o município existia (não inventa dado antes disso)
    df[cols] = interp.fillna(0).where(existe_2d)
    return df


def populacao_ajustada(painel: pd.DataFrame) -> pd.Series:
    """Variação anual da população (em log), corrigida nas quebras de série.

    Em alguns anos a diferença em relação ao ano anterior mistura crescimento real
    com mudança de base da série:

    * 2007, 2010 e 2022: o valor vem de contagem ou censo, e não da projeção. Ex.: o
      Sul "perde" 1,5% da população entre a estimativa de 2021 e o Censo 2022;
    * 2008 e 2013: as estimativas voltam a ser projeções, recalibradas pela contagem
      de 2007 e pelo Censo 2010 (o Sul "ganha" 3,8% de 2012 para 2013);
    * 2023 e 2024: a estimativa de 2024 inclui o ajuste de cobertura do Censo 2022
      (ver abaixo; 2023 é interpolado entre 2022 e 2024).

    Nesses anos, a variação é substituída pela taxa média anual entre os dois
    levantamentos oficiais que cercam o ano (2007→2010 ou 2010→2022).
    """
    df = painel.sort_values(["cod_ibge", "ano"])
    lp = np.log(df["populacao"])
    dlog = lp - lp.groupby(df["cod_ibge"]).shift(1)

    largo = lp.groupby([df["cod_ibge"], df["ano"]]).first().unstack("ano")
    taxa_07_10 = (largo[2010] - largo[2007]) / 3
    taxa_10_22 = (largo[2022] - largo[2010]) / 12
    # Pescaria Brava, Balneário Rincão e Pinto Bandeira não têm 2010: usa 2012→2022 (população desde 2012)
    taxa_10_22 = taxa_10_22.fillna((largo[2022] - largo[2012]) / 10)
    intercensitario = {2008: taxa_07_10, 2010: taxa_07_10, 2013: taxa_10_22, 2022: taxa_10_22}
    for ano, taxa in intercensitario.items():
        mask = df["ano"] == ano
        dlog.loc[mask] = df.loc[mask, "cod_ibge"].map(taxa).to_numpy()
    dlog.loc[df["ano"] == 2007] = np.nan  # sem censo anterior na série; fora da janela de modelagem

    # 2023–2024: a estimativa de 2024 já incorpora o ajuste de cobertura do Censo 2022
    # (subenumeração), e a população do Sul "cresce" ~1,9% ao ano contra ~0,7% no período
    # 2010–2022. Descontamos esse excesso médio da região, preservando as diferenças entre
    # municípios.
    pop22 = np.exp(largo[2022])
    taxa_ref = np.average(taxa_10_22.dropna(), weights=pop22[taxa_10_22.dropna().index])
    for ano in (2023, 2024):
        mask = (df["ano"] == ano) & dlog.notna()
        excesso = np.average(dlog[mask], weights=df.loc[mask, "populacao"]) - taxa_ref
        dlog.loc[mask] -= excesso
    return dlog.reindex(painel.index)


# ---------------------------------------------------------------------------
# Engenharia de variáveis
# ---------------------------------------------------------------------------

PORTES = [0, 5_000, 10_000, 20_000, 50_000, 100_000, 500_000, np.inf]
ROTULOS_PORTE = ["ate_5mil", "5_10mil", "10_20mil", "20_50mil", "50_100mil", "100_500mil", "acima_500mil"]

PRINCIPAIS_LAVOURAS = ["soja", "milho", "trigo", "fumo", "arroz"]


def _log1p_dif(atual: pd.Series, anterior: pd.Series) -> pd.Series:
    """Variação em log que tolera zeros: log(1+x_t) - log(1+x_{t-1})."""
    return np.log1p(atual.clip(lower=0)) - np.log1p(anterior.clip(lower=0))


def constroi_dataset(painel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gera o dataset de modelagem e o dicionário das suas colunas."""
    dic: list[dict] = []

    def reg(col: str, desc: str, tipo: str, disponibilidade: str = "-", unidade: str = "-") -> None:
        dic.append({"coluna": col, "descricao": desc, "tipo": tipo,
                    "disponibilidade": disponibilidade, "unidade": unidade})

    p = trata_ausentes_agro(painel).sort_values(["cod_ibge", "ano"]).reset_index(drop=True)
    g = p.groupby("cod_ibge")

    def lag(col: str, k: int) -> pd.Series:
        return g[col].shift(k)

    out = p[["cod_ibge", "ano", "municipio", "uf", "cod_uf", "regiao_intermediaria", "regiao_imediata"]].copy()
    reg("cod_ibge", "Código IBGE do município (7 dígitos)", "identificador")
    reg("ano", "Ano-alvo t (ano cujo PIB se quer prever)", "identificador")
    reg("municipio", "Nome do município", "identificador")
    reg("uf", "Sigla da UF", "categórica", "fixa")
    reg("cod_uf", "Código IBGE da UF", "identificador")
    reg("regiao_intermediaria", "Região geográfica intermediária do IBGE (21 no Sul)", "categórica", "fixa")
    reg("regiao_imediata", "Região geográfica imediata do IBGE (96 no Sul)", "categórica", "fixa")

    # ----- Alvos -------------------------------------------------------------
    pib_l1 = lag("pib", 1)
    out["pib"] = p["pib"]
    out["cresc_pib_pct"] = (p["pib"] / pib_l1 - 1) * 100
    out["cresc_pib_log"] = np.log(p["pib"] / pib_l1)
    out["categoria_crescimento"] = categoriza_crescimento(out["cresc_pib_pct"])
    reg("pib", "ALVO (regressão): PIB do ano t a preços correntes", "alvo", "t", "mil R$")
    reg("cresc_pib_pct", "ALVO auxiliar: crescimento do PIB em relação a t-1, (PIB_t/PIB_t-1 - 1) × 100", "alvo", "t", "%")
    reg("cresc_pib_log", "ALVO de treino da regressão: log(PIB_t / PIB_t-1)", "alvo", "t", "log")
    reg("categoria_crescimento", "ALVO (classificação): categoria de crescimento do PIB no ano t (5 classes)", "alvo", "t")

    # ----- Histórico do PIB (até t-1) ----------------------------------------
    out["pib_l1"] = pib_l1
    out["log_pib_l1"] = np.log(pib_l1)
    out["log_pib_l2"] = np.log(lag("pib", 2))
    out["cresc_pib_l1"] = np.log(pib_l1 / lag("pib", 2))
    out["cresc_pib_l2"] = np.log(lag("pib", 2) / lag("pib", 3))
    out["cresc_pib_media_3a"] = np.log(pib_l1 / lag("pib", 4)) / 3
    out["log_pib_pc_l1"] = np.log(pib_l1 * 1000 / lag("populacao", 1))
    reg("pib_l1", "PIB do ano anterior (t-1); base para reconstruir o PIB previsto", "explicativa", "t-1", "mil R$")
    reg("log_pib_l1", "log do PIB em t-1", "explicativa", "t-1", "log")
    reg("log_pib_l2", "log do PIB em t-2", "explicativa", "t-2", "log")
    reg("cresc_pib_l1", "Crescimento do PIB em t-1: log(PIB_t-1 / PIB_t-2)", "explicativa", "t-1", "log")
    reg("cresc_pib_l2", "Crescimento do PIB em t-2: log(PIB_t-2 / PIB_t-3)", "explicativa", "t-2", "log")
    reg("cresc_pib_media_3a", "Crescimento médio anual do PIB de t-4 a t-1 (em log)", "explicativa", "t-1", "log")
    reg("log_pib_pc_l1", "log do PIB per capita em t-1 (R$ por habitante)", "explicativa", "t-1", "log")

    # Crescimento do PIB da UF e da região imediata em t-1 (contexto regional; soma dos municípios)
    for nivel, nome in [("cod_uf", "uf"), ("regiao_imediata", "reg_imediata")]:
        soma = p.groupby([nivel, "ano"])["pib"].transform("sum").where(p["pib"].notna())
        soma_l1 = soma.groupby(p["cod_ibge"]).shift(1)
        soma_l2 = soma.groupby(p["cod_ibge"]).shift(2)
        out[f"cresc_pib_{nome}_l1"] = np.log(soma_l1 / soma_l2)
        reg(f"cresc_pib_{nome}_l1", f"Crescimento do PIB agregado da {nome.replace('_', ' ')} em t-1 (log)",
            "explicativa", "t-1", "log")

    # ----- Estrutura setorial (até t-3) ---------------------------------------
    k = DEFASAGEM_SETORIAL
    vab_l = lag("vab_total", k)
    for setor in ["agro", "industria", "servicos", "adm_publica"]:
        out[f"part_{setor}_l{k}"] = lag(f"vab_{setor}", k) / vab_l
        reg(f"part_{setor}_l{k}", f"Participação do setor {setor.replace('_', ' ')} no VAB em t-{k}",
            "explicativa", f"t-{k}", "fração")
    out[f"part_impostos_l{k}"] = lag("impostos", k) / lag("pib", k)
    reg(f"part_impostos_l{k}", f"Participação dos impostos líquidos no PIB em t-{k}", "explicativa", f"t-{k}", "fração")

    # ----- Agropecuária do ano t (PAM e PPM) -----------------------------------
    base = pib_l1  # escala: quanto a variação do agro representa do PIB do município
    out["pam_valor_sobre_pib_l1"] = p["pam_valor_total"] / base
    out["choque_pam"] = (p["pam_valor_total"] - lag("pam_valor_total", 1)) / base
    out["choque_ppm"] = (p["ppm_valor_total"] - lag("ppm_valor_total", 1)) / base
    out["var_pam_valor"] = _log1p_dif(p["pam_valor_total"], lag("pam_valor_total", 1))
    out["var_ppm_valor"] = _log1p_dif(p["ppm_valor_total"], lag("ppm_valor_total", 1))
    reg("pam_valor_sobre_pib_l1", "Valor da produção agrícola em t dividido pelo PIB de t-1 (intensidade agrícola)",
        "explicativa", "t", "fração")
    reg("choque_pam", "Variação do valor da produção agrícola (t vs t-1) como fração do PIB de t-1",
        "explicativa", "t", "fração")
    reg("choque_ppm", "Variação do valor da produção de origem animal (t vs t-1) como fração do PIB de t-1",
        "explicativa", "t", "fração")
    reg("var_pam_valor", "Variação do valor da produção agrícola: log(1+PAM_t) - log(1+PAM_t-1)", "explicativa", "t", "log")
    reg("var_ppm_valor", "Variação do valor da produção de origem animal (log1p)", "explicativa", "t", "log")

    # Choque agropecuário agregado da UF e da região imediata: uma quebra de safra afeta
    # também o comércio e os serviços dos municípios vizinhos, não só quem planta.
    for nivel, nome in [("cod_uf", "uf"), ("regiao_imediata", "reg_imediata")]:
        for fonte, rotulo in [("pam_valor_total", "pam"), ("ppm_valor_total", "ppm")]:
            agg = p.groupby([nivel, "ano"])[[fonte, "pib"]].sum(min_count=1)
            agg_l1 = agg.groupby(level=0).shift(1)
            choque = (agg[fonte] - agg_l1[fonte]) / agg_l1["pib"]
            chave = pd.MultiIndex.from_arrays([p[nivel], p["ano"]])
            out[f"choque_{rotulo}_{nome}"] = choque.reindex(chave).to_numpy()
            reg(f"choque_{rotulo}_{nome}",
                f"Variação do valor da {'PAM' if rotulo == 'pam' else 'PPM'} agregada da "
                f"{'UF' if nome == 'uf' else 'região imediata'} (t vs t-1) como fração do PIB agregado de t-1",
                "explicativa", "t", "fração")

    for cultura in PRINCIPAIS_LAVOURAS:
        col = f"pam_valor_{cultura}"
        out[f"choque_{cultura}"] = (p[col] - lag(col, 1)) / base
        reg(f"choque_{cultura}", f"Variação do valor da produção de {cultura} (t vs t-1) como fração do PIB de t-1",
            "explicativa", "t", "fração")

    plantada = p["pam_area_plantada_total"]
    out["perda_area"] = np.where(plantada > 0, 1 - p["pam_area_colhida_total"] / plantada, 0.0)
    reg("perda_area", "Fração da área plantada que não foi colhida em t (indicador de quebra de safra)",
        "explicativa", "t", "fração")

    for reb in ["bovino", "suino_total", "galinaceos_total"]:
        col = f"ppm_reb_{reb}"
        out[f"var_reb_{reb}"] = _log1p_dif(p[col], lag(col, 1))
        reg(f"var_reb_{reb}", f"Variação do efetivo do rebanho {reb.replace('_', ' ')} (log1p, t vs t-1)",
            "explicativa", "t", "log")

    # ----- População do ano t ------------------------------------------------
    out["log_populacao"] = np.log(p["populacao"])
    out["var_populacao"] = populacao_ajustada(p)
    out["porte"] = pd.cut(p["populacao"], PORTES, labels=ROTULOS_PORTE, right=False)
    reg("log_populacao", "log da população residente em t", "explicativa", "t", "log")
    reg("var_populacao", "Variação anual da população em log, corrigida nas quebras de série (2008, 2010, 2013, 2022; "
        "2023–2024 descontado o ajuste de cobertura do Censo 2022)",
        "explicativa", "t", "log")
    reg("porte", "Porte do município pela população em t (7 faixas)", "categórica", "t")

    # ----- Contexto estadual do ano t (PIA e PAS) --------------------------------
    for col, desc in [
        ("uf_pia_receita_liquida", "receita líquida de vendas da indústria"),
        ("uf_pia_valor_transformacao", "valor da transformação industrial"),
        ("uf_pia_pessoal_ocupado", "pessoal ocupado na indústria"),
        ("uf_pas_receita_bruta", "receita bruta de serviços"),
        ("uf_pas_pessoal_ocupado", "pessoal ocupado em serviços"),
    ]:
        novo = col.replace("uf_", "var_uf_")
        out[novo] = np.log(p[col] / lag(col, 1))
        reg(novo, f"Variação (log, t vs t-1) da {desc} da UF", "explicativa", "t", "log")

    # ----- Contexto nacional do ano t (Contas Nacionais Trimestrais) ---------------
    # O PIB trimestral do Brasil sai bem antes do PIB municipal: em 2026 os 4 trimestres de
    # 2024 já são conhecidos. Funciona como termômetro do crescimento nominal do ano.
    for col, desc in [("br_pib", "PIB"), ("br_vab_agro", "VAB da agropecuária"),
                      ("br_vab_industria", "VAB da indústria"), ("br_vab_servicos", "VAB dos serviços")]:
        novo = col.replace("br_", "var_br_")
        out[novo] = np.log(p[col] / lag(col, 1))
        reg(novo, f"Variação nominal (log, t vs t-1) do {desc} do Brasil", "explicativa", "t", "log")

    out["agro_imputado"] = p["agro_imputado"]
    reg("agro_imputado", "1 se algum dado de PAM/PPM do município no ano t foi imputado", "indicador", "t", "0/1")

    # ----- Recorte -----------------------------------------------------------
    out = out[out["ano"].between(PRIMEIRO_ANO_ALVO, ANO_PREVISAO)].reset_index(drop=True)
    out["amostra_padrao"] = (out["ano"] >= PRIMEIRO_ANO_AMOSTRA_PADRAO).astype(int)
    out["conjunto"] = np.select(
        [out["ano"] == ANO_PREVISAO, out["ano"] == 2023],
        ["previsao_2024", "teste_2023"],
        default="treino",
    )
    reg("amostra_padrao", f"1 se o ano-alvo está na janela padrão de modelagem (>= {PRIMEIRO_ANO_AMOSTRA_PADRAO})",
        "indicador")
    reg("conjunto", "treino (alvos até 2022), teste_2023 (Etapa A) ou previsao_2024 (Etapa B, alvo desconhecido)",
        "indicador")

    dicionario = pd.DataFrame(dic).set_index("coluna").loc[list(out.columns)].reset_index()
    return out, dicionario


def colunas_modelo(dataset: pd.DataFrame, incluir_brasil: bool = False) -> tuple[list[str], list[str]]:
    """Listas de variáveis explicativas numéricas e categóricas usadas pelos modelos.

    As variações do PIB do Brasil (``var_br_*``) ficam no dataset, mas fora do modelo
    padrão: no experimento do notebook 03 elas não melhoraram a validação. Como são
    iguais para todos os municípios num ano, há só tantos valores distintos quanto anos
    de treino.
    """
    excluir = {"cod_ibge", "ano", "municipio", "cod_uf", "pib", "cresc_pib_pct", "cresc_pib_log",
               "categoria_crescimento", "pib_l1", "amostra_padrao", "conjunto", "regiao_imediata"}
    categoricas = ["uf", "regiao_intermediaria", "porte"]
    numericas = [c for c in dataset.columns if c not in excluir and c not in categoricas]
    if not incluir_brasil:
        numericas = [c for c in numericas if not c.startswith("var_br_")]
    return numericas, categoricas


def preprocessador(numericas: list[str], categoricas: list[str]):
    """Pipeline de pré-processamento do scikit-learn (ajustar só no treino).

    * Numéricas: imputação pela mediana + padronização (média 0, desvio 1).
    * Categóricas: one-hot (categorias novas no teste são ignoradas).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = Pipeline([("imputa", SimpleImputer(strategy="median")), ("padroniza", StandardScaler())])
    cat = Pipeline([("imputa", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, numericas), ("cat", cat, categoricas)], verbose_feature_names_out=False)


def carrega_dataset() -> pd.DataFrame:
    """Lê o dataset tratado restaurando os tipos categóricos."""
    df = pd.read_csv(ARQ_DATASET)
    df["categoria_crescimento"] = pd.Categorical(df["categoria_crescimento"], categories=CATEGORIAS, ordered=True)
    df["porte"] = pd.Categorical(df["porte"], categories=ROTULOS_PORTE, ordered=True)
    return df


if __name__ == "__main__":
    painel = pd.read_csv(ARQ_PAINEL)
    dataset, dicionario = constroi_dataset(painel)
    dataset.to_csv(ARQ_DATASET, index=False, encoding="utf-8")
    dicionario.to_csv(ARQ_DICIONARIO_DATASET, index=False, encoding="utf-8")
    print(f"Dataset: {dataset.shape[0]:,} linhas × {dataset.shape[1]} colunas -> {ARQ_DATASET}")
    print(f"Dicionário: {len(dicionario)} colunas -> {ARQ_DICIONARIO_DATASET}")
