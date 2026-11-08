"""Integração das fontes brutas em um painel município × ano (Fase 2).

Lê os CSVs de ``data/raw/`` (gerados por :mod:`src.coleta`), converte cada fonte
para o formato largo (uma coluna por variável) e junta tudo em uma grade completa
de 1.191 municípios × anos (2002–2024).

Saídas em ``data/processed/``:

* ``painel_municipios.csv``: o painel integrado;
* ``dicionario_painel.csv``: nome, descrição, unidade, fonte e nível de cada coluna.

Uso pela linha de comando (na raiz do projeto)::

    python -m src.integracao
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.coleta import ANO_FINAL, ANO_INICIAL, DIR_RAW, LAVOURAS, N_MUNICIPIOS_SUL, RAIZ, para_numerico

DIR_PROCESSED = RAIZ / "data" / "processed"
ARQ_PAINEL = DIR_PROCESSED / "painel_municipios.csv"
ARQ_DICIONARIO = DIR_PROCESSED / "dicionario_painel.csv"

ANOS = list(range(ANO_INICIAL, ANO_FINAL + 1))

# Anos em que a série de população muda de base: o valor vem de uma contagem ou de um censo,
# e não da projeção usada nos anos anteriores. A variação em relação ao ano anterior
# mistura crescimento real e correção de base.
ANOS_QUEBRA_POPULACAO = [2007, 2010, 2022]

# Dicionário de variáveis: preenchido à medida que as colunas são criadas.
_DICIONARIO: list[dict] = []


def _registra(coluna: str, descricao: str, unidade: str, fonte: str, nivel: str = "Município") -> None:
    _DICIONARIO.append({"coluna": coluna, "descricao": descricao, "unidade": unidade,
                        "fonte": fonte, "nivel": nivel})


def _le(nome: str) -> pd.DataFrame:
    df = pd.read_csv(DIR_RAW / f"{nome}.csv", dtype={"valor_raw": str})
    df["valor"] = para_numerico(df["valor_raw"])
    df["cod_local"] = df["cod_local"].astype(int)
    return df


def _pivota(df: pd.DataFrame, coluna_nome: str) -> pd.DataFrame:
    """Formato longo -> largo, indexado por (cod_ibge, ano). Falha se houver duplicatas."""
    chave = ["cod_local", "ano", coluna_nome]
    duplicadas = df.duplicated(chave).sum()
    if duplicadas:
        raise ValueError(f"{duplicadas} linhas duplicadas para a chave {chave}")
    largo = df.pivot(index=["cod_local", "ano"], columns=coluna_nome, values="valor")
    largo.columns.name = None
    largo.index = largo.index.set_names(["cod_ibge", "ano"])
    return largo


# ---------------------------------------------------------------------------
# Fontes municipais
# ---------------------------------------------------------------------------

def municipios() -> pd.DataFrame:
    """Lista oficial de municípios com a hierarquia territorial (atributos fixos no tempo)."""
    mun = pd.read_csv(DIR_RAW / "municipios_sul.csv")
    textos = {
        "cod_ibge": "Código IBGE do município (7 dígitos)",
        "municipio": "Nome do município",
        "cod_uf": "Código IBGE da UF",
        "uf": "Sigla da UF",
        "cod_mesorregiao": "Código da mesorregião geográfica (divisão de 1989)",
        "mesorregiao": "Mesorregião geográfica",
        "cod_microrregiao": "Código da microrregião geográfica (divisão de 1989)",
        "microrregiao": "Microrregião geográfica",
        "cod_regiao_intermediaria": "Código da região geográfica intermediária (divisão de 2017)",
        "regiao_intermediaria": "Região geográfica intermediária",
        "cod_regiao_imediata": "Código da região geográfica imediata (divisão de 2017)",
        "regiao_imediata": "Região geográfica imediata",
    }
    for col, desc in textos.items():
        _registra(col, desc, "-", "IBGE – API de Localidades")
    return mun


VARIAVEIS_PIB = {
    37: ("pib", "Produto Interno Bruto a preços correntes"),
    543: ("impostos", "Impostos, líquidos de subsídios, sobre produtos, a preços correntes"),
    498: ("vab_total", "Valor adicionado bruto total, a preços correntes"),
    513: ("vab_agro", "VAB da agropecuária, a preços correntes"),
    517: ("vab_industria", "VAB da indústria, a preços correntes"),
    6575: ("vab_servicos", "VAB dos serviços, exceto administração pública, a preços correntes"),
    525: ("vab_adm_publica", "VAB da administração, defesa, educação e saúde públicas e seguridade social"),
}


def pib() -> pd.DataFrame:
    df = _le("pib_municipios")
    df["nome"] = df["cod_variavel"].map(lambda c: VARIAVEIS_PIB[c][0])
    largo = _pivota(df, "nome")[[v[0] for v in VARIAVEIS_PIB.values()]]
    for col, desc in VARIAVEIS_PIB.values():
        obs = " Disponível até 2021 (o IBGE publicou só o PIB total para 2022–2023)." if col != "pib" else ""
        _registra(col, desc + "." + obs, "mil R$", "SIDRA 5938 – PIB dos Municípios")
    return largo


def _slug(texto: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return "".join(ch if ch.isalnum() else "_" for ch in s.lower()).strip("_")


VARIAVEIS_PAM = {
    215: ("valor", "Valor da produção", "mil R$"),
    8331: ("area_plantada", "Área plantada ou destinada à colheita", "ha"),
    216: ("area_colhida", "Área colhida", "ha"),
    214: ("qtd", "Quantidade produzida", "t"),
}


def pam() -> pd.DataFrame:
    df = _le("pam_lavouras")
    cod_prod = "Produto das lavouras temporárias e permanentes (Código)"
    df = df[~((df[cod_prod] == 0) & (df["cod_variavel"] == 214))]  # toneladas somadas entre culturas não fazem sentido
    df["nome"] = [
        f"pam_{VARIAVEIS_PAM[v][0]}_{_slug(LAVOURAS[c])}" for v, c in zip(df["cod_variavel"], df[cod_prod])
    ]
    largo = _pivota(df, "nome")
    ordem = [f"pam_{VARIAVEIS_PAM[v][0]}_{_slug(LAVOURAS[c])}"
             for c in LAVOURAS for v in VARIAVEIS_PAM if not (c == 0 and v == 214)]
    for v_cod in VARIAVEIS_PAM:
        for c_cod, cultura in LAVOURAS.items():
            if c_cod == 0 and v_cod == 214:
                continue
            nome, desc, unid = VARIAVEIS_PAM[v_cod]
            alvo = "total das lavouras temporárias e permanentes" if c_cod == 0 else cultura
            _registra(f"pam_{nome}_{_slug(cultura)}", f"{desc} – {alvo}", unid,
                      "SIDRA 5457 – Produção Agrícola Municipal")
    return largo[ordem]


REBANHOS = {
    2670: ("bovino", "Bovino"),
    2675: ("bubalino", "Bubalino"),
    2672: ("equino", "Equino"),
    32794: ("suino_total", "Suíno – total"),
    32795: ("suino_matrizes", "Suíno – matrizes"),
    2681: ("caprino", "Caprino"),
    2677: ("ovino", "Ovino"),
    32796: ("galinaceos_total", "Galináceos – total"),
    32793: ("galinaceos_galinhas", "Galináceos – galinhas"),
    2680: ("codornas", "Codornas"),
}


def ppm_rebanhos() -> pd.DataFrame:
    df = _le("ppm_rebanhos")
    df["nome"] = df["Tipo de rebanho (Código)"].map(lambda c: f"ppm_reb_{REBANHOS[c][0]}")
    largo = _pivota(df, "nome")
    for nome, desc in REBANHOS.values():
        _registra(f"ppm_reb_{nome}", f"Efetivo do rebanho – {desc}", "cabeças",
                  "SIDRA 3939 – Pesquisa da Pecuária Municipal")
    return largo[[f"ppm_reb_{n}" for n, _ in REBANHOS.values()]]


PRODUTOS_ANIMAIS = {
    0: ("total", "Total", None),
    2682: ("leite", "Leite", "mil litros"),
    2685: ("ovos_galinha", "Ovos de galinha", "mil dúzias"),
    2686: ("ovos_codorna", "Ovos de codorna", "mil dúzias"),
    2687: ("mel", "Mel de abelha", "kg"),
    2683: ("casulos", "Casulos do bicho-da-seda", "kg"),
    2684: ("la", "Lã", "kg"),
}


def ppm_origem_animal() -> pd.DataFrame:
    df = _le("ppm_origem_animal")
    cod = "Tipo de produto de origem animal (Código)"
    df = df[~((df[cod] == 0) & (df["cod_variavel"] == 106))]  # quantidade "Total" mistura unidades
    df["nome"] = [
        f"ppm_{'valor' if v == 215 else 'qtd'}_{PRODUTOS_ANIMAIS[c][0]}" for v, c in zip(df["cod_variavel"], df[cod])
    ]
    largo = _pivota(df, "nome")
    ordem = []
    for c, (nome, desc, unid) in PRODUTOS_ANIMAIS.items():
        ordem.append(f"ppm_valor_{nome}")
        _registra(f"ppm_valor_{nome}", f"Valor da produção de origem animal – {desc}", "mil R$",
                  "SIDRA 74 – Pesquisa da Pecuária Municipal")
    for c, (nome, desc, unid) in PRODUTOS_ANIMAIS.items():
        if unid:
            ordem.append(f"ppm_qtd_{nome}")
            _registra(f"ppm_qtd_{nome}", f"Quantidade produzida – {desc}", unid,
                      "SIDRA 74 – Pesquisa da Pecuária Municipal")
    return largo[ordem]


def populacao(grade: pd.MultiIndex) -> pd.DataFrame:
    """População por município e ano, com 2023 interpolado e marcação das quebras de série."""
    df = _le("populacao")
    pop = df.set_index(["cod_local", "ano"])[["valor", "fonte"]]
    pop.index = pop.index.set_names(["cod_ibge", "ano"])
    pop = pop.rename(columns={"valor": "populacao", "fonte": "populacao_fonte"}).reindex(grade)

    # 2023: sem estimativa oficial -> média geométrica entre Censo 2022 e estimativa 2024
    # (equivale a supor crescimento a taxa constante entre os dois anos).
    largo = pop["populacao"].unstack("ano")
    interp = np.sqrt(largo[2022] * largo[2024])
    idx_2023 = pd.MultiIndex.from_arrays([interp.index, np.full(len(interp), 2023)], names=["cod_ibge", "ano"])
    pop.loc[idx_2023, "populacao"] = interp.to_numpy()
    pop.loc[idx_2023, "populacao_fonte"] = "Interpolação geométrica (Censo 2022 e estimativa 2024)"

    pop["populacao_interpolada"] = (pop.index.get_level_values("ano") == 2023).astype(int)
    pop["populacao_quebra_serie"] = pop.index.get_level_values("ano").isin(ANOS_QUEBRA_POPULACAO).astype(int)

    fonte_pop = "SIDRA 6579 (estimativas), 793 (Contagem 2007), 202 (Censo 2010), 4709 (Censo 2022)"
    _registra("populacao", "População residente (estimativa, contagem ou censo, conforme o ano)", "pessoas", fonte_pop)
    _registra("populacao_fonte", "Origem do valor de população do ano", "-", fonte_pop)
    _registra("populacao_interpolada", "1 se a população do ano foi interpolada (2023, sem estimativa oficial)",
              "0/1", "Calculado")
    _registra("populacao_quebra_serie",
              "1 nos anos de contagem/censo (2007, 2010, 2022), em que a série muda de base", "0/1", "Calculado")
    return pop


# ---------------------------------------------------------------------------
# Fontes estaduais (UF)
# ---------------------------------------------------------------------------

# Código da variável no SIDRA -> nome harmonizado (as séries antiga e nova usam códigos diferentes
# para o número de unidades locais da PIA e para os salários da PAS).
VARIAVEIS_PIA = {
    706: "uf_pia_unidades_locais", 13816: "uf_pia_unidades_locais",
    631: "uf_pia_pessoal_ocupado",
    673: "uf_pia_salarios",
    864: "uf_pia_receita_liquida",
    810: "uf_pia_valor_bruto_producao",
    811: "uf_pia_valor_transformacao",
}
VARIAVEIS_PAS = {
    630: "uf_pas_empresas",
    631: "uf_pas_pessoal_ocupado",
    672: "uf_pas_receita_bruta",
    673: "uf_pas_salarios", 857: "uf_pas_salarios",
}


def uf_contexto() -> pd.DataFrame:
    """PIA-Empresa e PAS por UF e ano, com as séries antiga (até 2023) e nova (2024) harmonizadas."""
    partes = []
    for nome, mapa in [("pia_empresa_uf", VARIAVEIS_PIA), ("pas_uf", VARIAVEIS_PAS)]:
        df = _le(nome)
        df["nome"] = df["cod_variavel"].map(mapa)
        largo = df.pivot(index=["cod_local", "ano"], columns="nome", values="valor")
        largo.index = largo.index.set_names(["cod_uf", "ano"])
        partes.append(largo)
    uf = pd.concat(partes, axis=1)
    uf.columns.name = None
    uf["uf_serie_nova_pia_pas"] = (uf.index.get_level_values("ano") >= 2024).astype(int)

    desc_pia = {
        "uf_pia_unidades_locais": ("Unidades locais industriais (empresas com 5+ pessoas ocupadas)", "unidades"),
        "uf_pia_pessoal_ocupado": ("Pessoal ocupado na indústria em 31/12", "pessoas"),
        "uf_pia_salarios": ("Salários, retiradas e outras remunerações na indústria", "mil R$"),
        "uf_pia_receita_liquida": ("Receita líquida de vendas da indústria", "mil R$"),
        "uf_pia_valor_bruto_producao": ("Valor bruto da produção industrial", "mil R$"),
        "uf_pia_valor_transformacao": ("Valor da transformação industrial", "mil R$"),
    }
    desc_pas = {
        "uf_pas_empresas": ("Número de empresas de serviços não financeiros", "unidades"),
        "uf_pas_pessoal_ocupado": ("Pessoal ocupado em serviços em 31/12", "pessoas"),
        "uf_pas_receita_bruta": ("Receita bruta de serviços", "mil R$"),
        "uf_pas_salarios": ("Salários, retiradas e outras remunerações em serviços", "mil R$"),
    }
    for col, (desc, unid) in desc_pia.items():
        _registra(col, desc + " (valor da UF)", unid, "SIDRA 5603 (2007–2023) e 10455 (2024) – PIA-Empresa", "UF")
    for col, (desc, unid) in desc_pas.items():
        _registra(col, desc + " (valor da UF)", unid, "SIDRA 2715 (2007–2023) e 10762 (2024) – PAS", "UF")
    _registra("uf_serie_nova_pia_pas", "1 quando PIA/PAS vêm da série nova do IBGE (2024)", "0/1", "Calculado", "UF")
    return uf


# ---------------------------------------------------------------------------
# Montagem
# ---------------------------------------------------------------------------

def monta_painel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monta o painel completo município × ano e o dicionário das colunas."""
    _DICIONARIO.clear()
    mun = municipios()
    _registra("ano", "Ano de referência", "-", "-")

    # Grade completa: todo município em todo ano, mesmo sem dado (ex.: PIB de 2024, que será previsto).
    grade = pd.MultiIndex.from_product([mun["cod_ibge"], ANOS], names=["cod_ibge", "ano"])

    blocos = [pib(), pam(), ppm_rebanhos(), ppm_origem_animal()]
    painel = pd.concat([b.reindex(grade) for b in blocos], axis=1)
    painel = painel.join(populacao(grade))

    painel = painel.reset_index().merge(mun, on="cod_ibge", how="left", validate="many_to_one")
    painel = painel.merge(uf_contexto().reset_index(), on=["cod_uf", "ano"], how="left", validate="many_to_one")

    # Ordem das colunas: identificação, depois as variáveis na ordem em que foram registradas.
    ordem = [d["coluna"] for d in _DICIONARIO]
    ordem = ["cod_ibge", "ano"] + [c for c in ordem if c not in ("cod_ibge", "ano")]
    painel = painel[ordem].sort_values(["cod_ibge", "ano"]).reset_index(drop=True)

    # Verificações de integridade
    assert painel["cod_ibge"].nunique() == N_MUNICIPIOS_SUL
    assert len(painel) == N_MUNICIPIOS_SUL * len(ANOS)
    assert not painel.duplicated(["cod_ibge", "ano"]).any()

    dicionario = pd.DataFrame(_DICIONARIO).drop_duplicates("coluna").set_index("coluna").loc[ordem].reset_index()
    return painel, dicionario


def salva(painel: pd.DataFrame, dicionario: pd.DataFrame) -> None:
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    painel.to_csv(ARQ_PAINEL, index=False, encoding="utf-8")
    dicionario.to_csv(ARQ_DICIONARIO, index=False, encoding="utf-8")


if __name__ == "__main__":
    painel, dicionario = monta_painel()
    salva(painel, dicionario)
    print(f"Painel: {painel.shape[0]:,} linhas × {painel.shape[1]} colunas -> {ARQ_PAINEL}")
    print(f"Dicionário: {len(dicionario)} variáveis -> {ARQ_DICIONARIO}")
