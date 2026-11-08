"""Coleta dos dados brutos do IBGE (SIDRA + API de Localidades) para a Região Sul.

Cada função baixa uma fonte e grava um CSV em ``data/raw/`` em formato longo,
mantendo o valor original do SIDRA como texto (coluna ``valor_raw``). A conversão
para número e o tratamento dos símbolos especiais ficam para o pré-processamento
(ver :func:`para_numerico`).

Uso pela linha de comando (na raiz do projeto)::

    python -m src.coleta            # baixa só o que ainda não existe em data/raw/
    python -m src.coleta --force    # baixa tudo de novo
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[1]
DIR_RAW = RAIZ / "data" / "raw"

SIDRA_URL = "https://apisidra.ibge.gov.br/values"
LOCALIDADES_URL = "https://servicodados.ibge.gov.br/api/v1/localidades"

UFS_SUL = {41: "PR", 42: "SC", 43: "RS"}
COD_REGIAO_SUL = 4
N_MUNICIPIOS_SUL = 1191  # 399 (PR) + 295 (SC) + 497 (RS)

# Filtro territorial do SIDRA: todos os municípios (n6) dentro das UFs (n3) do Sul.
NIVEL_MUNICIPIOS_SUL = "n6/in n3 " + ",".join(str(uf) for uf in UFS_SUL)
NIVEL_UFS_SUL = "n3/" + ",".join(str(uf) for uf in UFS_SUL)

ANO_INICIAL = 2002  # primeiro ano da série do PIB dos Municípios (tabela 5938)
ANO_FINAL = 2024    # último ano necessário (ano a ser previsto)


# ---------------------------------------------------------------------------
# Cliente da API do SIDRA
# ---------------------------------------------------------------------------

def _get_json(url: str, tentativas: int = 5, timeout: int = 180) -> list | dict:
    """GET com novas tentativas e espera crescente (a API do SIDRA às vezes falha)."""
    for i in range(1, tentativas + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            # Força UTF-8: a API não declara o charset de forma confiável.
            return json.loads(r.content.decode("utf-8"))
        except (requests.RequestException, ValueError) as erro:
            if i == tentativas:
                raise RuntimeError(f"Falha ao acessar {url}: {erro}") from erro
            espera = 5 * i
            print(f"    tentativa {i} falhou ({erro.__class__.__name__}); nova tentativa em {espera}s")
            time.sleep(espera)
    raise AssertionError("inalcançável")


def consulta_sidra(
    tabela: int,
    variaveis: list[int],
    periodos: list[int] | str,
    nivel: str = NIVEL_MUNICIPIOS_SUL,
    classificacoes: dict[int, list[int] | str] | None = None,
) -> pd.DataFrame:
    """Faz uma consulta à API ``/values`` do SIDRA e devolve um DataFrame em formato longo.

    Parâmetros
    ----------
    tabela : número da tabela no SIDRA.
    variaveis : códigos das variáveis (``v``).
    periodos : lista de anos ou ``"all"``.
    nivel : recorte territorial no formato da API (padrão: municípios do Sul).
    classificacoes : ``{codigo_classificacao: [categorias]}`` ou ``"all"``.
    """
    per = periodos if isinstance(periodos, str) else ",".join(map(str, periodos))
    partes = [f"t/{tabela}", nivel, "v/" + ",".join(map(str, variaveis)), f"p/{per}"]
    for cod, cats in (classificacoes or {}).items():
        cats_txt = cats if isinstance(cats, str) else ",".join(map(str, cats))
        partes.append(f"c{cod}/{cats_txt}")
    url = f"{SIDRA_URL}/{'/'.join(partes)}?formato=json".replace(" ", "%20")

    dados = _get_json(url)
    cabecalho, linhas = dados[0], dados[1:]
    df = pd.DataFrame(linhas).rename(columns=cabecalho)
    return df


def _padroniza_sidra(df: pd.DataFrame, nome_local: str = "Município") -> pd.DataFrame:
    """Renomeia as colunas do SIDRA para nomes curtos e estáveis."""
    ren = {
        f"{nome_local} (Código)": "cod_local",
        nome_local: "nome_local",
        "Ano (Código)": "ano",
        "Variável (Código)": "cod_variavel",
        "Variável": "variavel",
        "Unidade de Medida": "unidade",
        "Valor": "valor_raw",
    }
    df = df.rename(columns=ren)
    fixas = ["cod_local", "nome_local", "ano", "cod_variavel", "variavel", "unidade"]
    # Colunas de classificação (ex.: "Tipo de rebanho") ficam entre as fixas e o valor.
    extras = [
        c for c in df.columns
        if c not in fixas + ["valor_raw"]
        and not c.startswith(("Nível Territorial", "Unidade de Medida (Código)"))
        and not c.endswith("(Código)")
    ]
    cods_extras = [f"{c} (Código)" for c in extras if f"{c} (Código)" in df.columns]
    df = df[fixas + extras + cods_extras + ["valor_raw"]].copy()
    df["ano"] = df["ano"].astype(int)
    return df


def baixa_por_ano(
    tabela: int,
    variaveis: list[int],
    anos: range | list[int],
    nivel: str = NIVEL_MUNICIPIOS_SUL,
    classificacoes: dict[int, list[int] | str] | None = None,
    nome_local: str = "Município",
) -> pd.DataFrame:
    """Baixa uma tabela ano a ano, para não estourar o limite de valores por consulta da API."""
    partes = []
    for ano in anos:
        print(f"    tabela {tabela} | {ano}")
        df = consulta_sidra(tabela, variaveis, [ano], nivel=nivel, classificacoes=classificacoes)
        partes.append(_padroniza_sidra(df, nome_local))
    return pd.concat(partes, ignore_index=True)


# ---------------------------------------------------------------------------
# Conversão de valores
# ---------------------------------------------------------------------------

# Símbolos especiais do SIDRA (ver "Notas" de qualquer tabela):
#   "-"   zero absoluto, não resultante de arredondamento  -> 0
#   ".."  não se aplica dado numérico                       -> NaN
#   "..." dado numérico não disponível                      -> NaN
#   "X"   dado omitido para evitar identificar o informante -> NaN
SIMBOLOS_SIDRA = {"-": 0.0, "..": np.nan, "...": np.nan, "X": np.nan}


def para_numerico(serie: pd.Series) -> pd.Series:
    """Converte a coluna ``valor_raw`` do SIDRA em número, tratando os símbolos especiais."""
    s = serie.astype(str).str.strip()
    convertido = s.map(SIMBOLOS_SIDRA)
    numerico = pd.to_numeric(s.where(~s.isin(list(SIMBOLOS_SIDRA))), errors="coerce")
    return numerico.where(~s.isin(list(SIMBOLOS_SIDRA)), convertido).astype(float)


# ---------------------------------------------------------------------------
# Fontes
# ---------------------------------------------------------------------------

def coleta_municipios() -> pd.DataFrame:
    """Lista oficial dos municípios do Sul com a hierarquia territorial do IBGE."""
    dados = _get_json(f"{LOCALIDADES_URL}/regioes/{COD_REGIAO_SUL}/municipios")
    linhas = []
    for m in dados:
        micro = m["microrregiao"]
        meso = micro["mesorregiao"]
        imediata = m["regiao-imediata"]
        intermed = imediata["regiao-intermediaria"]
        linhas.append({
            "cod_ibge": m["id"],
            "municipio": m["nome"],
            "cod_uf": meso["UF"]["id"],
            "uf": meso["UF"]["sigla"],
            "cod_mesorregiao": meso["id"],
            "mesorregiao": meso["nome"],
            "cod_microrregiao": micro["id"],
            "microrregiao": micro["nome"],
            "cod_regiao_intermediaria": intermed["id"],
            "regiao_intermediaria": intermed["nome"],
            "cod_regiao_imediata": imediata["id"],
            "regiao_imediata": imediata["nome"],
        })
    df = pd.DataFrame(linhas).sort_values("cod_ibge").reset_index(drop=True)
    assert len(df) == N_MUNICIPIOS_SUL, f"esperados {N_MUNICIPIOS_SUL} municípios, vieram {len(df)}"
    return df


def coleta_pib() -> pd.DataFrame:
    """PIB dos Municípios (tabela 5938): PIB, impostos e VAB por setor, a preços correntes (mil R$)."""
    variaveis = [
        37,    # PIB a preços correntes
        543,   # Impostos, líquidos de subsídios, sobre produtos
        498,   # VAB total
        513,   # VAB agropecuária
        517,   # VAB indústria
        6575,  # VAB serviços (exceto administração pública)
        525,   # VAB administração, defesa, educação, saúde e seguridade social
    ]
    return baixa_por_ano(5938, variaveis, range(ANO_INICIAL, 2024))  # a série termina em 2023


# Principais lavouras da Região Sul (código da categoria na classificação 782).
LAVOURAS = {
    0: "Total",
    40124: "Soja",
    40122: "Milho",
    40127: "Trigo",
    40113: "Fumo",
    40102: "Arroz",
    40112: "Feijão",
    40119: "Mandioca",
    40274: "Uva",
    40260: "Maçã",
    40147: "Erva-mate",
}


def coleta_pam() -> pd.DataFrame:
    """Produção Agrícola Municipal (tabela 5457): áreas, quantidade e valor da produção."""
    variaveis = [8331, 216, 214, 215]  # área plantada, área colhida, quantidade, valor (mil R$)
    return baixa_por_ano(5457, variaveis, range(ANO_INICIAL, ANO_FINAL + 1),
                         classificacoes={782: list(LAVOURAS)})


def coleta_ppm_rebanhos() -> pd.DataFrame:
    """Pesquisa da Pecuária Municipal (tabela 3939): efetivo dos rebanhos por tipo."""
    return baixa_por_ano(3939, [105], range(ANO_INICIAL, ANO_FINAL + 1),
                         classificacoes={79: "all"})


def coleta_ppm_origem_animal() -> pd.DataFrame:
    """Pesquisa da Pecuária Municipal (tabela 74): quantidade e valor da produção de origem animal."""
    return baixa_por_ano(74, [106, 215], range(ANO_INICIAL, ANO_FINAL + 1),
                         classificacoes={80: "all"})


def coleta_populacao() -> pd.DataFrame:
    """População residente por município e ano.

    As estimativas anuais (tabela 6579) não existem nos anos de censo/contagem.
    Esses anos são preenchidos com a fonte oficial correspondente, e a coluna
    ``fonte`` registra a origem de cada valor:

    * 2007: Contagem da População 2007 (tabela 793)
    * 2010: Censo Demográfico 2010 (tabela 202)
    * 2022: Censo Demográfico 2022 (tabela 4709)

    Atenção: o IBGE **não publicou estimativas para 2023**, então esse ano fica
    ausente aqui e é interpolado no pré-processamento (entre o Censo 2022 e a
    estimativa de 2024). Pescaria Brava, Balneário Rincão e Pinto Bandeira só
    têm dados a partir de 2012/2013.
    """
    estim = consulta_sidra(6579, [9324], list(range(ANO_INICIAL, ANO_FINAL + 1)))
    estim = _padroniza_sidra(estim).assign(fonte="Estimativas de População (6579)")

    contagem = _padroniza_sidra(consulta_sidra(793, [93], [2007]))
    contagem = contagem.assign(fonte="Contagem da População 2007 (793)")

    censo10 = consulta_sidra(202, [93], [2010], classificacoes={2: [0], 1: [0]})
    censo10 = _padroniza_sidra(censo10).assign(fonte="Censo 2010 (202)")

    censo22 = _padroniza_sidra(consulta_sidra(4709, [93], [2022]))
    censo22 = censo22.assign(fonte="Censo 2022 (4709)")

    cols = ["cod_local", "nome_local", "ano", "cod_variavel", "variavel", "unidade", "valor_raw", "fonte"]
    pop = pd.concat([estim[cols], contagem[cols], censo10[cols], censo22[cols]], ignore_index=True)
    return pop.sort_values(["cod_local", "ano"]).reset_index(drop=True)


def coleta_pia_uf() -> pd.DataFrame:
    """Pesquisa Industrial Anual - Empresa, por UF.

    * Tabela 5603: série 2007–2023
    * Tabela 10455: série nova, 2024 (pode haver quebra metodológica)
    """
    antiga = consulta_sidra(5603, [706, 631, 673, 864, 810, 811], "all", nivel=NIVEL_UFS_SUL)
    antiga = _padroniza_sidra(antiga, "Unidade da Federação").assign(serie="antiga (5603)")
    nova = consulta_sidra(10455, [13816, 631, 673, 864, 810, 811], "all", nivel=NIVEL_UFS_SUL)
    nova = _padroniza_sidra(nova, "Unidade da Federação").assign(serie="nova (10455)")
    return pd.concat([antiga, nova], ignore_index=True)


def coleta_pas_uf() -> pd.DataFrame:
    """Pesquisa Anual de Serviços, por UF (total das atividades de serviços).

    * Tabela 2715: série 2007–2023. A UF vem na classificação 12354, não no nível territorial.
    * Tabela 10762: série nova, 2024 (UF no nível territorial n3).
    """
    cats_uf = {106795: 41, 106796: 42, 106797: 43}  # Paraná, Santa Catarina, Rio Grande do Sul
    antiga = consulta_sidra(2715, [630, 631, 672, 673], "all", nivel="n1/1",
                            classificacoes={12354: list(cats_uf), 12355: [107071]})
    antiga = _padroniza_sidra(antiga, "Brasil")
    col_uf = "Grandes regiões e unidades da federação de atuação das empresas"
    antiga["cod_local"] = antiga[f"{col_uf} (Código)"].astype(int).map(cats_uf).astype(str)
    antiga["nome_local"] = antiga[col_uf]
    antiga = antiga.drop(columns=[col_uf, f"{col_uf} (Código)"]).assign(serie="antiga (2715)")

    nova = consulta_sidra(10762, [630, 631, 857, 672], "all", nivel=NIVEL_UFS_SUL,
                          classificacoes={12355: [107071]})
    nova = _padroniza_sidra(nova, "Unidade da Federação").assign(serie="nova (10762)")
    return pd.concat([antiga, nova], ignore_index=True)


# Nome do arquivo em data/raw/ -> função de coleta
FONTES = {
    "municipios_sul": coleta_municipios,
    "pib_municipios": coleta_pib,
    "pam_lavouras": coleta_pam,
    "ppm_rebanhos": coleta_ppm_rebanhos,
    "ppm_origem_animal": coleta_ppm_origem_animal,
    "populacao": coleta_populacao,
    "pia_empresa_uf": coleta_pia_uf,
    "pas_uf": coleta_pas_uf,
}


def coleta_tudo(force: bool = False, fontes: list[str] | None = None) -> dict[str, Path]:
    """Baixa todas as fontes para ``data/raw/``. Sem ``force``, pula os arquivos que já existem."""
    DIR_RAW.mkdir(parents=True, exist_ok=True)
    caminhos = {}
    for nome in fontes or FONTES:
        caminho = DIR_RAW / f"{nome}.csv"
        caminhos[nome] = caminho
        if caminho.exists() and not force:
            print(f"[ok] {nome}: já existe ({caminho.name})")
            continue
        print(f"[baixando] {nome}")
        inicio = time.time()
        df = FONTES[nome]()
        df.to_csv(caminho, index=False, encoding="utf-8")
        print(f"[salvo] {nome}: {len(df):,} linhas em {time.time() - inicio:.0f}s")
    return caminhos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="baixa de novo mesmo se o arquivo existir")
    parser.add_argument("fontes", nargs="*", help=f"fontes específicas (padrão: todas): {', '.join(FONTES)}")
    args = parser.parse_args()
    desconhecidas = set(args.fontes) - set(FONTES)
    if desconhecidas:
        parser.error(f"fontes desconhecidas: {', '.join(sorted(desconhecidas))}")
    coleta_tudo(force=args.force, fontes=args.fontes or None)
