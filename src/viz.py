"""Estilo único dos gráficos do projeto (matplotlib) e funções de figura reutilizáveis.

Paleta validada (light mode) com o validador de paletas de visualização:

* Categórica (até 3 séries, válida para todos os pares): azul, laranja, verde-água.
* Divergente para as 5 categorias de crescimento: vermelho (retração), cinza
  neutro (estabilidade) e azul (crescimento). Cada braço é monotônico em
  luminosidade a partir do neutro.
* Sequencial: rampa de azul.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.coleta import RAIZ

DIR_FIGURAS = RAIZ / "reports" / "figures"

# Superfície e tinta
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_SUAVE = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"

# Categórica (ordem fixa; nunca reciclar). As 3 primeiras valem para qualquer par de cores
# (dispersão, facetas); as 8 valem para pares adjacentes (linhas, barras).
CATEGORICA_8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CATEGORICA = CATEGORICA_8[:3]
CORES_UF = {"PR": CATEGORICA[0], "SC": CATEGORICA[1], "RS": CATEGORICA[2]}
# Modelos: baselines em cinza (referência), modelos treinados na ordem categórica
CORES_MODELOS = {
    "Persistência (PIB_t = PIB_t-1)": "#898781",
    "Ridge": CATEGORICA_8[0],
    "Huber": CATEGORICA_8[1],
    "Random Forest": CATEGORICA_8[2],
    "LightGBM": CATEGORICA_8[3],
    "Ensemble (Ridge + RF + LightGBM)": CATEGORICA_8[4],
}

# Divergente: categorias de crescimento (retração → crescimento)
CORES_CATEGORIAS = {
    "Forte retração": "#c23b3a",
    "Retração": "#f09a8f",
    "Estabilidade": "#d9d8d2",
    "Crescimento moderado": "#86b6ef",
    "Forte crescimento": "#2a78d6",
}

# Sequencial (azul, claro → escuro)
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
CMAP_SEQUENCIAL = LinearSegmentedColormap.from_list("azul", RAMPA_AZUL)
# Divergente contínua (correlações): vermelho ← cinza → azul
CMAP_DIVERGENTE = LinearSegmentedColormap.from_list(
    "verm_azul", ["#a32f2f", "#c23b3a", "#f09a8f", "#f0efec", "#86b6ef", "#2a78d6", "#1c5cab"]
)


def aplica_estilo() -> None:
    """Configura o matplotlib: marcas finas, grade discreta, sem bordas superior e direita."""
    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 10,
        "text.color": TINTA,
        "axes.labelcolor": TINTA_SECUNDARIA,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.edgecolor": EIXO,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.6,
        "xtick.color": TINTA_SUAVE,
        "ytick.color": TINTA_SUAVE,
        "xtick.labelcolor": TINTA_SECUNDARIA,
        "ytick.labelcolor": TINTA_SECUNDARIA,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICA),
    })


def salva_figura(fig: plt.Figure, nome: str) -> Path:
    """Salva em reports/figures/<nome>.png (200 dpi) e devolve o caminho."""
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    caminho = DIR_FIGURAS / f"{nome}.png"
    fig.savefig(caminho)
    return caminho


def barras_categorias_por_ano(tabela_pct, ax=None, titulo=None):
    """Barras horizontais 100% empilhadas: participação de cada categoria de crescimento por ano.

    ``tabela_pct``: DataFrame com anos no índice e as 5 categorias nas colunas (somando 100).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(tabela_pct) + 1.6))
    anos = [str(a) for a in tabela_pct.index]
    esquerda = [0.0] * len(tabela_pct)
    for cat, cor in CORES_CATEGORIAS.items():
        valores = tabela_pct[cat].to_numpy()
        ax.barh(anos, valores, left=esquerda, color=cor, edgecolor=SUPERFICIE, linewidth=1.5,
                height=0.72, label=cat)
        for i, (v, e) in enumerate(zip(valores, esquerda)):
            if v >= 7:  # rótulo só onde cabe
                cor_txt = "#ffffff" if cat in ("Forte retração", "Forte crescimento") else TINTA
                ax.text(e + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8, color=cor_txt)
        esquerda = [e + v for e, v in zip(esquerda, valores)]
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("% dos municípios")
    ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0, -0.12 if len(tabela_pct) > 6 else -0.25),
              handlelength=1.2, columnspacing=1.2)
    if titulo:
        ax.set_title(titulo)
    return ax
