# -*- coding: utf-8 -*-
"""Estilo visual compartilhado pelos projetos do portfólio."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

BLUSH = "#FFE5EC"
PASTEL = "#FFC2D1"
BLOSSOM = "#FFB3C6"
CANDY = "#FF8FAB"
ROUGE = "#FB6F92"

CARMIM = "#C9184A"
VINHO = "#A4133C"
TINTA = "#590D22"

PALETA = [BLUSH, PASTEL, BLOSSOM, CANDY, ROUGE]
PALETA_4 = [BLUSH, BLOSSOM, ROUGE, VINHO]
GRADE = "#F6E3E8"
NEUTRO = "#9C8389"
ARTEFATO = "#E9D6DC"

CMAP_ROSA = LinearSegmentedColormap.from_list("rosa", [BLUSH, CANDY, TINTA])


def aplicar_estilo():
    sns.set_theme(style="white", font_scale=0.85)
    mpl.rcParams.update({
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": TINTA, "axes.labelcolor": TINTA,
        "text.color": TINTA, "xtick.color": TINTA, "ytick.color": TINTA,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "grid.color": GRADE, "grid.linewidth": 0.9, "axes.grid": True,
        "axes.axisbelow": True,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "font.size": 9,
        "legend.frameon": False, "legend.labelcolor": TINTA,
    })


def rosa_claro(ax):
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)


def anota_fonte(fig, texto="Fonte: elaborado pelo autor (2026)."):
    fig.text(0.5, -0.02, texto, ha="center", fontsize=6.5, color=NEUTRO,
              style="italic")


plt = plt
