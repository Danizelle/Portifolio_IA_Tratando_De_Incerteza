<div align="center">

# Portfólio de IA — Tratando Incerteza

**Redes Bayesianas · Modelos Ocultos de Markov · Filtros de Kalman**

[![UnB](https://img.shields.io/badge/UnB-FGA0221%20Intelig%C3%AAncia%20Artificial-A4133C?style=flat-square)](https://fga.unb.br)
[![Python](https://img.shields.io/badge/Python-3.13-FB6F92?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Matplotlib%20%7C%20Seaborn-FFB3C6?style=flat-square)](https://numpy.org/)

*Implementação do zero de três modelos probabilísticos aplicados à área de saúde, com cenários simulados e reprodutibilidade garantida por semente fixa.*

</div>

---

## Visão Geral dos Projetos

| # | Modelo | Problema | Algoritmos Implementados | Script |
|:-:|:-------|:---------|:-------------------------|:-------|
| **1** | **Rede Bayesiana** | Diagnóstico diferencial entre dengue, chikungunya, zika e outras febres | Enumeração, eliminação de variáveis, amostragem por rejeição e ponderação por verossimilhança | [`p1_rede_bayesiana_arboviroses.py`](p1_rede_bayesiana_arboviroses.py) |
| **2** | **Modelo Oculto de Markov (HMM)** | Estimação das fases do sono a partir de sinais de um *smartwatch* | *Forward-backward*, Viterbi, estimação supervisionada e Baum-Welch | [`p2_hmm_fases_do_sono.py`](p2_hmm_fases_do_sono.py) |
| **3** | **Filtro de Kalman** | Monitoramento contínuo de glicose (CGM) e alerta precoce de hipoglicemia | Kalman com entrada de controle, teste de consistência ($\chi^2$) e previsão de 30 min | [`p3_kalman_glicose.py`](p3_kalman_glicose.py) |

---

## Estrutura do Repositório

```text
.
├── estilo_rosa.py                    # Paleta e identidade visual compartilhada dos gráficos
├── p1_rede_bayesiana_arboviroses.py  # Projeto 1 — Rede Bayesiana (~5 s)
├── p2_hmm_fases_do_sono.py           # Projeto 2 — Modelo Oculto de Markov (~7 s)
├── p3_kalman_glicose.py              # Projeto 3 — Filtro de Kalman (~10 s)
├── figs/                             # Figuras (.png) geradas pelos experimentos
└── results/                          # Métricas (.json) e saídas de terminal (.txt)
```

---

## Como Executar

**1. Instale as dependências:**

```bash
pip install numpy matplotlib seaborn
```

**2. Execute os projetos:**

```bash
python p1_rede_bayesiana_arboviroses.py
python p2_hmm_fases_do_sono.py
python p3_kalman_glicose.py
```

> Cada script é autocontido: simula os dados, executa os métodos, gera os gráficos em `figs/` e salva os resultados em `results/`.

---

<div align="center">
<sub>
<b>Danielle Rodrigues Silva</b> · Matrícula 211061574<br>
FGA0221 — Inteligência Artificial · Prof. Fabiano Araujo Soares, Dr.<br>
Universidade de Brasília — Faculdade de Ciências e Tecnologias em Engenharia (UnB/FGA) · 2026
</sub>
</div>
