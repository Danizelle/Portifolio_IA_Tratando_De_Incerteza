# -*- coding: utf-8 -*-
"""Projeto 1: Rede Bayesiana — dengue, chikungunya ou zika?"""

import itertools
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from estilo_rosa import (aplicar_estilo, PALETA_4, CARMIM, VINHO, ROUGE, TINTA,
                         NEUTRO, CMAP_ROSA, rosa_claro)

SEMENTE = 2026
os.makedirs("figs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. Estrutura e parâmetros da rede

DOENCAS = ["dengue", "chikungunya", "zika", "outra"]
SN = ["não", "sim"]


def binaria(p_sim):
    p = np.asarray(p_sim, dtype=float)
    return np.stack([1.0 - p, p], axis=-1)


REDE = {
    "Epoca": {
        "valores": SN, "pais": [],
        "cpt": np.array([0.65, 0.35]),
    },
    "Dia": {
        "valores": ["início (1-3)", "tardio (4-7)"], "pais": [],
        "cpt": np.array([0.60, 0.40]),
    },
    "Doenca": {
        "valores": DOENCAS, "pais": ["Epoca"],
        "cpt": np.array([
            [0.15, 0.04, 0.03, 0.78],
            [0.55, 0.10, 0.05, 0.30],
        ]),
    },
    "FebreAlta":    {"valores": SN, "pais": ["Doenca"],
                     "cpt": binaria([0.85, 0.90, 0.25, 0.50])},
    "Artralgia":    {"valores": SN, "pais": ["Doenca"],
                     "cpt": binaria([0.20, 0.90, 0.15, 0.10])},
    "Exantema":     {"valores": SN, "pais": ["Doenca"],
                     "cpt": binaria([0.40, 0.50, 0.90, 0.10])},
    "DorRetro":     {"valores": SN, "pais": ["Doenca"],
                     "cpt": binaria([0.60, 0.20, 0.20, 0.10])},
    "Conjuntivite": {"valores": SN, "pais": ["Doenca"],
                     "cpt": binaria([0.05, 0.15, 0.60, 0.05])},
    "Plaquetopenia": {
        "valores": SN, "pais": ["Doenca", "Dia"],
        "cpt": binaria([[0.25, 0.65],
                        [0.08, 0.12],
                        [0.03, 0.05],
                        [0.05, 0.05]]),
    },
    "NS1": {
        "valores": SN, "pais": ["Doenca", "Dia"],
        "cpt": binaria([[0.85, 0.45],
                        [0.02, 0.02],
                        [0.02, 0.02],
                        [0.02, 0.02]]),
    },
}
VARIAVEIS = list(REDE.keys())
CARD = {v: len(REDE[v]["valores"]) for v in VARIAVEIS}

ROTULO = {"Epoca": "Época\nepidêmica", "Dia": "Dia de\ndoença", "Doenca": "Doença",
          "FebreAlta": "Febre\nalta", "Artralgia": "Dor articular\nintensa",
          "Exantema": "Exantema", "DorRetro": "Dor\nretro-orbital",
          "Conjuntivite": "Conjuntivite", "Plaquetopenia": "Plaquetas\nbaixas",
          "NS1": "NS1\nreagente"}


def checar_cpts(rede):
    for v, info in rede.items():
        soma = info["cpt"].sum(axis=-1)
        assert np.allclose(soma, 1.0), f"CPT de {v} não soma 1"
        forma = tuple(CARD[p] for p in info["pais"]) + (CARD[v],)
        assert info["cpt"].shape == forma, f"forma errada na CPT de {v}"


checar_cpts(REDE)


def prob_local(rede, var, atrib):
    info = rede[var]
    indice = tuple(atrib[p] for p in info["pais"]) + (atrib[var],)
    return info["cpt"][indice]


# 2. Inferência exata por enumeração

def enumerar_todas(rede, variaveis, atrib):
    if not variaveis:
        return 1.0
    y, resto = variaveis[0], variaveis[1:]
    if y in atrib:
        return prob_local(rede, y, atrib) * enumerar_todas(rede, resto, atrib)
    total = 0.0
    for valor in range(CARD[y]):
        atrib[y] = valor
        total += prob_local(rede, y, atrib) * enumerar_todas(rede, resto, atrib)
    del atrib[y]
    return total


def consulta_enumeracao(rede, X, evidencia):
    dist = np.zeros(CARD[X])
    for x in range(CARD[X]):
        atrib = dict(evidencia)
        atrib[X] = x
        dist[x] = enumerar_todas(rede, list(rede.keys()), atrib)
    return dist / dist.sum()


# 3. Inferência exata por eliminação de variáveis

class Fator:
    def __init__(self, vars_, tabela):
        self.vars = list(vars_)
        self.tabela = np.asarray(tabela, dtype=float)

    def restringir(self, evidencia):
        idx, novas = [], []
        for v in self.vars:
            if v in evidencia:
                idx.append(evidencia[v])
            else:
                idx.append(slice(None))
                novas.append(v)
        return Fator(novas, self.tabela[tuple(idx)])

    def produto(self, outro):
        todas = self.vars + [v for v in outro.vars if v not in self.vars]

        def expandir(f):
            perm = [f.vars.index(v) for v in todas if v in f.vars]
            t = np.transpose(f.tabela, perm) if perm else f.tabela
            forma = [CARD[v] if v in f.vars else 1 for v in todas]
            return t.reshape(forma)

        return Fator(todas, expandir(self) * expandir(outro))

    def somar(self, var):
        eixo = self.vars.index(var)
        return Fator([v for v in self.vars if v != var], self.tabela.sum(axis=eixo))


def consulta_eliminacao(rede, X, evidencia, ordem=None):
    fatores = [Fator(info["pais"] + [v], info["cpt"]).restringir(evidencia)
               for v, info in rede.items()]
    ocultas = [v for v in rede if v != X and v not in evidencia]
    if ordem is None:
        ordem = []
        restantes = list(ocultas)
        while restantes:
            def tamanho(v):
                vizinhos = set()
                for f in fatores:
                    if v in f.vars:
                        vizinhos |= set(f.vars)
                return np.prod([CARD[u] for u in vizinhos])
            v = min(restantes, key=tamanho)
            ordem.append(v)
            restantes.remove(v)
    for v in ordem:
        envolvidos = [f for f in fatores if v in f.vars]
        fatores = [f for f in fatores if v not in f.vars]
        prod = envolvidos[0]
        for f in envolvidos[1:]:
            prod = prod.produto(f)
        fatores.append(prod.somar(v))
    resultado = fatores[0]
    for f in fatores[1:]:
        resultado = resultado.produto(f)
    dist = resultado.tabela.reshape(-1)
    return dist / dist.sum()


# 4. Inferência aproximada por amostragem

def amostrar_direto(rede, rng, n):
    amostras = np.zeros((n, len(VARIAVEIS)), dtype=int)
    pos = {v: i for i, v in enumerate(VARIAVEIS)}
    for v in VARIAVEIS:
        info = rede[v]
        if info["pais"]:
            idx_pais = tuple(amostras[:, pos[p]] for p in info["pais"])
            probs = info["cpt"][idx_pais]
        else:
            probs = np.broadcast_to(info["cpt"], (n, CARD[v]))
        u = rng.random(n)[:, None]
        amostras[:, pos[v]] = (u > np.cumsum(probs, axis=1)).sum(axis=1)
    return amostras


def consulta_rejeicao(rede, X, evidencia, n, rng):
    a = amostrar_direto(rede, rng, n)
    pos = {v: i for i, v in enumerate(VARIAVEIS)}
    ok = np.ones(n, dtype=bool)
    for v, val in evidencia.items():
        ok &= a[:, pos[v]] == val
    if ok.sum() == 0:
        return np.full(CARD[X], np.nan), 0.0
    cont = np.bincount(a[ok, pos[X]], minlength=CARD[X]).astype(float)
    return cont / cont.sum(), ok.mean()


def consulta_ponderacao(rede, X, evidencia, n, rng):
    pos = {v: i for i, v in enumerate(VARIAVEIS)}
    amostras = np.zeros((n, len(VARIAVEIS)), dtype=int)
    peso = np.ones(n)
    for v in VARIAVEIS:
        info = rede[v]
        if info["pais"]:
            idx_pais = tuple(amostras[:, pos[p]] for p in info["pais"])
            probs = info["cpt"][idx_pais]
        else:
            probs = np.broadcast_to(info["cpt"], (n, CARD[v]))
        if v in evidencia:
            amostras[:, pos[v]] = evidencia[v]
            peso *= probs[:, evidencia[v]]
        else:
            u = rng.random(n)[:, None]
            amostras[:, pos[v]] = (u > np.cumsum(probs, axis=1)).sum(axis=1)
    dist = np.bincount(amostras[:, pos[X]], weights=peso, minlength=CARD[X])
    return dist / dist.sum()


# 5. Experimentos

def ev(**kw):
    return dict(kw)


def main():
    rng = np.random.default_rng(SEMENTE)
    res = {}

    max_dif = 0.0
    for _ in range(200):
        k = rng.integers(0, 8)
        obs = rng.choice([v for v in VARIAVEIS if v != "Doenca"], size=k, replace=False)
        e = {v: int(rng.integers(0, CARD[v])) for v in obs}
        d1 = consulta_enumeracao(REDE, "Doenca", e)
        d2 = consulta_eliminacao(REDE, "Doenca", e)
        max_dif = max(max_dif, np.abs(d1 - d2).max())
    assert max_dif < 1e-12, "enumeração e eliminação discordam"
    res["max_diferenca_enum_vs_ve"] = max_dif

    casos = {
        "A": [("Época epidêmica, 2º dia", ev(Epoca=1, Dia=0)),
              ("+ febre alta", ev(FebreAlta=1)),
              ("+ dor retro-orbital", ev(DorRetro=1)),
              ("+ sem dor articular intensa", ev(Artralgia=0)),
              ("+ plaquetas baixas", ev(Plaquetopenia=1)),
              ("+ NS1 reagente", ev(NS1=1))],
        "B": [("Época epidêmica, 2º dia", ev(Epoca=1, Dia=0)),
              ("+ exantema", ev(Exantema=1)),
              ("+ conjuntivite", ev(Conjuntivite=1)),
              ("+ sem febre alta", ev(FebreAlta=0)),
              ("+ plaquetas normais", ev(Plaquetopenia=0)),
              ("+ NS1 não reagente", ev(NS1=0))],
    }
    res["casos"] = {}
    for nome, passos in casos.items():
        e, linhas = {}, []
        for rotulo, novo in passos:
            e.update(novo)
            linhas.append((rotulo, consulta_eliminacao(REDE, "Doenca", e).tolist()))
        res["casos"][nome] = linhas

    ctx = {}
    for ep in (0, 1):
        ctx[f"NS1+_so_epoca{ep}"] = consulta_eliminacao(
            REDE, "Doenca", ev(Epoca=ep, Dia=0, NS1=1))[0]
        ctx[f"febre_semNS1_epoca{ep}"] = consulta_eliminacao(
            REDE, "Doenca", ev(Epoca=ep, Dia=0, FebreAlta=1, DorRetro=1))[0]
    for dia in (0, 1):
        base = ev(Epoca=1, Dia=dia, FebreAlta=1, DorRetro=1)
        ctx[f"antes_NS1_dia{dia}"] = consulta_eliminacao(REDE, "Doenca", base)[0]
        ctx[f"NS1neg_dia{dia}"] = consulta_eliminacao(REDE, "Doenca", {**base, "NS1": 0})[0]
    sem_dia = ev(Epoca=1, FebreAlta=1, DorRetro=1, NS1=0)
    ctx["NS1neg_dia_desconhecido"] = consulta_eliminacao(REDE, "Doenca", sem_dia)[0]
    res["contexto"] = ctx

    N_POP = 20000
    pop = amostrar_direto(REDE, rng, N_POP)
    pos = {v: i for i, v in enumerate(VARIAVEIS)}
    verdade = pop[:, pos["Doenca"]]
    sintomas = ["FebreAlta", "Artralgia", "Exantema", "DorRetro", "Conjuntivite"]
    conjuntos = {
        "Só a época (taxa-base)": ["Epoca", "Dia"],
        "Época + sintomas": ["Epoca", "Dia"] + sintomas,
        "Época + sintomas + plaquetas": ["Epoca", "Dia"] + sintomas + ["Plaquetopenia"],
        "Época + sintomas + exames": ["Epoca", "Dia"] + sintomas + ["Plaquetopenia", "NS1"],
        "Sintomas + exames, sem a época": ["Dia"] + sintomas + ["Plaquetopenia", "NS1"],
    }

    def diagnosticar(rede, obs_vars, amostras):
        cache, pred, post = {}, np.zeros(len(amostras), int), np.zeros((len(amostras), 4))
        for i, linha in enumerate(amostras):
            chave = tuple(linha[pos[v]] for v in obs_vars)
            if chave not in cache:
                cache[chave] = consulta_eliminacao(rede, "Doenca", dict(zip(obs_vars, chave)))
            post[i] = cache[chave]
            pred[i] = int(np.argmax(cache[chave]))
        return pred, post

    res["populacao"] = {}
    for nome, obs_vars in conjuntos.items():
        t0 = time.perf_counter()
        pred, post = diagnosticar(REDE, obs_vars, pop)
        dt = time.perf_counter() - t0
        recall = [float((pred[verdade == d] == d).mean()) for d in range(4)]
        logloss = float(-np.mean(np.log(post[np.arange(N_POP), verdade] + 1e-12)))
        res["populacao"][nome] = {"acuracia": float((pred == verdade).mean()),
                                  "recall": recall, "logloss": logloss, "tempo_s": dt}
        if nome == "Época + sintomas + exames":
            matriz = np.zeros((4, 4), int)
            for v_, p_ in zip(verdade, pred):
                matriz[v_, p_] += 1
            res["matriz_confusao"] = matriz.tolist()
    res["prevalencia_pop"] = np.bincount(verdade, minlength=4).tolist()

    def perturbar(rede, sigma, rng_):
        nova = {}
        for v, info in rede.items():
            cpt = info["cpt"]
            logit = np.log(cpt) + rng_.normal(0, sigma, cpt.shape)
            p = np.exp(logit)
            nova[v] = {**info, "cpt": p / p.sum(axis=-1, keepdims=True)}
        return nova

    obs_full = conjuntos["Época + sintomas + exames"]
    acc_pert = {}
    for sigma in (0.25, 0.5, 1.0):
        accs = []
        for r in range(10):
            rede_p = perturbar(REDE, sigma, np.random.default_rng(100 + r))
            pred, _ = diagnosticar(rede_p, obs_full, pop[:5000])
            accs.append(float((pred == verdade[:5000]).mean()))
        acc_pert[str(sigma)] = (float(np.mean(accs)), float(np.min(accs)))
    res["robustez"] = acc_pert

    e_teste = ev(Epoca=1, Dia=1, FebreAlta=1, Artralgia=0, Exantema=1,
                 DorRetro=0, Conjuntivite=0, Plaquetopenia=0, NS1=0)
    exato = consulta_eliminacao(REDE, "Doenca", e_teste)

    def cronometrar(f, rep):
        t0 = time.perf_counter()
        for _ in range(rep):
            f()
        return (time.perf_counter() - t0) / rep

    e_pouca = ev(FebreAlta=1, NS1=0)
    tempos = {
        "enumeracao": cronometrar(lambda: consulta_enumeracao(REDE, "Doenca", e_pouca), 20),
        "eliminacao": cronometrar(lambda: consulta_eliminacao(REDE, "Doenca", e_pouca), 200),
        "ponderacao_10000": cronometrar(
            lambda: consulta_ponderacao(REDE, "Doenca", e_pouca, 10000, rng), 20),
        "rejeicao_10000": cronometrar(
            lambda: consulta_rejeicao(REDE, "Doenca", e_pouca, 10000, rng), 20),
    }
    res["tempos_s"] = tempos

    Ns = [100, 300, 1000, 3000, 10000, 30000, 100000]
    erro_lw, erro_rj, aceitas = [], [], []
    for n in Ns:
        el, er, ac = [], [], []
        for r in range(30):
            dl = consulta_ponderacao(REDE, "Doenca", e_teste, n, rng)
            dr, a = consulta_rejeicao(REDE, "Doenca", e_teste, n, rng)
            el.append(np.abs(dl - exato).max())
            er.append(np.abs(dr - exato).max() if a > 0 else 1.0)
            ac.append(a)
        erro_lw.append(float(np.mean(el)))
        erro_rj.append(float(np.mean(er)))
        aceitas.append(float(np.mean(ac)))
    res["amostragem"] = {"N": Ns, "erro_ponderacao": erro_lw,
                         "erro_rejeicao": erro_rj, "frac_aceita": aceitas,
                         "exato": exato.tolist()}

    # 6. Figuras
    aplicar_estilo()
    cores = {"dengue": PALETA_4[2], "chikungunya": PALETA_4[3],
             "zika": PALETA_4[1], "outra": PALETA_4[0]}

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    PB_CAIXA = "#D9D9D9"
    PB_BORDA = "#000000"
    PB_TEXTO = "#000000"
    baixo = ["FebreAlta", "Artralgia", "Exantema", "DorRetro", "Conjuntivite",
             "Plaquetopenia", "NS1"]
    posxy = {"Epoca": (0.22, 0.86), "Dia": (0.84, 0.86), "Doenca": (0.40, 0.50)}
    for i, v in enumerate(baixo):
        posxy[v] = (0.06 + i * 0.145, 0.12)
    for v, (x, y) in posxy.items():
        for p in REDE[v]["pais"]:
            px, py = posxy[p]
            ax.annotate("", xy=(x, y + 0.085), xytext=(px, py - 0.085),
                        arrowprops=dict(arrowstyle="-|>", color=PB_BORDA, lw=1.1))
    for v, (x, y) in posxy.items():
        caixa = FancyBboxPatch((x - 0.058, y - 0.085), 0.116, 0.17,
                               boxstyle="square,pad=0.012",
                               fc=PB_CAIXA, ec=PB_BORDA, lw=1.1)
        ax.add_patch(caixa)
        ax.text(x, y, ROTULO[v], ha="center", va="center", fontsize=7.5,
                color=PB_TEXTO, fontweight="bold")
    ax.text(0.485, 0.50, "dengue · chikungunya\n· zika · outra", ha="left", va="center",
            fontsize=7, style="italic", color=PB_TEXTO)
    ax.text(0.06, 0.29, "sintomas", fontsize=7.5, color=PB_TEXTO, ha="center", fontweight="bold")
    ax.text(0.94, 0.29, "exames", fontsize=7.5, color=PB_TEXTO, ha="center", fontweight="bold")
    ax.set_xlim(-0.01, 1.02); ax.set_ylim(0.0, 0.97); ax.axis("off")
    fig.tight_layout()
    fig.savefig("figs/fig01_p1_rede.png", dpi=220)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9), sharex=True)
    for ax, (nome, titulo) in zip(axs, [("A", "(a) Paciente A"), ("B", "(b) Paciente B")]):
        linhas = res["casos"][nome]
        rot = [l[0] for l in linhas][::-1]
        dados = np.array([l[1] for l in linhas])[::-1]
        esq = np.zeros(len(rot))
        for j, d in enumerate(DOENCAS):
            ax.barh(rot, dados[:, j], left=esq, color=cores[d], label=d, height=0.68,
                    edgecolor="white", linewidth=1.1)
            for i, val in enumerate(dados[:, j]):
                if val > 0.12:
                    txt = ">99%" if val > 0.995 else f"{val:.0%}"
                    cor_tx = "white" if j in (0, 1) else TINTA
                    ax.text(esq[i] + val / 2, i, txt,
                            ha="center", va="center", fontsize=6.5, color=cor_tx)
            esq += dados[:, j]
        ax.set_title(titulo, fontsize=9, loc="left")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0%", "50%", "100%"])
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    axs[0].legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(1.05, -0.1),
                  frameon=False)
    fig.tight_layout()
    fig.savefig("figs/fig02_p1_casos.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8),
                            gridspec_kw={"width_ratios": [1, 1.25]})
    m = np.array(res["matriz_confusao"])
    mn = m / m.sum(axis=1, keepdims=True)
    axs[0].imshow(mn, cmap=CMAP_ROSA, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            axs[0].text(j, i, f"{mn[i, j]:.0%}".replace(".", ","), ha="center",
                        va="center", fontsize=7,
                        color="white" if mn[i, j] > 0.55 else TINTA)
    curtos = ["dengue", "chik.", "zika", "outra"]
    axs[0].set_xticks(range(4)); axs[0].set_xticklabels(curtos, fontsize=7)
    axs[0].set_yticks(range(4)); axs[0].set_yticklabels(curtos, fontsize=7)
    axs[0].set_xlabel("diagnóstico da rede"); axs[0].set_ylabel("doença real")
    axs[0].set_title("(a) Matriz de confusão", fontsize=8, loc="left")
    axs[0].grid(False)
    axs[1].loglog(Ns, erro_lw, "o-", color=ROUGE, lw=1.6, ms=4,
                  label="ponderação por verossimilhança")
    axs[1].loglog(Ns, erro_rj, "s--", color=VINHO, lw=1.3, ms=4, label="rejeição")
    ref = erro_lw[0] * np.sqrt(Ns[0] / np.array(Ns))
    axs[1].loglog(Ns, ref, ":", color=NEUTRO, label=r"$\propto 1/\sqrt{N}$")
    axs[1].set_xlabel("amostras sorteadas (N)")
    axs[1].set_ylabel("erro máximo vs. exato")
    axs[1].set_title("(b) Inferência aproximada", fontsize=8, loc="left")
    axs[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig("figs/fig03_p1_avaliacao.png", dpi=220)
    plt.close(fig)

    # 7. Saída
    with open("results/p1_resultados.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)

    print("Diferença máxima enumeração × eliminação:", f"{max_dif:.1e}")
    for nome, linhas in res["casos"].items():
        print(f"\nPaciente {nome}")
        for rot, d in linhas:
            print(f"  {rot:32s} " + "  ".join(f"{n[:5]}={p:6.1%}" for n, p in zip(DOENCAS, d)))
    print("\nContexto (P(dengue)):")
    for k, v in ctx.items():
        print(f"  {k:28s} {v:6.1%}")
    print("\nPopulação simulada (", N_POP, "pacientes; prevalência:",
          res["prevalencia_pop"], ")")
    for nome, r in res["populacao"].items():
        print(f"  {nome:34s} acurácia {r['acuracia']:.3f}  log-perda {r['logloss']:.3f}  "
              "recall " + " ".join(f"{x:.2f}" for x in r["recall"]))
    print("  matriz de confusão:", res["matriz_confusao"])
    print("\nRobustez (acurácia média, mínima) com CPTs perturbadas:", acc_pert)
    print("\nTempos por consulta (s):", {k: f"{v*1e3:.2f} ms" for k, v in tempos.items()})
    print("Amostragem: erro LW", [f"{x:.4f}" for x in erro_lw])
    print("            erro RJ", [f"{x:.4f}" for x in erro_rj])
    print("            aceitas", [f"{x:.4f}" for x in aceitas])


if __name__ == "__main__":
    main()
