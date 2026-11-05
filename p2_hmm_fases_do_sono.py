# -*- coding: utf-8 -*-
"""Projeto 2: Modelo Oculto de Markov (HMM) — fases do sono via smartwatch."""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from estilo_rosa import (aplicar_estilo, BLUSH, BLOSSOM, ROUGE, VINHO, CARMIM,
                         TINTA, NEUTRO, CMAP_ROSA)

SEMENTE = 2026
os.makedirs("figs", exist_ok=True)
os.makedirs("results", exist_ok=True)

FASES = ["W", "L", "P", "R"]
NOMES = ["Vigília", "Sono leve", "Sono profundo", "REM"]
W, L, P, R = range(4)
EPOCA_S = 30
EPOCAS_H = 3600 // EPOCA_S


# 1. Simulador de noites

def gerar_hipnograma(rng):
    seq = []

    def bloco(fase, minutos):
        seq.extend([fase] * max(0, int(round(minutos * 60 / EPOCA_S))))

    bloco(W, rng.uniform(8, 25))
    duracao_total = rng.uniform(7.0, 8.5) * 60
    k = 0
    while len(seq) * EPOCA_S / 60 < duracao_total - 10:
        ciclo = rng.normal(90, 10)
        profundo = max(0.0, rng.normal(45 - 13 * k, 5))
        rem = min(40.0, max(3.0, rng.normal(8 + 8 * k, 3)))
        leve = max(15.0, ciclo - profundo - rem)
        l1 = leve * rng.uniform(0.55, 0.75)
        for fase, mins in ((L, l1), (P, profundo), (L, leve - l1), (R, rem)):
            restante = mins
            while restante > 0:
                pedaco = min(restante, rng.exponential(25) + 2) if fase == L else restante
                bloco(fase, pedaco)
                restante -= pedaco
                if fase == L and restante > 0 and rng.random() < 0.35:
                    bloco(W, rng.uniform(0.5, 2.5))
        if rng.random() < 0.5:
            bloco(W, rng.uniform(1, 4))
        k += 1
    bloco(W, rng.uniform(3, 12))
    return np.array(seq, dtype=int)


def gerar_sinais(hip, rng):
    n = len(hip)
    base_fc = rng.normal(62, 6)
    base_vfc = rng.normal(np.log(40), 0.25)
    t = np.arange(n) / n

    desloc_fc = np.array([7.0, 0.0, -3.0, 2.0])[hip]
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.85 * ar[i - 1] + rng.normal(0, 1.0)
    fc = base_fc + desloc_fc - 4.0 * t + ar + rng.normal(0, 1.0, n)
    fc += np.where(hip == R, rng.normal(0, 3.0, n), 0.0)

    desloc_vfc = np.array([-0.25, 0.0, 0.22, -0.12])[hip]
    log_vfc = base_vfc + desloc_vfc + rng.normal(0, 0.30, n)

    p_mexe = np.array([0.85, 0.20, 0.05, 0.12])[hip]
    media = np.array([40.0, 4.0, 2.0, 3.0])[hip]
    mov = np.where(rng.random(n) < p_mexe, rng.poisson(media), 0)
    return np.column_stack([fc, np.exp(log_vfc), mov]).astype(float)


def preparar_observacoes(sinais):
    x = np.column_stack([sinais[:, 0], np.log(sinais[:, 1])])
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return np.column_stack([x, (sinais[:, 2] > 0).astype(float)])


# 2. Modelo Oculto de Markov (HMM)

def log_emissoes(o, mu, var, q):
    x, mexeu = o[:, :2], o[:, 2:3]
    dif = x[:, None, :] - mu[None, :, :]
    lg = -0.5 * np.sum(dif ** 2 / var[None] + np.log(2 * np.pi * var[None]), axis=2)
    lb = mexeu * np.log(q[None, :]) + (1 - mexeu) * np.log(1 - q[None, :])
    return lg + lb


def logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    s = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return s if keepdims else np.squeeze(s, axis=axis)


def forward_backward(logB, log_pi, log_A):
    T, N = logB.shape
    log_alfa = np.zeros((T, N))
    log_beta = np.zeros((T, N))
    log_alfa[0] = log_pi + logB[0]
    for t in range(1, T):
        log_alfa[t] = logB[t] + logsumexp(log_alfa[t - 1][:, None] + log_A, axis=0)
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(log_A + (logB[t + 1] + log_beta[t + 1])[None, :], axis=1)
    log_veross = logsumexp(log_alfa[-1])
    gama = np.exp(log_alfa + log_beta - log_veross)
    log_xi = (log_alfa[:-1, :, None] + log_A[None] +
              (logB[1:] + log_beta[1:])[:, None, :] - log_veross)
    xi_soma = np.exp(logsumexp(log_xi, axis=0))
    return gama, xi_soma, log_veross


def viterbi(logB, log_pi, log_A):
    T, N = logB.shape
    delta = log_pi + logB[0]
    psi = np.zeros((T, N), dtype=int)
    for t in range(1, T):
        cand = delta[:, None] + log_A
        psi[t] = np.argmax(cand, axis=0)
        delta = cand[psi[t], np.arange(N)] + logB[t]
    caminho = np.zeros(T, dtype=int)
    caminho[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        caminho[t - 1] = psi[t, caminho[t]]
    return caminho, float(np.max(delta))


class HMMSono:
    def __init__(self, pi, A, mu, var, q):
        self.pi, self.A, self.mu, self.var, self.q = (
            np.array(pi, float), np.array(A, float), np.array(mu, float),
            np.array(var, float), np.array(q, float))

    def lb(self, o):
        return log_emissoes(o, self.mu, self.var, self.q)

    def logs(self):
        return np.log(self.pi + 1e-300), np.log(self.A + 1e-300)

    def decodificar(self, x):
        lp, lA = self.logs()
        return viterbi(self.lb(x), lp, lA)[0]

    def posteriores(self, x):
        lp, lA = self.logs()
        return forward_backward(self.lb(x), lp, lA)[0]

    def sem_memoria(self, x):
        lb = self.lb(x)
        priori = np.log(self.frac + 1e-300) if hasattr(self, "frac") else 0.0
        return np.argmax(lb + priori, axis=1)

    @staticmethod
    def supervisionado(noites_x, noites_y):
        N = 4
        cont = np.ones((N, N))
        ini = np.ones(N)
        xs, ys = np.vstack(noites_x), np.concatenate(noites_y)
        for y in noites_y:
            ini[y[0]] += 1
            np.add.at(cont, (y[:-1], y[1:]), 1)
        mu = np.array([xs[ys == i, :2].mean(axis=0) for i in range(N)])
        var = np.array([xs[ys == i, :2].var(axis=0) for i in range(N)]) + 1e-3
        q = np.array([(xs[ys == i, 2].sum() + 1) / ((ys == i).sum() + 2) for i in range(N)])
        m = HMMSono(ini / ini.sum(), cont / cont.sum(axis=1, keepdims=True), mu, var, q)
        m.frac = np.bincount(ys, minlength=N) / len(ys)
        return m

    def baum_welch(self, noites_x, iteracoes=30, tol=1e-3, ao_iterar=None):
        historico = []
        for it in range(iteracoes):
            lp, lA = self.logs()
            s_pi = np.zeros(4); s_xi = np.zeros((4, 4))
            s_g = np.zeros(4); s_gx = np.zeros((4, 2)); s_gx2 = np.zeros((4, 2))
            s_gm = np.zeros(4)
            total = 0.0
            for o in noites_x:
                g, xi, ll = forward_backward(self.lb(o), lp, lA)
                total += ll
                s_pi += g[0]; s_xi += xi
                s_g += g.sum(axis=0)
                s_gx += g.T @ o[:, :2]; s_gx2 += g.T @ (o[:, :2] ** 2)
                s_gm += g.T @ o[:, 2]
            historico.append(total)
            if ao_iterar is not None:
                ao_iterar(self)
            self.pi = (s_pi + 1e-3) / (s_pi + 1e-3).sum()
            self.A = (s_xi + 1e-3) / (s_xi + 1e-3).sum(axis=1, keepdims=True)
            self.mu = s_gx / s_g[:, None]
            self.var = np.maximum(s_gx2 / s_g[:, None] - self.mu ** 2, 1e-3)
            self.q = np.clip(s_gm / s_g, 1e-3, 1 - 1e-3)
            self.frac = s_g / s_g.sum()
            if it > 0 and (historico[-1] - historico[-2]) < tol * abs(historico[-2]) * 1e-2:
                break
        return historico


# 3. Métricas

def kappa_cohen(y, yhat, n=4):
    m = np.zeros((n, n))
    np.add.at(m, (y, yhat), 1)
    po = np.trace(m) / m.sum()
    pe = (m.sum(axis=0) * m.sum(axis=1)).sum() / m.sum() ** 2
    return (po - pe) / (1 - pe)


def resumo_noite(h):
    sono = h != W
    return {"tts_min": sono.sum() * EPOCA_S / 60,
            "rem_pct": 100 * (h == R).sum() / max(1, sono.sum()),
            "prof_pct": 100 * (h == P).sum() / max(1, sono.sum()),
            "transicoes": int((h[1:] != h[:-1]).sum())}


def avaliar(pares):
    y = np.concatenate([a for a, _ in pares]); yh = np.concatenate([b for _, b in pares])
    m = np.zeros((4, 4), int)
    np.add.at(m, (y, yh), 1)
    r_v = [resumo_noite(a) for a, _ in pares]; r_p = [resumo_noite(b) for _, b in pares]
    erro = {k: float(np.mean([abs(p[k] - v[k]) for v, p in zip(r_v, r_p)])) for k in r_v[0]}
    return {"acuracia": float((y == yh).mean()), "kappa": float(kappa_cohen(y, yh)),
            "recall": (np.diag(m) / m.sum(axis=1)).tolist(), "matriz": m.tolist(),
            "erro_medio_indicadores": erro,
            "transicoes_medias": float(np.mean([p["transicoes"] for p in r_p]))}


# 4. Experimento

def main():
    rng = np.random.default_rng(SEMENTE)
    N_TREINO, N_TESTE = 20, 10
    noites = []
    for _ in range(N_TREINO + N_TESTE):
        h = gerar_hipnograma(rng)
        s = gerar_sinais(h, rng)
        noites.append((h, s, preparar_observacoes(s)))
    treino, teste = noites[:N_TREINO], noites[N_TREINO:]
    res = {"noites_treino": N_TREINO, "noites_teste": N_TESTE,
           "epocas_por_noite": [len(n[0]) for n in noites]}
    y_all = np.concatenate([n[0] for n in noites])
    res["proporcao_fases_real"] = (np.bincount(y_all, minlength=4) / len(y_all)).tolist()

    t0 = time.perf_counter()
    hmm_sup = HMMSono.supervisionado([n[2] for n in treino], [n[0] for n in treino])
    res["tempo_treino_supervisionado_s"] = time.perf_counter() - t0

    mu0 = np.array([[1.0, -0.5],
                    [0.0, 0.0],
                    [-0.8, 1.0],
                    [0.4, -0.5]])
    q0 = np.array([0.8, 0.2, 0.05, 0.1])
    A0 = np.full((4, 4), 0.01) + np.eye(4) * 0.96
    A0 /= A0.sum(axis=1, keepdims=True)
    hmm_bw = HMMSono([0.7, 0.1, 0.1, 0.1], A0, mu0, np.ones((4, 2)), q0)
    palpite = HMMSono([0.7, 0.1, 0.1, 0.1], A0, mu0, np.ones((4, 2)), q0)
    palpite.frac = np.full(4, 0.25)
    t0 = time.perf_counter()
    acc_it = []

    def medir(m):
        acc_it.append(float(np.mean(np.concatenate(
            [m.decodificar(n[2]) == n[0] for n in teste]))))

    hist = hmm_bw.baum_welch([n[2] for n in treino], iteracoes=40, ao_iterar=medir)
    res["baum_welch_acuracia_teste"] = acc_it
    res["tempo_baum_welch_s"] = time.perf_counter() - t0
    res["baum_welch_logveross"] = hist

    metodos = {
        "Sem memória (só emissões)": lambda x: hmm_sup.sem_memoria(x),
        "HMM supervisionado + forward-backward": lambda x: np.argmax(hmm_sup.posteriores(x), 1),
        "HMM supervisionado + Viterbi": lambda x: hmm_sup.decodificar(x),
        "Palpite do especialista + Viterbi": lambda x: palpite.decodificar(x),
        "Baum-Welch (sem rótulos) + Viterbi": lambda x: hmm_bw.decodificar(x),
    }
    res["metodos"] = {}
    previsoes = {}
    for nome, f in metodos.items():
        t0 = time.perf_counter()
        prev = [f(n[2]) for n in teste]
        dt = (time.perf_counter() - t0) / N_TESTE
        previsoes[nome] = prev
        r = avaliar([(n[0], p) for n, p in zip(teste, prev)])
        r["tempo_por_noite_ms"] = 1e3 * dt
        res["metodos"][nome] = r
    res["transicoes_reais_medias"] = float(np.mean([resumo_noite(n[0])["transicoes"]
                                                    for n in teste]))
    res["A_supervisionado"] = hmm_sup.A.tolist()
    res["A_baum_welch"] = hmm_bw.A.tolist()
    res["mu_supervisionado"] = hmm_sup.mu.tolist()
    res["mu_baum_welch"] = hmm_bw.mu.tolist()
    res["q_supervisionado"] = hmm_sup.q.tolist()
    res["q_baum_welch"] = hmm_bw.q.tolist()
    res["duracao_media_min_modelo"] = (0.5 / (1 - np.diag(hmm_sup.A))).tolist()

    # 5. Figuras
    aplicar_estilo()
    plt.rcParams.update({"font.size": 8.5})
    cores = [BLUSH, BLOSSOM, VINHO, ROUGE]
    noite = teste[0]
    h, s, x = noite
    horas = np.arange(len(h)) / EPOCAS_H
    vit = previsoes["HMM supervisionado + Viterbi"][0]
    sm = previsoes["Sem memória (só emissões)"][0]
    gama = hmm_sup.posteriores(x)
    ordem_plot = np.array([3, 1, 0, 2])
    nivel = {W: 3, R: 2, L: 1, P: 0}

    fig, axs = plt.subplots(4, 1, figsize=(7.4, 6.0), sharex=True,
                            gridspec_kw={"height_ratios": [1.1, 1, 1, 0.9]})
    ax = axs[0]
    ax.plot(horas, s[:, 0], color=CARMIM, lw=0.7)
    ax.set_ylabel("FC (bpm)", color=CARMIM)
    ax2 = ax.twinx()
    ax2.bar(horas, s[:, 2], width=1 / EPOCAS_H, color=NEUTRO, alpha=0.55)
    ax2.set_ylabel("movimento", color=NEUTRO)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    ax.set_title("(a) O que o relógio mede", loc="left", fontsize=8.5)

    def hipno(ax, seq, titulo, cor):
        y = np.array([nivel[v] for v in seq])
        ax.step(horas, y, where="post", color=cor, lw=1.1)
        ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(["Prof.", "Leve", "REM", "Vigília"])
        ax.set_title(titulo, loc="left", fontsize=8.5)
        yr = np.where(seq == R, 2, np.nan)
        ax.plot(horas, yr, color=ROUGE, lw=2.4, solid_capstyle="butt")

    hipno(axs[1], h, "(b) Hipnograma real", TINTA)
    hipno(axs[2], vit, "(c) Estimado pelo HMM (Viterbi)  ·  em cinza: classificador sem memória",
          VINHO)
    y_sm = np.array([nivel[v] for v in sm])
    axs[2].step(horas, y_sm + 0.0, where="post", color=NEUTRO, lw=0.5, zorder=0, alpha=0.8)
    base = np.zeros(len(h))
    for i in [P, L, R, W]:
        axs[3].fill_between(horas, base, base + gama[:, i], step="post",
                            color=cores[i], lw=0, label=NOMES[i])
        base += gama[:, i]
    axs[3].set_ylim(0, 1); axs[3].set_ylabel("P(fase | sinais)")
    axs[3].set_title("(d) Probabilidade de cada fase (forward-backward)", loc="left", fontsize=8.5)
    axs[3].legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.62))
    axs[3].set_xlabel("horas desde que deitou")
    fig.tight_layout()
    fig.savefig("figs/fig04_p2_noite.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1, 1.2]})
    m = np.array(res["metodos"]["HMM supervisionado + Viterbi"]["matriz"], float)
    mn = m / m.sum(axis=1, keepdims=True)
    axs[0].imshow(mn, cmap=CMAP_ROSA, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            axs[0].text(j, i, f"{mn[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if mn[i, j] > 0.55 else TINTA)
    axs[0].set_xticks(range(4)); axs[0].set_xticklabels(NOMES, fontsize=6.5, rotation=20)
    axs[0].set_yticks(range(4)); axs[0].set_yticklabels(NOMES, fontsize=6.5)
    axs[0].set_xlabel("estimado (Viterbi)"); axs[0].set_ylabel("real")
    axs[0].set_title("(a) Matriz de confusão", loc="left", fontsize=8)
    axs[0].grid(False)
    it = np.arange(len(hist))
    axs[1].plot(it, np.array(hist) / sum(len(n[0]) for n in treino),
                "o-", ms=3.5, lw=1.6, color=ROUGE)
    axs[1].set_xlabel("iteração do Baum-Welch (0 = palpite do especialista)")
    axs[1].set_ylabel("log-verossimilhança por época", color=ROUGE)
    axb = axs[1].twinx()
    axb.plot(it, 100 * np.array(acc_it), "s--", ms=3.5, lw=1.4, color=VINHO)
    axb.set_ylabel("acurácia no teste (%)", color=VINHO)
    axb.spines["top"].set_visible(False)
    axb.grid(False)
    axs[1].set_title("(b) Baum-Welch: verossimilhança sobe, acurácia cai", loc="left", fontsize=8)
    fig.tight_layout()
    fig.savefig("figs/fig05_p2_avaliacao.png", dpi=220)
    plt.close(fig)

    # 6. Saída
    with open("results/p2_resultados.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    print("Épocas por noite:", min(res["epocas_por_noite"]), "a", max(res["epocas_por_noite"]))
    print("Proporção real das fases (W, L, P, R):",
          [f"{p:.1%}" for p in res["proporcao_fases_real"]])
    print(f"Transições reais por noite: {res['transicoes_reais_medias']:.1f}")
    for nome, r in res["metodos"].items():
        e = r["erro_medio_indicadores"]
        print(f"{nome:40s} acc {r['acuracia']:.3f} kappa {r['kappa']:.3f} "
              f"recall {' '.join(f'{v:.2f}' for v in r['recall'])} | trans {r['transicoes_medias']:.0f}"
              f" | erro TTS {e['tts_min']:.1f} min, REM {e['rem_pct']:.1f} pp, "
              f"prof {e['prof_pct']:.1f} pp | {r['tempo_por_noite_ms']:.1f} ms/noite")
    print("Acurácia no teste ao longo do Baum-Welch:", [f"{a:.3f}" for a in acc_it])
    print("Baum-Welch: iterações", len(hist), "log-veross. inicial/final",
          f"{hist[0]:.0f} / {hist[-1]:.0f}", f"tempo {res['tempo_baum_welch_s']:.1f} s")
    print("Duração média por fase segundo o modelo (min):",
          [f"{d:.1f}" for d in res["duracao_media_min_modelo"]])
    print("A (supervisionado):\n", np.round(hmm_sup.A, 3))
    print("A (Baum-Welch):\n", np.round(hmm_bw.A, 3))
    print("mu sup:\n", np.round(hmm_sup.mu, 2), "\nmu BW:\n", np.round(hmm_bw.mu, 2))
    print("q sup:", np.round(hmm_sup.q, 3), " q BW:", np.round(hmm_bw.q, 3))


if __name__ == "__main__":
    main()
