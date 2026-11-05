# -*- coding: utf-8 -*-
"""Projeto 3: Filtro de Kalman — monitor contínuo de glicose (CGM)."""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from estilo_rosa import (aplicar_estilo, BLUSH, BLOSSOM, CANDY, ROUGE, VINHO,
                         CARMIM, TINTA, NEUTRO, ARTEFATO)

os.makedirs("figs", exist_ok=True)
os.makedirs("results", exist_ok=True)

MIN_DIA = 24 * 60
LIMIAR_HIPO = 70.0
HORIZONTE = 30
PERIODO_CGM = 5

# 1. Paciente e sensor simulados

PAC = dict(Gb=140.0, p1=0.003, p2=0.025, p3=1.4e-5,
           VG=112.0, AG=0.8, tmaxG=55.0,
           VI=8.4, ke=0.138, tmaxI=55.0,
           tau_isf=10.0)
RAZAO_IC = 10.0


def gerar_rotina(dias, rng):
    T = dias * MIN_DIA
    carb_real = np.zeros(T); carb_reg = np.zeros(T)
    insulina = np.zeros(T)
    exercicio = np.zeros(T, dtype=bool)
    nao_reg = []
    for d in range(dias):
        o = d * MIN_DIA
        refeicoes = [(7.5, rng.uniform(35, 60)), (12.5, rng.uniform(50, 90)),
                     (20.0, rng.uniform(45, 80))]
        if rng.random() < 0.6:
            refeicoes.append((16.0, rng.uniform(15, 30)))
        for h, g in refeicoes:
            t = o + int(h * 60 + rng.normal(0, 20))
            carb_real[t] += g
            estimado = g * np.exp(rng.normal(0, 0.25))
            carb_reg[t] += estimado
            dose = estimado / RAZAO_IC
            if rng.random() < 0.12:
                dose *= rng.uniform(1.4, 1.8)
            atraso = int(rng.uniform(15, 30)) if rng.random() < 0.1 else 0
            insulina[t + atraso] += round(dose * 2) / 2
        if rng.random() < 0.43:
            t0 = o + int(18 * 60 + rng.normal(0, 30))
            exercicio[t0:t0 + int(rng.uniform(40, 60))] = True
        if rng.random() < 0.3:
            t = o + int(rng.uniform(10, 22) * 60)
            g = rng.uniform(10, 20)
            carb_real[t] += g
            nao_reg.append(t)
    return carb_real, carb_reg, insulina, exercicio, nao_reg


def simular_paciente(carb_real, insulina, exercicio, rng, G0=130.0):
    p = PAC
    T = len(carb_real)
    G, X, D1, D2, S1, S2, Ip, Gi = G0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, G0
    sens = 1.0
    abaixo, ultima_corr = 0, -10 ** 6
    carb = carb_real.copy()
    Gp = np.zeros(T); Gint = np.zeros(T); correcoes = []
    for t in range(T):
        if G < 65:
            abaixo += 1
        else:
            abaixo = 0
        if abaixo >= 10 and t - ultima_corr > 30:
            carb[t] += 15.0; ultima_corr = t; correcoes.append(t)
        D1 += p["AG"] * carb[t] * 1000.0
        S1 += insulina[t] * 1000.0
        if exercicio[t]:
            sens = 1.6
        else:
            sens = 1.0 + (sens - 1.0) * np.exp(-1 / 240)
        e = 0.0025 if exercicio[t] else 0.0
        Ra = D2 / p["tmaxG"]
        dG = -(p["p1"] + X * sens + e) * G + p["p1"] * p["Gb"] + Ra / p["VG"]
        dX = -p["p2"] * X + p["p3"] * Ip
        dIp = S2 / (p["tmaxI"] * p["VI"]) - p["ke"] * Ip
        D1, D2 = D1 - D1 / p["tmaxG"], D2 + (D1 - D2) / p["tmaxG"]
        S1, S2 = S1 - S1 / p["tmaxI"], S2 + (S1 - S2) / p["tmaxI"]
        G = max(G + dG, 20.0); X += dX; Ip += dIp
        Gi += (G - Gi) / p["tau_isf"]
        Gp[t], Gint[t] = G, Gi
    return Gp, Gint, correcoes


def simular_sensor(Gint, rng):
    T = len(Gint)
    y = np.full(T, np.nan)
    artefato = np.zeros(T, dtype=bool)
    v, vies = 0.0, rng.normal(0, 3)
    idx = np.arange(0, T, PERIODO_CGM)
    queda = np.zeros(T)
    dias = T // MIN_DIA
    for d in range(dias):
        o = d * MIN_DIA
        if rng.random() < 0.3:
            t0 = o + int(rng.uniform(0, 5) * 60)
            dur = int(rng.uniform(20, 45))
            prof = rng.uniform(30, 60)
            perfil = np.sin(np.linspace(0, np.pi, dur)) ** 0.5
            queda[t0:t0 + dur] = prof * perfil
            artefato[t0:t0 + dur] = True
    falha = np.zeros(T, dtype=bool)
    for d in range(dias):
        if rng.random() < 0.25:
            t0 = d * MIN_DIA + int(rng.uniform(0, 24) * 60)
            falha[t0:t0 + int(rng.uniform(20, 60))] = True
    for k, t in enumerate(idx):
        if k > 0 and t % (7 * MIN_DIA) == 0:
            vies = rng.normal(0, 3)
        vies += rng.normal(0, 0.1)
        sd = 3.0 + 0.05 * Gint[t]
        v = 0.6 * v + np.sqrt(1 - 0.6 ** 2) * rng.normal(0, sd)
        if not falha[t]:
            y[t] = max(40.0, Gint[t] + vies + v - queda[t])
    return y, artefato


# 2. Filtro de Kalman

FILTRO = dict(Gb=140.0, p1=0.003, tm=40.0, ti=60.0, ts=10.0,
              kc=6.0, ki=60.0, erro_carb=0.30, erro_ins=0.10, sigma_v=10.0)
NX = 7
iG, iGi, iM1, iM2, iI1, iI2, iD = range(NX)


def montar_modelo(f):
    A = np.zeros((NX, NX))
    A[iG, iG] = -f["p1"]; A[iG, iM2] = 1 / f["tm"]; A[iG, iI2] = -1 / f["ti"]; A[iG, iD] = 1
    A[iGi, iG] = 1 / f["ts"]; A[iGi, iGi] = -1 / f["ts"]
    A[iM1, iM1] = -1 / f["tm"]; A[iM2, iM1] = 1 / f["tm"]; A[iM2, iM2] = -1 / f["tm"]
    A[iI1, iI1] = -1 / f["ti"]; A[iI2, iI1] = 1 / f["ti"]; A[iI2, iI2] = -1 / f["ti"]
    F = np.eye(NX) + A
    B = np.zeros((NX, 3))
    B[iM1, 0] = f["kc"]; B[iI1, 1] = f["ki"]; B[iG, 2] = f["p1"] * f["Gb"]
    H = np.zeros((1, NX)); H[0, iGi] = 1.0
    return F, B, H


class KalmanGlicose:
    def __init__(self, f, q_d, usar_entradas=True, rejeitar_outliers=True):
        self.f = f
        self.F, self.B, self.H = montar_modelo(f)
        self.Q = np.diag([0.5 ** 2, 0.1 ** 2, 0.0, 0.0, 0.0, 0.0, q_d])
        self.R = f["sigma_v"] ** 2
        self.usar_entradas = usar_entradas
        self.rejeitar = rejeitar_outliers
        self.x = None
        self.P = None
        self.rejeitadas = []
        self.seguidas = 0

    def iniciar(self, y0):
        self.x = np.array([y0, y0, 0, 0, 0, 0, 0.0])
        self.P = np.diag([20.0 ** 2, 15.0 ** 2, 1.0, 1.0, 1.0, 1.0, 0.05 ** 2])

    def predizer(self, carb, ins):
        u = np.array([carb, ins, 1.0]) if self.usar_entradas else np.array([0.0, 0.0, 1.0])
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
        if self.usar_entradas:
            self.P[iM1, iM1] += (self.f["erro_carb"] * self.f["kc"] * carb) ** 2
            self.P[iI1, iI1] += (self.f["erro_ins"] * self.f["ki"] * ins) ** 2

    def atualizar(self, y, t):
        if np.isnan(y):
            return "ausente"
        nu = y - (self.H @ self.x)[0]
        S = (self.H @ self.P @ self.H.T)[0, 0] + self.R
        if self.rejeitar and nu ** 2 / S > 10.83 and self.seguidas < 2:
            self.seguidas += 1
            self.rejeitadas.append(t)
            return "rejeitada"
        self.seguidas = 0
        K = (self.P @ self.H.T / S).ravel()
        self.x = self.x + K * nu
        IKH = np.eye(NX) - np.outer(K, self.H)
        self.P = IKH @ self.P @ IKH.T + self.R * np.outer(K, K)
        return "usada"

    def prever_a_frente(self, minutos):
        x, P = self.x.copy(), self.P.copy()
        u = np.array([0.0, 0.0, 1.0])
        for _ in range(minutos):
            x = self.F @ x + self.B @ u
            P = self.F @ P @ self.F.T + self.Q
        return x[iG], np.sqrt(P[iG, iG])


def rodar_filtro(y, carb_reg, insulina, q_d, usar_entradas=True, rejeitar=True,
                 com_previsao=True):
    T = len(y)
    kf = KalmanGlicose(FILTRO, q_d, usar_entradas, rejeitar)
    t0 = int(np.flatnonzero(~np.isnan(y))[0])
    kf.iniciar(y[t0])
    est = np.full(T, np.nan); sd = np.full(T, np.nan); prev = np.full(T, np.nan)
    estados = {"usada": 0, "rejeitada": 0, "ausente": 0}
    for t in range(t0 + 1, T):
        kf.predizer(carb_reg[t], insulina[t])
        if t % PERIODO_CGM == 0:
            estados[kf.atualizar(y[t], t)] += 1
            if com_previsao:
                prev[t] = kf.prever_a_frente(HORIZONTE)[0]
        est[t], sd[t] = kf.x[iG], np.sqrt(kf.P[iG, iG])
    return est, sd, prev, estados, kf.rejeitadas


# 3. Alarmes e métricas

def eventos_hipo(G, minimo=15):
    abaixo = G < LIMIAR_HIPO
    ev, t = [], 0
    while t < len(G):
        if abaixo[t]:
            s = t
            while t < len(G) and abaixo[t]:
                t += 1
            if t - s >= minimo:
                if ev and s - ev[-1][1] < 30:
                    ev[-1] = (ev[-1][0], t)
                else:
                    ev.append((s, t))
        t += 1
    return ev


def alarmes(condicao, refratario=60):
    out, ultimo = [], -10 ** 9
    for t in np.flatnonzero(condicao):
        if t - ultimo >= refratario:
            out.append(int(t)); ultimo = t
    return out


def avaliar_alarmes(al, eventos, dias):
    al = np.array(al)
    antecedencias, detectados = [], 0
    for s, e in eventos:
        dentro = al[(al >= s - 60) & (al <= e)] if len(al) else []
        if len(dentro):
            detectados += 1
            antecedencias.append(s - dentro[0])
    falsos = 0
    for a in al:
        if not any((s - 60 <= a <= e) for s, e in eventos):
            falsos += 1
    return {"eventos": len(eventos), "detectados": detectados,
            "sensibilidade": detectados / max(1, len(eventos)),
            "antecedencia_mediana_min": float(np.median(antecedencias)) if antecedencias else None,
            "antecedencias": [int(a) for a in antecedencias],
            "alarmes": int(len(al)), "falsos": falsos,
            "falsos_por_dia": falsos / dias}


def metricas_estimativa(est, G, mascara):
    m = mascara & ~np.isnan(est)
    e = est[m] - G[m]
    atrasos = []
    for L in range(0, 21):
        idx = np.flatnonzero(m)
        idx = idx[idx >= L]
        atrasos.append(np.sqrt(np.mean((est[idx] - G[idx - L]) ** 2)))
    return {"rmse": float(np.sqrt(np.mean(e ** 2))),
            "mard_pct": float(100 * np.mean(np.abs(e) / G[m])),
            "atraso_min": int(np.argmin(atrasos))}


def tendencia_cgm(y, horizonte=HORIZONTE):
    T = len(y)
    prev = np.full(T, np.nan)
    tt = np.arange(0, T, PERIODO_CGM)
    for k in range(3, len(tt)):
        janela = y[tt[k - 3:k + 1]]
        if np.isnan(janela).any():
            continue
        a, b = np.polyfit(np.arange(4) * PERIODO_CGM, janela, 1)
        prev[tt[k]] = b + a * (3 * PERIODO_CGM + horizonte)
    return prev


# 4. Experimento

def gerar_cenario(dias, semente):
    rng = np.random.default_rng(semente)
    carb_real, carb_reg, ins, exer, nao_reg = gerar_rotina(dias, rng)
    G, Gi, corr = simular_paciente(carb_real, ins, exer, rng)
    y, artef = simular_sensor(Gi, rng)
    return dict(dias=dias, carb_real=carb_real, carb_reg=carb_reg, ins=ins, exer=exer,
                nao_reg=nao_reg, G=G, Gi=Gi, corr=corr, y=y, artef=artef)


def main():
    res = {}
    cal = gerar_cenario(10, semente=11)
    leituras_cal = (np.arange(len(cal["y"])) % PERIODO_CGM == 0)
    grade = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    curva = {"com_entradas": [], "sem_entradas": []}
    for q in grade:
        for nome, usar in (("com_entradas", True), ("sem_entradas", False)):
            est, *_ = rodar_filtro(cal["y"], cal["carb_reg"], cal["ins"], q, usar,
                                   com_previsao=False)
            curva[nome].append(metricas_estimativa(est, cal["G"], leituras_cal)["rmse"])
    q_com = grade[int(np.argmin(curva["com_entradas"]))]
    q_sem = grade[int(np.argmin(curva["sem_entradas"]))]
    res["calibracao"] = {"grade_q_d": grade, "rmse": curva, "q_com": q_com, "q_sem": q_sem}

    DIAS = 28
    cen = gerar_cenario(DIAS, semente=2026)
    G, y = cen["G"], cen["y"]
    T = len(G)
    leituras = (np.arange(T) % PERIODO_CGM == 0)
    t0 = time.perf_counter()
    est, sd, prev, estados, rejeitadas = rodar_filtro(y, cen["carb_reg"], cen["ins"], q_com)
    tempo = time.perf_counter() - t0
    est_s, sd_s, prev_s, _, _ = rodar_filtro(y, cen["carb_reg"], cen["ins"], q_sem,
                                             usar_entradas=False)
    est_nr, _, prev_nr, _, _ = rodar_filtro(y, cen["carb_reg"], cen["ins"], q_com,
                                            rejeitar=False)
    ym = np.full(T, np.nan)
    tt = np.arange(0, T, PERIODO_CGM)
    for k in range(2, len(tt)):
        janela = y[tt[k - 2:k + 1]]
        if not np.isnan(janela).all():
            ym[tt[k]] = np.nanmean(janela)
    prev_tend = tendencia_cgm(y)

    res["sensor"] = {"leituras_validas_pct": float(100 * np.mean(~np.isnan(y[leituras]))),
                     "minutos_com_artefato": int(cen["artef"].sum()),
                     "noites_com_artefato": int(len({t // MIN_DIA for t in
                                                     np.flatnonzero(cen["artef"])}))}
    res["kalman_leituras"] = estados
    rej = np.array(rejeitadas)
    res["rejeitadas_em_artefato"] = int(cen["artef"][rej].sum()) if len(rej) else 0
    res["tempo_filtro_s"] = tempo
    res["tempo_por_passo_us"] = 1e6 * tempo / T

    res["estimativa"] = {
        "Sensor (leitura bruta)": metricas_estimativa(y, G, leituras),
        "Média móvel (3 leituras)": metricas_estimativa(ym, G, leituras),
        "Kalman sem entradas": metricas_estimativa(est_s, G, leituras),
        "Kalman com entradas": metricas_estimativa(est, G, leituras),
    }
    m = leituras & ~np.isnan(est)
    res["cobertura_2sigma_pct"] = float(100 * np.mean(np.abs(est[m] - G[m]) <= 2 * sd[m]))
    res["sigma_medio"] = float(np.nanmean(sd[m]))

    ev = eventos_hipo(G)
    res["eventos_hipo"] = ev
    res["tempo_abaixo_70_pct"] = float(100 * np.mean(G < LIMIAR_HIPO))
    res["tempo_70_180_pct"] = float(100 * np.mean((G >= 70) & (G <= 180)))
    res["glicose_media"] = float(G.mean())
    res["correcoes_paciente"] = len(cen["corr"])
    cond = {
        "Alarme do sensor (leitura < 70)": leituras & (y < LIMIAR_HIPO),
        "Tendência do sensor (30 min)": leituras & (prev_tend < LIMIAR_HIPO),
        "Kalman sem entradas (30 min)": leituras & ((prev_s < LIMIAR_HIPO) | (est_s < LIMIAR_HIPO)),
        "Kalman com entradas (30 min)": leituras & ((prev < LIMIAR_HIPO) | (est < LIMIAR_HIPO)),
        "Kalman com entradas, sem rejeição": leituras & ((prev_nr < LIMIAR_HIPO) | (est_nr < LIMIAR_HIPO)),
    }
    res["alarmes"] = {}
    lista_alarmes = {}
    for nome, c in cond.items():
        al = alarmes(c)
        lista_alarmes[nome] = al
        res["alarmes"][nome] = avaliar_alarmes(al, ev, DIAS)
    for nome, al in lista_alarmes.items():
        res["alarmes"][nome]["falsos_em_artefato"] = int(sum(
            cen["artef"][a] and not any(s - 60 <= a <= e for s, e in ev) for a in al))

    # 5. Figuras
    aplicar_estilo()
    plt.rcParams.update({"font.size": 8.5})
    COR = {"G": TINTA, "cgm": NEUTRO, "kf": CANDY, "hipo": BLUSH}

    dia = int(np.argmax([sum(1 for s, _ in ev if d * MIN_DIA <= s < (d + 1) * MIN_DIA)
                         for d in range(DIAS)]))
    a, b = dia * MIN_DIA + 6 * 60, (dia + 1) * MIN_DIA + 6 * 60
    h = (np.arange(a, b) - dia * MIN_DIA) / 60
    fig, axs = plt.subplots(2, 1, figsize=(7.4, 3.9), sharex=True,
                            gridspec_kw={"height_ratios": [3.2, 1]})
    ax = axs[0]
    ax.axhspan(0, LIMIAR_HIPO, color=COR["hipo"], alpha=0.5, lw=0)
    ax.fill_between(h, est[a:b] - 2 * sd[a:b], est[a:b] + 2 * sd[a:b], color=COR["kf"],
                    alpha=0.18, lw=0, label="Kalman ± 2σ")
    ax.plot(h, G[a:b], color=COR["G"], lw=1.3, label="glicose real (sangue)")
    tl = np.arange(a, b)[leituras[a:b]]
    ax.plot((tl - dia * MIN_DIA) / 60, y[tl], ".", ms=2.8, color=COR["cgm"],
            label="leituras do sensor")
    ax.plot(h, est[a:b], color=COR["kf"], lw=1.0, label="estimativa do Kalman")
    art = cen["artef"][a:b]
    if art.any():
        for s_, e_ in _trechos(art):
            ax.axvspan(h[s_], h[e_ - 1], color=ARTEFATO, alpha=0.8, lw=0)
    ymax = 320
    marc = {"Alarme do sensor (leitura < 70)": ("v", VINHO, ymax - 20),
            "Kalman com entradas (30 min)": ("v", ROUGE, ymax - 5)}
    for nome, (mk, cor, yy) in marc.items():
        al = [t for t in lista_alarmes[nome] if a <= t < b]
        ax.plot([(t - dia * MIN_DIA) / 60 for t in al], [yy] * len(al), mk, color=cor,
                ms=5, label=nome.split(" (")[0] + ": alarme", clip_on=False)
    ax.set_ylim(40, ymax); ax.set_ylabel("glicose (mg/dL)")
    ax.legend(fontsize=6.5, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.22), frameon=False)
    ax2 = axs[1]
    hc = (np.flatnonzero(cen["carb_reg"][a:b]) + a - dia * MIN_DIA) / 60
    ax2.bar(hc, cen["carb_reg"][a:b][cen["carb_reg"][a:b] > 0], width=0.25,
            color=CARMIM, label="carboidrato registrado (g)")
    hi = (np.flatnonzero(cen["ins"][a:b]) + a - dia * MIN_DIA) / 60
    ax2.bar(hi + 0.15, 10 * cen["ins"][a:b][cen["ins"][a:b] > 0], width=0.25,
            color=VINHO, label="insulina (U × 10)")
    for s_, e_ in _trechos(cen["exer"][a:b]):
        ax2.axvspan(h[s_], h[e_ - 1], facecolor=BLOSSOM, alpha=0.5, hatch="//",
                    edgecolor="white", lw=0, label="exercício")
    nr = [(t - dia * MIN_DIA) / 60 for t in cen["nao_reg"] + cen["corr"] if a <= t < b]
    ax2.plot(nr, [5] * len(nr), "x", color=TINTA, ms=5, label="comeu sem registrar")
    ax2.legend(fontsize=6.5, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.35))
    ax2.set_xlabel("hora do dia"); ax2.set_yticks([]); ax2.grid(False)
    ax2.set_xticks(range(6, 31, 3)); ax2.set_xticklabels([f"{x % 24}h" for x in range(6, 31, 3)])
    fig.tight_layout()
    fig.savefig("figs/fig06_p3_dia.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    melhor, ganho = None, -1e9
    for s, e in ev:
        ak = [t for t in lista_alarmes["Kalman com entradas (30 min)"] if s - 60 <= t <= e]
        asn = [t for t in lista_alarmes["Alarme do sensor (leitura < 70)"] if s - 60 <= t <= e]
        if ak:
            g = (asn[0] if asn else e) - ak[0]
            if 10 < g < 60 and g > ganho:
                melhor, ganho = (s, e, ak[0], asn[0] if asn else None), g
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    s, e, tk, ts_ = melhor
    a2, b2 = s - 120, e + 45
    hh = (np.arange(a2, b2) - s) / 1
    ax.axhspan(0, LIMIAR_HIPO, color=COR["hipo"], alpha=0.5, lw=0)
    ax.plot(hh, G[a2:b2], color=COR["G"], lw=1.3, label="glicose real")
    tl = np.arange(a2, b2)[leituras[a2:b2]]
    ax.plot(tl - s, y[tl], "o", ms=2.5, color=COR["cgm"], label="sensor")
    ax.plot(hh, est[a2:b2], color=COR["kf"], lw=1.0, label="Kalman")
    kf = KalmanGlicose(FILTRO, q_com)
    kf.iniciar(y[np.flatnonzero(~np.isnan(y))[0]])
    for t in range(int(np.flatnonzero(~np.isnan(y))[0]) + 1, tk + 1):
        kf.predizer(cen["carb_reg"][t], cen["ins"][t])
        if t % PERIODO_CGM == 0:
            kf.atualizar(y[t], t)
    traj = []
    for k in range(HORIZONTE + 1):
        traj.append(kf.prever_a_frente(k)[0])
    ax.plot(np.arange(tk, tk + HORIZONTE + 1) - s, traj, "--", color=ROUGE, lw=1.8,
            label="previsão de 30 min (no alerta)")
    ax.axvline(tk - s, color=ROUGE, lw=0.8, ls=":")
    ax.annotate("alerta do Kalman", (tk - s, 175), fontsize=7, color=ROUGE, ha="right")
    if ts_ is not None:
        ax.axvline(ts_ - s, color=VINHO, lw=0.8, ls=":")
        ax.annotate("alarme do sensor", (ts_ - s, 175), fontsize=7, color=VINHO)
    ax.set_xlabel("minutos em relação ao início da hipoglicemia")
    ax.set_ylabel("glicose (mg/dL)"); ax.set_ylim(40, 200)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("(a) Episódio de hipoglicemia", loc="left", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    res["episodio_figura"] = {"inicio": int(s), "alerta_kalman": int(tk - s),
                              "alarme_sensor": None if ts_ is None else int(ts_ - s)}
    ax = axs[1]
    ax.semilogx(grade, curva["com_entradas"], "o-", color=ROUGE, label="com entradas")
    ax.semilogx(grade, curva["sem_entradas"], "s--", color=VINHO, label="sem entradas")
    ax.axhline(res_rmse_bruto := metricas_estimativa(cal["y"], cal["G"], leituras_cal)["rmse"],
               color=NEUTRO, lw=0.9, ls=":", label="sensor bruto")
    ax.set_xlabel(r"variância $q_d$ da perturbação")
    ax.set_ylabel("RMSE (mg/dL)")
    ax.set_title("(b) Ajuste de Q", loc="left", fontsize=8.5)
    ax.legend(fontsize=6.5)
    res["calibracao"]["rmse_sensor_bruto"] = res_rmse_bruto
    fig.tight_layout()
    fig.savefig("figs/fig07_p3_evento_ajuste.png", dpi=220)
    plt.close(fig)

    # 6. Saída
    with open("results/p3_resultados.json", "w", encoding="utf-8") as fjs:
        json.dump(res, fjs, ensure_ascii=False, indent=2, default=float)
    print("Calibração: q_d com entradas =", q_com, "| sem entradas =", q_sem)
    print("  RMSE com:", [f"{v:.1f}" for v in curva["com_entradas"]])
    print("  RMSE sem:", [f"{v:.1f}" for v in curva["sem_entradas"]])
    print(f"Teste ({DIAS} dias): glicose média {res['glicose_media']:.0f} mg/dL, "
          f"{res['tempo_70_180_pct']:.0f}% no alvo, {res['tempo_abaixo_70_pct']:.1f}% < 70; "
          f"{len(ev)} hipoglicemias; {res['correcoes_paciente']} correções do paciente")
    print("Sensor:", res["sensor"], "| Kalman:", estados,
          "| rejeitadas em artefato:", res["rejeitadas_em_artefato"])
    for nome, r in res["estimativa"].items():
        print(f"  {nome:28s} RMSE {r['rmse']:5.1f}  MARD {r['mard_pct']:4.1f}%  atraso {r['atraso_min']} min")
    print(f"  cobertura do intervalo ±2σ: {res['cobertura_2sigma_pct']:.1f}% (σ médio {res['sigma_medio']:.1f})")
    for nome, r in res["alarmes"].items():
        print(f"  {nome:36s} detectou {r['detectados']}/{r['eventos']}  antecedência mediana "
              f"{r['antecedencia_mediana_min']} min  falsos {r['falsos']} "
              f"({r['falsos_por_dia']:.2f}/dia; {r['falsos_em_artefato']} em artefato)")
    print(f"Tempo do filtro: {tempo:.2f} s para {T} passos ({res['tempo_por_passo_us']:.0f} µs/passo)")
    print("Episódio da figura:", res["episodio_figura"])


def _trechos(mask):
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


if __name__ == "__main__":
    main()
