#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S&P 500 Opportunity Monitor
Escanea el S&P 500, puntúa oportunidades con un modelo compuesto
(técnico 45% / fundamental 30% / sentimiento 15% / macro-Polymarket 10%)
y genera un dashboard HTML con justificativos y gráficos.

Uso:  python monitor.py
Salida: dashboard.html
"""

import json
import math
import os
import re
import sys
import time
import warnings
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------- Configuración ---------------------------------
N_CANDIDATES = 28      # candidatos técnicos que pasan a análisis profundo
N_PICKS = 12           # oportunidades finales en el dashboard
N_WATCHLIST = 10       # watchlist (siguientes en ranking)
CHART_DAYS = 130       # días de historia para los gráficos
W_TECH, W_FUND, W_SENT, W_MACRO = 0.45, 0.30, 0.15, 0.10

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}

# P/E forward de referencia por sector (aprox. histórica, para contexto relativo)
SECTOR_PE_REF = {
    "Technology": 27.0, "Information Technology": 27.0,
    "Communication Services": 19.0, "Consumer Cyclical": 22.0,
    "Consumer Discretionary": 22.0, "Consumer Defensive": 19.0,
    "Consumer Staples": 19.0, "Healthcare": 18.0, "Health Care": 18.0,
    "Financial Services": 14.0, "Financials": 14.0, "Industrials": 19.0,
    "Energy": 12.0, "Utilities": 16.0, "Real Estate": 34.0,
    "Basic Materials": 16.0, "Materials": 16.0,
}
DEFENSIVE_SECTORS = {"Consumer Defensive", "Consumer Staples", "Healthcare",
                     "Health Care", "Utilities"}
RATE_SENSITIVE_SECTORS = {"Technology", "Information Technology", "Real Estate",
                          "Consumer Cyclical", "Consumer Discretionary",
                          "Communication Services"}

POS_WORDS = set("""beat beats beating upgrade upgraded surge surges soar soars rally rallies
growth record strong bullish buy outperform raise raised raises gains gain jump jumps top
tops win wins boost boosts expand expands breakthrough approval approved accelerates
accelerate exceeds exceed profit profitable momentum upside innovative partnership deal
contract award awarded launches launch""".split())
NEG_WORDS = set("""miss misses missed downgrade downgraded cut cuts fall falls fell drop
drops dropped weak weakness bearish sell underperform lawsuit probe investigation recall
warning warns warned layoffs layoff decline declines plunge plunges slump slumps fears
fear risk risks losses loss delays delay halted halt fraud fine fined penalty downside""".split())


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------- Universo --------------------------------------
def get_sp500_universe():
    """Lista de tickers del S&P 500 desde Wikipedia (con caché local)."""
    cache = os.path.join(DATA_DIR, "sp500_universe.csv")
    try:
        html = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=UA, timeout=30).text
        df = pd.read_html(StringIO(html))[0]
        df = df.rename(columns={"Symbol": "ticker", "Security": "name",
                                "GICS Sector": "sector"})
        df = df[["ticker", "name", "sector"]]
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        df.to_csv(cache, index=False)
        log(f"Universo S&P 500: {len(df)} tickers (Wikipedia)")
        return df
    except Exception as e:
        if os.path.exists(cache):
            df = pd.read_csv(cache)
            log(f"Universo desde caché: {len(df)} tickers ({e})")
            return df
        raise RuntimeError(f"No pude obtener el universo S&P 500: {e}")


# ----------------------------- Indicadores -----------------------------------
def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def atr(high, low, close, n=14):
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(high, low, close, n=14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def compute_features(px):
    """px: DataFrame con Open/High/Low/Close/Volume de un ticker."""
    c, h, l, v = px["Close"], px["High"], px["Low"], px["Volume"]
    f = pd.DataFrame(index=px.index)
    f["close"] = c
    f["volume"] = v
    f["sma20"] = c.rolling(20).mean()
    f["sma50"] = c.rolling(50).mean()
    f["sma200"] = c.rolling(200).mean()
    f["rsi"] = rsi(c)
    f["macd"], f["macd_sig"], f["macd_hist"] = macd(c)
    mid = f["sma20"]
    sd = c.rolling(20).std()
    f["bb_up"], f["bb_lo"] = mid + 2 * sd, mid - 2 * sd
    f["atr"] = atr(h, l, c)
    f["adx"], f["pdi"], f["mdi"] = adx(h, l, c)
    f["vol20"] = v.rolling(20).mean()
    f["relvol"] = v / f["vol20"]
    f["hi52"] = c.rolling(252, min_periods=120).max()
    f["lo52"] = c.rolling(252, min_periods=120).min()
    f["ret_1m"] = c.pct_change(21)
    f["ret_3m"] = c.pct_change(63)
    f["ret_6m"] = c.pct_change(126)
    return f


# ----------------------------- Score técnico ---------------------------------
def technical_score(f):
    """Devuelve (score 0-100, setup, señales dict) usando la última fila."""
    r = f.iloc[-1]
    prev = f.iloc[-2]
    need = ["close", "sma50", "sma200", "rsi", "macd_hist", "atr", "hi52"]
    if any(pd.isna(r[k]) for k in need):
        return None

    sig = {}
    close = r["close"]
    dist_hi = close / r["hi52"] - 1            # negativo: % debajo del máx 52s
    dist_lo = close / r["lo52"] - 1
    macd_cross_days = None
    hist = f["macd_hist"].tail(8).values
    for i in range(len(hist) - 1, 0, -1):
        if hist[i] > 0 and hist[i - 1] <= 0:
            macd_cross_days = len(hist) - 1 - i
            break

    trend_up = close > r["sma50"] > r["sma200"]
    above200 = close > r["sma200"]

    # ---- subscore MOMENTUM (0-100)
    m = 0.0
    if trend_up:
        m += 25
    elif above200:
        m += 12
    if 48 <= r["rsi"] <= 68:
        m += 15
    elif 68 < r["rsi"] <= 75:
        m += 8
    if r["macd_hist"] > 0 and r["macd_hist"] >= prev["macd_hist"]:
        m += 15
    elif r["macd_hist"] > 0:
        m += 8
    if macd_cross_days is not None and macd_cross_days <= 5:
        m += 5
    if dist_hi > -0.05:
        m += 12
    elif dist_hi > -0.10:
        m += 6
    m += float(np.clip((r["relvol"] - 0.9) / (2.0 - 0.9), 0, 1)) * 10
    if r["adx"] > 20 and r["pdi"] > r["mdi"]:
        m += 10
    m += float(np.clip(r["ret_6m"] / 0.35, 0, 1)) * 8

    # ---- subscore REVERSIÓN (0-100)  (pullback dentro de tendencia alcista)
    rv = 0.0
    if above200:
        rv += 25                                # tendencia mayor intacta
    if r["rsi"] <= 35:
        rv += 20
    elif r["rsi"] <= 42:
        rv += 10
    if r["rsi"] > prev["rsi"]:
        rv += 10                                # RSI girando al alza
    if not pd.isna(r["bb_lo"]) and close <= r["bb_lo"] * 1.02:
        rv += 15                                # sobre banda inferior
    if r["macd_hist"] > prev["macd_hist"]:
        rv += 10                                # histograma mejorando
    if r["relvol"] > 1.3:
        rv += 8                                 # volumen de capitulación
    if r["ret_6m"] > 0:
        rv += 12                                # fortaleza de fondo

    if m >= rv:
        setup, score = "MOMENTUM", m
    else:
        setup, score = "REVERSION", rv

    sig.update(dict(
        close=round(float(close), 2), rsi=round(float(r["rsi"]), 1),
        rsi_prev=round(float(prev["rsi"]), 1),
        macd_hist=round(float(r["macd_hist"]), 3),
        macd_hist_prev=round(float(prev["macd_hist"]), 3),
        macd_cross_days=macd_cross_days,
        sma50=round(float(r["sma50"]), 2), sma200=round(float(r["sma200"]), 2),
        trend_up=bool(trend_up), above200=bool(above200),
        dist_hi52=round(float(dist_hi) * 100, 1),
        dist_lo52=round(float(dist_lo) * 100, 1),
        relvol=round(float(r["relvol"]), 2),
        adx=round(float(r["adx"]), 1),
        pdi_gt_mdi=bool(r["pdi"] > r["mdi"]),
        atr=round(float(r["atr"]), 2),
        ret_1m=round(float(r["ret_1m"]) * 100, 1),
        ret_3m=round(float(r["ret_3m"]) * 100, 1),
        ret_6m=round(float(r["ret_6m"]) * 100, 1),
        bb_lo=round(float(r["bb_lo"]), 2) if not pd.isna(r["bb_lo"]) else None,
        bb_up=round(float(r["bb_up"]), 2) if not pd.isna(r["bb_up"]) else None,
    ))
    return dict(score=round(min(score, 100), 1), setup=setup, signals=sig)


# ----------------------------- Fundamental ------------------------------------
def fetch_fundamentals(ticker):
    out = {}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    g = info.get
    out = dict(
        name=g("shortName") or g("longName"),
        sector=g("sector"), industry=g("industry"),
        market_cap=g("marketCap"),
        pe_trailing=g("trailingPE"), pe_forward=g("forwardPE"),
        peg=g("trailingPegRatio") or g("pegRatio"),
        rev_growth=g("revenueGrowth"), earn_growth=g("earningsGrowth"),
        margins=g("profitMargins"), roe=g("returnOnEquity"),
        debt_to_equity=g("debtToEquity"),
        fcf=g("freeCashflow"),
        target_mean=g("targetMeanPrice"),
        rec_mean=g("recommendationMean"),
        rec_key=g("recommendationKey"),
        n_analysts=g("numberOfAnalystOpinions"),
        beta=g("beta"),
        next_earnings=None,
    )
    try:
        cal = yf.Ticker(ticker).calendar
        ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if ed:
            out["next_earnings"] = str(ed[0])
    except Exception:
        pass
    return out


def fundamental_score(fund, price):
    """0-100. Datos faltantes puntúan neutral (mitad del peso del ítem)."""
    pts, mx = 0.0, 0.0

    def add(cond_pts, max_pts, available):
        nonlocal pts, mx
        mx += max_pts
        pts += cond_pts if available else max_pts * 0.5

    sector = fund.get("sector")
    pe_ref = SECTOR_PE_REF.get(sector, 20.0)
    pe = fund.get("pe_forward")
    if pe and pe > 0:
        ratio = pe / pe_ref
        add(20 * float(np.clip((1.35 - ratio) / 0.7, 0, 1)), 20, True)
    else:
        add(0, 20, False)

    peg = fund.get("peg")
    add(15 * float(np.clip((2.2 - peg) / 1.5, 0, 1)) if peg and peg > 0 else 0,
        15, bool(peg and peg > 0))

    rg = fund.get("rev_growth")
    add(15 * float(np.clip(rg / 0.20, 0, 1)) if rg is not None else 0,
        15, rg is not None)

    mgn = fund.get("margins")
    add(10 * float(np.clip(mgn / 0.22, 0, 1)) if mgn is not None else 0,
        10, mgn is not None)

    roe = fund.get("roe")
    add(10 * float(np.clip(roe / 0.25, 0, 1)) if roe is not None else 0,
        10, roe is not None)

    fcf, mcap = fund.get("fcf"), fund.get("market_cap")
    if fcf and mcap:
        fcf_yield = fcf / mcap
        add(10 * float(np.clip(fcf_yield / 0.05, 0, 1)), 10, True)
        fund["fcf_yield"] = round(fcf_yield * 100, 1)
    else:
        add(0, 10, False)

    tgt = fund.get("target_mean")
    if tgt and price:
        upside = tgt / price - 1
        fund["target_upside"] = round(upside * 100, 1)
        add(12 * float(np.clip(upside / 0.25, 0, 1)), 12, True)
    else:
        add(0, 12, False)

    rec = fund.get("rec_mean")
    add(8 * float(np.clip((3.0 - rec) / 1.5, 0, 1)) if rec else 0, 8, bool(rec))

    return round(pts / mx * 100, 1) if mx else 50.0


# ----------------------------- Sentimiento ------------------------------------
def fetch_news(ticker, limit=8):
    items = []
    try:
        raw = yf.Ticker(ticker).news or []
        for it in raw[:limit]:
            c = it.get("content", it)
            title = c.get("title")
            if not title:
                continue
            url = None
            cu = c.get("canonicalUrl") or c.get("clickThroughUrl")
            if isinstance(cu, dict):
                url = cu.get("url")
            url = url or it.get("link")
            prov = c.get("provider")
            prov = prov.get("displayName") if isinstance(prov, dict) else (c.get("publisher"))
            date = c.get("pubDate") or c.get("displayTime")
            items.append(dict(title=title, url=url, provider=prov, date=str(date or "")[:10]))
    except Exception:
        pass
    return items


def news_sentiment(items):
    """(-1..1, n_pos, n_neg) según léxico simple sobre titulares."""
    pos = neg = 0
    for it in items:
        words = re.findall(r"[a-z']+", it["title"].lower())
        p = sum(w in POS_WORDS for w in words)
        n = sum(w in NEG_WORDS for w in words)
        it["tone"] = "pos" if p > n else ("neg" if n > p else "neu")
        pos += p
        neg += n
    tot = pos + neg
    return ((pos - neg) / tot if tot else 0.0), pos, neg


def fetch_social_buzz():
    """Menciones en Reddit (r/wallstreetbets y otros subs financieros) vía ApeWisdom."""
    buzz = {}
    try:
        for page in (1, 2, 3):
            r = requests.get(
                f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}",
                headers=UA, timeout=20)
            if r.status_code != 200:
                break
            for it in r.json().get("results", []):
                buzz[it["ticker"]] = dict(
                    mentions=int(it.get("mentions") or 0),
                    upvotes=int(it.get("upvotes") or 0),
                    rank=it.get("rank"),
                    mentions_prev=int(it.get("mentions_24h_ago") or 0))
            time.sleep(0.4)
    except Exception:
        pass
    return buzz


def fetch_stocktwits(ticker):
    """Ratio bullish/bearish en Stocktwits (si la API responde)."""
    try:
        r = requests.get(
            f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
            headers=UA, timeout=12)
        if r.status_code != 200:
            return None
        msgs = r.json().get("messages", [])
        bull = bear = 0
        for m in msgs:
            s = ((m.get("entities") or {}).get("sentiment") or {})
            if s.get("basic") == "Bullish":
                bull += 1
            elif s.get("basic") == "Bearish":
                bear += 1
        tot = bull + bear
        return dict(bull=bull, bear=bear,
                    ratio=round(bull / tot, 2) if tot else None,
                    n_msgs=len(msgs))
    except Exception:
        return None


def sentiment_score(news_sc, buzz, buzz_max, st):
    """0-100 combinando noticias, buzz de Reddit (ApeWisdom) y Stocktwits."""
    s = 50.0 + news_sc * 30.0                       # noticias: ±30
    if buzz and buzz_max > 0:
        m = buzz["mentions"]
        s += min(math.log1p(m) / math.log1p(buzz_max), 1.0) * 8   # buzz: +8
        if m >= 5 and m > buzz.get("mentions_prev", 0) * 1.5:
            s += 4                                  # menciones acelerando: +4
    if st and st.get("ratio") is not None and (st["bull"] + st["bear"]) >= 5:
        s += (st["ratio"] - 0.5) * 20               # stocktwits: ±10
    return round(float(np.clip(s, 0, 100)), 1)


# ----------------------------- Polymarket -------------------------------------
MACRO_KEYWORDS = ["fed ", "fed?", "rate cut", "rate hike", "interest rate",
                  "recession", "s&p", "sp500", "inflation", "cpi", "tariff",
                  "powell", "gdp", "unemployment"]


def _poly_get(url, params):
    """GET a Polymarket; si el DNS local bloquea el dominio (común con ISPs),
    resuelve por DNS-over-HTTPS y conecta con curl --resolve."""
    import subprocess
    import urllib.parse
    try:
        r = requests.get(url, params=params, headers=UA, timeout=25)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        host = urllib.parse.urlparse(url).netloc
        ips = []
        try:
            dj = requests.get("https://dns.google/resolve",
                              params={"name": host, "type": "A"}, timeout=15).json()
            ips = [a["data"] for a in dj.get("Answer", []) if a.get("type") == 1]
        except Exception:
            pass
        ips = ips or ["104.18.34.205", "172.64.153.51"]
        qs = urllib.parse.urlencode(params)
        for ip in ips:
            try:
                out = subprocess.run(
                    ["curl", "-s", "-m", "25", "--resolve", f"{host}:443:{ip}",
                     f"{url}?{qs}", "-H", "User-Agent: Mozilla/5.0"],
                    capture_output=True, timeout=30)
                if out.returncode == 0 and out.stdout:
                    return json.loads(out.stdout)
            except Exception:
                continue
        raise


def _market_outcomes(mk):
    """[(label, prob)] de un mercado de Polymarket."""
    try:
        prices = [float(x) for x in json.loads(mk.get("outcomePrices") or "[]")]
        outcomes = json.loads(mk.get("outcomes") or "[]")
    except Exception:
        return []
    return list(zip([str(o) for o in outcomes], prices))


def fetch_polymarket():
    """Eventos macro de Polymarket (tag 'economy', API Gamma sin key).
    Devuelve (mercados para mostrar, contexto agregado)."""
    display, ctx = [], dict(p_recession=None, p_cut=None, p_hike=None)
    try:
        evs = _poly_get("https://gamma-api.polymarket.com/events",
                        {"closed": "false", "limit": 40, "order": "volume",
                         "ascending": "false", "tag_slug": "economy"})
        for ev in evs:
            title = ev.get("title") or ""
            tl = title.lower()
            vol = float(ev.get("volume") or 0)
            if vol < 5e5:
                continue
            mks = ev.get("markets") or []
            # mejor outcome del evento (para mostrar)
            best = None
            for mk in mks:
                outs = _market_outcomes(mk)
                if not outs:
                    continue
                yes = dict((o.lower(), p) for o, p in outs).get("yes", outs[0][1])
                lbl = mk.get("groupItemTitle") or ""
                if best is None or yes > best["yes"]:
                    best = dict(yes=yes, label=lbl)
            if not best:
                continue
            display.append(dict(
                question=title + (f" → {best['label']}" if best["label"] else ""),
                yes=round(best["yes"] * 100, 1),
                volume=round(vol / 1e6, 1),
                end=str(ev.get("endDate") or "")[:10]))
            # contexto agregado
            if "recession" in tl and ctx["p_recession"] is None:
                ctx["p_recession"] = round(best["yes"] * 100, 1) \
                    if not best["label"] else None
                for mk in mks:
                    for o, p in _market_outcomes(mk):
                        if o.lower() == "yes":
                            ctx["p_recession"] = round(p * 100, 1)
            if "fed decision" in tl and ctx["p_cut"] is None:
                p_dec = 0.0
                for mk in mks:
                    lbl = (mk.get("groupItemTitle") or "").lower()
                    if "decrease" in lbl or "cut" in lbl:
                        for o, p in _market_outcomes(mk):
                            if o.lower() == "yes":
                                p_dec += p
                ctx["p_cut"] = round(min(p_dec, 1.0) * 100, 1)
            if "hike" in tl and "fed" in tl and ctx["p_hike"] is None:
                for mk in mks:
                    for o, p in _market_outcomes(mk):
                        if o.lower() == "yes":
                            ctx["p_hike"] = round(p * 100, 1)
                            break
        display = display[:8]
        log(f"Polymarket: {len(display)} eventos macro | "
            f"p_cut={ctx['p_cut']} p_recession={ctx['p_recession']}")
    except Exception as e:
        log(f"Polymarket no disponible ({e})")
    return display, ctx


def macro_score(sector, ctx, regime):
    """0-100: fit del sector con el contexto macro + régimen de mercado."""
    s = 50.0
    if regime == "RISK-ON":
        s += 10
    elif regime == "RISK-OFF":
        s -= 10
    pr = ctx.get("p_recession")
    if pr is not None:
        if pr >= 35:
            s += 15 if sector in DEFENSIVE_SECTORS else -8
        elif pr <= 15:
            s += 8 if sector in RATE_SENSITIVE_SECTORS else 0
    pc = ctx.get("p_cut")
    if pc is not None and pc >= 55 and sector in RATE_SENSITIVE_SECTORS:
        s += 12
    return round(float(np.clip(s, 0, 100)), 1)


# ----------------------------- Justificativos ---------------------------------
def build_tech_bullets(setup, sig):
    tech = []
    if setup == "MOMENTUM":
        if sig["trend_up"]:
            tech.append(f"Estructura alcista completa: precio ${sig['close']} > SMA50 "
                        f"(${sig['sma50']}) > SMA200 (${sig['sma200']}).")
        elif sig["above200"]:
            tech.append(f"Precio sobre la SMA200 (${sig['sma200']}): tendencia de fondo alcista.")
        if sig["macd_cross_days"] is not None and sig["macd_cross_days"] <= 5:
            tech.append(f"Cruce alcista de MACD hace {sig['macd_cross_days']} rueda(s): "
                        f"señal fresca de entrada.")
        elif sig["macd_hist"] > 0:
            tech.append(f"MACD en zona positiva con histograma "
                        f"{'expandiéndose' if sig['macd_hist'] >= sig['macd_hist_prev'] else 'positivo'} "
                        f"({sig['macd_hist']:+.3f}).")
        tech.append(f"RSI en {sig['rsi']}: impulso {'saludable, sin sobrecompra extrema' if sig['rsi'] <= 68 else 'fuerte (vigilar sobrecompra)'}.")
        if sig["dist_hi52"] > -5:
            tech.append(f"A solo {abs(sig['dist_hi52']):.1f}% del máximo de 52 semanas: "
                        f"zona de ruptura con poca resistencia por encima.")
        if sig["adx"] > 20 and sig["pdi_gt_mdi"]:
            tech.append(f"ADX {sig['adx']} con +DI > -DI: la tendencia tiene fuerza direccional.")
    else:
        tech.append(f"Setup de reversión: RSI en {sig['rsi']} "
                    f"({'sobreventa' if sig['rsi'] <= 35 else 'zona baja'})"
                    f"{', girando al alza desde ' + str(sig['rsi_prev']) if sig['rsi'] > sig['rsi_prev'] else ''}.")
        if sig["above200"]:
            tech.append(f"La tendencia mayor sigue intacta (precio sobre SMA200 en ${sig['sma200']}): "
                        f"es un pullback, no un cambio de tendencia.")
        if sig["bb_lo"] and sig["close"] <= sig["bb_lo"] * 1.02:
            tech.append(f"Precio testeando la banda inferior de Bollinger (${sig['bb_lo']}): "
                        f"extensión estadística de -2σ.")
        if sig["macd_hist"] > sig["macd_hist_prev"]:
            tech.append("Histograma MACD mejorando: el momentum vendedor se agota.")
        if sig["ret_6m"] > 0:
            tech.append(f"Fortaleza de fondo: {sig['ret_6m']:+.1f}% en 6 meses pese al pullback actual.")

    if sig["relvol"] >= 1.2:
        tech.append(f"Volumen {sig['relvol']}x su promedio de 20 ruedas: participación institucional.")
    tech.append(f"Retornos: {sig['ret_1m']:+.1f}% (1m) · {sig['ret_3m']:+.1f}% (3m) · "
                f"{sig['ret_6m']:+.1f}% (6m).")
    return tech


def build_thesis(t, setup, sig, fund, f_score, news_items, news_sc, bz, st, m_score, ctx):
    fnd, snt, mac = [], [], []
    tech = build_tech_bullets(setup, sig)

    pe = fund.get("pe_forward")
    ref = SECTOR_PE_REF.get(fund.get("sector"), 20.0)
    if pe and pe > 0:
        rel = "descuento" if pe < ref else "prima"
        fnd.append(f"P/E forward {pe:.1f} vs ~{ref:.0f} de referencia del sector: "
                   f"cotiza con {rel} de {abs(pe / ref - 1) * 100:.0f}%.")
    if fund.get("peg"):
        fnd.append(f"PEG {fund['peg']:.2f}: "
                   f"{'valuación razonable vs crecimiento' if fund['peg'] < 1.8 else 'crecimiento ya bien pagado'}.")
    if fund.get("rev_growth") is not None:
        fnd.append(f"Crecimiento de ingresos: {fund['rev_growth'] * 100:+.1f}% interanual.")
    if fund.get("margins") is not None:
        fnd.append(f"Margen neto {fund['margins'] * 100:.1f}%"
                   + (f" · ROE {fund['roe'] * 100:.0f}%" if fund.get("roe") else "") + ".")
    if fund.get("fcf_yield") is not None:
        fnd.append(f"FCF yield {fund['fcf_yield']}%: generación de caja real.")
    if fund.get("target_upside") is not None:
        fnd.append(f"Target promedio de analistas ${fund['target_mean']:.0f} "
                   f"({fund['target_upside']:+.1f}% de upside, "
                   f"{fund.get('n_analysts') or '?'} analistas, "
                   f"consenso '{fund.get('rec_key') or 'n/d'}').")
    if not fnd:
        fnd.append("Datos fundamentales limitados en esta corrida: el peso se asignó neutral.")

    tone = "positivo" if news_sc > 0.15 else ("negativo" if news_sc < -0.15 else "neutro")
    snt.append(f"Tono de noticias recientes: {tone} "
               f"(score léxico {news_sc:+.2f} sobre {len(news_items)} titulares).")
    if bz and bz["mentions"] > 0:
        delta = bz["mentions"] - bz.get("mentions_prev", 0)
        snt.append(f"{bz['mentions']} menciones en subreddits financieros "
                   f"(r/wallstreetbets y otros, vía ApeWisdom), rank #{bz['rank']} "
                   f"de todo el mercado ({'+' if delta >= 0 else ''}{delta} vs 24h antes).")
    else:
        snt.append("Sin buzz relevante en Reddit: sin riesgo de trade saturado por retail.")
    if st and st.get("ratio") is not None:
        snt.append(f"Stocktwits: {st['bull']} bullish vs {st['bear']} bearish "
                   f"({st['ratio'] * 100:.0f}% bullish) en los últimos {st['n_msgs']} mensajes.")

    if ctx.get("p_recession") is not None:
        mac.append(f"Polymarket asigna {ctx['p_recession']:.0f}% a recesión: "
                   + ("entorno que favorece defensivos."
                      if ctx["p_recession"] >= 35 else "riesgo de cola contenido."))
    if ctx.get("p_cut") is not None:
        mac.append(f"Probabilidad de recorte de tasas (Polymarket): {ctx['p_cut']:.0f}%"
                   + (" — viento de cola para duración larga/growth." if ctx["p_cut"] >= 55 else "."))
    sec = fund.get("sector") or "n/d"
    mac.append(f"Fit macro del sector ({sec}): {m_score}/100.")

    return dict(tecnico=tech, fundamental=fnd, sentimiento=snt, macro=mac)


def trade_plan(f, sig, setup):
    close = sig["close"]
    a = sig["atr"]
    lows = f["close"].tail(11)
    swing_low = float(lows.min())
    if setup == "MOMENTUM":
        stop = min(swing_low, close - 2 * a)
    else:
        stop = float(f["close"].tail(6).min()) - 0.5 * a
    stop = round(stop, 2)
    risk = close - stop
    if risk <= 0:
        risk = 2 * a
        stop = round(close - risk, 2)
    t1 = round(close + 2 * risk, 2)
    t2 = round(close + 3 * risk, 2)
    hi52 = close / (1 + sig["dist_hi52"] / 100) if sig["dist_hi52"] < 0 else None
    shares_per_10k = math.floor(100.0 / risk) if risk > 0 else None  # riesgo 1% de $10k
    return dict(entry=close, stop=stop, t1=t1, t2=t2,
                rr1=2.0, rr2=3.0, risk_pct=round(risk / close * 100, 1),
                hi52=round(hi52, 2) if hi52 else None,
                shares_per_10k=shares_per_10k)


# ----------------------------- Backtest ---------------------------------------
def run_backtest(raw):
    """Evalúa los picks de scans históricos contra la evolución posterior.
    Reglas: entrada al cierre del día del scan; primer toque del objetivo 2R
    (+2R) o del stop (-1R); doble toque el mismo día cuenta como stop
    (conservador); si en HORIZON ruedas no tocó ninguno, cierra a mercado."""
    import glob
    HORIZON = 40
    files = sorted(glob.glob(os.path.join(DATA_DIR, "scan_*.json")))
    try:
        spy_close = raw["SPY"]["Close"].dropna()
    except Exception:
        spy_close = None
    signals, n_scans, first, last = [], 0, None, None
    for fp in files:
        try:
            with open(fp) as fh:
                sc = json.load(fh)
        except Exception:
            continue
        as_of, picks = sc.get("as_of"), sc.get("picks") or []
        if not as_of or not picks:
            continue
        n_scans += 1
        first, last = first or as_of, as_of
        cutoff = pd.Timestamp(as_of)
        for p in picks:
            t, plan = p.get("ticker"), p.get("plan") or {}
            entry, stop, t1 = plan.get("entry"), plan.get("stop"), plan.get("t1")
            if not (t and entry and stop and t1) or entry <= stop:
                continue
            try:
                px = raw[t].dropna(how="all")
            except Exception:
                continue
            fwd = px[px.index > cutoff].head(HORIZON)
            risk = entry - stop
            outcome, r_mult, days, exitp = "ABIERTA", None, int(len(fwd)), None
            for i, (_, row) in enumerate(fwd.iterrows(), 1):
                lo, hi = row.get("Low"), row.get("High")
                if pd.isna(lo) or pd.isna(hi):
                    continue
                if lo <= stop:
                    outcome, r_mult, days, exitp = "STOP", -1.0, i, stop
                    break
                if hi >= t1:
                    outcome, r_mult, days, exitp = "TARGET", 2.0, i, t1
                    break
            closes = fwd["Close"].dropna()
            if outcome == "ABIERTA" and len(closes):
                exitp = float(closes.iloc[-1])
                r_mult = (exitp - entry) / risk
                if len(fwd) >= HORIZON:
                    outcome = "TIEMPO"
            r10 = (round((float(closes.iloc[9]) / entry - 1) * 100, 2)
                   if len(closes) >= 10 else None)
            alpha10 = None
            if r10 is not None and spy_close is not None:
                try:
                    s_entry = float(spy_close[spy_close.index <= cutoff].iloc[-1])
                    s_fwd = spy_close[spy_close.index > cutoff]
                    if len(s_fwd) >= 10:
                        alpha10 = round(r10 - (float(s_fwd.iloc[9]) / s_entry - 1) * 100, 2)
                except Exception:
                    pass
            signals.append(dict(
                date=as_of, ticker=t, setup=p.get("setup"),
                score=p.get("score_total"), entry=entry, stop=stop, t1=t1,
                outcome=outcome, r=None if r_mult is None else round(r_mult, 2),
                days=days, ret10=r10, alpha10=alpha10))

    def _agg(rows):
        cl = [s for s in rows if s["outcome"] in ("STOP", "TARGET", "TIEMPO")
              and s["r"] is not None]
        rs = [s for s in rows if s["outcome"] in ("STOP", "TARGET")]
        wins = [s for s in rs if s["r"] > 0]
        pos = sum(s["r"] for s in cl if s["r"] > 0)
        neg = sum(-s["r"] for s in cl if s["r"] < 0)
        a10 = [s["alpha10"] for s in rows if s["alpha10"] is not None]
        return dict(
            n=len(rows), closed=len(cl),
            win_rate=round(len(wins) / len(rs) * 100, 1) if rs else None,
            avg_r=round(sum(s["r"] for s in cl) / len(cl), 2) if cl else None,
            profit_factor=round(pos / neg, 2) if neg > 0 else None,
            avg_alpha10=round(sum(a10) / len(a10), 2) if a10 else None)

    by_setup = {k: _agg([s for s in signals if s["setup"] == k])
                for k in ("MOMENTUM", "REVERSION")}
    _bucket = lambda s: "70+" if (s.get("score") or 0) >= 70 else \
        ("60-70" if (s.get("score") or 0) >= 60 else "<60")
    by_bucket = {b: _agg([s for s in signals if _bucket(s) == b])
                 for b in ("70+", "60-70", "<60")}
    closed = [s for s in signals if s["outcome"] in ("STOP", "TARGET", "TIEMPO")
              and s["r"] is not None]
    eq, cum = [], 0.0
    for s in sorted(closed, key=lambda x: x["date"]):
        cum += s["r"]
        eq.append(dict(date=s["date"], ticker=s["ticker"], eq=round(cum, 2)))
    signals.sort(key=lambda x: x["date"], reverse=True)
    return dict(n_scans=n_scans, first=first, last=last, horizon=HORIZON,
                overall=_agg(signals), by_setup=by_setup, by_bucket=by_bucket,
                n_open=len([s for s in signals if s["outcome"] == "ABIERTA"]),
                equity=eq, signals=signals[:80])


# ----------------------------- Serie para gráficos -----------------------------
def chart_series(f):
    tail = f.tail(CHART_DAYS)
    def col(name, nd=2):
        return [None if pd.isna(x) else round(float(x), nd) for x in tail[name]]
    return dict(
        dates=[d.strftime("%Y-%m-%d") for d in tail.index],
        close=col("close"), sma50=col("sma50"), sma200=col("sma200"),
        bb_up=col("bb_up"), bb_lo=col("bb_lo"),
        rsi=col("rsi", 1), macd=col("macd", 3), macd_sig=col("macd_sig", 3),
        macd_hist=col("macd_hist", 3),
        volume=[None if pd.isna(x) else int(x) for x in tail["volume"]],
        vol20=[None if pd.isna(x) else int(x) for x in tail["vol20"]],
    )


# ----------------------------- Pipeline ---------------------------------------
def main():
    t0 = time.time()
    universe = get_sp500_universe()
    tickers = universe["ticker"].tolist()
    meta = universe.set_index("ticker").to_dict("index")

    log(f"Descargando 1 año de OHLCV para {len(tickers)} tickers + SPY + ^VIX ...")
    raw = yf.download(tickers + ["SPY", "^VIX"], period="1y", interval="1d",
                      group_by="ticker", threads=True, progress=False,
                      auto_adjust=True)
    log(f"Descarga completa ({time.time() - t0:.0f}s)")

    # --- Régimen de mercado (SPY) y breadth
    spy = raw["SPY"].dropna()
    spy_f = compute_features(spy)
    spy_last = spy_f.iloc[-1]
    vix_close = None
    try:
        vix_close = round(float(raw["^VIX"]["Close"].dropna().iloc[-1]), 1)
    except Exception:
        pass

    above50 = above200 = valid = 0
    feats = {}
    scored = []
    for t in tickers:
        try:
            px = raw[t].dropna(how="all")
            if len(px) < 60 or pd.isna(px["Close"].iloc[-1]):
                continue
            f = compute_features(px)
            r = f.iloc[-1]
            valid += 1
            if not pd.isna(r["sma50"]) and r["close"] > r["sma50"]:
                above50 += 1
            if not pd.isna(r["sma200"]) and r["close"] > r["sma200"]:
                above200 += 1
            ts = technical_score(f)
            if ts is None:
                continue
            feats[t] = f
            scored.append(dict(ticker=t, **ts))
        except Exception:
            continue

    breadth50 = round(above50 / valid * 100, 1) if valid else None
    breadth200 = round(above200 / valid * 100, 1) if valid else None

    spy_trend_up = spy_last["close"] > spy_last["sma50"] > spy_last["sma200"]
    spy_above200 = spy_last["close"] > spy_last["sma200"]
    if spy_trend_up and (breadth50 or 0) > 55:
        regime = "RISK-ON"
    elif not spy_above200 or (breadth200 or 0) < 40:
        regime = "RISK-OFF"
    else:
        regime = "NEUTRAL"
    log(f"Régimen: {regime} | breadth>SMA50: {breadth50}% | >SMA200: {breadth200}% | VIX: {vix_close}")

    scored.sort(key=lambda x: -x["score"])
    # candidatos: mezcla de setups para diversificar
    momentum = [s for s in scored if s["setup"] == "MOMENTUM"][:int(N_CANDIDATES * 0.65)]
    reversion = [s for s in scored if s["setup"] == "REVERSION"][:N_CANDIDATES - len(momentum)]
    candidates = momentum + reversion
    log(f"Candidatos técnicos: {len(candidates)} "
        f"({len(momentum)} momentum, {len(reversion)} reversión)")

    # --- Focus list: tickers del usuario que siempre reciben análisis profundo
    scored_by = {s["ticker"]: s for s in scored}
    focus_file = os.path.join(DATA_DIR, "focus.txt")
    focus = []
    if os.path.exists(focus_file):
        with open(focus_file) as fh:
            focus = [ln.strip().upper().replace(".", "-") for ln in fh
                     if ln.strip() and not ln.strip().startswith("#")]
    in_cand = {c["ticker"] for c in candidates}
    for t in focus:
        if t in scored_by and t not in in_cand:
            candidates.append(scored_by[t])
            in_cand.add(t)
    if focus:
        log(f"Focus list: {focus}")

    # --- Macro (Polymarket)
    poly, ctx = fetch_polymarket()

    # --- Buzz social (una sola pasada, ApeWisdom agrega subreddits financieros)
    log("Buscando menciones en Reddit (ApeWisdom) ...")
    buzz_all = fetch_social_buzz()
    buzz_max = max((b["mentions"] for b in buzz_all.values()), default=0)
    log(f"ApeWisdom: {len(buzz_all)} tickers con menciones, máx: {buzz_max}")

    # --- Análisis profundo por candidato
    results = []
    for i, c in enumerate(candidates):
        t = c["ticker"]
        log(f"  [{i + 1}/{len(candidates)}] {t}: fundamentals + noticias + stocktwits")
        fund = fetch_fundamentals(t)
        if not fund.get("sector"):
            fund["sector"] = meta.get(t, {}).get("sector")
        if not fund.get("name"):
            fund["name"] = meta.get(t, {}).get("name", t)
        price = c["signals"]["close"]
        f_score = fundamental_score(fund, price)
        news_items = fetch_news(t)
        news_sc, _, _ = news_sentiment(news_items)
        st = fetch_stocktwits(t)
        bz = buzz_all.get(t)
        s_score = sentiment_score(news_sc, bz, buzz_max, st)
        m_score = macro_score(fund.get("sector"), ctx, regime)
        total = round(W_TECH * c["score"] + W_FUND * f_score
                      + W_SENT * s_score + W_MACRO * m_score, 1)
        thesis = build_thesis(t, c["setup"], c["signals"], fund, f_score,
                              news_items, news_sc, bz, st, m_score, ctx)
        plan = trade_plan(feats[t], c["signals"], c["setup"])
        results.append(dict(
            ticker=t, name=fund.get("name") or t, sector=fund.get("sector"),
            industry=fund.get("industry"), setup=c["setup"],
            score_total=total, score_tech=c["score"], score_fund=f_score,
            score_sent=s_score, score_macro=m_score,
            signals=c["signals"], fundamentals=fund, news=news_items,
            news_sentiment=round(news_sc, 2), reddit=bz,
            stocktwits=st, thesis=thesis, plan=plan,
        ))
        time.sleep(0.4)

    results.sort(key=lambda x: -x["score_total"])
    picks = results[:N_PICKS]
    watch = results[N_PICKS:N_PICKS + N_WATCHLIST]

    # análisis profundo consultable para todos los candidatos (solapa Consultar)
    deep = {r["ticker"]: {k: v for k, v in r.items() if k != "chart"}
            for r in results}

    for p in picks:
        p["chart"] = chart_series(feats[p["ticker"]])

    # análisis técnico consultable para TODO el universo (solapa Consultar)
    consult = {}
    for t, s in scored_by.items():
        f = feats[t]
        consult[t] = dict(
            name=meta.get(t, {}).get("name", t),
            sector=meta.get(t, {}).get("sector"),
            setup=s["setup"], score_tech=s["score"], signals=s["signals"],
            plan=trade_plan(f, s["signals"], s["setup"]),
            tech=build_tech_bullets(s["setup"], s["signals"]),
            chart=chart_series(f),
            day_chg=round((f["close"].iloc[-1] / f["close"].iloc[-2] - 1) * 100, 2),
        )

    # cambio diario
    for p in picks + watch:
        f = feats[p["ticker"]]
        p["day_chg"] = round((f["close"].iloc[-1] / f["close"].iloc[-2] - 1) * 100, 2)

    # --- Backtest sobre scans históricos (crece solo con cada corrida)
    backtest = run_backtest(raw)
    log(f"Backtest: {backtest['n_scans']} scans, {backtest['overall']['n']} señales "
        f"({backtest['overall']['closed']} cerradas, {backtest['n_open']} abiertas)")

    spy_tail = spy_f.tail(CHART_DAYS)
    data = dict(
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        as_of=str(spy_f.index[-1].date()),
        regime=regime,
        breadth50=breadth50, breadth200=breadth200, vix=vix_close,
        spy=dict(
            close=round(float(spy_last["close"]), 2),
            sma50=round(float(spy_last["sma50"]), 2),
            sma200=round(float(spy_last["sma200"]), 2),
            ret_1m=round(float(spy_last["ret_1m"]) * 100, 1),
            ret_6m=round(float(spy_last["ret_6m"]) * 100, 1),
            dates=[d.strftime("%Y-%m-%d") for d in spy_tail.index],
            series=[round(float(x), 2) for x in spy_tail["close"]],
            sma50_s=[None if pd.isna(x) else round(float(x), 2) for x in spy_tail["sma50"]],
            sma200_s=[None if pd.isna(x) else round(float(x), 2) for x in spy_tail["sma200"]],
        ),
        polymarket=poly, macro_ctx=ctx,
        weights=dict(tech=W_TECH, fund=W_FUND, sent=W_SENT, macro=W_MACRO),
        universe_size=valid, focus=focus, consult=consult, deep=deep,
        backtest=backtest,
        picks=picks, watchlist=[
            {k: w[k] for k in ("ticker", "name", "sector", "setup", "score_total",
                               "score_tech", "score_fund", "score_sent",
                               "score_macro", "day_chg", "signals")}
            for w in watch],
    )

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(x) for x in o]
        if isinstance(o, (np.floating, float)):
            return None if (isinstance(o, float) and (math.isnan(o) or math.isinf(o))) or \
                (isinstance(o, np.floating) and (np.isnan(o) or np.isinf(o))) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    data = clean(data)
    with open(os.path.join(DATA_DIR, f"scan_{data['as_of']}.json"), "w") as fh:
        json.dump({k: v for k, v in data.items()
                   if k not in ("consult", "backtest")}, fh)

    with open(os.path.join(BASE_DIR, "template.html"), encoding="utf-8") as fh:
        template = fh.read()
    html = template.replace("/*__DATA__*/",
                            "window.DATA = " + json.dumps(data, ensure_ascii=False) + ";")
    out = os.path.join(BASE_DIR, "dashboard.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    log(f"✅ Dashboard generado: {out}  ({time.time() - t0:.0f}s total)")
    log(f"   Top pick: {picks[0]['ticker']} ({picks[0]['score_total']}/100, {picks[0]['setup']})")


if __name__ == "__main__":
    main()
