# ultra-bot/main.py
# Multi-agent backtest for stocks and crypto, long-only, about 3 years.
# Past performance is not a guarantee. Paper trading only.

import time
import numpy as np
import pandas as pd
import yfinance as yf

SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "BTC-USD", "ETH-USD"]
MACRO_SYMBOL, VIX_SYMBOL = "SPY", "^VIX"
START_CAPITAL = 100.0
TEST_YEARS = 3
RISK_PER_TRADE = 0.015     # 1.5% Kapitalrisiko pro Trade
MAX_POSITION_PCT = 0.25    # max. 25 % des Kapitals in einer Position
MAX_OPEN = 4               # Bis zu 4 Positionen gleichzeitig
MAX_DRAWDOWN = 0.15        # 15% Drawdown-Limit
FEE = 0.001                # 0,1 % pro Seite
SLIPPAGE = 0.0005          # 0,05 % pro Seite
STOP_ATR = 2.0             # Initialer Stop-Loss bei 2x ATR
TRAIL_ATR = 3.0            # Trailing Stop zieht bei 3x ATR nach
MAX_HOLD_DAYS = 60         # Maximale Haltedauer

def load_real(symbol):
    print(f"Lade {symbol} von Yahoo Finance...", flush=True)
    try:
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(years=TEST_YEARS + 1)
        df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True, multi_level_index=False)
        if df.empty: raise RuntimeError("Keine Daten geladen.")
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if df.index.tz is not None: df.index = df.index.tz_localize(None).normalize()
        return df
    except Exception as exc:
        raise SystemExit(f"Fehler bei {symbol}: {exc}")

def add_indicators(df):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    pc = c.shift(1)
    df["ema20"], df["ema50"], df["ema200"] = (c.ewm(span=s, adjust=False).mean() for s in (20, 50, 200))
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    
    # RSI
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    ls = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + g / ls.replace(0, np.nan))
    
    # MACD
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    df["vol_ratio"] = v / v.rolling(20).mean()
    df["high20"] = h.rolling(20).max().shift(1)
    return df

def agent_makro(spy_row, vix_row):
    return spy_row["Close"] > spy_row["ema200"] and vix_row["Close"] < 35

def agent_score(r, vix_row):
    score = 0
    if r["Close"] > r["ema50"]: score += 1.5
    if r["ema50"] > r["ema200"]: score += 1.5
    if r["rsi"] > 50: score += 1.0
    if r["macd_hist"] > 0: score += 1.5
    if r["Close"] > r["high20"]: score += 2.0 
    elif abs(r["Close"] - r["ema20"]) < r["atr"] and r["Close"] > r["ema50"]: score += 1.5
    if r["vol_ratio"] > 1.2: score += 1.0
    
    if r["Close"] > (r["ema20"] * 1.15) or vix_row["Close"] > 40: return 0 
    return score

def agent_risiko(price, atr, equity, cash):
    stop = price - STOP_ATR * atr
    risk_per_share = price - stop
    if risk_per_share <= 0: return None
    shares = (equity * RISK_PER_TRADE) / risk_per_share
    shares = min(shares, equity * MAX_POSITION_PCT / price, cash / (price * (1 + FEE)))
    return (shares, stop) if shares > 0 else None

def run():
    print("Lade Daten...")
    data = {s: add_indicators(load_real(s)) for s in SYMBOLS + [MACRO_SYMBOL, VIX_SYMBOL]}
    spy, vix = data[MACRO_SYMBOL], data[VIX_SYMBOL]
    all_dates = sorted(set().union(*[set(data[s].index) for s in SYMBOLS]))
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=TEST_YEARS)
    dates = [d for d in all_dates if d >= cutoff]
    cash, peak, halted = START_CAPITAL, START_CAPITAL, False
    positions, pending, trades, curve = {}, [], [], []
    
    print(f"Starte Backtest für {len(dates)} Tage...")
    for date in dates:
        for sym in list(positions):
            if date not in data[sym].index: continue
            row, p = data[sym].loc[date], positions[sym]
            p["days"] += 1
            exit_price = None
            if row["High"] > p["entry"] + TRAIL_ATR * p["atr"]:
                new_stop = row["Close"] - STOP_ATR * p["atr"]
                if new_stop > p["stop"]: p["stop"] = new_stop
            if row["Open"] <= p["stop"]: exit_price = row["Open"]
            elif row["Low"] <= p["stop"]: exit_price = p["stop"]
            elif p["days"] >= MAX_HOLD_DAYS: exit_price = row["Close"]
            
            if exit_price is not None:
                exit_price *= (1 - SLIPPAGE)
                proceeds = p["shares"] * exit_price * (1 - FEE)
                cash += proceeds
                trades.append({"symbol": sym, "entry_date": p["date"], "exit_date": date,
                               "entry": p["entry"], "exit": exit_price,
                               "pnl": proceeds - p["cost"], "days": p["days"]})
                del positions[sym]

        equity = cash + sum(p["shares"] * data[s].loc[:date, "Close"].iloc[-1] for s, p in positions.items())
        peak = max(peak, equity)
        if equity < peak * (1 - MAX_DRAWDOWN): halted = True
        curve.append((date, equity))

        if not halted:
            for sym, atr, score in sorted(pending, key=lambda x: -x[2]):
                if sym in positions or len(positions) >= MAX_OPEN or date not in data[sym].index: continue
                price = data[sym].loc[date, "Open"] * (1 + SLIPPAGE)
                plan = agent_risiko(price, atr, equity, cash)
                if plan is None: continue
                shares, stop = plan
                cost = shares * price * (1 + FEE)
                cash -= cost
                positions[sym] = {"shares": shares, "entry": price, "stop": stop,
                                  "atr": atr, "cost": cost, "date": date, "days": 0}
        pending = []

        if halted or date not in spy.index or date not in vix.index: continue
        if not agent_makro(spy.loc[date], vix.loc[date]): continue
            
        for sym in SYMBOLS:
            if sym in positions or date not in data[sym].index: continue
            r = data[sym].loc[date]
            if r[["ema200", "rsi", "atr", "high20", "vol_ratio"]].isna().any(): continue
            score = agent_score(r, vix.loc[date])
            if score >= 6.5: pending.append((sym, r["atr"], score))

    for sym, p in positions.items():
        last = data[sym]["Close"].iloc[-1]
        proceeds = p["shares"] * last * (1 - FEE)
        cash += proceeds
        trades.append({"symbol": sym, "entry_date": p["date"], "exit_date": dates[-1],
                       "entry": p["entry"], "exit": last, "pnl": proceeds - p["cost"], "days": p["days"]})
    
    tr = pd.DataFrame(trades)
    eq = pd.Series(dict(curve))
    print("\n=== ERGEBNIS ===")
    if tr.empty:
        print("Keine Trades - Marktbedingungen wurden nicht erfüllt.")
        return
    final = START_CAPITAL + tr["pnl"].sum()
    wins, losses = tr[tr.pnl > 0], tr[tr.pnl <= 0]
    pf = wins.pnl.sum() / abs(losses.pnl.sum()) if len(losses) and losses.pnl.sum() else float("inf")
    dd = ((eq / eq.cummax()) - 1).min()
    
    print(f"Startkapital:   {START_CAPITAL:.2f}  ->  Endkapital: {final:.2f}")
    print(f"Rendite:        {final / START_CAPITAL - 1:+.1%}")
    print(f"Max. Drawdown:  {dd:.1%}")
    print(f"Trades:         {len(tr)}")
    print(f"Trefferquote:   {len(wins) / len(tr):.1%}")
    print(f"Profit Factor:  {pf:.2f}")
    print("\nPro Symbol:")
    print(tr.groupby("symbol")["pnl"].agg(["count", "sum"]).round(2).to_string())

if __name__ == "__main__":
    run()