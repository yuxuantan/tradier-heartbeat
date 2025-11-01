#!/usr/bin/env python3
"""
heartbeat_tradier.py
Runs every minute (kept awake via `caffeinate`) and performs 4 checks:

1) Positions endpoint responds within 10s and ≥1 position.
2) Orders endpoint responds within 10s and has ≥1 ACTIVE order.
3) Preview single-leg BUY PUT (ATM, today or nearest expiry) at mid price returns status 'ok' within 10s.
4) Balance drawdown guard: today's total equity vs. yesterday's (from historical-balances or local store);
   if drop > BALANCE_DD_LIMIT_PCT (default 3%), fail & alert.

Notes
- Prints raw responses for every API call (trimmed to MAX_PRINT_CHARS).
- Handles Tradier's array/object flip in JSON (see docs' Response Format note).
"""

import os, time, json, smtplib, traceback, requests, atexit, subprocess
from datetime import datetime, date, timedelta
from email.mime.text import MIMEText
from dotenv import load_dotenv
from playsound import playsound

load_dotenv()

# === Keep system awake while this script runs ===
# (Use -ims if you want display to be able to sleep; -dims keeps display on)
caffeinate = subprocess.Popen(["caffeinate", "-dims"])
atexit.register(lambda: caffeinate.terminate())

# === CONFIG ===
TRADIER_ACCESS_TOKEN = os.getenv("TRADIER_ACCESS_TOKEN")
ACCOUNT_ID           = os.getenv("TRADIER_ACCOUNT_ID")
BASE_URL             = os.getenv("TRADIER_BASE_URL", "https://api.tradier.com/v1").rstrip("/")
HEARTBEAT_SYMBOL     = os.getenv("HEARTBEAT_SYMBOL", "SPX").upper()
ORDER_QTY            = int(os.getenv("ORDER_QTY", "1"))

# Balance check config
BALANCE_STORE        = os.getenv("BALANCE_STORE", "balance_store.json")
BALANCE_DD_LIMIT_PCT = float(os.getenv("BALANCE_DD_LIMIT_PCT", "3.0"))  # 3% default

# Email + alert
SMTP_HOST  = os.getenv("SMTP_HOST")
SMTP_PORT  = int(os.getenv("SMTP_PORT", "587")) if os.getenv("SMTP_HOST") else None
SMTP_USER  = os.getenv("SMTP_USER")
SMTP_PASS  = os.getenv("SMTP_PASS")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO   = os.getenv("EMAIL_TO")

# HTTP
HEADERS     = {"Authorization": f"Bearer {TRADIER_ACCESS_TOKEN}", "Accept": "application/json"}
REQ_TIMEOUT = 8
SLA_SECS    = 10

# Logging
MAX_PRINT_CHARS = int(os.getenv("MAX_PRINT_CHARS", "2000"))

def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def today_str(): return date.today().isoformat()

# === Alert (email + sound) ===
def alert_with_vol(volume_level: int):
    volume_level = max(0, min(100, int(volume_level)))
    subprocess.run(["osascript", "-e", "set volume output muted false",
                               "-e", f"set volume output volume {volume_level}"], check=True)
    print("🔈 Alert sound played on Mac")
    try:
        playsound("alarm2.wav")
    except Exception as e:
        print(f"[{now()}] ⚠️ Could not play sound: {e}")

def send_alert(subject, body):
    if not (SMTP_HOST and SMTP_PORT and EMAIL_FROM and EMAIL_TO):
        print(f"[{now()}] ⚠️ Email not sent (SMTP vars missing)\n{body}")
    else:
        try:
            msg = MIMEText(body, "plain")
            msg["Subject"], msg["From"], msg["To"] = subject, EMAIL_FROM, EMAIL_TO
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
                s.starttls()
                if SMTP_USER and SMTP_PASS:
                    s.login(SMTP_USER, SMTP_PASS)
                s.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())
            print(f"[{now()}] 📧 Alert email sent.")
        except Exception as e:
            print(f"[{now()}] ❌ Email send failed: {e}")
    alert_with_vol(60)

# === Utilities ===
def _pretty(obj):
    try: return json.dumps(obj, indent=2, sort_keys=True)
    except Exception: return str(obj)

def _print_response(tag, url, status_code, elapsed, content, is_json_guess=True):
    body = _pretty(content) if is_json_guess else str(content)
    if len(body) > MAX_PRINT_CHARS:
        body = body[:MAX_PRINT_CHARS] + "\n…(truncated)…"
    print(f"[{now()}] 🔎 {tag}\nURL: {url}\nStatus: {status_code} | Elapsed: {elapsed:.2f}s\nBody:\n{body}\n")

def timed_req(method, url, data=None, tag="API"):
    t0 = time.monotonic()
    try:
        if method == "get":
            r = requests.get(url, headers=HEADERS, timeout=REQ_TIMEOUT)
        else:
            r = requests.post(url, headers={**HEADERS,"Content-Type":"application/x-www-form-urlencoded"},
                              data=data, timeout=REQ_TIMEOUT)
        elapsed = time.monotonic() - t0

        # Try JSON, else text (Tradier sometimes mislabels content-type)
        is_json = False
        content_type = (r.headers.get("Content-Type") or "").lower()
        if "application/json" in content_type:
            try:
                content = r.json(); is_json = True
            except Exception:
                content = r.text
        else:
            try:
                content = r.json(); is_json = True
            except Exception:
                content = r.text

        _print_response(tag, url, r.status_code, elapsed, content, is_json_guess=is_json)
        ok = r.ok and elapsed <= SLA_SECS
        return ok, elapsed, content, r.status_code, None if ok else f"HTTP {r.status_code}"
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"[{now()}] ❌ {tag} request error: {e} (elapsed {elapsed:.2f}s)")
        return False, elapsed, None, None, str(e)

# === API wrappers ===
def get_positions():      return timed_req("get", f"{BASE_URL}/accounts/{ACCOUNT_ID}/positions", tag="Positions")
def get_orders():         return timed_req("get", f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders", tag="Orders")
def get_quote(sym):       return timed_req("get", f"{BASE_URL}/markets/quotes?symbols={sym}", tag=f"Quote {sym}")
def get_expirations(sym): return timed_req("get", f"{BASE_URL}/markets/options/expirations?symbol={sym}&includeAllRoots=true", tag=f"Expirations {sym}")
def get_chain(sym, exp):  return timed_req("get", f"{BASE_URL}/markets/options/chains?symbol={sym}&expiration={exp}", tag=f"Chain {sym} {exp}")
def get_option_quotes(symbols):
    if isinstance(symbols, (list, tuple)): symbols = ",".join(symbols)
    url = f"{BASE_URL}/markets/quotes?symbols={symbols}"
    return timed_req("get", url, tag=f"Option Quotes ({symbols})")

# Balances endpoints (current + historical)
def get_balances_now():   # current balances
    return timed_req("get", f"{BASE_URL}/accounts/{ACCOUNT_ID}/balances", tag="Balances Now")
def get_historical_balances():  # daily historical balances (if enabled for your account)
    return timed_req("get", f"{BASE_URL}/accounts/{ACCOUNT_ID}/historical-balances?period=WEEK", tag="Historical Balances")

def preview_single_option(underlying_symbol, option_symbol, side, qty, limit_price):
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders?preview=true"
    data = {
        "class": "option",
        "symbol": underlying_symbol,
        "option_symbol": option_symbol,
        "side": side,                 # "buy_to_open"
        "quantity": str(qty),
        "type": "limit",
        "duration": "day",
        "price": f"{limit_price:.2f}",
    }
    try:
        print(f"[{now()}] 🧾 Preview payload (single):\n{json.dumps(data, indent=2)}")
    except Exception:
        pass
    return timed_req("post", url, data, tag="Preview Single Option")

# === JSON helpers ===
def safe_get(d, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

ACTIVE_STATUSES = {"open","pending","live","accepted","partially_filled"}

def count_active_orders(js):
    raw = safe_get(js, "orders", "order", default=[])
    if isinstance(raw, dict): raw=[raw]
    n_total  = len(raw)
    n_active = sum(1 for o in raw if str(o.get("status","")).lower() in ACTIVE_STATUSES)
    return n_total, n_active

def nearest_valid_expiration(symbol, prefer=date.today().isoformat()):
    ok, te, exps, _, err = get_expirations(symbol)
    if not ok or not isinstance(exps, dict):
        return None, f"expirations error: {err or 'bad payload'}"
    raw = safe_get(exps, "expirations", "date", default=[])
    if isinstance(raw, str): raw=[raw]
    if not raw: return None, "no expirations available"
    if prefer in raw: return prefer, None
    try:
        future = sorted([d for d in raw if d >= prefer])
        return (future[0] if future else sorted(raw)[-1]), None
    except Exception:
        return raw[0], None

def pick_atm_put_symbol(chain_json, underlying_price):
    if not isinstance(chain_json, dict):
        return None, "chain_json is not a dict"
    options = safe_get(chain_json, "options", "option", default=[])
    if isinstance(options, dict): options=[options]
    puts = [o for o in options if str(o.get("option_type") or o.get("right") or "").lower() == "put"]
    if not puts: return None, "no put options in chain"
    def sval(o):
        try: return float(o.get("strike"))
        except: return float("inf")
    puts.sort(key=lambda o: abs(sval(o)-underlying_price))
    sym = puts[0].get("symbol") or puts[0].get("option_symbol")
    if not sym: return None, "no option symbol on selected put"
    return sym, None

def mid_from_bidask(bid, ask):
    try:
        b = float(bid) if bid is not None else None
        a = float(ask) if ask is not None else None
        if b is not None and a is not None and a >= b:
            return (a + b) / 2.0
        if a is not None: return a
        if b is not None: return b
    except Exception:
        pass
    return None

def compute_option_mid(option_symbol):
    ok, t, quotes, _, err = get_option_quotes(option_symbol)
    if not ok or not isinstance(quotes, dict):
        return None, f"option quotes error: {err or 'bad payload'}"
    q = safe_get(quotes, "quotes", "quote")
    if isinstance(q, list): q = q[0] if q else {}
    mb = mid_from_bidask(q.get("bid"), q.get("ask"))
    if mb is None:
        try:
            mb = float(q.get("last"))
        except Exception:
            mb = None
    if mb is None:
        return None, "insufficient quote data for mid"
    return max(mb, 0.01), {"bid": q.get("bid"), "ask": q.get("ask"), "last": q.get("last")}

# === Local balance store (fallback for #4) ===
def read_balance_store():
    try:
        with open(BALANCE_STORE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def write_balance_store(store):
    try:
        with open(BALANCE_STORE, "w") as f:
            json.dump(store, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[{now()}] ⚠️ Could not write balance store: {e}")

# === Checks ===
def check_positions():
    ok, t, js, status, err = get_positions()
    if not ok or not isinstance(js, dict):
        return [f"Check 1/4 FAILED - positions API error ({err}), {t:.2f}s"], None
    items = safe_get(js, "positions", "position", default=[])
    if isinstance(items, dict): items=[items]
    n = len(items)
    if n >= 1 and t <= SLA_SECS:
        print(f"✅ Check 1/4 passed - detected {n} positions (> minimum 1). Response {t:.2f}s.")
        return [], js
    return [f"Check 1/4 FAILED - {n} positions (need ≥1) or slow (> {SLA_SECS}s, {t:.2f}s)"], js

def check_orders():
    ok, t, js, status, err = get_orders()
    if not ok or not isinstance(js, dict):
        return [f"Check 2/4 FAILED - orders API error ({err}), {t:.2f}s"]
    n_total, n_active = count_active_orders(js)
    if n_active >= 1 and t <= SLA_SECS:
        print(f"✅ Check 2/4 passed - detected {n_active} active order(s) (total returned {n_total}). Response {t:.2f}s.")
        return []
    return [f"Check 2/4 FAILED - zero active orders (active={n_active}, total={n_total}) or slow (> {SLA_SECS}s, {t:.2f}s)"]

def check_preview_single_put():
    # Underlying price
    ok, tq, q, _, err = get_quote(HEARTBEAT_SYMBOL)
    if not ok or not isinstance(q, dict):
        return [f"Check 3/4 FAILED - quote error {err or 'bad payload'}"], 0
    qobj = safe_get(q, "quotes", "quote")
    if isinstance(qobj, list): qobj = qobj[0] if qobj else {}
    try:
        under_px = float(qobj.get("last") or qobj.get("close") or qobj.get("bid") or 0)
    except: under_px = 0.0
    if under_px <= 0:
        return [f"Check 3/4 FAILED - could not derive underlying price"], 0

    # Expiration (today preferred)
    prefer = date.today().isoformat()
    exp, exp_err = nearest_valid_expiration(HEARTBEAT_SYMBOL, prefer=prefer)
    if not exp:
        return [f"Check 3/4 FAILED - {exp_err}"], 0

    # Chain -> ATM put
    ok, tc, chain, _, err = get_chain(HEARTBEAT_SYMBOL, exp)
    if not ok or not isinstance(chain, dict):
        return [f"Check 3/4 FAILED - chain error {err or 'bad payload'} for {exp}"], 0
    opt_sym, pick_err = pick_atm_put_symbol(chain, under_px)
    if not opt_sym:
        return [f"Check 3/4 FAILED - {pick_err or 'ATM put selection error'} for {exp}"], 0

    # Mid price
    mid, details = compute_option_mid(opt_sym)
    if mid is None:
        return [f"Check 3/4 FAILED - cannot compute mid price ({details})"], 0
    try:
        print(f"[{now()}] 🔢 ATM PUT mid: {mid:.2f} (bid={details['bid']}, ask={details['ask']}, last={details['last']})")
    except Exception:
        print(f"[{now()}] 🔢 ATM PUT mid: {mid}")

    # Preview single-leg buy_to_open PUT @ mid
    ok, tp, pv, _, err = preview_single_option(
        underlying_symbol=HEARTBEAT_SYMBOL,
        option_symbol=opt_sym,
        side="buy_to_open",
        qty=ORDER_QTY,
        limit_price=mid
    )
    if not ok or not isinstance(pv, dict):
        return [f"Check 3/4 FAILED - preview error {err or 'bad payload'}, {tp:.2f}s"], 0
    status_txt = str(safe_get(pv, "order", "status", default="ok")).lower()
    if "ok" in status_txt:
        print(f"✅ Check 3/4 passed - single-leg ATM PUT preview OK for {exp}. Response {tp:.2f}s.")
        return [], tp
    return [f"Check 3/4 FAILED - preview status='{status_txt}' for {exp}, {tp:.2f}s"], tp

def _extract_total_equity_from_balances_now(js):
    # Based on Tradier's balances response (e.g., "balances": {"total_equity": ...})
    # See: /v1/accounts/{account_id}/balances docs. 
    total_equity = safe_get(js, "balances", "total_equity")
    if total_equity is None:
        # Some accounts might expose "equity" or "account_equity"
        total_equity = safe_get(js, "balances", "equity")
    try:
        return float(total_equity) if total_equity is not None else None
    except Exception:
        return None

def _extract_series_from_historical(js):
    """
    Expected shape (best effort):
    {
      "historical_balances": {
        "balance": [
          {"date": "2025-10-30", "value": 12345.67, ...},
          {"date": "2025-10-31", "value": 12444.10, ...},
          ...
        ]
      }
    }
    Handles array/object flip as well.
    """
    root = safe_get(js, "historical_balances", "balances")
    if not isinstance(root, dict):
        return []
    bal = root.get("balance")
    if bal is None: return []
    if isinstance(bal, dict): bal = [bal]
    out = []
    for row in bal:
        try:
            d = str(row.get("date"))
            te = row.get("value")
            
            te = float(te) if te is not None else None
            if d and te is not None:
                out.append({"date": d, "total_equity": te})
        except Exception:
            continue
    # sort ascending by date
    out.sort(key=lambda x: x["date"])
    return out

def _load_and_update_local_balance_store(today_equity):
    store = read_balance_store()
    today = today_str()
    # save today's equity
    if today_equity is not None:
        store[today] = today_equity
        write_balance_store(store)
    # Find most recent prior day with equity
    prior_dates = [d for d in store.keys() if d < today]
    prior_dates.sort()
    return (store.get(prior_dates[-1]) if prior_dates else None)

def check_balance_drawdown():
    errs = []

    # 1) Get current balances
    ok_now, t_now, js_now, _, err_now = get_balances_now()
    if not ok_now or not isinstance(js_now, dict):
        return [f"Check 4/4 FAILED - balances API error ({err_now}), {t_now:.2f}s"]

    today_equity = _extract_total_equity_from_balances_now(js_now)
    if today_equity is None:
        return [f"Check 4/4 FAILED - could not parse today's total equity from balances payload"]

    # 2) Try historical balances from API
    ok_hist, t_hist, js_hist, code_hist, err_hist = get_historical_balances()
    if ok_hist and isinstance(js_hist, dict):
        series = _extract_series_from_historical(js_hist)
        # pick the most recent date prior to today (data updated nightly)
        prev = None
        if series:
            # Get last record that is strictly < today
            prior = [r for r in series if r["date"] < today_str()]
            if prior:
                prev = prior[-1]["total_equity"]
        if prev is not None:
            drop_pct = (prev - today_equity) / prev * 100.0
            if drop_pct > BALANCE_DD_LIMIT_PCT:
                return [f"Check 4/4 FAILED - equity drop {drop_pct:.2f}% exceeds {BALANCE_DD_LIMIT_PCT:.2f}% (prev={prev:.2f}, today={today_equity:.2f})"]
            print(f"✅ Check 4/4 passed - equity drop {drop_pct:.2f}% within limit (prev={prev:.2f}, today={today_equity:.2f}).")
            # also update local store for redundancy
            _load_and_update_local_balance_store(today_equity)
            return []
        # If historical exists but no prior day yet, fall back to local store below
    else:
        print(f"[{now()}] ⚠️ Historical balances API unavailable; falling back to local store.")

    # 3) Fallback to local store
    prev_local = _load_and_update_local_balance_store(today_equity)
    if prev_local is None:
        print(f"ℹ️ Check 4/4 fallback - no prior balance available yet; stored today's value for future comparisons.")
        return []
    drop_pct = (prev_local - today_equity) / prev_local * 100.0
    if drop_pct > BALANCE_DD_LIMIT_PCT:
        return [f"Check 4/4 FAILED - equity drop {drop_pct:.2f}% exceeds {BALANCE_DD_LIMIT_PCT:.2f}% (prev={prev_local:.2f}, today={today_equity:.2f}) [local]"]
    print(f"✅ Check 4/4 passed (local) - equity drop {drop_pct:.2f}% within limit (prev={prev_local:.2f}, today={today_equity:.2f}).")
    return []

# === Orchestration ===
def run_checks():
    print(f"\n=== Tradier Heartbeat @ {now()} ===")
    errs=[]
    c1, _ = check_positions();           errs+=c1
    errs   += check_orders()
    c3, _ = check_preview_single_put();  errs+=c3
    errs   += check_balance_drawdown()   # new Check 4

    if errs:
        print("\n".join(f"❌ {e}" for e in errs))
        send_alert(f"[Automation Alert] Tradier Heartbeat Failures ({len(errs)}) @ {now()}", "\n".join(errs))
    else:
        print("✅ All 4 checks passed successfully.\n")

def main():
    # test alert hook once at start (optional: comment out)
    send_alert("[Automation Alert] TEST Alert MAC", "")

    while True:
        try:
            run_checks()
        except Exception as e:
            tb=traceback.format_exc()
            send_alert("[ALERT] Heartbeat Script Crash", f"{e}\n{tb}")
            print(tb)
        time.sleep(60)

if __name__ == "__main__":
    main()
