#!/usr/bin/env python3
"""
heartbeat_tradier.py
1-minute heartbeat with explicit success logs, full API response prints,
and simplified Check #3 = single-leg ATM PUT preview at mid price.

Checks:
1) Positions endpoint responds within 10s and >=1 position.
2) Orders endpoint responds within 10s and has ≥1 ACTIVE order.
3) Preview single-leg BUY PUT (ATM, today or nearest expiry) at mid price returns status 'ok' within 10s.
"""

import os, time, json, smtplib, traceback, requests
from datetime import datetime, date
from email.mime.text import MIMEText
from dotenv import load_dotenv
from playsound import playsound
import subprocess

load_dotenv()


# keep system awake while this Python script runs
caffeinate = subprocess.Popen(["caffeinate", "-dims"])

# === CONFIG ===
TRADIER_ACCESS_TOKEN = os.getenv("TRADIER_ACCESS_TOKEN")
ACCOUNT_ID           = os.getenv("TRADIER_ACCOUNT_ID")
BASE_URL             = os.getenv("TRADIER_BASE_URL", "https://api.tradier.com/v1").rstrip("/")
HEARTBEAT_SYMBOL     = os.getenv("HEARTBEAT_SYMBOL", "SPX").upper()  # underlying for quote/chain and for preview 'symbol'
ORDER_QTY            = int(os.getenv("ORDER_QTY", "1"))

SMTP_HOST  = os.getenv("SMTP_HOST")
SMTP_PORT  = int(os.getenv("SMTP_PORT", "587")) if os.getenv("SMTP_HOST") else None
SMTP_USER  = os.getenv("SMTP_USER")
SMTP_PASS  = os.getenv("SMTP_PASS")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO   = os.getenv("EMAIL_TO")

HEADERS    = {"Authorization": f"Bearer {TRADIER_ACCESS_TOKEN}", "Accept": "application/json"}
REQ_TIMEOUT = 8
SLA_SECS    = 10

# Limit body print length for readability
MAX_PRINT_CHARS = int(os.getenv("MAX_PRINT_CHARS", "2000"))

def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def alert_with_vol(volume_level):
    """
    Sets the main output volume of the Mac.
    volume_level should be an integer between 0 (mute) and 100 (max volume).
    """
    if not 0 <= volume_level <= 100:
        raise ValueError("Volume level must be between 0 and 100.")

    # Execute the AppleScript command using osascript
    subprocess.run(["osascript", "-e", "set volume output muted false", "-e", f"set volume output volume {volume_level}"], check=True)
    print("🔈Alert sound played on mac")
    playsound("alarm2.wav")




def _pretty(obj):
    try: return json.dumps(obj, indent=2, sort_keys=True)
    except Exception: return str(obj)

def _print_response(tag, url, status_code, elapsed, content, is_json_guess=True):
    body = _pretty(content) if is_json_guess else str(content)
    if len(body) > MAX_PRINT_CHARS:
        body = body[:MAX_PRINT_CHARS] + "\n…(truncated)…"
    print(f"[{now()}] 🔎 {tag}\nURL: {url}\nStatus: {status_code} | Elapsed: {elapsed:.2f}s\nBody:\n{body}\n")

def send_alert(subject, body):
    if not (SMTP_HOST and SMTP_PORT and EMAIL_FROM and EMAIL_TO):
        print(f"[{now()}] ⚠️ Email not sent (SMTP vars missing)\n{body}")
        return
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
    
    alert_with_vol(50)

def timed_req(method, url, data=None, tag="API"):
    t0 = time.monotonic()
    try:
        if method == "get":
            r = requests.get(url, headers=HEADERS, timeout=REQ_TIMEOUT)
        else:
            r = requests.post(url, headers={**HEADERS,"Content-Type":"application/x-www-form-urlencoded"},
                              data=data, timeout=REQ_TIMEOUT)
        elapsed = time.monotonic() - t0

        # Try JSON else text (some endpoints mislabel content-type)
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

# --- API wrappers
def get_positions():      return timed_req("get", f"{BASE_URL}/accounts/{ACCOUNT_ID}/positions", tag="Positions")
def get_orders():         return timed_req("get", f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders", tag="Orders")
def get_quote(sym):       return timed_req("get", f"{BASE_URL}/markets/quotes?symbols={sym}", tag=f"Quote {sym}")
def get_expirations(sym): return timed_req("get", f"{BASE_URL}/markets/options/expirations?symbol={sym}&includeAllRoots=true", tag=f"Expirations {sym}")
def get_chain(sym, exp):  return timed_req("get", f"{BASE_URL}/markets/options/chains?symbol={sym}&expiration={exp}", tag=f"Chain {sym} {exp}")

def get_option_quotes(symbols):
    if isinstance(symbols, (list, tuple)): symbols = ",".join(symbols)
    url = f"{BASE_URL}/markets/quotes?symbols={symbols}"
    return timed_req("get", url, tag=f"Option Quotes ({symbols})")

def preview_single_option(underlying_symbol, option_symbol, side, qty, limit_price):
    """
    Preview a SINGLE-LEG option order:
      class=option, symbol=<underlying>, option_symbol=<occ>, side=buy_to_open, quantity, type=limit, duration=day, price=<mid>
    """
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
    # debug payload
    try:
        print(f"[{now()}] 🧾 Preview payload (single):\n{json.dumps(data, indent=2)}")
    except Exception:
        pass

    return timed_req("post", url, data, tag="Preview Single Option")

# --- Helpers
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
    n_total = len(raw)
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
        today_s = prefer
        future = sorted([d for d in raw if d >= today_s])
        if future: return future[0], None
        return sorted(raw)[-1], None
    except Exception:
        return raw[0], None

def pick_atm_put_symbol(chain_json, underlying_price):
    """Return OCC/OPRA symbol of the ATM PUT (closest strike to underlying)."""
    if not isinstance(chain_json, dict):
        return None, "chain_json is not a dict"
    options = safe_get(chain_json, "options", "option", default=[])
    if isinstance(options, dict): options=[options]
    puts = [o for o in options if str(o.get("option_type") or o.get("right") or "").lower() == "put"]
    if not puts:
        return None, "no put options in chain"
    def sval(o):
        try: return float(o.get("strike"))
        except: return float("inf")
    puts.sort(key=lambda o: abs(sval(o)-underlying_price))
    sym = puts[0].get("symbol") or puts[0].get("option_symbol")
    if not sym:
        return None, "no option symbol on selected put"
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

# --- Checks
def check_positions():
    ok, t, js, status, err = get_positions()
    if not ok or not isinstance(js, dict):
        return [f"Check 1/3 FAILED - positions API error ({err}), {t:.2f}s"], None
    items = safe_get(js, "positions", "position", default=[])
    if isinstance(items, dict): items=[items]
    n = len(items)
    if n >= 1 and t <= SLA_SECS:
        print(f"✅ Check 1/3 passed - detected {n} positions (> minimum 1). Response {t:.2f}s.")
        return [], js
    return [f"Check 1/3 FAILED - {n} positions (need ≥1) or slow (> {SLA_SECS}s, {t:.2f}s)"], js

def check_orders():
    ok, t, js, status, err = get_orders()
    if not ok or not isinstance(js, dict):
        return [f"Check 2/3 FAILED - orders API error ({err}), {t:.2f}s"]
    n_total, n_active = count_active_orders(js)
    if n_active >= 1 and t <= SLA_SECS:
        print(f"✅ Check 2/3 passed - detected {n_active} active order(s) (total returned {n_total}). Response {t:.2f}s.")
        return []
    return [f"Check 2/3 FAILED - zero active orders (active={n_active}, total={n_total}) or slow (> {SLA_SECS}s, {t:.2f}s)"]

def check_preview_single_put():
    # 1) underlying quote
    ok, tq, q, _, err = get_quote(HEARTBEAT_SYMBOL)
    if not ok or not isinstance(q, dict):
        return [f"Check 3/3 FAILED - quote error {err or 'bad payload'}"], 0
    qobj = safe_get(q, "quotes", "quote")
    if isinstance(qobj, list): qobj = qobj[0] if qobj else {}
    try:
        under_px = float(qobj.get("last") or qobj.get("close") or qobj.get("bid") or 0)
    except: under_px = 0.0
    if under_px <= 0:
        return [f"Check 3/3 FAILED - could not derive underlying price"], 0

    # 2) expiration (today preferred)
    prefer = date.today().isoformat()
    exp, exp_err = nearest_valid_expiration(HEARTBEAT_SYMBOL, prefer=prefer)
    if not exp:
        return [f"Check 3/3 FAILED - {exp_err}"], 0

    # 3) chain -> ATM put symbol
    ok, tc, chain, _, err = get_chain(HEARTBEAT_SYMBOL, exp)
    if not ok or not isinstance(chain, dict):
        return [f"Check 3/3 FAILED - chain error {err or 'bad payload'} for {exp}"], 0
    opt_sym, pick_err = pick_atm_put_symbol(chain, under_px)
    if not opt_sym:
        return [f"Check 3/3 FAILED - {pick_err or 'ATM put selection error'} for {exp}"], 0

    # 4) compute mid for the put
    mid, details = compute_option_mid(opt_sym)
    if mid is None:
        return [f"Check 3/3 FAILED - cannot compute mid price ({details})"], 0
    try:
        print(f"[{now()}] 🔢 ATM PUT mid: {mid:.2f} (bid={details['bid']}, ask={details['ask']}, last={details['last']})")
    except Exception:
        print(f"[{now()}] 🔢 ATM PUT mid: {mid}")

    # 5) preview single-leg buy_to_open PUT @ mid
    ok, tp, pv, _, err = preview_single_option(
        underlying_symbol=HEARTBEAT_SYMBOL,
        option_symbol=opt_sym,
        side="buy_to_open",
        qty=ORDER_QTY,
        limit_price=mid
    )
    if not ok or not isinstance(pv, dict):
        return [f"Check 3/3 FAILED - preview error {err or 'bad payload'}, {tp:.2f}s"], 0
    status_txt = str(safe_get(pv, "order", "status", default="ok")).lower()
    if "ok" in status_txt:
        print(f"✅ Check 3/3 passed - single-leg ATM PUT preview OK for {exp}. Response {tp:.2f}s.")
        return [], tp
    return [f"Check 3/3 FAILED - preview status='{status_txt}' for {exp}, {tp:.2f}s"], tp

# --- Orchestration
def run_checks():
    print(f"\n=== Tradier Heartbeat @ {now()} ===")
    errs=[]
    c1, _ = check_positions(); errs+=c1
    errs += check_orders()
    c3, _ = check_preview_single_put(); errs+=c3
    if errs:
        print("\n".join(f"❌ {e}" for e in errs))
        send_alert(f"[Automation Alert] Tradier Heartbeat Failures ({len(errs)}) @ {now()}", "\n".join(errs))

    else:
        print("✅ All 3 checks passed successfully.\n")

def main():
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
