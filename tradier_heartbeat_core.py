#!/usr/bin/env python3
"""
Shared Tradier heartbeat engine.

This module centralizes the monitoring/check logic and exposes runtime knobs so
small wrappers can run it in different environments:
- local mode: keep-awake + rising-volume alarm
- CI mode: email-only alerts, no local audio interaction
"""

import atexit
import json
import os
import shutil
import smtplib
import subprocess
import threading
import time
import traceback
from datetime import date, datetime
from email.mime.text import MIMEText

import requests
from dotenv import load_dotenv

load_dotenv()

# === CONFIG ===
TRADIER_ACCESS_TOKEN = os.getenv("TRADIER_ACCESS_TOKEN")
ACCOUNT_ID = os.getenv("TRADIER_ACCOUNT_ID")
BASE_URL = os.getenv("TRADIER_BASE_URL", "https://api.tradier.com/v1").rstrip("/")
HEARTBEAT_SYMBOL = os.getenv("HEARTBEAT_SYMBOL", "SPX").upper()
ORDER_QTY = int(os.getenv("ORDER_QTY", "1"))

# Balance check config
BALANCE_STORE = os.getenv("BALANCE_STORE", "balance_store.json")
BALANCE_DD_LIMIT_PCT = float(os.getenv("BALANCE_DD_LIMIT_PCT", "7"))  # 2% default

# Email + alert
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587")) if os.getenv("SMTP_HOST") else None
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO = os.getenv("EMAIL_TO")

# HTTP
HEADERS = {"Authorization": f"Bearer {TRADIER_ACCESS_TOKEN}", "Accept": "application/json"}
REQ_TIMEOUT = 10  # per request timeout seconds
SLA_SECS = 10
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))  # immediate retries per request on timeout only

# Consecutive-failure alert thresholds
FAILURE_CATEGORY_DEFAULT = "default"
FAILURE_CATEGORY_BALANCE_DRAWDOWN = "balance_drawdown"
FAILURE_CATEGORY_SCRIPT_CRASH = "script_crash"
FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN = "quote_preview_chain"
FAILURE_CATEGORY_SLOW_RESPONSE = "slow_response"

FAILURE_THRESHOLDS = {
    FAILURE_CATEGORY_DEFAULT: int(os.getenv("ALERT_CONSEC_DEFAULT", "1")),
    FAILURE_CATEGORY_BALANCE_DRAWDOWN: int(os.getenv("ALERT_CONSEC_BALANCE_DRAWDOWN", "1")),
    FAILURE_CATEGORY_SCRIPT_CRASH: int(os.getenv("ALERT_CONSEC_SCRIPT_CRASH", "1")),
    FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN: int(os.getenv("ALERT_CONSEC_QUOTE_PREVIEW_CHAIN", "2")),
    FAILURE_CATEGORY_SLOW_RESPONSE: int(os.getenv("ALERT_CONSEC_SLOW_RESPONSE", "3")),
}

# Runtime mode toggles (configured by wrappers)
ENABLE_SOUND_ALERT = False
MAX_PRINT_CHARS = 50
_CAFFEINATE_PROC = None
RUN_CONTEXT = "Unknown"
_CONSECUTIVE_FAILURE_COUNTS = {}
SPX_PRICE_RANGE = None


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today_str():
    return date.today().isoformat()


# === Runtime mode ===
def _default_run_context():
    if str(os.getenv("GITHUB_ACTIONS", "")).lower() == "true":
        return "GitHub Actions"
    return "Local PC"


def _stop_keep_awake():
    global _CAFFEINATE_PROC
    if _CAFFEINATE_PROC is None:
        return
    try:
        _CAFFEINATE_PROC.terminate()
    except Exception:
        pass
    _CAFFEINATE_PROC = None


def _start_keep_awake():
    global _CAFFEINATE_PROC
    if _CAFFEINATE_PROC is not None:
        return
    try:
        _CAFFEINATE_PROC = subprocess.Popen(["caffeinate", "-dims"])
        print(f"[{now()}] ☕️ Started caffeinate keep-awake process.")
    except Exception as exc:
        print(f"[{now()}] ⚠️ Could not start caffeinate: {exc}")


def configure_runtime(enable_sound_alert=False, keep_awake=False, max_print_chars_default=50, run_context=None):
    global ENABLE_SOUND_ALERT, MAX_PRINT_CHARS, RUN_CONTEXT
    ENABLE_SOUND_ALERT = bool(enable_sound_alert)
    MAX_PRINT_CHARS = int(os.getenv("MAX_PRINT_CHARS", str(max_print_chars_default)))
    RUN_CONTEXT = (run_context or os.getenv("HEARTBEAT_RUN_CONTEXT") or _default_run_context()).strip()
    if keep_awake:
        _start_keep_awake()


atexit.register(_stop_keep_awake)


def configure_spx_price_range(low, high):
    global SPX_PRICE_RANGE
    low = float(low)
    high = float(high)
    if low <= 0 or high <= 0:
        raise ValueError("SPX range bounds must be positive numbers.")
    if low > high:
        raise ValueError("SPX range low cannot be greater than high.")
    SPX_PRICE_RANGE = (low, high)


def clear_spx_price_range():
    global SPX_PRICE_RANGE
    SPX_PRICE_RANGE = None


def _prompt_positive_float(prompt):
    while True:
        raw = input(prompt).strip().replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if value <= 0:
            print("Please enter a positive number.")
            continue
        return value


def prompt_spx_price_range():
    while True:
        print(f"\nConfigure allowed {HEARTBEAT_SYMBOL} price range for this heartbeat run.")
        low = _prompt_positive_float(f"{HEARTBEAT_SYMBOL} range low: ")
        high = _prompt_positive_float(f"{HEARTBEAT_SYMBOL} range high: ")
        try:
            configure_spx_price_range(low, high)
        except ValueError as exc:
            print(f"{exc} Please enter the range again.")
            continue
        print(f"[{now()}] {HEARTBEAT_SYMBOL} range check enabled: {low:.2f} to {high:.2f}.")
        return


# === Alerting ===
def _play_alarm_sequence(stop_flag=None):
    sound_files = ["alarm-bell-47839.mp3", "alarm-bell-47839.mp3", "alarm-bell-47839.mp3", "alarm2.wav"]
    afplay_path = shutil.which("afplay")

    # Prefer afplay on macOS because we can interrupt playback immediately when user hits ENTER.
    if afplay_path:
        for sound_file in sound_files:
            if stop_flag and stop_flag.get("stop"):
                return
            try:
                proc = subprocess.Popen([afplay_path, sound_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as exc:
                print(f"[{now()}] ⚠️ Could not start afplay for {sound_file}: {exc}")
                continue

            while proc.poll() is None:
                if stop_flag and stop_flag.get("stop"):
                    try:
                        proc.terminate()
                        proc.wait(timeout=1)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    return
                time.sleep(0.05)
        return

    try:
        from playsound import playsound
    except Exception as exc:
        print(f"[{now()}] ⚠️ Could not import playsound for local alarm: {exc}")
        return

    try:
        for sound_file in sound_files:
            if stop_flag and stop_flag.get("stop"):
                return
            playsound(sound_file)
    except Exception as exc:
        print(f"[{now()}] ⚠️ Could not play alert sound: {exc}")


def alert_with_rising_volume():
    if not ENABLE_SOUND_ALERT:
        return

    stop_flag = {"stop": False}

    def wait_for_user():
        try:
            input("\nPress ENTER to stop the alert...\n")
            stop_flag["stop"] = True
        except Exception:
            # Non-interactive terminal (or input unavailable): keep alert looping.
            pass

    threading.Thread(target=wait_for_user, daemon=True).start()

    volume = int(os.getenv("ALERT_START_VOLUME", "20"))
    step = int(os.getenv("ALERT_VOLUME_STEP", "10"))
    sleep_secs = float(os.getenv("ALERT_LOOP_SLEEP_SECS", "0.5"))
    print("🚨 Starting alert...")

    while not stop_flag["stop"]:
        safe_volume = max(0, min(100, int(volume)))
        try:
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    "set volume output muted false",
                    "-e",
                    f"set volume output volume {safe_volume}",
                ],
                check=True,
            )
            print(f"🔊 Volume: {safe_volume}")
        except Exception as exc:
            print(f"[{now()}] ⚠️ Could not set Mac system volume: {exc}")

        _play_alarm_sequence(stop_flag=stop_flag)

        volume += step
        if volume > 100:
            volume = 100

        time.sleep(sleep_secs)

    print("🛑 Alert stopped by user.")


def send_alert(subject, body):
    if not (SMTP_HOST and SMTP_PORT and EMAIL_FROM and EMAIL_TO):
        print(f"[{now()}] ⚠️ Email not sent (SMTP vars missing)\n{body}")
    else:
        try:
            msg = MIMEText(body, "plain")
            msg["Subject"], msg["From"], msg["To"] = subject, EMAIL_FROM, EMAIL_TO
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as smtp_conn:
                smtp_conn.starttls()
                if SMTP_USER and SMTP_PASS:
                    smtp_conn.login(SMTP_USER, SMTP_PASS)
                smtp_conn.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())
            print(f"[{now()}] 📧 Alert email sent.")
        except Exception as exc:
            print(f"[{now()}] ❌ Email send failed: {exc}")

    alert_with_rising_volume()


def context_subject(title):
    return f"[Automation Alert][{RUN_CONTEXT}] {title}"


# === Utilities ===
def build_failure(message, category=FAILURE_CATEGORY_DEFAULT, key=None, details=None):
    failure = {
        "message": message,
        "category": category,
        "key": key or message,
    }
    if details:
        failure["details"] = details
    return failure


def _failure_threshold(category):
    return FAILURE_THRESHOLDS.get(category, FAILURE_THRESHOLDS[FAILURE_CATEGORY_DEFAULT])


def _apply_failure_thresholds(failures):
    global _CONSECUTIVE_FAILURE_COUNTS

    next_counts = {}
    alert_failures = []

    for failure in failures:
        key = failure["key"]
        if key in next_counts:
            count = next_counts[key]
        else:
            count = _CONSECUTIVE_FAILURE_COUNTS.get(key, 0) + 1
            next_counts[key] = count

        threshold = _failure_threshold(failure["category"])
        failure["count"] = count
        failure["threshold"] = threshold

        if count >= threshold:
            alert_failures.append(failure)
            continue

        print(
            f"[{now()}] 🔕 Alert suppressed for {key}: "
            f"{count}/{threshold} consecutive failure(s)."
        )

    _CONSECUTIVE_FAILURE_COUNTS = next_counts
    return alert_failures


def _format_failure_for_alert(failure):
    line = failure["message"]
    threshold = failure.get("threshold", 1)
    count = failure.get("count", 1)
    if threshold > 1:
        line = f"{line} [consecutive {count}/{threshold}]"
    details = failure.get("details")
    if details:
        return f"{line}\n{details}"
    return line


def _pretty(obj):
    try:
        return json.dumps(obj, indent=2, sort_keys=True)
    except Exception:
        return str(obj)


def _print_response(tag, url, status_code, elapsed, content, is_json_guess=True, attempt=None):
    body = _pretty(content) if is_json_guess else str(content)
    if len(body) > MAX_PRINT_CHARS:
        body = body[:MAX_PRINT_CHARS] + "\n…(truncated)…"
    att = f" (attempt {attempt})" if attempt is not None else ""
    print(
        f"[{now()}] 🔎 {tag}{att}\n"
        f"URL: {url}\n"
        f"Status: {status_code} | Elapsed: {elapsed:.2f}s\n"
        f"Body:\n{body}\n"
    )


def _do_http(method, url, data=None, tag="API", attempt=None):
    """One attempt only. Returns (ok, elapsed, content, status_code, err_tag)."""
    start = time.monotonic()
    try:
        if method == "get":
            resp = requests.get(url, headers=HEADERS, timeout=REQ_TIMEOUT)
        else:
            resp = requests.post(
                url,
                headers={**HEADERS, "Content-Type": "application/x-www-form-urlencoded"},
                data=data,
                timeout=REQ_TIMEOUT,
            )
        elapsed = time.monotonic() - start

        is_json = False
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "application/json" in content_type:
            try:
                content = resp.json()
                is_json = True
            except Exception:
                content = resp.text
        else:
            try:
                content = resp.json()
                is_json = True
            except Exception:
                content = resp.text

        _print_response(tag, url, resp.status_code, elapsed, content, is_json_guess=is_json, attempt=attempt)
        if resp.ok and elapsed <= SLA_SECS:
            return True, elapsed, content, resp.status_code, None
        if resp.ok:
            return False, elapsed, content, resp.status_code, "slow_response"
        return False, elapsed, content, resp.status_code, f"HTTP {resp.status_code}"
    except requests.exceptions.Timeout:
        elapsed = time.monotonic() - start
        print(f"[{now()}] ⏱️ Timeout in {tag} after {elapsed:.2f}s (attempt {attempt})")
        return False, elapsed, None, None, "timeout"
    except Exception as exc:
        elapsed = time.monotonic() - start
        print(f"[{now()}] ❌ {tag} request error: {exc} (elapsed {elapsed:.2f}s) (attempt {attempt})")
        return False, elapsed, None, None, str(exc)


def timed_req(method, url, data=None, tag="API", retries=MAX_RETRIES):
    """
    Immediate retries (no backoff) for timeout errors only.
    Returns final (ok, elapsed_last, content_last, status_code_last, err_tag_last).
    """
    last_result = (False, 0.0, None, None, "timeout")
    for attempt in range(1, retries + 1):
        ok, elapsed, content, status_code, err = _do_http(method, url, data=data, tag=tag, attempt=attempt)
        last_result = (ok, elapsed, content, status_code, err)
        if ok:
            return last_result
        if err != "timeout":
            return last_result
    return last_result


# === API wrappers ===
def get_positions():
    return timed_req("get", f"{BASE_URL}/accounts/{ACCOUNT_ID}/positions", tag="Positions")


def get_orders():
    return timed_req("get", f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders", tag="Orders")


def get_quote(sym):
    return timed_req("get", f"{BASE_URL}/markets/quotes?symbols={sym}", tag=f"Quote {sym}")


def get_expirations(sym):
    return timed_req(
        "get",
        f"{BASE_URL}/markets/options/expirations?symbol={sym}&includeAllRoots=true",
        tag=f"Expirations {sym}",
    )


def get_chain(sym, exp):
    return timed_req(
        "get",
        f"{BASE_URL}/markets/options/chains?symbol={sym}&expiration={exp}",
        tag=f"Chain {sym} {exp}",
    )


def get_option_quotes(symbols):
    if isinstance(symbols, (list, tuple)):
        symbols = ",".join(symbols)
    url = f"{BASE_URL}/markets/quotes?symbols={symbols}"
    return timed_req("get", url, tag=f"Option Quotes ({symbols})")


def get_balances_now():
    return timed_req("get", f"{BASE_URL}/accounts/{ACCOUNT_ID}/balances", tag="Balances Now")


def get_historical_balances():
    return timed_req(
        "get",
        f"{BASE_URL}/accounts/{ACCOUNT_ID}/historical-balances?period=WEEK",
        tag="Historical Balances",
    )


def preview_single_option(underlying_symbol, option_symbol, side, qty, limit_price):
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders?preview=true"
    data = {
        "class": "option",
        "symbol": underlying_symbol,
        "option_symbol": option_symbol,
        "side": side,
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
def safe_get(dct, *path, default=None):
    cur = dct
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _extract_quote_object(quote_payload):
    qobj = safe_get(quote_payload, "quotes", "quote")
    if isinstance(qobj, list):
        return qobj[0] if qobj else {}
    return qobj if isinstance(qobj, dict) else {}


def _extract_underlying_price_from_quote(quote_payload):
    qobj = _extract_quote_object(quote_payload)
    for field in ("last", "close", "bid", "ask"):
        try:
            value = qobj.get(field)
            if value is None:
                continue
            price = float(value)
            if price > 0:
                return price
        except Exception:
            continue
    return None


ACTIVE_STATUSES = {"open", "pending", "live", "accepted", "partially_filled"}


def count_active_orders(js):
    raw = safe_get(js, "orders", "order", default=[])
    if isinstance(raw, dict):
        raw = [raw]
    n_total = len(raw)
    n_active = sum(1 for item in raw if str(item.get("status", "")).lower() in ACTIVE_STATUSES)
    return n_total, n_active


def pick_atm_put_symbol(chain_json, underlying_price):
    if not isinstance(chain_json, dict):
        return None, "chain_json is not a dict"
    options = safe_get(chain_json, "options", "option", default=[])
    if isinstance(options, dict):
        options = [options]
    puts = [opt for opt in options if str(opt.get("option_type") or opt.get("right") or "").lower() == "put"]
    if not puts:
        return None, "no put options in chain"

    def strike_val(opt):
        try:
            return float(opt.get("strike"))
        except Exception:
            return float("inf")

    puts.sort(key=lambda opt: abs(strike_val(opt) - underlying_price))
    sym = puts[0].get("symbol") or puts[0].get("option_symbol")
    if not sym:
        return None, "no option symbol on selected put"
    return sym, None


def mid_from_bidask(bid, ask):
    try:
        bid_val = float(bid) if bid is not None else None
        ask_val = float(ask) if ask is not None else None
        if bid_val is not None and ask_val is not None and ask_val >= bid_val:
            return (ask_val + bid_val) / 2.0
        if ask_val is not None:
            return ask_val
        if bid_val is not None:
            return bid_val
    except Exception:
        pass
    return None


def compute_option_mid(option_symbol):
    ok, elapsed, quotes, _, err = get_option_quotes(option_symbol)
    if not ok or not isinstance(quotes, dict):
        if err == "slow_response":
            return None, f"slow_response:{elapsed:.2f}"
        return None, f"option quotes error: {err or 'bad payload'}"
    qobj = safe_get(quotes, "quotes", "quote")
    if isinstance(qobj, list):
        qobj = qobj[0] if qobj else {}

    mid = mid_from_bidask(qobj.get("bid"), qobj.get("ask"))
    if mid is None:
        try:
            mid = float(qobj.get("last"))
        except Exception:
            mid = None
    if mid is None:
        return None, "insufficient quote data for mid"
    return max(mid, 0.01), {"bid": qobj.get("bid"), "ask": qobj.get("ask"), "last": qobj.get("last")}


# === Local balance store (fallback for #4) ===
def read_balance_store():
    try:
        with open(BALANCE_STORE, "r") as file_obj:
            return json.load(file_obj)
    except Exception:
        return {}


def write_balance_store(store):
    try:
        with open(BALANCE_STORE, "w") as file_obj:
            json.dump(store, file_obj, indent=2, sort_keys=True)
    except Exception as exc:
        print(f"[{now()}] ⚠️ Could not write balance store: {exc}")


def _load_and_update_local_balance_store(today_equity):
    store = read_balance_store()
    today = today_str()
    if today_equity is not None:
        store[today] = today_equity
        write_balance_store(store)
    prior_dates = [d for d in store.keys() if d < today]
    prior_dates.sort()
    return store.get(prior_dates[-1]) if prior_dates else None


# === Checks ===
def check_spx_price_range():
    if SPX_PRICE_RANGE is None:
        return []

    low, high = SPX_PRICE_RANGE
    ok, elapsed, quote_payload, _, err = get_quote(HEARTBEAT_SYMBOL)
    if not ok:
        if err == "timeout":
            return [
                build_failure(
                    f"Check 2/4 FAILED - {HEARTBEAT_SYMBOL} quote timeout x{MAX_RETRIES}",
                    key="spx_price_range_quote_timeout",
                )
            ]
        if err == "slow_response":
            return [
                build_failure(
                    f"Check 2/4 FAILED - slow {HEARTBEAT_SYMBOL} quote response (> {SLA_SECS}s, {elapsed:.2f}s)",
                    category=FAILURE_CATEGORY_SLOW_RESPONSE,
                    key="spx_price_range_quote_slow",
                )
            ]
        return [
            build_failure(
                f"Check 2/4 FAILED - {HEARTBEAT_SYMBOL} quote error {err or 'bad payload'}",
                key="spx_price_range_quote_error",
            )
        ]

    if not isinstance(quote_payload, dict):
        return [build_failure(f"Check 2/4 FAILED - {HEARTBEAT_SYMBOL} quote bad payload", key="spx_price_range_bad_payload")]

    current_price = _extract_underlying_price_from_quote(quote_payload)
    if current_price is None:
        return [
            build_failure(
                f"Check 2/4 FAILED - could not derive {HEARTBEAT_SYMBOL} current price",
                key="spx_price_range_no_price",
            )
        ]

    if current_price < low or current_price > high:
        return [
            build_failure(
                f"Check 2/4 FAILED - {HEARTBEAT_SYMBOL} price {current_price:.2f} outside range "
                f"{low:.2f}-{high:.2f}",
                key="spx_price_out_of_range",
            )
        ]

    print(
        f"✅ Check 2/4 passed - {HEARTBEAT_SYMBOL} price {current_price:.2f} within range "
        f"{low:.2f}-{high:.2f}. Response {elapsed:.2f}s."
    )
    return []


def check_positions():
    ok, elapsed, js, _, err = get_positions()
    if not ok:
        if err == "timeout":
            return [build_failure(f"Check 1/4 FAILED - positions API timeout x{MAX_RETRIES}", key="positions_timeout")], None
        if err == "slow_response":
            return [
                build_failure(
                    f"Check 1/4 FAILED - slow positions response (> {SLA_SECS}s, {elapsed:.2f}s)",
                    category=FAILURE_CATEGORY_SLOW_RESPONSE,
                    key="positions_slow",
                )
            ], js
        return [build_failure(f"Check 1/4 FAILED - positions API error ({err}), {elapsed:.2f}s", key="positions_error")], None

    if not isinstance(js, dict):
        return [build_failure("Check 1/4 FAILED - positions bad payload", key="positions_bad_payload")], js
    items = safe_get(js, "positions", "position", default=[])
    if isinstance(items, dict):
        items = [items]
    count = len(items)
    print(f"✅ Check 1/4 passed - detected {count} positions (minimum 0). Response {elapsed:.2f}s.")
    return [], js


def check_orders():
    ok, elapsed, js, _, err = get_orders()
    if not ok:
        if err == "timeout":
            return [build_failure(f"Check 2/4 FAILED - orders API timeout x{MAX_RETRIES}", key="orders_timeout")]
        if err == "slow_response":
            return [
                build_failure(
                    f"Check 2/4 FAILED - slow orders response (> {SLA_SECS}s, {elapsed:.2f}s)",
                    category=FAILURE_CATEGORY_SLOW_RESPONSE,
                    key="orders_slow",
                )
            ]
        return [build_failure(f"Check 2/4 FAILED - orders API error ({err}), {elapsed:.2f}s", key="orders_error")]

    if not isinstance(js, dict):
        return [build_failure("Check 2/4 FAILED - orders bad payload", key="orders_bad_payload")]
    total, active = count_active_orders(js)
    if active >= 1:
        print(
            f"✅ Check 2/4 passed - detected {active} active order(s) "
            f"(total returned {total}). Response {elapsed:.2f}s."
        )
        return []
    return [build_failure(f"Check 2/4 FAILED - zero active orders (active={active}, total={total})", key="orders_zero_active")]


def check_preview_single_put():
    ok, quote_elapsed, quote_payload, _, err = get_quote(HEARTBEAT_SYMBOL)
    if not ok:
        if err == "timeout":
            return [
                build_failure(
                    f"Check 3/4 FAILED - quote timeout x{MAX_RETRIES}",
                    category=FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN,
                    key="quote_timeout",
                )
            ], 0
        if err == "slow_response":
            return [
                build_failure(
                    f"Check 3/4 FAILED - slow quote response (> {SLA_SECS}s, {quote_elapsed:.2f}s)",
                    category=FAILURE_CATEGORY_SLOW_RESPONSE,
                    key="quote_slow",
                )
            ], 0
        return [
            build_failure(
                f"Check 3/4 FAILED - quote error {err or 'bad payload'}",
                category=FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN,
                key="quote_error",
            )
        ], 0
    if not isinstance(quote_payload, dict):
        return [build_failure("Check 3/4 FAILED - quote bad payload", FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN, "quote_bad_payload")], 0

    under_px = _extract_underlying_price_from_quote(quote_payload)
    if under_px is None:
        return [
            build_failure(
                "Check 3/4 FAILED - could not derive underlying price",
                category=FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN,
                key="quote_underlying_price",
            )
        ], 0

    ok, exp_elapsed, exps, _, err = get_expirations(HEARTBEAT_SYMBOL)
    if not ok:
        if err == "timeout":
            return [
                build_failure(
                    f"Check 3/4 FAILED - expirations timeout x{MAX_RETRIES}",
                    category=FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN,
                    key="expirations_timeout",
                )
            ], 0
        if err == "slow_response":
            return [
                build_failure(
                    f"Check 3/4 FAILED - slow expirations response (> {SLA_SECS}s, {exp_elapsed:.2f}s)",
                    category=FAILURE_CATEGORY_SLOW_RESPONSE,
                    key="expirations_slow",
                )
            ], 0
        return [
            build_failure(
                f"Check 3/4 FAILED - expirations error {err or 'bad payload'}",
                category=FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN,
                key="expirations_error",
            )
        ], 0
    if not isinstance(exps, dict):
        return [
            build_failure(
                "Check 3/4 FAILED - expirations bad payload",
                category=FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN,
                key="expirations_bad_payload",
            )
        ], 0

    raw = safe_get(exps, "expirations", "date", default=[])
    if isinstance(raw, str):
        raw = [raw]
    if not raw:
        return [
            build_failure(
                "Check 3/4 FAILED - no expirations available",
                category=FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN,
                key="expirations_none",
            )
        ], 0

    prefer = date.today().isoformat()
    exp = (
        prefer
        if prefer in raw
        else (sorted([d for d in raw if d >= prefer])[0] if any(d >= prefer for d in raw) else sorted(raw)[-1])
    )

    ok, chain_elapsed, chain, _, err = get_chain(HEARTBEAT_SYMBOL, exp)
    if not ok:
        if err == "timeout":
            return [
                build_failure(
                    f"Check 3/4 FAILED - chain timeout x{MAX_RETRIES}",
                    category=FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN,
                    key="chain_timeout",
                )
            ], 0
        if err == "slow_response":
            return [
                build_failure(
                    f"Check 3/4 FAILED - slow chain response (> {SLA_SECS}s, {chain_elapsed:.2f}s) for {exp}",
                    category=FAILURE_CATEGORY_SLOW_RESPONSE,
                    key="chain_slow",
                )
            ], 0
        return [
            build_failure(
                f"Check 3/4 FAILED - chain error {err or 'bad payload'} for {exp}",
                category=FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN,
                key="chain_error",
            )
        ], 0
    if not isinstance(chain, dict):
        return [
            build_failure(
                f"Check 3/4 FAILED - chain bad payload for {exp}",
                category=FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN,
                key="chain_bad_payload",
            )
        ], 0

    opt_sym, pick_err = pick_atm_put_symbol(chain, under_px)
    if not opt_sym:
        return [
            build_failure(
                f"Check 3/4 FAILED - {pick_err or 'ATM put selection error'} for {exp}",
                category=FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN,
                key="chain_option_selection",
            )
        ], 0

    mid, details = compute_option_mid(opt_sym)
    if mid is None:
        if isinstance(details, str) and details.startswith("slow_response:"):
            elapsed_txt = details.split(":", 1)[1]
            return [
                build_failure(
                    f"Check 3/4 FAILED - slow option quotes response (> {SLA_SECS}s, {elapsed_txt}s)",
                    category=FAILURE_CATEGORY_SLOW_RESPONSE,
                    key="option_quotes_slow",
                )
            ], 0
        return [
            build_failure(
                f"Check 3/4 FAILED - cannot compute mid price ({details})",
                category=FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN,
                key="option_mid_failure",
            )
        ], 0

    try:
        print(
            f"[{now()}] 🔢 ATM PUT mid: {mid:.2f} "
            f"(bid={details['bid']}, ask={details['ask']}, last={details['last']})"
        )
    except Exception:
        print(f"[{now()}] 🔢 ATM PUT mid: {mid}")

    ok, preview_elapsed, preview_payload, _, err = preview_single_option(
        underlying_symbol=HEARTBEAT_SYMBOL,
        option_symbol=opt_sym,
        side="buy_to_open",
        qty=ORDER_QTY,
        limit_price=mid,
    )
    if not ok:
        if err == "timeout":
            return [
                build_failure(
                    f"Check 3/4 FAILED - preview timeout x{MAX_RETRIES}",
                    category=FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN,
                    key="preview_timeout",
                )
            ], 0
        if err == "slow_response":
            return [
                build_failure(
                    f"Check 3/4 FAILED - slow preview response (> {SLA_SECS}s, {preview_elapsed:.2f}s)",
                    category=FAILURE_CATEGORY_SLOW_RESPONSE,
                    key="preview_slow",
                )
            ], 0
        return [
            build_failure(
                f"Check 3/4 FAILED - preview error {err or 'bad payload'}, {preview_elapsed:.2f}s",
                category=FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN,
                key="preview_error",
            )
        ], 0

    if not isinstance(preview_payload, dict):
        return [
            build_failure(
                "Check 3/4 FAILED - preview bad payload",
                category=FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN,
                key="preview_bad_payload",
            )
        ], 0
    status_txt = str(safe_get(preview_payload, "order", "status", default="ok")).lower()
    if "ok" in status_txt:
        print(f"✅ Check 3/4 passed - single-leg ATM PUT preview OK for {exp}. Response {preview_elapsed:.2f}s.")
        return [], preview_elapsed
    return [
        build_failure(
            f"Check 3/4 FAILED - preview status='{status_txt}' for {exp}, {preview_elapsed:.2f}s",
            category=FAILURE_CATEGORY_QUOTE_PREVIEW_CHAIN,
            key="preview_status",
        )
    ], preview_elapsed


def _extract_total_equity_from_balances_now(js):
    total_equity = safe_get(js, "balances", "total_equity")
    if total_equity is None:
        total_equity = safe_get(js, "balances", "equity")
    try:
        return float(total_equity) if total_equity is not None else None
    except Exception:
        return None


def _extract_series_from_historical(js):
    root = safe_get(js, "historical_balances", "balances")
    if not isinstance(root, dict):
        return []

    bal = root.get("balance")
    if bal is None:
        return []
    if isinstance(bal, dict):
        bal = [bal]

    out = []
    for row in bal:
        try:
            row_date = str(row.get("date"))
            total_equity = row.get("value")
            total_equity = float(total_equity) if total_equity is not None else None
            if row_date and total_equity is not None:
                out.append({"date": row_date, "total_equity": total_equity})
        except Exception:
            continue

    out.sort(key=lambda x: x["date"])
    return out


def check_balance_drawdown():
    ok_now, elapsed_now, js_now, _, err_now = get_balances_now()
    if not ok_now:
        if err_now == "timeout":
            return [
                build_failure(
                    f"Check 4/4 FAILED - balances now timeout x{MAX_RETRIES}",
                    category=FAILURE_CATEGORY_BALANCE_DRAWDOWN,
                    key="balances_timeout",
                )
            ]
        if err_now == "slow_response":
            return [
                build_failure(
                    f"Check 4/4 FAILED - slow balances response (> {SLA_SECS}s, {elapsed_now:.2f}s)",
                    category=FAILURE_CATEGORY_SLOW_RESPONSE,
                    key="balances_slow",
                )
            ]
        return [
            build_failure(
                f"Check 4/4 FAILED - balances API error ({err_now}), {elapsed_now:.2f}s",
                category=FAILURE_CATEGORY_BALANCE_DRAWDOWN,
                key="balances_error",
            )
        ]
    if not isinstance(js_now, dict):
        return [
            build_failure(
                "Check 4/4 FAILED - balances now bad payload",
                category=FAILURE_CATEGORY_BALANCE_DRAWDOWN,
                key="balances_bad_payload",
            )
        ]

    today_equity = _extract_total_equity_from_balances_now(js_now)
    if today_equity is None:
        return [
            build_failure(
                "Check 4/4 FAILED - could not parse today's total equity from balances payload",
                category=FAILURE_CATEGORY_BALANCE_DRAWDOWN,
                key="balances_parse_error",
            )
        ]

    ok_hist, _, js_hist, _, err_hist = get_historical_balances()
    prev = None
    if ok_hist and isinstance(js_hist, dict):
        series = _extract_series_from_historical(js_hist)
        if series:
            prior = [row for row in series if row["date"] < today_str()]
            if prior:
                prev = prior[-1]["total_equity"]
    elif err_hist == "timeout":
        print(f"[{now()}] ⚠️ Historical balances timeout x{MAX_RETRIES}; using local store fallback")
    else:
        print(f"[{now()}] ⚠️ Historical balances API unavailable; using local store fallback")

    if prev is None:
        prev = _load_and_update_local_balance_store(today_equity)
        if prev is None:
            print("ℹ️ Check 4/4 fallback - no prior balance available yet; stored today's value for future comparisons.")
            return []

    drop_pct = (prev - today_equity) / prev * 100.0
    if drop_pct > BALANCE_DD_LIMIT_PCT:
        return [
            build_failure(
                f"Check 4/4 FAILED - equity drop {drop_pct:.2f}% exceeds {BALANCE_DD_LIMIT_PCT:.2f}% "
                f"(prev={prev:.2f}, today={today_equity:.2f})",
                category=FAILURE_CATEGORY_BALANCE_DRAWDOWN,
                key="balance_drawdown",
            )
        ]

    print(
        f"✅ Check 4/4 passed - equity drop {drop_pct:.2f}% within limit "
        f"(prev={prev:.2f}, today={today_equity:.2f})."
    )
    _load_and_update_local_balance_store(today_equity)
    return []


# === Orchestration ===
def run_checks():
    started_at = datetime.now().astimezone()
    started_monotonic = time.monotonic()
    print(f"\n=== Tradier Heartbeat @ {now()} ===")
    failures = []
    check1_failures, _ = check_positions()
    failures += check1_failures
    # failures += check_orders()
    price_range_failures = check_spx_price_range()
    failures += price_range_failures
    check3_failures, _ = check_preview_single_put()
    failures += check3_failures
    balance_failures = check_balance_drawdown()
    failures += balance_failures

    alert_failures = _apply_failure_thresholds(failures)

    checks = [
        {
            "id": "positions",
            "label": "Positions API",
            "status": "failed" if check1_failures else "passed",
            "failures": check1_failures,
        },
        {
            "id": "price_range",
            "label": f"{HEARTBEAT_SYMBOL} Price Guard",
            "status": "skipped" if SPX_PRICE_RANGE is None else ("failed" if price_range_failures else "passed"),
            "failures": price_range_failures,
        },
        {
            "id": "option_preview",
            "label": "Option Order Preview",
            "status": "failed" if check3_failures else "passed",
            "failures": check3_failures,
        },
        {
            "id": "balance_drawdown",
            "label": "Balance Drawdown",
            "status": "failed" if balance_failures else "passed",
            "failures": balance_failures,
        },
    ]

    def build_result():
        finished_at = datetime.now().astimezone()
        return {
            "status": "healthy" if not failures else "degraded",
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_seconds": round(time.monotonic() - started_monotonic, 3),
            "checks": checks,
            "failures": failures,
            "alert_failures": alert_failures,
        }

    if not failures:
        print("✅ All 4 checks passed successfully.\n")
        return build_result()

    print("\n".join(f"❌ {failure['message']}" for failure in failures))
    if not alert_failures:
        return build_result()

    send_alert(
        context_subject(f"Tradier Heartbeat Failures ({len(alert_failures)}) @ {now()}"),
        "\n\n".join(_format_failure_for_alert(failure) for failure in alert_failures),
    )
    return build_result()


def run_forever(send_initial_test_alert=True):
    if send_initial_test_alert:
        send_alert(context_subject("TEST Alert"), "")

    while True:
        try:
            run_checks()
        except Exception as exc:
            tb = traceback.format_exc()
            crash_failure = build_failure(
                f"Heartbeat Script Crash - {exc}",
                category=FAILURE_CATEGORY_SCRIPT_CRASH,
                key="script_crash",
                details=tb,
            )
            alert_failures = _apply_failure_thresholds([crash_failure])
            if alert_failures:
                send_alert(
                    context_subject("Heartbeat Script Crash"),
                    "\n\n".join(_format_failure_for_alert(failure) for failure in alert_failures),
                )
            print(tb)
        time.sleep(60)


def run(
    enable_sound_alert=False,
    keep_awake=False,
    max_print_chars_default=50,
    send_initial_test_alert=True,
    run_context=None,
    prompt_spx_range=False,
):
    configure_runtime(
        enable_sound_alert=enable_sound_alert,
        keep_awake=keep_awake,
        max_print_chars_default=max_print_chars_default,
        run_context=run_context,
    )
    if prompt_spx_range:
        prompt_spx_price_range()
    run_forever(send_initial_test_alert=send_initial_test_alert)


if __name__ == "__main__":
    run(
        enable_sound_alert=False,
        keep_awake=False,
        max_print_chars_default=50,
        send_initial_test_alert=True,
        run_context="GitHub Actions",
    )
