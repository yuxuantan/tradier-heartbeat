#!/usr/bin/env python3
"""Local-only web dashboard for the Tradier heartbeat engine."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from datetime import date, datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import tradier_heartbeat_core as heartbeat


ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = ROOT / "dashboard"


def _iso_now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class HeartbeatController:
    """Runs heartbeat checks on a background thread and exposes safe snapshots."""

    def __init__(self, interval_seconds=60, history_limit=120):
        self.interval_seconds = max(15, int(interval_seconds))
        self.history_limit = history_limit
        self.started_at = _iso_now()
        self.monitoring = True
        self.checking = False
        self.last_result = None
        self.run_count = 0
        self.next_run_at_epoch = None
        self.history = []
        self.last_error = None
        self._original_system_volume = None
        self._original_system_muted = None
        self._last_system_volume = None
        self._manual_run_requested = False
        self._shutdown = False
        self._condition = threading.Condition()
        self._worker = threading.Thread(target=self._run_loop, name="tradier-heartbeat", daemon=True)

    def start(self):
        self._worker.start()

    def shutdown(self):
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()
        self._worker.join(timeout=2)
        self.restore_system_volume()

    def _read_system_volume(self):
        volume_result = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        muted_result = subprocess.run(
            ["osascript", "-e", "output muted of (get volume settings)"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return int(volume_result.stdout.strip()), muted_result.stdout.strip().lower() == "true"

    def raise_system_volume(self, step=5):
        """Raise macOS volume by a small step, preserving the pre-alarm state."""
        if sys.platform != "darwin" or not shutil.which("osascript"):
            return {"supported": False, "changed": False}

        step = max(1, min(10, int(step)))
        with self._condition:
            try:
                current_volume, current_muted = self._read_system_volume()
                if self._original_system_volume is None:
                    self._original_system_volume = current_volume
                    self._original_system_muted = current_muted
                target = min(100, current_volume + step)
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        "set volume output muted false",
                        "-e",
                        f"set volume output volume {target}",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self._last_system_volume = target
                return {"supported": True, "changed": target != current_volume or current_muted, "volume": target}
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                return {"supported": True, "changed": False, "error": str(exc)}

    def restore_system_volume(self):
        """Restore the volume and mute state captured before alarm escalation."""
        with self._condition:
            volume = self._original_system_volume
            muted = self._original_system_muted
            if volume is None or sys.platform != "darwin" or not shutil.which("osascript"):
                return {"supported": sys.platform == "darwin", "restored": False}
            try:
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        f"set volume output volume {int(volume)}",
                        "-e",
                        f"set volume output muted {'true' if muted else 'false'}",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if self._original_system_volume == volume:
                    self._original_system_volume = None
                    self._original_system_muted = None
                    self._last_system_volume = None
                return {"supported": True, "restored": True, "volume": int(volume), "muted": bool(muted)}
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                return {"supported": True, "restored": False, "error": str(exc)}

    def set_monitoring(self, enabled):
        with self._condition:
            self.monitoring = bool(enabled)
            if self.monitoring:
                self.next_run_at_epoch = time.time()
            else:
                self.next_run_at_epoch = None
            self._condition.notify_all()

    def request_run(self):
        with self._condition:
            if self.checking or self._manual_run_requested:
                return False
            self._manual_run_requested = True
            self._condition.notify_all()
            return True

    def configure(self, payload):
        with self._condition:
            if self.checking:
                raise ValueError("Wait for the active heartbeat cycle to finish before changing settings.")

        interval = self.interval_seconds
        price_range = None
        enabled = payload.get("price_range_enabled", heartbeat.SPX_PRICE_RANGE is not None)
        if not isinstance(enabled, bool):
            raise ValueError("price_range_enabled must be true or false.")

        if "interval_seconds" in payload:
            interval = int(payload["interval_seconds"])
            if not 15 <= interval <= 3600:
                raise ValueError("Interval must be between 15 and 3600 seconds.")

        if enabled:
            low = float(payload.get("price_range_low"))
            high = float(payload.get("price_range_high"))
            if low <= 0 or high <= 0:
                raise ValueError("SPX range bounds must be positive numbers.")
            if low > high:
                raise ValueError("SPX range low cannot be greater than high.")
            price_range = (low, high)

        with self._condition:
            if enabled:
                heartbeat.configure_spx_price_range(*price_range)
            else:
                heartbeat.clear_spx_price_range()
            self.interval_seconds = interval

            if self.monitoring:
                self.next_run_at_epoch = time.time() + self.interval_seconds
            self._condition.notify_all()

    def _run_loop(self):
        next_due = 0.0
        while True:
            with self._condition:
                while True:
                    if self._shutdown:
                        return
                    now_epoch = time.time()
                    scheduled = self.monitoring and now_epoch >= next_due
                    if self._manual_run_requested or scheduled:
                        self._manual_run_requested = False
                        self.checking = True
                        self.next_run_at_epoch = None
                        break
                    self.next_run_at_epoch = next_due if self.monitoring else None
                    wait_for = max(0.1, next_due - now_epoch) if self.monitoring else None
                    self._condition.wait(timeout=wait_for)

            try:
                result = heartbeat.run_checks()
                self._record_result(result)
            except Exception as exc:
                self._record_crash(exc)
            finally:
                with self._condition:
                    self.checking = False
                    next_due = time.time() + self.interval_seconds
                    self.next_run_at_epoch = next_due if self.monitoring else None
                    self._condition.notify_all()

    def _record_result(self, result):
        with self._condition:
            self.last_result = result
            self.last_error = None
            self.run_count += 1
            self.history.append(
                {
                    "finished_at": result["finished_at"],
                    "status": result["status"],
                    "duration_seconds": result["duration_seconds"],
                    "failed_checks": sum(1 for item in result["checks"] if item["status"] == "failed"),
                }
            )
            self.history = self.history[-self.history_limit :]

    def _record_crash(self, exc):
        with self._condition:
            finished_at = _iso_now()
            self.last_error = str(exc)
            self.run_count += 1
            self.last_result = {
                "status": "error",
                "started_at": finished_at,
                "finished_at": finished_at,
                "duration_seconds": 0,
                "checks": [],
                "failures": [{"message": str(exc), "category": "script_crash", "details": traceback.format_exc()}],
                "alert_failures": [],
            }
            self.history.append(
                {"finished_at": finished_at, "status": "error", "duration_seconds": 0, "failed_checks": 1}
            )
            self.history = self.history[-self.history_limit :]

    def _balance_history(self):
        store = heartbeat.read_balance_store()
        points = []
        today = date.today()
        cutoff = today - timedelta(days=29)
        for day, value in sorted(store.items()):
            try:
                balance_date = date.fromisoformat(str(day))
                if balance_date < cutoff or balance_date > today:
                    continue
                points.append({"date": str(day), "value": float(value)})
            except (TypeError, ValueError):
                continue
        return points

    def snapshot(self):
        with self._condition:
            price_range = heartbeat.SPX_PRICE_RANGE
            return {
                "service": {
                    "status": "checking" if self.checking else (self.last_result or {}).get("status", "starting"),
                    "monitoring": self.monitoring,
                    "checking": self.checking,
                    "started_at": self.started_at,
                    "run_count": self.run_count,
                    "next_run_at_epoch": self.next_run_at_epoch,
                    "last_error": self.last_error,
                },
                "config": {
                    "symbol": heartbeat.HEARTBEAT_SYMBOL,
                    "interval_seconds": self.interval_seconds,
                    "sla_seconds": heartbeat.SLA_SECS,
                    "drawdown_limit_pct": heartbeat.BALANCE_DD_LIMIT_PCT,
                    "price_range_enabled": price_range is not None,
                    "price_range_low": price_range[0] if price_range else None,
                    "price_range_high": price_range[1] if price_range else None,
                    "credentials_configured": bool(heartbeat.TRADIER_ACCESS_TOKEN and heartbeat.ACCOUNT_ID),
                    "email_configured": bool(heartbeat.SMTP_HOST and heartbeat.EMAIL_FROM and heartbeat.EMAIL_TO),
                },
                "last_result": self.last_result,
                "history": list(self.history),
                "balance_history": self._balance_history(),
                "server_time": _iso_now(),
            }


class DashboardHandler(BaseHTTPRequestHandler):
    controller = None

    def log_message(self, fmt, *args):
        print(f"[{heartbeat.now()}] Dashboard {self.address_string()} - {fmt % args}")

    def _security_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'",
        )

    def _send_bytes(self, body, content_type, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid content length.") from exc
        if length > 16_384:
            raise ValueError("Request body is too large.")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            self._json(self.controller.snapshot())
            return

        assets = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        asset = assets.get(path)
        if not asset:
            self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        filename, content_type = asset
        try:
            body = (DASHBOARD_DIR / filename).read_bytes()
        except OSError:
            self._json({"error": "Dashboard asset unavailable"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_bytes(body, content_type)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/run":
                accepted = self.controller.request_run()
                self._json({"accepted": accepted}, HTTPStatus.ACCEPTED if accepted else HTTPStatus.CONFLICT)
            elif path == "/api/monitoring":
                if "enabled" not in payload:
                    raise ValueError("The enabled field is required.")
                self.controller.set_monitoring(bool(payload["enabled"]))
                self._json({"ok": True, "monitoring": bool(payload["enabled"])})
            elif path == "/api/config":
                self.controller.configure(payload)
                self._json({"ok": True})
            elif path == "/api/alarm/volume":
                self._json(self.controller.raise_system_volume(payload.get("step", 5)))
            elif path == "/api/alarm/reset":
                self._json(self.controller.restore_system_volume())
            else:
                self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (TypeError, ValueError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(host="127.0.0.1", port=8787, open_browser=True):
    heartbeat.configure_runtime(
        enable_sound_alert=False,
        keep_awake=True,
        max_print_chars_default=100,
        run_context="Local Dashboard",
    )

    range_low = os.getenv("HEARTBEAT_RANGE_LOW")
    range_high = os.getenv("HEARTBEAT_RANGE_HIGH")
    if range_low and range_high:
        heartbeat.configure_spx_price_range(range_low, range_high)

    controller = HeartbeatController(interval_seconds=int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "60")))
    DashboardHandler.controller = controller
    server = DashboardServer((host, port), DashboardHandler)
    controller.start()
    url = f"http://{host}:{server.server_port}"
    print(f"[{heartbeat.now()}] Tradier Heartbeat dashboard running at {url}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print(f"\n[{heartbeat.now()}] Stopping dashboard...")
    finally:
        server.shutdown()
        server.server_close()
        controller.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tradier Heartbeat localhost dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8787, help="Dashboard port (default: 8787)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")
    args = parser.parse_args(argv)
    serve(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
