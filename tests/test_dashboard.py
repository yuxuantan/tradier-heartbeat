import threading
import time
import unittest
import urllib.request
from datetime import date, timedelta
from unittest import mock

import tradier_heartbeat_core as core
from tradier_dashboard import DashboardHandler, DashboardServer, HeartbeatController


def sample_result(status="healthy"):
    failure = [] if status == "healthy" else [core.build_failure("A check failed", key="test_failure")]
    return {
        "status": status,
        "started_at": "2026-07-13T10:00:00+08:00",
        "finished_at": "2026-07-13T10:00:01+08:00",
        "duration_seconds": 1.25,
        "checks": [
            {
                "id": "positions",
                "label": "Positions API",
                "status": "passed" if status == "healthy" else "failed",
                "failures": failure,
            }
        ],
        "failures": failure,
        "alert_failures": failure,
    }


class CoreResultTests(unittest.TestCase):
    @mock.patch.object(core, "send_alert")
    @mock.patch.object(core, "_apply_failure_thresholds", return_value=[])
    @mock.patch.object(core, "check_balance_drawdown", return_value=[])
    @mock.patch.object(core, "check_preview_single_put", return_value=([], 0.2))
    @mock.patch.object(core, "check_spx_price_range", return_value=[])
    @mock.patch.object(core, "check_positions", return_value=([], {}))
    def test_run_checks_returns_structured_healthy_result(self, *_mocks):
        original_range = core.SPX_PRICE_RANGE
        core.SPX_PRICE_RANGE = (5000, 7000)
        try:
            result = core.run_checks()
        finally:
            core.SPX_PRICE_RANGE = original_range

        self.assertEqual(result["status"], "healthy")
        self.assertEqual([item["id"] for item in result["checks"]], [
            "positions", "price_range", "option_preview", "balance_drawdown"
        ])
        self.assertTrue(all(item["status"] == "passed" for item in result["checks"]))

    @mock.patch.object(core, "send_alert")
    @mock.patch.object(core, "_apply_failure_thresholds", return_value=[])
    @mock.patch.object(core, "check_balance_drawdown", return_value=[])
    @mock.patch.object(core, "check_preview_single_put", return_value=([], 0.2))
    @mock.patch.object(core, "check_spx_price_range", return_value=[])
    @mock.patch.object(core, "check_positions", return_value=([], {}))
    def test_disabled_price_guard_is_reported_as_skipped(self, *_mocks):
        original_range = core.SPX_PRICE_RANGE
        core.SPX_PRICE_RANGE = None
        try:
            result = core.run_checks()
        finally:
            core.SPX_PRICE_RANGE = original_range

        price_check = next(item for item in result["checks"] if item["id"] == "price_range")
        self.assertEqual(price_check["status"], "skipped")


class ControllerTests(unittest.TestCase):
    def test_controller_runs_immediately_and_records_result(self):
        completed = threading.Event()

        def fake_checks():
            completed.set()
            return sample_result()

        controller = HeartbeatController(interval_seconds=3600)
        with mock.patch.object(core, "run_checks", side_effect=fake_checks):
            controller.start()
            self.assertTrue(completed.wait(timeout=1))
            deadline = time.time() + 1
            while controller.snapshot()["service"]["run_count"] == 0 and time.time() < deadline:
                time.sleep(0.01)
            snapshot = controller.snapshot()
            controller.shutdown()

        self.assertEqual(snapshot["service"]["run_count"], 1)
        self.assertEqual(snapshot["last_result"]["status"], "healthy")
        self.assertEqual(snapshot["history"][0]["duration_seconds"], 1.25)

    def test_configuration_validates_interval_and_range(self):
        controller = HeartbeatController(interval_seconds=60)
        with self.assertRaisesRegex(ValueError, "between 15 and 3600"):
            controller.configure({"interval_seconds": 5, "price_range_enabled": False})
        with self.assertRaisesRegex(ValueError, "low cannot be greater"):
            controller.configure({
                "interval_seconds": 60,
                "price_range_enabled": True,
                "price_range_low": 6000,
                "price_range_high": 5000,
            })

    def test_balance_history_only_contains_past_30_calendar_days(self):
        controller = HeartbeatController(interval_seconds=60)
        today = date.today()
        store = {
            (today - timedelta(days=31)).isoformat(): 100.0,
            (today - timedelta(days=29)).isoformat(): 101.0,
            (today - timedelta(days=1)).isoformat(): 102.0,
            today.isoformat(): 103.0,
        }
        with mock.patch.object(core, "read_balance_store", return_value=store):
            points = controller._balance_history()

        self.assertEqual([point["value"] for point in points], [101.0, 102.0, 103.0])

    @mock.patch("tradier_dashboard.sys.platform", "darwin")
    @mock.patch("tradier_dashboard.shutil.which", return_value="/usr/bin/osascript")
    @mock.patch("tradier_dashboard.subprocess.run")
    def test_alarm_volume_rises_then_restores_original_mac_state(self, run_mock, _which):
        controller = HeartbeatController(interval_seconds=60)
        controller._read_system_volume = mock.Mock(return_value=(42, True))

        raised = controller.raise_system_volume(step=5)
        restored = controller.restore_system_volume()

        self.assertEqual(raised["volume"], 47)
        self.assertTrue(raised["changed"])
        self.assertEqual(restored, {"supported": True, "restored": True, "volume": 42, "muted": True})
        raise_command = run_mock.call_args_list[0].args[0]
        restore_command = run_mock.call_args_list[1].args[0]
        self.assertIn("set volume output volume 47", raise_command)
        self.assertIn("set volume output volume 42", restore_command)
        self.assertIn("set volume output muted true", restore_command)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.controller = HeartbeatController(interval_seconds=60)
        self.controller.last_result = sample_result()
        DashboardHandler.controller = self.controller
        self.server = DashboardServer(("127.0.0.1", 0), DashboardHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)

    def test_dashboard_asset_and_status_api_are_served(self):
        with urllib.request.urlopen(f"{self.base_url}/", timeout=1) as response:
            page = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertIn("Tradier Heartbeat", page)

        with urllib.request.urlopen(f"{self.base_url}/api/status", timeout=1) as response:
            payload = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertIn('"status":"healthy"', payload)


if __name__ == "__main__":
    unittest.main()
