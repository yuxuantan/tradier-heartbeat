#!/usr/bin/env python3
"""Local runner: keep awake + rising-volume local alarm."""

from tradier_heartbeat_core import run


if __name__ == "__main__":
    run(
        enable_sound_alert=True,
        keep_awake=True,
        max_print_chars_default=100,
        send_initial_test_alert=True,
    )
