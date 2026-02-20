#!/usr/bin/env python3
"""CI runner: no local alarm and no keep-awake interaction."""

from tradier_heartbeat_core import run


if __name__ == "__main__":
    run(
        enable_sound_alert=False,
        keep_awake=False,
        max_print_chars_default=50,
        send_initial_test_alert=True,
    )
