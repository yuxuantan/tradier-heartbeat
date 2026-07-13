#!/usr/bin/env python3
"""Local runner for the browser dashboard or legacy terminal heartbeat."""

import sys


if __name__ == "__main__":
    if "--cli" in sys.argv:
        from tradier_heartbeat_core import run

        sys.argv.remove("--cli")
        run(
            enable_sound_alert=True,
            keep_awake=True,
            max_print_chars_default=100,
            send_initial_test_alert=True,
            run_context="Local PC",
            prompt_spx_range=True,
        )
    else:
        from tradier_dashboard import main

        main()
