"""
platform_compat/permissions_check.py
CLI and diagnostic utility to verify macOS system permissions for GlideText.

Can be run directly via:
    python -m platform_compat.permissions_check

Checks:
  1. Microphone Access (TCC / AVFoundation)
  2. Accessibility Access (ApplicationServices AXIsProcessTrusted)
  3. Input Monitoring Access (Quartz CGPreflightListenEventAccess)
"""

import os
import platform
import sys
from typing import Tuple


def run_diagnostics(prompt: bool = True) -> int:
    """Run interactive or automated diagnostic check of required permissions."""
    print("=" * 60)
    print("       GlideText macOS Permissions & Health Diagnostic")
    print("=" * 60)

    print(f"Platform:         {sys.platform}")
    print(f"OS Version:       {platform.mac_ver()[0] or platform.platform()}")
    print(f"Architecture:     {platform.machine()} ({'Apple Silicon' if platform.machine() == 'arm64' else 'Intel'})")
    print(f"Python Executable:{sys.executable}")
    print(f"Process PID:      {os.getpid()}")
    print("-" * 60)

    if sys.platform != "darwin":
        print("[INFO] Not running on macOS. All macOS permissions are considered allowed / N/A.")
        print("=" * 60)
        return 0

    import platform_compat

    all_passed = True

    # 1. Microphone Check
    print("Checking [1/3] Microphone Permission...")
    mic_ok, mic_msg = platform_compat.check_microphone_permission()
    if mic_ok:
        print("  [PASS] Microphone: GRANTED")
    else:
        all_passed = False
        print("  [FAIL] Microphone: DENIED or NOT DETERMINED")
        print(f"         {mic_msg}")
        print("         -> Fix: Open System Settings > Privacy & Security > Microphone")
        print("                 and toggle ON for your terminal or Python.")

    # 2. Accessibility Check
    print("Checking [2/3] Accessibility Permission (synthetic keystrokes / text injection)...")
    acc_ok, acc_msg = platform_compat.check_accessibility_permission(prompt=prompt)
    if acc_ok:
        print("  [PASS] Accessibility: GRANTED")
    else:
        all_passed = False
        print("  [FAIL] Accessibility: DENIED or MISSING")
        print(f"         {acc_msg}")
        print("         -> Fix: Open System Settings > Privacy & Security > Accessibility")
        print("                 and enable permission for the app running GlideText.")

    # 3. Input Monitoring Check
    print("Checking [3/3] Input Monitoring Permission (global hotkey listener)...")
    inp_ok, inp_msg = platform_compat.check_input_monitoring_permission(prompt=prompt)
    if inp_ok:
        print("  [PASS] Input Monitoring: GRANTED")
    else:
        all_passed = False
        print("  [FAIL] Input Monitoring: DENIED or MISSING")
        print(f"         {inp_msg}")
        print("         -> Fix: Open System Settings > Privacy & Security > Input Monitoring")
        print("                 and enable permission for your terminal or Python.")

    print("-" * 60)
    if all_passed:
        print("[SUCCESS] All required permissions are granted! GlideText is ready to run.")
        print("=" * 60)
        return 0
    else:
        print("[WARNING] One or more permissions are missing.")
        print("GlideText requires Accessibility and Input Monitoring to capture hotkeys")
        print("and inject polished text into target applications.")
        print("Note: Do NOT run GlideText with sudo. Grant permissions via System Settings.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(run_diagnostics(prompt=True))
