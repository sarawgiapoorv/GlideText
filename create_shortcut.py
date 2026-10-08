import os
import sys
import subprocess

if sys.platform != "win32":
    print("[GlideText] Desktop .lnk shortcuts are only applicable to Windows. On macOS, use Launch_GlideText.command.")
    sys.exit(0)


project_dir = os.path.dirname(os.path.abspath(__file__))
bat_path = os.path.join(project_dir, "Launch_GlideText.bat")
user_profile = os.environ.get("USERPROFILE", os.path.expanduser("~"))

desktops = [
    os.path.join(user_profile, "OneDrive", "Desktop"),
    os.path.join(user_profile, "Desktop")
]

for d in desktops:
    if os.path.exists(d):
        # Clean up legacy shortcut if present
        legacy_lnk = os.path.join(d, "LocalFlow.lnk")
        if os.path.exists(legacy_lnk):
            try:
                os.remove(legacy_lnk)
                print(f"Removed legacy shortcut: {legacy_lnk}")
            except Exception as e:
                print(f"Could not remove legacy shortcut: {e}")

        lnk = os.path.join(d, "GlideText.lnk")
        ps = f"""
$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut('{lnk}')
$s.TargetPath = '{bat_path}'
$s.WorkingDirectory = '{project_dir}'
$s.Description = 'GlideText - Speech to Mind Voice Dictation'
$s.Save()
"""
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
        print(f"Shortcut created at: {lnk} (exists: {os.path.exists(lnk)})")
