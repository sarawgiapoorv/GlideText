Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "c:\Users\jai shree shyam\OneDrive\Desktop\LocalFlow"
WshShell.Run "pythonw.exe main.py", 0, False
Set WshShell = Nothing
