Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = scriptDir

venvPythonw = scriptDir & "\.venv\Scripts\pythonw.exe"
If fso.FileExists(venvPythonw) Then
    cmd = """" & venvPythonw & """" & " main.py"
Else
    cmd = "pythonw.exe main.py"
End If

WshShell.Run cmd, 0, False
Set WshShell = Nothing
Set fso = Nothing
