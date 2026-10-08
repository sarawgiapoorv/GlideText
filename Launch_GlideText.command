#!/usr/bin/env bash
# GlideText macOS Launcher
# Activates the local virtual environment and starts GlideText.

set -e

# Resolve script directory (handles symlinks and arbitrary cwd)
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
PROJECT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
cd "$PROJECT_DIR"

# Locate Python executable in .venv or venv
if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON_EXEC="$PROJECT_DIR/.venv/bin/python"
elif [ -f "$PROJECT_DIR/venv/bin/python" ]; then
    PYTHON_EXEC="$PROJECT_DIR/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXEC="$(command -v python3)"
else
    PYTHON_EXEC="python"
fi

# Execute GlideText passing through all command-line arguments
exec "$PYTHON_EXEC" "$PROJECT_DIR/main.py" "$@"
