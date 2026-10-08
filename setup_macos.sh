#!/usr/bin/env bash
# GlideText macOS Automated Setup Script
# Checks system requirements, creates Python virtual environment,
# installs dependencies, and runs permissions verification.

set -e

echo "=================================================="
echo "    GlideText macOS Setup & Diagnostic Tool       "
echo "=================================================="

# 1. Verify operating system
OS_NAME="$(uname -s)"
if [ "$OS_NAME" != "Darwin" ]; then
    echo "[-] Error: setup_macos.sh must be run on macOS (detected: $OS_NAME)."
    exit 1
fi

ARCH="$(uname -m)"
echo "[+] Detected Architecture: $ARCH ($( [ "$ARCH" = "arm64" ] && echo "Apple Silicon" || echo "Intel Mac" ))"

# 2. Check Homebrew
if command -v brew >/dev/null 2>&1; then
    BREW_BIN="$(command -v brew)"
    echo "[+] Found Homebrew at: $BREW_BIN"
elif [ -x "/opt/homebrew/bin/brew" ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
    echo "[+] Found Homebrew at: /opt/homebrew/bin/brew"
elif [ -x "/usr/local/bin/brew" ]; then
    eval "$(/usr/local/bin/brew shellenv)"
    echo "[+] Found Homebrew at: /usr/local/bin/brew"
else
    echo "[!] Warning: Homebrew is not installed. Recommended for installing Python and Node.js."
    echo "    Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
fi

# 3. Check Python 3.10+
PYTHON_CMD=""
for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        VER="$("$cand" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        MAJOR="$("$cand" -c 'import sys; print(sys.version_info.major)')"
        MINOR="$("$cand" -c 'import sys; print(sys.version_info.minor)')"
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_CMD="$cand"
            echo "[+] Found compatible Python: $cand ($VER)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "[-] Error: Python 3.10+ was not found."
    echo "    Please install Python via Homebrew: brew install python@3.11"
    exit 1
fi

# 4. Check Node.js & npm (Optional for FreeLLMAPI)
if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    NODE_VER="$(node -v)"
    echo "[+] Found Node.js ($NODE_VER) and npm (FreeLLMAPI supported)"
else
    echo "[!] Note: Node.js/npm not detected. FreeLLMAPI local server requires Node.js."
    echo "    Install via Homebrew if desired: brew install node"
fi

# 5. Check Ollama (Optional for Tier 3 Local LLM)
if command -v ollama >/dev/null 2>&1 || [ -x "/opt/homebrew/bin/ollama" ] || [ -d "/Applications/Ollama.app" ]; then
    echo "[+] Found Ollama (Offline Tier 3 LLM supported)"
else
    echo "[!] Note: Ollama not detected. Optional for 100% offline local polishing."
    echo "    Download from https://ollama.com if desired."
fi

# 6. Create & activate Virtual Environment
PROJECT_DIR="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    echo "[*] Creating Python virtual environment in .venv..."
    "$PYTHON_CMD" -m venv .venv
else
    echo "[+] Virtual environment .venv already exists."
fi

source .venv/bin/activate
echo "[+] Active Python: $(which python)"

# 7. Install dependencies
echo "[*] Upgrading pip and wheel..."
pip install --upgrade pip wheel

echo "[*] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 8. Ensure launcher is executable
if [ -f "Launch_GlideText.command" ]; then
    chmod +x Launch_GlideText.command
    echo "[+] Made Launch_GlideText.command executable."
fi

# 9. Run permissions diagnostic
echo ""
echo "=================================================="
echo "    Running macOS Permissions Diagnostic          "
echo "=================================================="
python -m platform_compat.permissions_check || true

echo ""
echo "=================================================="
echo "    Setup Complete!                               "
echo "=================================================="
echo "To run GlideText:"
echo "  1. Double-click Launch_GlideText.command in Finder"
echo "  or in your terminal:"
echo "     source .venv/bin/activate"
echo "     python main.py"
echo "=================================================="
