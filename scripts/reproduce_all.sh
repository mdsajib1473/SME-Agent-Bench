#!/usr/bin/env bash
# Reproduce everything: task validation, unit tests, the full experiment and the analysis.
#
# Usage (Linux, macOS, or Git Bash on Windows):
#     bash scripts/reproduce_all.sh
#
# The experiment step is resumable: run the script again after an interruption and it
# continues results/main where it stopped.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TAG="main"

if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
else
    echo "STOP: the virtual environment is missing. Create it with:"
    echo "  python3.11 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
    exit 1
fi

stop() {
    echo
    echo "STOP: $1"
    exit 1
}

step() {
    local name="$1"
    shift
    echo
    echo "== $name"
    "$PYTHON" "$@" || stop "$name failed (exit code $?). Fix the problem above and run the script again."
}

# Ollama must be running and every model in config.yaml must exist; agents.llm.check_ollama
# names the missing model and the ollama create command for it.
echo "== Checking Ollama and the models in config.yaml"
if ! "$PYTHON" -c "
import sys
from agents.llm import check_ollama, load_config, native_base, ollama_get
c = load_config()
p = check_ollama(c['models'], c)
if p:
    print(p)
    sys.exit(1)
print('ollama', ollama_get(native_base(c['ollama_base_url']) + '/api/version', 10)['version'], 'with', ', '.join(c['models']))
"; then
    if ! command -v ollama > /dev/null 2>&1; then
        stop "Ollama is not installed or not on PATH. Install it from ollama.com, start it, create the models (see README.md) and run the script again."
    fi
    stop "Ollama or a model is missing (see the message above). Start Ollama (ollama serve) and create the models with: ollama create <tag> -f models/<tag>.Modelfile"
fi

step "Task validation" eval/validate_tasks.py
step "Unit tests" -m pytest -q
step "Full experiment (about 15 GPU hours)" run.py --tag "$TAG" --runs 3 --resume
step "Analysis" -m analysis.run_all --tag "$TAG"

echo
echo "Done. Tables, figures and summary.md are in results/$TAG/analysis/"
