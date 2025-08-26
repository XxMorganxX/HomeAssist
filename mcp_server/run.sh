#!/usr/bin/env bash

# Strict mode
set -Eeuo pipefail

# Resolve directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Choose a virtual environment
# Prefer existing project venv if present, otherwise create a dedicated one
DEFAULT_PROJECT_VENV="${PROJECT_ROOT}/venv"
MCP_VENV="${PROJECT_ROOT}/.venv_mcp"

if [[ -d "${DEFAULT_PROJECT_VENV}" ]]; then
  VENV_DIR="${DEFAULT_PROJECT_VENV}"
else
  VENV_DIR="${MCP_VENV}"
fi

PYTHON_BIN="python3"

echo "[run.sh] Project root: ${PROJECT_ROOT}"
echo "[run.sh] Using venv: ${VENV_DIR}"

# Create venv if missing
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[run.sh] Creating virtual environment..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# Activate venv
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

# Ensure pip is up to date
python -m pip install --upgrade pip >/dev/null

# Install MCP server dependencies
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
if [[ -f "${REQ_FILE}" ]]; then
  echo "[run.sh] Installing MCP server dependencies from ${REQ_FILE}..."
  pip install -r "${REQ_FILE}"
else
  echo "[run.sh] WARNING: requirements.txt not found at ${REQ_FILE}. Skipping install."
fi

# Export PYTHONPATH to ensure project root imports work for tools
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Defaults (can be overridden by environment variables)
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-3000}"
TRANSPORT="${TRANSPORT:-http}"

echo "[run.sh] Host: ${HOST}"
echo "[run.sh] Port: ${PORT}"
echo "[run.sh] Transport: ${TRANSPORT}"

echo "[run.sh] Starting MCP server..."

# If flags are provided, pass them straight through. Otherwise use defaults.
if [[ $# -gt 0 ]]; then
  exec python -m mcp_server.server "$@"
else
  exec python -m mcp_server.server --host "${HOST}" --port "${PORT}" --transport "${TRANSPORT}"
fi


