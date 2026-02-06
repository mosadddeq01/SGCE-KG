#!/usr/bin/env bash
# quick_clear.sh  -- fast, safe, project-local cache clear
set -euo pipefail

PWD_ESC=$(printf '%s\n' "$PWD" | sed 's/\//\\\//g')
echo "Project root: $PWD"

# 1) Kill only Python/jupyter processes that reference this project path
PIDS=$(ps aux | grep -E 'python|ipykernel|jupyter' | grep -v grep | grep -F "$PWD" | awk '{print $2}' | sort -u || true)
if [[ -n "$PIDS" ]]; then
  echo "Killing processes: $PIDS"
  kill $PIDS 2>/dev/null || kill -9 $PIDS 2>/dev/null || true
else
  echo "No project-linked python/jupyter processes found."
fi

# 2) Remove __pycache__ and *.pyc under project (fast)
echo "Removing __pycache__ and .pyc (project-local)..."
find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
find . -type f -name "*.pyc" -delete || true

# 3) Remove Jupyter checkpoints (project-local)
echo "Removing .ipynb_checkpoints (project-local)..."
find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} + || true

echo "Done. Restart VSCode if needed."
s