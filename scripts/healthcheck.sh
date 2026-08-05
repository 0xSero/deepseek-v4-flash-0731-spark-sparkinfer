#!/usr/bin/env bash
set -euo pipefail
curl -fsS --connect-timeout 2 --max-time 5 http://127.0.0.1:8000/health >/dev/null
