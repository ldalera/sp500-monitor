#!/bin/zsh
# Corre el scan diario y abre el dashboard en el navegador.
cd "$(dirname "$0")"
./venv/bin/python monitor.py && open dashboard.html
