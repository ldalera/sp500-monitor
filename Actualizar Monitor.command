#!/bin/zsh
cd "$(dirname "$0")"
echo "📈 Actualizando el monitor S&P 500 (~2 min)..."
./venv/bin/python monitor.py && open dashboard.html && echo "✅ Listo. Podés cerrar esta ventana."
