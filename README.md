# S&P 500 · Monitor de Oportunidades

Recomendador diario de oportunidades de compra en el S&P 500. Escanea las ~503
acciones del índice, las puntúa con un modelo compuesto y genera un dashboard
HTML (`dashboard.html`) con el justificativo completo y gráficos de cada pick.

## Uso diario

```bash
./run_daily.sh
```

Corre el scan (~2 minutos) y abre `dashboard.html` en el navegador.
Ideal correrlo después del cierre de NY (18:30 ART) o antes de la apertura.

Para automatizarlo todos los días hábiles a las 08:30, agregá a tu crontab
(`crontab -e`):

```
30 8 * * 1-5 /Users/ldalera/Downloads/sp500-monitor/run_daily.sh >> /tmp/sp500-monitor.log 2>&1
```

## Modelo de scoring

`Score = 45% técnico + 30% fundamental + 15% sentimiento + 10% macro`

| Pilar | Fuentes / señales |
|---|---|
| **Técnico (45%)** | RSI(14), MACD(12,26,9) con detección de cruces frescos, ADX(14) +DI/-DI, SMA 20/50/200, Bollinger(20,2σ), ATR(14), volumen relativo vs 20 ruedas, distancia al máx/mín de 52 semanas, momentum 1/3/6 meses. Clasifica cada acción en setup **MOMENTUM** (continuación) o **REVERSIÓN** (pullback en tendencia alcista). |
| **Fundamental (30%)** | P/E forward vs referencia sectorial, PEG, crecimiento de ingresos, margen neto, ROE, FCF yield, upside al target de analistas, consenso (Yahoo Finance). |
| **Sentimiento (15%)** | Léxico sobre titulares de noticias, menciones en subreddits financieros con aceleración 24h (ApeWisdom), ratio bullish/bearish de Stocktwits cuando responde. |
| **Macro (10%)** | Probabilidades de Polymarket (decisiones de la Fed, recesión) + régimen de mercado (SPY vs SMA50/200, breadth del índice, VIX) con tilt sectorial. |

Cada pick incluye **plan de trade**: stop por estructura (mínimo swing 10 ruedas)
o 2×ATR, objetivos a 2R y 3R, y tamaño de posición para arriesgar 1% del capital.

## Solapa "Consultar ticker"

Escribí cualquier ticker del S&P 500 (con autocompletado) y ves al instante su
análisis técnico completo: score, setup, lectura técnica, plan de trade y los 4
gráficos, con los datos del scan del día. Los que fueron candidatos del día
muestran además fundamental, sentimiento y noticias.

**Focus list**: agregá tickers (uno por línea) en `data/focus.txt` para que
reciban análisis profundo (fundamental + sentimiento) en TODAS las corridas,
estén o no entre los mejores candidatos técnicos.

## Solapa "Backtest"

Evalúa automáticamente cada pick histórico contra la evolución posterior del
precio: primer toque del objetivo 2R (+2R) o del stop (−1R), con cierre a
mercado a las 40 ruedas. Muestra win rate, R promedio, profit factor, alfa a
10 ruedas vs SPY, curva de R acumulado y desglose por setup y por score.
Se alimenta de los `data/scan_*.json` que cada corrida diaria guarda, así que
mejora sola a medida que se acumula historia (los primeros resultados cerrados
aparecen a los pocos días; estadística con algo de sustancia, en 4-8 semanas).

## Versión en la nube (celular incluido)

El repo incluye `.github/workflows/daily.yml`: GitHub Actions corre el scan
todos los días hábiles a las 22:30 UTC (después del cierre de NY) y publica el
dashboard en GitHub Pages. Resultado: una URL pública tipo
`https://<usuario>.github.io/sp500-monitor/` que se puede abrir desde cualquier
dispositivo, incluido el celular (el layout es responsive).

- **Actualización manual desde el celu**: en la app de GitHub → repo → Actions
  → "Scan diario S&P 500" → Run workflow.
- **Editar la focus list desde el celu**: editá `data/focus.txt` en la app de
  GitHub; el próximo scan la toma.
- El histórico diario (`data/scan_*.json`) se guarda automáticamente en el repo.

## Archivos

- `monitor.py` — pipeline completo (datos → indicadores → scoring → HTML)
- `template.html` — template del dashboard (el JSON del día se inyecta acá)
- `dashboard.html` — salida del día (autocontenida, salvo Chart.js por CDN)
- `data/scan_YYYY-MM-DD.json` — histórico de cada corrida
- `run_daily.sh` — corre el scan y abre el dashboard

## Parámetros ajustables (arriba de `monitor.py`)

- `N_PICKS` / `N_WATCHLIST` / `N_CANDIDATES` — tamaño de la selección
- `W_TECH, W_FUND, W_SENT, W_MACRO` — pesos del modelo
- `SECTOR_PE_REF` — referencias de valuación sectorial

## Notas

- Sin API keys: usa Yahoo Finance, Polymarket (API Gamma), ApeWisdom y Stocktwits.
- Si tu ISP bloquea `polymarket.com` (común en Argentina), el script lo resuelve
  solo vía DNS-over-HTTPS + `curl --resolve`.
- ⚠️ Herramienta de análisis, no asesoramiento financiero. Validá siempre con tu
  propio criterio y gestión de riesgo.
