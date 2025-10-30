@echo off
setlocal
set URL=http://127.0.0.1:9090/invocations
set CSV=%~1
if "%CSV%"=="" set CSV=tmp\sample.csv
if not exist predictions mkdir predictions
curl.exe -s -X POST -H "Content-Type: text/csv" --data-binary "@%CSV%" %URL% > predictions\preds_from_curl.json
echo [OK] Predicciones guardadas en predictions\preds_from_curl.json
endlocal
