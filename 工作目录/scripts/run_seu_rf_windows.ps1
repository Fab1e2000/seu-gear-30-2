$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& '.\.venv\Scripts\python.exe' -m pip freeze | Set-Content -Encoding utf8 requirements-lock.txt
& '.\.venv\Scripts\python.exe' -u '.\train_seu_rf_cv.py' --root $PSScriptRoot *> training.log
if ($LASTEXITCODE -ne 0) { throw "Training failed; see training.log" }
