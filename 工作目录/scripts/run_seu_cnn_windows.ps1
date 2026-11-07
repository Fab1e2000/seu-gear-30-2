$ErrorActionPreference = 'Stop'
$root = 'D:\SEU-training-20261104\cnn-search'
$python = 'D:\SEU-training-20261104\.venv\Scripts\python.exe'
Set-Location $root
& $python -u train_seu_cnn.py --smoke *> smoke.log
if ($LASTEXITCODE -ne 0) { throw 'CNN smoke failed; see smoke.log' }
& $python -u train_seu_cnn.py *> training.log
if ($LASTEXITCODE -ne 0) { throw 'CNN search failed; see training.log' }
