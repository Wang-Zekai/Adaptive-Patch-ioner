$ErrorActionPreference = 'Stop'
$setupPython = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
& $setupPython (Join-Path $PSScriptRoot 'setup_assets.py') @args
exit $LASTEXITCODE
