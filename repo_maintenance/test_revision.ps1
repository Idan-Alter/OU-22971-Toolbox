# Focused S1-S3 acceptance using disposable listeners and a fixture checkout.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$fixture = Join-Path $repo ('test_logs/revision_fixtures/' + [guid]::NewGuid().ToString())
$null = New-Item -ItemType Directory -Path $fixture -Force
$tokens = $null; $errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'test_repo.ps1'), [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
foreach ($fn in $ast.EndBlock.Statements | Where-Object { $_ -is [Management.Automation.Language.FunctionDefinitionAst] }) {
    Invoke-Expression $fn.Extent.Text
}
$RepoRoot = $repo
$SandboxRoot = $fixture
$DumpRoot = $fixture
$PowerShellExe = (Get-Command powershell.exe).Source
$script:Results = New-Object 'System.Collections.Generic.List[object]'
$script:HarnessPythonCommand = $null
$python = (Get-HarnessPythonCommand).file_path
$listenerCode = 'import socket,time; s=socket.socket(); s.bind((''127.0.0.1'',5000)); s.listen(); time.sleep(90)'
$service = $null; $other = $null
try {
    if (@(Get-TcpListeningProcessIds -Port 5000).Count) { throw 'Fixture requires free port 5000.' }
    $service = Start-LoggedBackgroundProcess -Id fixture-owned -WorkingDirectory $repo -FilePath $python -Arguments @('-c', $listenerCode) -WaitForPort 5000
    if (-not (Wait-TcpPort -Port 5000 -TimeoutSec 15 -OwnedProcess $service.process)) {
        "root=$($service.process.Id), exited=$($service.process.HasExited), listeners=$(@(Get-TcpListeningProcessIds -Port 5000)), descendants=$(@(Get-ProcessDescendantIds -RootProcessId $service.process.Id))"
        throw 'Owned listener failed readiness.'
    }
    if ((Stop-BackgroundProcess -Service $service).status -ne 'passed') { throw 'Owned listener cleanup failed.' }
    $service = $null
    $failed = Start-LoggedBackgroundProcess -Id fixture-failed -WorkingDirectory $repo -FilePath $python -Arguments @('-c', 'raise SystemExit(7)') -WaitForPort 5000
    if (Wait-TcpPort -Port 5000 -TimeoutSec 10 -OwnedProcess $failed.process) { throw 'Failed startup reported ready.' }
    $null = Stop-BackgroundProcess -Service $failed
    $other = Start-LoggedBackgroundProcess -Id fixture-unrelated -WorkingDirectory $repo -FilePath $python -Arguments @('-c', $listenerCode) -WaitForPort 5000
    if (-not (Wait-TcpPort -Port 5000 -TimeoutSec 15 -OwnedProcess $other.process)) { throw 'Unrelated fixture not ready.' }
    $rejected = $false
    try { Start-LoggedBackgroundProcess -Id fixture-rejected -WorkingDirectory $repo -FilePath $python -Arguments @('-c', 'raise SystemExit(0)') -WaitForPort 5000 | Out-Null }
    catch { $rejected = $_.Exception.Message -match 'occupied' }
    if (-not $rejected -or $other.process.HasExited) { throw 'Existing listener was reused or stopped.' }
    $failed.wait_for_port = 5000
    if ((Stop-BackgroundProcess -Service $failed).status -ne 'failed') { throw 'Unrelated remaining listener did not fail cleanup.' }
    if (-not (Wait-TcpPort -Port 5000 -TimeoutSec 2 -OwnedProcess $other.process)) { throw 'Unrelated listener did not survive cleanup.' }
    $null = Stop-BackgroundProcess -Service $other
    $other = $null
    'S1 passed: occupied listener survives; owned listener stops; early failure is not ready.'

    $checkout = Join-Path $fixture 'checkout'
    $null = New-Item -ItemType Directory -Path (Join-Path $checkout 'repo_maintenance'), (Join-Path $checkout 'Distributed_DL/5_scaling_strategies'), (Join-Path $checkout 'outputs') -Force
    Copy-Item -LiteralPath (Join-Path $repo '.gitignore') -Destination $checkout
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'clean_ignored.ps1') -Destination (Join-Path $checkout 'repo_maintenance')
    $sentinel = Join-Path $checkout 'Distributed_DL/5_scaling_strategies/sentinel.txt'
    $generated = Join-Path $checkout 'outputs/generated.txt'
    Set-Content -LiteralPath $sentinel -Value 'live fixture'
    Set-Content -LiteralPath $generated -Value 'generated fixture'
    Set-Content -LiteralPath (Join-Path $checkout 'Distributed_DL/5_scaling_strategies/README.md') -Value 'tracked fixture readme'
    & git -C $checkout init --quiet
    & git -C $checkout add .gitignore repo_maintenance/clean_ignored.ps1 Distributed_DL/5_scaling_strategies/README.md
    $preview = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $checkout 'repo_maintenance/clean_ignored.ps1') -WhatIf 2>&1 | Out-String
    if ($preview -notmatch 'Skipping preserved path: Distributed_DL/5_scaling_strategies' -or $preview -notmatch 'outputs') { throw 'Cleanup preview failed.' }
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $checkout 'repo_maintenance/clean_ignored.ps1')
    if (-not (Test-Path $sentinel) -or (Test-Path $generated)) { throw 'Cleanup fixture failed.' }
    'S2 passed: live sentinel preserved, ordinary generated artifact removed.'

    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'test_repo.ps1') -Destination (Join-Path $checkout 'repo_maintenance')
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'scripts') -Destination (Join-Path $checkout 'repo_maintenance') -Recurse
    foreach ($part in 'MLOps','Ray','Distributed_DL') {
        $null = New-Item -ItemType Directory -Path (Join-Path $checkout $part) -Force
        Copy-Item -LiteralPath (Join-Path $repo "$part/environment.yml") -Destination (Join-Path $checkout $part)
    }
    $ErrorActionPreference = 'Continue'
    $missingOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $checkout 'repo_maintenance/test_repo.ps1') -SessionName missing_fixture -SetupEnvs -BuildDocker -SkipCleanIgnored 2>&1 | Out-String
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    foreach ($month in '2020-01','2020-04','2020-08') { if ($missingOutput -notmatch "green_tripdata_$month.parquet") { throw "Missing preflight path $month" } }
    if ($exitCode -eq 0 -or $missingOutput -match '\[env\]|\[docker\]') { throw 'Prerequisite preflight ran expensive work.' }
    'S3 passed: all missing data listed before setup/services.'
    "Fixture logs: $fixture"
}
finally {
    if ($service) { $null = Stop-BackgroundProcess -Service $service }
    if ($other) { $null = Stop-BackgroundProcess -Service $other }
}
