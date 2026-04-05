<#
.SYNOPSIS
Runs the maintainer smoke test for this repository and writes session logs under test_logs.

.DESCRIPTION
By default, reuses existing course environments and Docker images, then starts the local
MLflow server and Ray lesson cluster, runs the curated scripts and notebooks, compile-checks
helper modules, uses the harness helper runners under repo_maintenance\scripts, writes a
session summary plus per-task stdout/stderr dumps, and optionally runs
repo_maintenance\clean_ignored.ps1 at the end. Use -SetupEnvs and -BuildDocker when you want
the harness to refresh environments or rebuild images first. The harness applies a few
test-only adjustments, such as suppressing MLflow URL printing during harness runs and
rewriting selected notebook cells for bounded local smoke execution.

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File .\repo_maintenance\test_repo.ps1
Run the smoke test against existing environments and images, and write logs under test_logs\<timestamp>\.

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File .\repo_maintenance\test_repo.ps1 -SessionName maintainer_check -SetupEnvs -BuildDocker
Run the smoke test with a stable session folder name after refreshing environments and rebuilding Docker images.

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File .\repo_maintenance\test_repo.ps1 -SkipCleanIgnored
Reuse existing environments and images, and keep ignored artifacts in place after the run for inspection.

.NOTES
Review test_logs\<session>\summary.md first, then drill into test_logs\<session>\dump\ as needed.
See repo_maintenance\README.md for the maintenance tooling overview.
You can also run: Get-Help .\repo_maintenance\test_repo.ps1 -Full
#>
[CmdletBinding()]
param(
    [string]$SessionName = (Get-Date -Format 'yyyyMMdd_HHmmss'),
    [switch]$AllowNotebookShell,
    [switch]$SkipEnvSetup = $true,
    [switch]$SkipDockerBuild = $true,
    [switch]$SetupEnvs,
    [switch]$BuildDocker,
    [switch]$SkipCleanIgnored
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (& git -C $ScriptDir rev-parse --show-toplevel).Trim()
if (-not $RepoRoot) {
    throw 'Could not determine git repository root.'
}

$LogRoot = Join-Path $RepoRoot "test_logs\$SessionName"
$DumpRoot = Join-Path $LogRoot 'dump'
$SandboxRoot = Join-Path $LogRoot 'sandboxes'
$SummaryJsonPath = Join-Path $LogRoot 'summary.json'
$SummaryMdPath = Join-Path $LogRoot 'summary.md'
$CleanIgnoredScript = Join-Path $ScriptDir 'clean_ignored.ps1'
$NotebookRunner = Join-Path $ScriptDir 'scripts\run_notebook_code.py'
$LoggedProcessRunner = Join-Path $ScriptDir 'scripts\run_logged_subprocess.py'
$PowerShellExe = (Get-Command powershell.exe -ErrorAction Stop).Source
$script:DistributedDlImageName = 'ou22971-distributed-dl-devcontainer'
$script:DistributedDlContainerName = (('ou22971-td-harness-' + $SessionName) -replace '[^A-Za-z0-9_.-]', '-').ToLowerInvariant()
$script:Results = New-Object System.Collections.Generic.List[object]
$script:CoveredPythonPaths = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
$script:HarnessPythonCommand = $null
$script:MlflowHarnessEnv = @{
    MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT = 'true'
}

$null = New-Item -ItemType Directory -Force -Path $LogRoot, $DumpRoot, $SandboxRoot
$env:MPLBACKEND = 'Agg'
$env:PYTHONUNBUFFERED = '1'

if (-not (Test-Path -LiteralPath $NotebookRunner)) {
    throw "Notebook runner not found: $NotebookRunner"
}

if (-not (Test-Path -LiteralPath $LoggedProcessRunner)) {
    throw "Logged process runner not found: $LoggedProcessRunner"
}

if ($SetupEnvs) {
    $SkipEnvSetup = $false
}

if ($BuildDocker) {
    $SkipDockerBuild = $false
}

function Get-CondaExecutable {
    if ($env:CONDA_EXE -and (Test-Path -LiteralPath $env:CONDA_EXE)) {
        return (Resolve-Path -LiteralPath $env:CONDA_EXE).Path
    }

    $command = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    throw 'Could not find conda. Make sure conda is on PATH or CONDA_EXE is set.'
}

function Get-CommandExecutable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    return (Get-Command $Name -ErrorAction Stop).Source
}

function Test-DockerContainerExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ContainerName
    )

    $containerNames = & $DockerExe @('ps', '-a', '--format', '{{.Names}}')
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not query Docker containers.'
    }

    return (@($containerNames | Where-Object { $_ -eq $ContainerName }).Count -gt 0)
}

function Get-HarnessPythonCommand {
    if ($null -ne $script:HarnessPythonCommand) {
        return $script:HarnessPythonCommand
    }

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        $script:HarnessPythonCommand = [pscustomobject]@{
            file_path = $python.Source
            arguments = @()
        }
        return $script:HarnessPythonCommand
    }

    $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        $script:HarnessPythonCommand = [pscustomobject]@{
            file_path = $pyLauncher.Source
            arguments = @('-3')
        }
        return $script:HarnessPythonCommand
    }

    throw 'Could not find python.exe or py.exe for the harness helper.'
}

function Get-RelativeRepoPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not [System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }

    $repoFull = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\', '/')
    $targetFull = [System.IO.Path]::GetFullPath($Path)

    if ([System.IO.Path]::GetPathRoot($repoFull) -ne [System.IO.Path]::GetPathRoot($targetFull)) {
        return $targetFull
    }

    $repoUri = New-Object System.Uri(($repoFull + [System.IO.Path]::DirectorySeparatorChar))
    $targetUri = New-Object System.Uri($targetFull)
    $relativePath = $repoUri.MakeRelativeUri($targetUri).ToString()
    return [System.Uri]::UnescapeDataString($relativePath).Replace('/', [System.IO.Path]::DirectorySeparatorChar)
}

function Format-CommandLine {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$Arguments = @()
    )

    $parts = @($FilePath) + $Arguments
    $quoted = foreach ($part in $parts) {
        $text = [string]$part
        if ($text -match '[\s"]') {
            '"' + ($text -replace '"', '\"') + '"'
        }
        else {
            $text
        }
    }

    return ($quoted -join ' ')
}

function ConvertTo-PowerShellLiteral {
    param(
        [AllowNull()]
        [object]$Value
    )

    if ($null -eq $Value) {
        return '$null'
    }

    $text = [string]$Value
    return "'" + ($text -replace "'", "''") + "'"
}

function Get-WorkspaceLinuxPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $relativePath = Get-RelativeRepoPath -Path $Path
    $normalized = $relativePath -replace '\\', '/'
    if ([string]::IsNullOrWhiteSpace($normalized) -or $normalized -eq '.') {
        return '/workspace'
    }

    return '/workspace/' + $normalized.TrimStart('/')
}

function Get-DistributedDlRelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $repoRelativePath = Get-RelativeRepoPath -Path $Path
    $distributedDlRelativePath = $repoRelativePath -replace '^[Dd]istributed_DL[\\/]', ''
    return ($distributedDlRelativePath -replace '\\', '/')
}

function New-InvocationWrapper {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Id,
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)]
        [string]$StdoutPath,
        [Parameter(Mandatory = $true)]
        [string]$StderrPath,
        [hashtable]$EnvironmentOverrides = @{},
        [switch]$CaptureExitCode
    )

    $wrapperPath = Join-Path $SandboxRoot "$Id.invoke.ps1"
    $exitCodePath = if ($CaptureExitCode) { Join-Path $SandboxRoot "$Id.exitcode.txt" } else { $null }

    if ($CaptureExitCode -and (Test-Path -LiteralPath $exitCodePath)) {
        Remove-Item -LiteralPath $exitCodePath -Force
    }

    $lines = New-Object System.Collections.Generic.List[string]
    $null = $lines.Add('$ErrorActionPreference = ''Stop''')
    $null = $lines.Add('$filePath = ' + (ConvertTo-PowerShellLiteral $FilePath))
    $null = $lines.Add('$arguments = @(')
    foreach ($argument in $Arguments) {
        $null = $lines.Add('    ' + (ConvertTo-PowerShellLiteral $argument))
    }
    $null = $lines.Add(')')
    $null = $lines.Add('$stdoutPath = ' + (ConvertTo-PowerShellLiteral $StdoutPath))
    $null = $lines.Add('$stderrPath = ' + (ConvertTo-PowerShellLiteral $StderrPath))
    if ($CaptureExitCode) {
        $null = $lines.Add('$exitCodePath = ' + (ConvertTo-PowerShellLiteral $exitCodePath))
    }
    $null = $lines.Add('$environmentOverrides = @{}')
    foreach ($entry in $EnvironmentOverrides.GetEnumerator()) {
        $null = $lines.Add('$environmentOverrides[' + (ConvertTo-PowerShellLiteral $entry.Key) + '] = ' + (ConvertTo-PowerShellLiteral $entry.Value))
    }
    $null = $lines.Add('Set-Location ' + (ConvertTo-PowerShellLiteral $WorkingDirectory))
    $null = $lines.Add('foreach ($entry in $environmentOverrides.GetEnumerator()) {')
    $null = $lines.Add('    Set-Item -Path ("Env:" + [string]$entry.Key) -Value ([string]$entry.Value)')
    $null = $lines.Add('}')
    $null = $lines.Add('$code = 0')
    $null = $lines.Add('try {')
    $null = $lines.Add('    & $filePath @arguments 1> $stdoutPath 2> $stderrPath')
    $null = $lines.Add('    if ($null -ne $LASTEXITCODE) {')
    $null = $lines.Add('        $code = [int]$LASTEXITCODE')
    $null = $lines.Add('    }')
    $null = $lines.Add('    elseif (-not $?) {')
    $null = $lines.Add('        $code = 1')
    $null = $lines.Add('    }')
    $null = $lines.Add('}')
    $null = $lines.Add('catch {')
    $null = $lines.Add('    ($_ | Out-String) | Out-File -FilePath $stderrPath -Encoding utf8 -Append')
    $null = $lines.Add('    $code = 1')
    $null = $lines.Add('}')
    if ($CaptureExitCode) {
        $null = $lines.Add('Set-Content -LiteralPath $exitCodePath -Value ([string]$code) -Encoding ascii -NoNewline')
    }
    $null = $lines.Add('exit $code')

    Set-Content -LiteralPath $wrapperPath -Value $lines -Encoding ascii

    return [pscustomobject]@{
        wrapper_path = $wrapperPath
        exit_code_path = $exitCodePath
    }
}

function Add-Result {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Result
    )

    $script:Results.Add($Result)
    return $Result
}

function Invoke-LoggedProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Id,
        [Parameter(Mandatory = $true)]
        [string]$Category,
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [hashtable]$EnvironmentOverrides = @{},
        [int]$TimeoutSec = 1800,
        [string]$Target = '',
        [string]$Note = ''
    )

    $stdoutPath = Join-Path $DumpRoot "$Id.stdout.txt"
    $stderrPath = Join-Path $DumpRoot "$Id.stderr.txt"
    $commandLine = Format-CommandLine -FilePath $FilePath -Arguments $Arguments
    $runnerSpecPath = Join-Path $SandboxRoot "$Id.invoke.json"
    $runnerResultPath = Join-Path $SandboxRoot "$Id.result.json"
    $runnerSpec = [ordered]@{
        command = @($FilePath) + $Arguments
        cwd = $WorkingDirectory
        stdout_path = $stdoutPath
        stderr_path = $stderrPath
        timeout_sec = $TimeoutSec
        env_overrides = $EnvironmentOverrides
        result_path = $runnerResultPath
    }
    $runnerSpec | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $runnerSpecPath -Encoding utf8
    $startTime = Get-Date

    Write-Host "[$Category] $Id"
    Write-Host "  cwd: $(Get-RelativeRepoPath $WorkingDirectory)"
    Write-Host "  cmd: $commandLine"

    $pythonCommand = Get-HarnessPythonCommand
    & $pythonCommand.file_path @($pythonCommand.arguments + @($LoggedProcessRunner, $runnerSpecPath))
    $runnerInvocationExitCode = $LASTEXITCODE

    $endTime = Get-Date
    $durationSec = [math]::Round(($endTime - $startTime).TotalSeconds, 2)
    $runnerResult = $null
    if (Test-Path -LiteralPath $runnerResultPath) {
        $runnerResult = Get-Content -LiteralPath $runnerResultPath -Raw | ConvertFrom-Json
    }

    $timedOut = if ($null -ne $runnerResult) { [bool]$runnerResult.timed_out } else { $false }
    $exitCode = $null
    if (($null -ne $runnerResult) -and (-not $timedOut) -and ($null -ne $runnerResult.exit_code)) {
        $exitCode = [int]$runnerResult.exit_code
    }
    elseif (($null -eq $runnerResult) -and ($runnerInvocationExitCode -ne 0)) {
        $exitCode = $runnerInvocationExitCode
    }

    $noteParts = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($Note)) {
        $null = $noteParts.Add($Note)
    }
    if ($null -eq $runnerResult) {
        $null = $noteParts.Add('Harness runner did not produce a result file.')
    }
    elseif (-not [string]::IsNullOrWhiteSpace([string]$runnerResult.runner_error)) {
        $null = $noteParts.Add("Harness runner error: $($runnerResult.runner_error)")
    }

    $combinedNote = $noteParts -join ' '
    $status = if ($timedOut) { 'timed_out' } elseif ($exitCode -eq 0) { 'passed' } else { 'failed' }

    return Add-Result ([pscustomobject]@{
            id = $Id
            category = $Category
            target = $Target
            status = $status
            exit_code = $exitCode
            timed_out = $timedOut
            duration_sec = $durationSec
            started_at = $startTime.ToString('o')
            finished_at = $endTime.ToString('o')
            working_dir = Get-RelativeRepoPath $WorkingDirectory
            command = $commandLine
            stdout_path = Get-RelativeRepoPath $stdoutPath
            stderr_path = Get-RelativeRepoPath $stderrPath
            note = $combinedNote
        })
}

function Start-LoggedBackgroundProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Id,
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [hashtable]$EnvironmentOverrides = @{},
        [int]$WaitForPort = 0
    )

    $stdoutPath = Join-Path $DumpRoot "$Id.stdout.txt"
    $stderrPath = Join-Path $DumpRoot "$Id.stderr.txt"
    $wrapper = New-InvocationWrapper `
        -Id $Id `
        -WorkingDirectory $WorkingDirectory `
        -FilePath $FilePath `
        -Arguments $Arguments `
        -StdoutPath $stdoutPath `
        -StderrPath $stderrPath `
        -EnvironmentOverrides $EnvironmentOverrides

    return [pscustomobject]@{
        id = $Id
        process = (Start-Process `
                -FilePath $PowerShellExe `
                -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $wrapper.wrapper_path) `
                -WorkingDirectory $WorkingDirectory `
                -NoNewWindow `
                -PassThru)
        stdout_path = $stdoutPath
        stderr_path = $stderrPath
        command = (Format-CommandLine -FilePath $FilePath -Arguments $Arguments)
        working_dir = $WorkingDirectory
        started_at = (Get-Date)
        wrapper_path = $wrapper.wrapper_path
        wait_for_port = if ($WaitForPort -gt 0) { $WaitForPort } else { $null }
    }
}

function Wait-TcpPort {
    param(
        [string]$HostName = '127.0.0.1',
        [Parameter(Mandatory = $true)]
        [int]$Port,
        [int]$TimeoutSec = 90
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $client = $null
        try {
            $client = New-Object System.Net.Sockets.TcpClient
            $asyncResult = $client.BeginConnect($HostName, $Port, $null, $null)
            if ($asyncResult.AsyncWaitHandle.WaitOne(1000, $false)) {
                $client.EndConnect($asyncResult) | Out-Null
                $client.Close()
                return $true
            }
        }
        catch {
        }
        finally {
            if ($null -ne $client) {
                $client.Close()
            }
        }

        Start-Sleep -Seconds 1
    }

    return $false
}

function Wait-TcpPortClosed {
    param(
        [string]$HostName = '127.0.0.1',
        [Parameter(Mandatory = $true)]
        [int]$Port,
        [int]$TimeoutSec = 90
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $client = $null
        try {
            $client = New-Object System.Net.Sockets.TcpClient
            $asyncResult = $client.BeginConnect($HostName, $Port, $null, $null)
            if (-not $asyncResult.AsyncWaitHandle.WaitOne(1000, $false)) {
                return $true
            }

            try {
                $client.EndConnect($asyncResult) | Out-Null
            }
            catch {
                return $true
            }
        }
        catch {
            return $true
        }
        finally {
            if ($null -ne $client) {
                $client.Close()
            }
        }

        Start-Sleep -Seconds 1
    }

    return $false
}

function Get-ProcessDescendantIds {
    param(
        [Parameter(Mandatory = $true)]
        [int]$RootProcessId
    )

    $childLookup = @{}
    foreach ($process in (Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)) {
        if (-not $childLookup.ContainsKey($process.ParentProcessId)) {
            $childLookup[$process.ParentProcessId] = New-Object System.Collections.Generic.List[int]
        }

        $childLookup[$process.ParentProcessId].Add([int]$process.ProcessId)
    }

    $descendants = New-Object System.Collections.Generic.List[int]
    $stack = New-Object System.Collections.Generic.Stack[int]
    $stack.Push($RootProcessId)

    while ($stack.Count -gt 0) {
        $currentId = $stack.Pop()
        if (-not $childLookup.ContainsKey($currentId)) {
            continue
        }

        foreach ($childId in $childLookup[$currentId]) {
            $descendants.Add($childId)
            $stack.Push($childId)
        }
    }

    return $descendants.ToArray()
}

function Stop-ProcessTree {
    param(
        [Parameter(Mandatory = $true)]
        [int]$RootProcessId
    )

    $processIds = New-Object System.Collections.Generic.List[int]
    foreach ($childId in (Get-ProcessDescendantIds -RootProcessId $RootProcessId)) {
        $processIds.Add([int]$childId)
    }
    $processIds.Add($RootProcessId)

    $orderedIds = $processIds |
        Sort-Object -Descending |
        Select-Object -Unique

    foreach ($processId in $orderedIds) {
        try {
            Stop-Process -Id $processId -Force -ErrorAction Stop
        }
        catch {
            $existing = Get-Process -Id $processId -ErrorAction SilentlyContinue
            if ($null -ne $existing) {
                throw
            }
        }
    }
}

function Get-TcpListeningProcessIds {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port
    )

    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($null -eq $connections) {
        return @()
    }

    return @($connections | Select-Object -ExpandProperty OwningProcess -Unique)
}

function Stop-BackgroundProcess {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Service,
        [string]$Note = ''
    )

    $endTime = Get-Date
    $status = 'passed'
    $noteParts = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($Note)) {
        $null = $noteParts.Add($Note)
    }
    try {
        if ($null -ne $Service.process) {
            Stop-ProcessTree -RootProcessId $Service.process.Id
        }

        if ($Service.PSObject.Properties.Name -contains 'wait_for_port') {
            $waitForPort = $Service.wait_for_port
            if ($null -ne $waitForPort) {
                $portClosed = Wait-TcpPortClosed -Port ([int]$waitForPort) -TimeoutSec 30
                if (-not $portClosed) {
                    foreach ($listenerPid in (Get-TcpListeningProcessIds -Port ([int]$waitForPort))) {
                        try {
                            Stop-ProcessTree -RootProcessId ([int]$listenerPid)
                        }
                        catch {
                        }
                    }

                    $portClosed = Wait-TcpPortClosed -Port ([int]$waitForPort) -TimeoutSec 15
                    if (-not $portClosed) {
                        throw "Port $waitForPort is still listening after service shutdown."
                    }
                }
            }
        }
    }
    catch {
        $status = 'failed'
        $null = $noteParts.Add($_.Exception.Message)
    }

    return Add-Result ([pscustomobject]@{
            id = "$($Service.id)_stop"
            category = 'cleanup'
            target = ''
            status = $status
            exit_code = $null
            timed_out = $false
            duration_sec = [math]::Round(($endTime - $Service.started_at).TotalSeconds, 2)
            started_at = $Service.started_at.ToString('o')
            finished_at = $endTime.ToString('o')
            working_dir = Get-RelativeRepoPath $Service.working_dir
            command = $Service.command
            stdout_path = Get-RelativeRepoPath $Service.stdout_path
            stderr_path = Get-RelativeRepoPath $Service.stderr_path
            note = ($noteParts -join ' ')
        })
}

function Get-YamlEnvName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$YamlPath
    )

    $nameLine = Get-Content -LiteralPath $YamlPath | Where-Object { $_ -match '^name:\s*' } | Select-Object -First 1
    if (-not $nameLine) {
        throw "Could not find an environment name in $YamlPath"
    }

    return ($nameLine -replace '^name:\s*', '').Trim()
}

function Get-EnvNameForRelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $normalized = $RelativePath -replace '\\', '/'
    if ($normalized.StartsWith('MLOps/')) {
        return $script:EnvSpecs['MLOps'].name
    }
    if ($normalized.StartsWith('Ray/')) {
        return $script:EnvSpecs['Ray'].name
    }
    if ($normalized.StartsWith('Distributed_DL/')) {
        return $script:EnvSpecs['Distributed_DL'].name
    }

    throw "Could not infer an environment for path: $RelativePath"
}

function Get-HarnessEnvOverridesForRelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $normalized = $RelativePath -replace '\\', '/'
    if ($normalized.StartsWith('MLOps/')) {
        return $script:MlflowHarnessEnv
    }

    return @{}
}

function New-CondaTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Id,
        [Parameter(Mandatory = $true)]
        [string]$Category,
        [Parameter(Mandatory = $true)]
        [string]$EnvName,
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,
        [Parameter(Mandatory = $true)]
        [string[]]$CommandArgs,
        [hashtable]$EnvironmentOverrides = @{},
        [int]$TimeoutSec = 1800,
        [string]$Target = '',
        [string]$Note = ''
    )

    return [pscustomobject]@{
        id = $Id
        category = $Category
        working_dir = $WorkingDirectory
        file_path = $script:CondaExe
        arguments = @('run', '--no-capture-output', '-n', $EnvName) + $CommandArgs
        env_overrides = $EnvironmentOverrides
        timeout_sec = $TimeoutSec
        target = $Target
        note = $Note
    }
}

function New-ProcessTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Id,
        [Parameter(Mandatory = $true)]
        [string]$Category,
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [hashtable]$EnvironmentOverrides = @{},
        [int]$TimeoutSec = 1800,
        [string]$Target = '',
        [string]$Note = ''
    )

    return [pscustomobject]@{
        id = $Id
        category = $Category
        working_dir = $WorkingDirectory
        file_path = $FilePath
        arguments = $Arguments
        env_overrides = $EnvironmentOverrides
        timeout_sec = $TimeoutSec
        target = $Target
        note = $Note
    }
}

function New-DockerExecTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Id,
        [Parameter(Mandatory = $true)]
        [string]$Category,
        [Parameter(Mandatory = $true)]
        [string]$ContainerName,
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,
        [Parameter(Mandatory = $true)]
        [string[]]$CommandArgs,
        [hashtable]$EnvironmentOverrides = @{},
        [int]$TimeoutSec = 1800,
        [string]$Target = '',
        [string]$Note = ''
    )

    $arguments = @('exec', '-w', (Get-WorkspaceLinuxPath -Path $WorkingDirectory))
    foreach ($entry in $EnvironmentOverrides.GetEnumerator()) {
        $arguments += @('-e', ([string]$entry.Key + '=' + [string]$entry.Value))
    }
    $arguments += $ContainerName
    $arguments += @('conda', 'run', '--no-capture-output', '-n', $script:EnvSpecs['Distributed_DL'].name)
    $arguments += $CommandArgs

    return New-ProcessTask `
        -Id $Id `
        -Category $Category `
        -WorkingDirectory $WorkingDirectory `
        -FilePath $DockerExe `
        -Arguments $arguments `
        -TimeoutSec $TimeoutSec `
        -Target $Target `
        -Note $Note
}

function Invoke-Task {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Task
    )

    return Invoke-LoggedProcess `
        -Id $Task.id `
        -Category $Task.category `
        -WorkingDirectory $Task.working_dir `
        -FilePath $Task.file_path `
        -Arguments $Task.arguments `
        -EnvironmentOverrides $Task.env_overrides `
        -TimeoutSec $Task.timeout_sec `
        -Target $Task.target `
        -Note $Task.note
}

function Ensure-CondaEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Key
    )

    $spec = $script:EnvSpecs[$Key]
    $envList = & $script:CondaExe env list --json | ConvertFrom-Json
    $exists = $false
    foreach ($envPath in $envList.envs) {
        if ([System.IO.Path]::GetFileName($envPath) -eq $spec.name) {
            $exists = $true
            break
        }
    }

    if ($Key -eq 'Ray') {
        if ($exists) {
            $removeResult = Invoke-LoggedProcess `
                -Id "$Key-env-remove" `
                -Category 'env' `
                -WorkingDirectory $RepoRoot `
                -FilePath $script:CondaExe `
                -Arguments @('env', 'remove', '-n', $spec.name, '-y') `
                -TimeoutSec 7200 `
                -Target $spec.file `
                -Note "Recreate Conda environment $($spec.name) to avoid stale pip pyarrow artifacts."
            if ($removeResult.status -ne 'passed') {
                return $removeResult
            }
        }

        return Invoke-LoggedProcess `
            -Id "$Key-env-create" `
            -Category 'env' `
            -WorkingDirectory $RepoRoot `
            -FilePath $script:CondaExe `
            -Arguments @('env', 'create', '-f', $spec.file) `
            -TimeoutSec 7200 `
            -Target $spec.file `
            -Note "Create Conda environment $($spec.name) from scratch."
    }

    $taskId = if ($exists) { "$Key-env-update" } else { "$Key-env-create" }
    $args = if ($exists) {
        @('env', 'update', '--prune', '-f', $spec.file)
    }
    else {
        @('env', 'create', '-f', $spec.file)
    }

    return Invoke-LoggedProcess `
        -Id $taskId `
        -Category 'env' `
        -WorkingDirectory $RepoRoot `
        -FilePath $script:CondaExe `
        -Arguments $args `
        -TimeoutSec 7200 `
        -Target $spec.file `
        -Note "Conda environment $($spec.name)"
}

function Add-CoveredPythonPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $null = $script:CoveredPythonPaths.Add($RelativePath)
}

function New-CondaPythonScriptTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Id,
        [Parameter(Mandatory = $true)]
        [string]$RelativePath,
        [string[]]$ScriptArgs = @(),
        [int]$TimeoutSec = 1800,
        [string]$Note = ''
    )

    $fullPath = Join-Path $RepoRoot $RelativePath
    $workingDir = Split-Path -Parent $fullPath
    $envName = Get-EnvNameForRelativePath -RelativePath $RelativePath
    $environmentOverrides = Get-HarnessEnvOverridesForRelativePath -RelativePath $RelativePath
    Add-CoveredPythonPath -RelativePath $RelativePath

    return New-CondaTask `
        -Id $Id `
        -Category 'script' `
        -EnvName $envName `
        -WorkingDirectory $workingDir `
        -CommandArgs (@('python', (Split-Path -Leaf $fullPath)) + $ScriptArgs) `
        -EnvironmentOverrides $environmentOverrides `
        -TimeoutSec $TimeoutSec `
        -Target $RelativePath `
        -Note $Note
}

function New-DistributedDlPythonScriptTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Id,
        [Parameter(Mandatory = $true)]
        [string]$RelativePath,
        [string[]]$ScriptArgs = @(),
        [int]$TimeoutSec = 1800,
        [string]$Note = ''
    )

    $fullPath = Join-Path $RepoRoot $RelativePath
    $workingDir = Join-Path $RepoRoot 'Distributed_DL'
    $scriptPath = Get-DistributedDlRelativePath -Path $fullPath
    Add-CoveredPythonPath -RelativePath $RelativePath

    return New-DockerExecTask `
        -Id $Id `
        -Category 'script' `
        -ContainerName $script:DistributedDlContainerName `
        -WorkingDirectory $workingDir `
        -CommandArgs (@('python', $scriptPath) + $ScriptArgs) `
        -TimeoutSec $TimeoutSec `
        -Target $RelativePath `
        -Note $Note
}

function New-DistributedDlTorchrunTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Id,
        [Parameter(Mandatory = $true)]
        [string]$RelativePath,
        [Parameter(Mandatory = $true)]
        [int]$ProcessCount,
        [string[]]$ScriptArgs = @(),
        [int]$TimeoutSec = 1800,
        [string]$Note = ''
    )

    $fullPath = Join-Path $RepoRoot $RelativePath
    $workingDir = Join-Path $RepoRoot 'Distributed_DL'
    $scriptPath = Get-DistributedDlRelativePath -Path $fullPath
    Add-CoveredPythonPath -RelativePath $RelativePath

    return New-DockerExecTask `
        -Id $Id `
        -Category 'script' `
        -ContainerName $script:DistributedDlContainerName `
        -WorkingDirectory $workingDir `
        -CommandArgs (@('torchrun', '--standalone', '--nproc_per_node', [string]$ProcessCount, $scriptPath) + $ScriptArgs) `
        -TimeoutSec $TimeoutSec `
        -Target $RelativePath `
        -Note $Note
}

function Get-NotebookReplacementPairs {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath,
        [Parameter(Mandatory = $true)]
        [string]$SandboxRootPath,
        [switch]$UseWorkspacePaths
    )

    $normalized = $RelativePath -replace '\\', '/'
    $pairs = New-Object System.Collections.Generic.List[string]

    switch ($normalized) {
        'Distributed_DL/4_ddp_on_cloud_gpus/colab_launcher.ipynb' {
            if ($UseWorkspacePaths) {
                $sandboxPath = Get-WorkspaceLinuxPath -Path $SandboxRootPath
                $localUnit4 = '/workspace/Distributed_DL/4_ddp_on_cloud_gpus'
            }
            else {
                $sandboxPath = ($SandboxRootPath -replace '\\', '/')
                $localUnit4 = ((Join-Path $RepoRoot 'Distributed_DL\4_ddp_on_cloud_gpus') -replace '\\', '/')
            }
            $null = $pairs.Add('/content/Distributed_DL=>' + $sandboxPath)
            $null = $pairs.Add('from urllib.request import urlretrieve=>import shutil')
            $null = $pairs.Add(
                'urlretrieve(f"{BASE_RAW_URL}/profile_ddp_gpu.py", PART4_ROOT / "profile_ddp_gpu.py")=>shutil.copy2(r"' +
                "$localUnit4/profile_ddp_gpu.py" +
                '", PART4_ROOT / "profile_ddp_gpu.py")'
            )
            break
        }
        'Ray/0_core_primitives/1_objects.ipynb' {
            $null = $pairs.Add('time.sleep(30)=>time.sleep(1)')
            break
        }
        'Ray/2_system_design/2_1_distributed_HPO/0_distributed_hpo.ipynb' {
            $null = $pairs.Add('NUM_BOOST_ROUND = 60=>NUM_BOOST_ROUND = 15')
            $null = $pairs.Add('REPORT_FAILURE_PROB = 0.5=>REPORT_FAILURE_PROB = 0.0')
            $null = $pairs.Add('local_study.optimize(local_objective, n_trials=6)=>local_study.optimize(local_objective, n_trials=2)')
            $null = $pairs.Add('n_trials=10=>n_trials=3')
            $null = $pairs.Add('n_trials=6=>n_trials=2')
            $null = $pairs.Add('max_concurrent_trials=3=>max_concurrent_trials=2')
            break
        }
    }

    return $pairs.ToArray()
}

function New-NotebookTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $fullPath = Join-Path $RepoRoot $RelativePath
    $workingDir = Split-Path -Parent $fullPath
    $envName = Get-EnvNameForRelativePath -RelativePath $RelativePath
    $environmentOverrides = Get-HarnessEnvOverridesForRelativePath -RelativePath $RelativePath
    $normalized = $RelativePath -replace '\\', '/'
    $taskId = ('notebook-' + ($normalized -replace '[^A-Za-z0-9]+', '-')).Trim('-').ToLowerInvariant()
    $sandboxPath = Join-Path $SandboxRoot (($normalized -replace '/', '\') -replace '\.ipynb$', '')
    $null = New-Item -ItemType Directory -Force -Path $sandboxPath

    if ($normalized.StartsWith('Distributed_DL/')) {
        $commandArgs = @(
            'python',
            (Get-WorkspaceLinuxPath -Path $NotebookRunner),
            (Get-WorkspaceLinuxPath -Path $fullPath),
            '--cwd',
            (Get-WorkspaceLinuxPath -Path $workingDir)
        )
        if ($AllowNotebookShell) {
            $commandArgs += '--allow-shell'
        }

        foreach ($pair in (Get-NotebookReplacementPairs -RelativePath $RelativePath -SandboxRootPath $sandboxPath -UseWorkspacePaths)) {
            $commandArgs += @('--replace', $pair)
        }

        return New-DockerExecTask `
            -Id $taskId `
            -Category 'notebook' `
            -ContainerName $script:DistributedDlContainerName `
            -WorkingDirectory $workingDir `
            -CommandArgs $commandArgs `
            -TimeoutSec 5400 `
            -Target $RelativePath `
            -Note 'Execute notebook code cells sequentially inside the Distributed DL devcontainer.'
    }

    $commandArgs = @('python', $NotebookRunner, $fullPath, '--cwd', $workingDir)
    if ($AllowNotebookShell) {
        $commandArgs += '--allow-shell'
    }

    foreach ($pair in (Get-NotebookReplacementPairs -RelativePath $RelativePath -SandboxRootPath $sandboxPath)) {
        $commandArgs += @('--replace', $pair)
    }

    return New-CondaTask `
        -Id $taskId `
        -Category 'notebook' `
        -EnvName $envName `
        -WorkingDirectory $workingDir `
        -CommandArgs $commandArgs `
        -EnvironmentOverrides $environmentOverrides `
        -TimeoutSec 5400 `
        -Target $RelativePath `
        -Note 'Execute notebook code cells sequentially.'
}

function New-CompileTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $taskId = ('compile-' + (($RelativePath -replace '\\', '/') -replace '[^A-Za-z0-9]+', '-')).Trim('-').ToLowerInvariant()
    $normalized = $RelativePath -replace '\\', '/'

    if ($normalized.StartsWith('Distributed_DL/')) {
        $fullPath = Join-Path $RepoRoot $RelativePath
        $workingDir = Split-Path -Parent $fullPath
        return New-DockerExecTask `
            -Id $taskId `
            -Category 'compile' `
            -ContainerName $script:DistributedDlContainerName `
            -WorkingDirectory $workingDir `
            -CommandArgs @('python', '-m', 'py_compile', (Split-Path -Leaf $fullPath)) `
            -TimeoutSec 300 `
            -Target $RelativePath `
            -Note 'Compile smoke-check for a Python helper inside the Distributed DL devcontainer.'
    }

    $envName = Get-EnvNameForRelativePath -RelativePath $RelativePath

    return New-CondaTask `
        -Id $taskId `
        -Category 'compile' `
        -EnvName $envName `
        -WorkingDirectory $RepoRoot `
        -CommandArgs @('python', '-m', 'py_compile', $RelativePath) `
        -TimeoutSec 300 `
        -Target $RelativePath `
        -Note 'Compile smoke-check for a Python helper not run directly.'
}

function Write-SummaryFiles {
    $counts = @{
        passed = @($script:Results | Where-Object { $_.status -eq 'passed' }).Count
        failed = @($script:Results | Where-Object { $_.status -eq 'failed' }).Count
        timed_out = @($script:Results | Where-Object { $_.status -eq 'timed_out' }).Count
    }
    $overallStatus = if (($counts.failed + $counts.timed_out) -eq 0) { 'passed' } else { 'failed' }

    $payload = [pscustomobject]@{
        session = $SessionName
        repo_root = $RepoRoot
        log_root = $LogRoot
        overall_status = $overallStatus
        created_at = (Get-Date).ToString('o')
        counts = $counts
        results = $script:Results
    }

    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $SummaryJsonPath -Encoding UTF8

    $lines = New-Object System.Collections.Generic.List[string]
    $null = $lines.Add('# Repo test summary')
    $null = $lines.Add('')
    $null = $lines.Add('- session: `' + $SessionName + '`')
    $null = $lines.Add('- overall_status: `' + $overallStatus + '`')
    $null = $lines.Add("- passed: $($counts.passed)")
    $null = $lines.Add("- failed: $($counts.failed)")
    $null = $lines.Add("- timed_out: $($counts.timed_out)")
    $null = $lines.Add('')

    $failedResults = @($script:Results | Where-Object { $_.status -ne 'passed' })
    if ($failedResults.Count -gt 0) {
        $null = $lines.Add('## Failures')
        $null = $lines.Add('')
        foreach ($result in $failedResults) {
            $null = $lines.Add(
                '- `[' + $result.category + '] ' + $result.id + '` status=`' + $result.status +
                '` target=`' + $result.target + '` stdout=`' + $result.stdout_path +
                '` stderr=`' + $result.stderr_path + '`'
            )
        }
        $null = $lines.Add('')
    }

    $null = $lines.Add('## Tasks')
    $null = $lines.Add('')
    foreach ($result in $script:Results) {
        $null = $lines.Add(
            '- `[' + $result.category + '] ' + $result.id + '` status=`' + $result.status +
            '` duration=`' + $result.duration_sec + 's` target=`' + $result.target + '`'
        )
    }

    $lines | Set-Content -LiteralPath $SummaryMdPath -Encoding UTF8
}

$script:CondaExe = Get-CondaExecutable
$DockerExe = Get-CommandExecutable -Name 'docker'
$script:EnvSpecs = @{
    MLOps = [pscustomobject]@{
        file = 'MLOps\environment.yml'
        name = Get-YamlEnvName -YamlPath (Join-Path $RepoRoot 'MLOps\environment.yml')
    }
    Ray = [pscustomobject]@{
        file = 'Ray\environment.yml'
        name = Get-YamlEnvName -YamlPath (Join-Path $RepoRoot 'Ray\environment.yml')
    }
    Distributed_DL = [pscustomobject]@{
        file = 'Distributed_DL\environment.yml'
        name = Get-YamlEnvName -YamlPath (Join-Path $RepoRoot 'Distributed_DL\environment.yml')
    }
}

$mlflowService = $null
$rayClusterStarted = $false
$distributedDlContainerStarted = $false

try {
    if (-not $SkipEnvSetup) {
        Ensure-CondaEnvironment -Key 'MLOps' | Out-Null
        Ensure-CondaEnvironment -Key 'Ray' | Out-Null
        Ensure-CondaEnvironment -Key 'Distributed_DL' | Out-Null
    }

    if (-not $SkipDockerBuild) {
        Invoke-Task (New-ProcessTask `
                -Id 'ray-docker-build' `
                -Category 'docker' `
                -WorkingDirectory (Join-Path $RepoRoot 'Ray\1_cluster_setup') `
                -FilePath $DockerExe `
                -Arguments @('compose', 'build') `
                -TimeoutSec 7200 `
                -Target 'Ray/1_cluster_setup/docker-compose.yml' `
                -Note 'Build the Ray lesson cluster images.') | Out-Null

        Invoke-Task (New-ProcessTask `
                -Id 'distributed-dl-devcontainer-build' `
                -Category 'docker' `
                -WorkingDirectory (Join-Path $RepoRoot 'Distributed_DL') `
                -FilePath $DockerExe `
                -Arguments @('build', '-f', '.devcontainer/Dockerfile', '-t', $script:DistributedDlImageName, '.') `
                -TimeoutSec 7200 `
                -Target 'Distributed_DL/.devcontainer/Dockerfile' `
                -Note 'Build the Distributed DL devcontainer image.') | Out-Null
    }

    if (Test-DockerContainerExists -ContainerName $script:DistributedDlContainerName) {
        & $DockerExe @('rm', '-f', $script:DistributedDlContainerName) 1> $null 2> $null
    }

    $distributedDlUpResult = Invoke-Task (New-ProcessTask `
            -Id 'distributed-dl-devcontainer-up' `
            -Category 'docker' `
            -WorkingDirectory (Join-Path $RepoRoot 'Distributed_DL') `
            -FilePath $DockerExe `
            -Arguments @(
                'run', '-d',
                '--name', $script:DistributedDlContainerName,
                '--mount', ('type=bind,source=' + $RepoRoot + ',target=/workspace'),
                '--workdir', '/workspace',
                $script:DistributedDlImageName,
                'bash', '-lc', 'sleep infinity'
            ) `
            -TimeoutSec 300 `
            -Target 'Distributed_DL/.devcontainer/devcontainer.json' `
            -Note 'Start the Distributed DL devcontainer used for all Distributed DL smoke tasks.')
    if ($distributedDlUpResult.status -eq 'passed') {
        $distributedDlContainerStarted = $true
    }

    $mlflowService = Start-LoggedBackgroundProcess `
        -Id 'mlflow-server' `
        -WorkingDirectory (Join-Path $RepoRoot 'MLOps') `
        -FilePath $script:CondaExe `
        -Arguments @(
            'run', '--no-capture-output', '-n', $script:EnvSpecs['MLOps'].name,
            'mlflow', 'server',
            '--workers', '1',
            '--port', '5000',
            '--backend-store-uri', 'sqlite:///mlflow_tracking/mlflow.db',
            '--default-artifact-root', 'mlflow_tracking/mlruns'
        ) `
        -EnvironmentOverrides $script:MlflowHarnessEnv `
        -WaitForPort 5000

    $mlflowReady = Wait-TcpPort -Port 5000 -TimeoutSec 120
    Add-Result ([pscustomobject]@{
            id = 'mlflow-server'
            category = 'service'
            target = 'http://127.0.0.1:5000'
            status = if ($mlflowReady) { 'passed' } else { 'failed' }
            exit_code = $null
            timed_out = $false
            duration_sec = [math]::Round(((Get-Date) - $mlflowService.started_at).TotalSeconds, 2)
            started_at = $mlflowService.started_at.ToString('o')
            finished_at = (Get-Date).ToString('o')
            working_dir = Get-RelativeRepoPath $mlflowService.working_dir
            command = $mlflowService.command
            stdout_path = Get-RelativeRepoPath $mlflowService.stdout_path
            stderr_path = Get-RelativeRepoPath $mlflowService.stderr_path
            note = 'Waited for the MLflow server to listen on port 5000.'
        }) | Out-Null
    if (-not $mlflowReady) {
        Stop-BackgroundProcess -Service $mlflowService -Note 'MLflow server did not become ready.' | Out-Null
        $mlflowService = $null
    }

    $scriptTasks = @(
        (New-CondaPythonScriptTask -Id 'mlops-unit1-generate-data' -RelativePath 'MLOps\1_conda_environments\generate_data.py' -ScriptArgs @('--outdir', 'data') -TimeoutSec 900),
        (New-CondaPythonScriptTask -Id 'mlops-unit1-ml-pipeline' -RelativePath 'MLOps\1_conda_environments\ml_pipeline.py' -ScriptArgs @('--data', 'data/clean.csv') -TimeoutSec 1800),
        (New-CondaPythonScriptTask -Id 'mlops-unit2-logging-wrapper' -RelativePath 'MLOps\2_logging_persistence\logging_wrapper.py' -ScriptArgs @('--data', '../1_conda_environments/data/clean.csv') -TimeoutSec 1800),
        (New-CondaPythonScriptTask -Id 'mlops-unit3-startup-test' -RelativePath 'MLOps\3_mlflow_setup\startup_test.py' -TimeoutSec 600),
        (New-CondaPythonScriptTask -Id 'mlops-unit4-generate-data-data1' -RelativePath 'MLOps\4_mlflow_logging\generate_data.py' -ScriptArgs @('--outdir', 'data1') -TimeoutSec 900),
        (New-CondaPythonScriptTask -Id 'mlops-unit4-generate-data-data2' -RelativePath 'MLOps\4_mlflow_logging\generate_data.py' -ScriptArgs @('--outdir', 'data2', '--seed', '1') -TimeoutSec 900),
        (New-CondaPythonScriptTask -Id 'mlops-unit4-mlflow-logging' -RelativePath 'MLOps\4_mlflow_logging\ml_pipeline_logging.py' -ScriptArgs @('--data', 'data1/clean.csv') -TimeoutSec 1800),
        (New-CondaPythonScriptTask -Id 'mlops-unit4-mlflow-autolog' -RelativePath 'MLOps\4_mlflow_logging\ml_pipeline_autolog.py' -ScriptArgs @('--data', 'data2/clean.csv') -TimeoutSec 1800),
        (New-CondaPythonScriptTask -Id 'mlops-unit5-optuna-xgboost' -RelativePath 'MLOps\5_xgboost_tuning\optuna_xgboost_mlflow.py' -ScriptArgs @('--n-trials', '3', '--num-boost-round', '25') -TimeoutSec 3600),
        (New-CondaPythonScriptTask -Id 'mlops-unit6-train-initial' -RelativePath 'MLOps\6_monitoring_data_drift\train_initial.py' -ScriptArgs @('--data-parquet', 'TLC_data/green_tripdata_2020-01.parquet') -TimeoutSec 3600),
        (New-CondaPythonScriptTask -Id 'mlops-unit6-check-drift' -RelativePath 'MLOps\6_monitoring_data_drift\check_drift.py' -ScriptArgs @('--ref-parquet', 'TLC_data/green_tripdata_2020-01.parquet', '--cur-parquet', 'TLC_data/green_tripdata_2020-04.parquet') -TimeoutSec 3600),
        (New-CondaPythonScriptTask -Id 'mlops-unit6-retrain' -RelativePath 'MLOps\6_monitoring_data_drift\retrain.py' -ScriptArgs @('--train-parquets', 'TLC_data/green_tripdata_2020-01.parquet', 'TLC_data/green_tripdata_2020-04.parquet', '--eval-parquet', 'TLC_data/green_tripdata_2020-08.parquet') -TimeoutSec 3600),
        (New-CondaPythonScriptTask -Id 'mlops-unit7-generate-data' -RelativePath 'MLOps\7_model_registry_deployment\generate_data.py' -TimeoutSec 900),
        (New-CondaPythonScriptTask -Id 'mlops-unit7-train-register' -RelativePath 'MLOps\7_model_registry_deployment\train_register.py' -TimeoutSec 3600),
        (New-CondaPythonScriptTask -Id 'mlops-unit7-flip-aliases' -RelativePath 'MLOps\7_model_registry_deployment\flip_aliases.py' -TimeoutSec 900),
        (New-CondaTask -Id 'ray-job-cluster-smoke' -Category 'script' -EnvName $script:EnvSpecs['Ray'].name -WorkingDirectory (Join-Path $RepoRoot 'Ray\1_cluster_setup') -CommandArgs @('ray', 'job', 'submit', '--address', 'http://127.0.0.1:8265', '--working-dir', '.', '--', 'python', 'smoke_test_job.py') -TimeoutSec 1800 -Target 'Ray/1_cluster_setup/smoke_test_job.py' -Note 'Submit the Ray smoke test to the local docker cluster.'),
        (New-CondaPythonScriptTask -Id 'ray-map-reduce-chunks' -RelativePath 'Ray\2_system_design\2_0_map_reduce\MR_chunks.py' -ScriptArgs @('--repeat', '2', '--docs-per-chunk', '2', '--reduce-batch-size', '2', '--straggler-delay-s', '0.1', '--reduce-delay-per-bucket-s', '0.0', '--top-k', '5') -TimeoutSec 1800),
        (New-DistributedDlTorchrunTask -Id 'distributed-dl-unit0-ddp-smoke' -RelativePath 'Distributed_DL\0_devcontainer_setup\1_ddp_smoke_test.py' -ProcessCount 2 -TimeoutSec 1800 -Note 'Run the unit 0 DDP smoke test inside the Distributed DL devcontainer.'),
        (New-DistributedDlTorchrunTask -Id 'distributed-dl-unit1-hello-ranks' -RelativePath 'Distributed_DL\1_collective_communication\1_hello_ranks.py' -ProcessCount 2 -TimeoutSec 900 -Note 'Run the hello ranks demo inside the Distributed DL devcontainer.'),
        (New-DistributedDlTorchrunTask -Id 'distributed-dl-unit1-send-recv' -RelativePath 'Distributed_DL\1_collective_communication\2_send_recv_demo.py' -ProcessCount 2 -TimeoutSec 900 -Note 'Run the send/recv demo inside the Distributed DL devcontainer.'),
        (New-DistributedDlTorchrunTask -Id 'distributed-dl-unit1-broadcast' -RelativePath 'Distributed_DL\1_collective_communication\3_broadcast_demo.py' -ProcessCount 2 -TimeoutSec 900 -Note 'Run the broadcast demo inside the Distributed DL devcontainer.'),
        (New-DistributedDlTorchrunTask -Id 'distributed-dl-unit1-reduce-allreduce' -RelativePath 'Distributed_DL\1_collective_communication\4_reduce_all_reduce_demo.py' -ProcessCount 2 -TimeoutSec 900 -Note 'Run the reduce/all-reduce demo inside the Distributed DL devcontainer.'),
        (New-DistributedDlTorchrunTask -Id 'distributed-dl-unit1-gather-allgather' -RelativePath 'Distributed_DL\1_collective_communication\5_gather_all_gather_demo.py' -ProcessCount 2 -TimeoutSec 900 -Note 'Run the gather/all-gather demo inside the Distributed DL devcontainer.'),
        (New-DistributedDlTorchrunTask -Id 'distributed-dl-unit1-scatter' -RelativePath 'Distributed_DL\1_collective_communication\6_scatter_demo.py' -ProcessCount 4 -TimeoutSec 900 -Note 'Run the scatter demo with the expected world size inside the Distributed DL devcontainer.'),
        (New-DistributedDlTorchrunTask -Id 'distributed-dl-unit1-barrier' -RelativePath 'Distributed_DL\1_collective_communication\7_barrier_demo.py' -ProcessCount 2 -TimeoutSec 900 -Note 'Run the barrier demo inside the Distributed DL devcontainer.'),
        (New-DistributedDlTorchrunTask -Id 'distributed-dl-unit1-async-allreduce' -RelativePath 'Distributed_DL\1_collective_communication\8_async_all_reduce_demo.py' -ProcessCount 2 -TimeoutSec 900 -Note 'Run the async all-reduce demo inside the Distributed DL devcontainer.'),
        (New-DistributedDlTorchrunTask -Id 'distributed-dl-unit2-manual-data-parallel' -RelativePath 'Distributed_DL\2_training_challenges\manual_data_parallel_demo.py' -ProcessCount 2 -ScriptArgs @('--steps', '2') -TimeoutSec 1800 -Note 'Run the manual data parallel training demo inside the Distributed DL devcontainer.'),
        (New-DistributedDlTorchrunTask -Id 'distributed-dl-unit3-profile-cpu-traces' -RelativePath 'Distributed_DL\3_profiler_cpu_traces\profile_manual_data_parallel.py' -ProcessCount 2 -ScriptArgs @('--steps', '2', '--trace-name', 'repo_test_smoke') -TimeoutSec 1800 -Note 'Run the CPU profiler trace demo inside the Distributed DL devcontainer.'),
        (New-DistributedDlTorchrunTask -Id 'distributed-dl-unit4-profile-ddp-gpu' -RelativePath 'Distributed_DL\4_ddp_on_cloud_gpus\profile_ddp_gpu.py' -ProcessCount 2 -ScriptArgs @('--cpu', '--steps', '2', '--dataset-size', '128', '--batch-size', '8', '--num-workers', '0', '--trace-name', 'repo_test_cpu') -TimeoutSec 3600 -Note 'Run the local CPU validation path for the Unit 4 GPU profiler inside the Distributed DL devcontainer.')
    )

    Add-CoveredPythonPath -RelativePath 'Ray\1_cluster_setup\smoke_test_job.py'

    $rayComposeDir = Join-Path $RepoRoot 'Ray\1_cluster_setup'
    Invoke-Task (New-ProcessTask `
            -Id 'ray-cluster-down-before-up' `
            -Category 'docker' `
            -WorkingDirectory $rayComposeDir `
            -FilePath $DockerExe `
            -Arguments @('compose', 'down', '-v') `
            -TimeoutSec 600 `
            -Target 'Ray/1_cluster_setup/docker-compose.yml' `
            -Note 'Reset the Ray lesson cluster before starting it.') | Out-Null

    $rayUpResult = Invoke-Task (New-ProcessTask `
            -Id 'ray-cluster-up' `
            -Category 'docker' `
            -WorkingDirectory $rayComposeDir `
            -FilePath $DockerExe `
            -Arguments @('compose', 'up', '-d', '--scale', 'ray-worker=1') `
            -TimeoutSec 1800 `
            -Target 'Ray/1_cluster_setup/docker-compose.yml' `
            -Note 'Start the Ray lesson cluster.')
    $rayReady = Wait-TcpPort -Port 8265 -TimeoutSec 180
    Add-Result ([pscustomobject]@{
            id = 'ray-cluster-ready'
            category = 'service'
            target = 'http://127.0.0.1:8265'
            status = if ($rayReady) { 'passed' } else { 'failed' }
            exit_code = $rayUpResult.exit_code
            timed_out = $false
            duration_sec = 0
            started_at = (Get-Date).ToString('o')
            finished_at = (Get-Date).ToString('o')
            working_dir = Get-RelativeRepoPath $rayComposeDir
            command = 'wait for ray dashboard on port 8265'
            stdout_path = $rayUpResult.stdout_path
            stderr_path = $rayUpResult.stderr_path
            note = 'Waited for the Ray dashboard port after docker compose up.'
        }) | Out-Null
    if ($rayUpResult.status -eq 'passed') {
        $rayClusterStarted = $true
    }

    foreach ($task in $scriptTasks) {
        Invoke-Task $task | Out-Null
    }

    $notebookTasks = Get-ChildItem -Path (Join-Path $RepoRoot 'MLOps'), (Join-Path $RepoRoot 'Ray'), (Join-Path $RepoRoot 'Distributed_DL') -Recurse -File -Filter '*.ipynb' |
    Sort-Object FullName |
    ForEach-Object {
        New-NotebookTask -RelativePath (Get-RelativeRepoPath $_.FullName)
    }

    foreach ($task in $notebookTasks) {
        Invoke-Task $task | Out-Null
    }

    $compileTasks = Get-ChildItem -Path (Join-Path $RepoRoot 'MLOps'), (Join-Path $RepoRoot 'Ray'), (Join-Path $RepoRoot 'Distributed_DL') -Recurse -File -Filter '*.py' |
    Sort-Object FullName |
    ForEach-Object {
        $relativePath = Get-RelativeRepoPath $_.FullName
        if (-not $script:CoveredPythonPaths.Contains($relativePath)) {
            New-CompileTask -RelativePath $relativePath
        }
    }

    foreach ($task in $compileTasks) {
        Invoke-Task $task | Out-Null
    }
}
finally {
    if ($null -ne $mlflowService) {
        Stop-BackgroundProcess -Service $mlflowService -Note 'Stopped the MLflow server at the end of the test session.' | Out-Null
    }

    if ($distributedDlContainerStarted -and (Test-DockerContainerExists -ContainerName $script:DistributedDlContainerName)) {
        Invoke-Task (New-ProcessTask `
                -Id 'distributed-dl-devcontainer-down' `
                -Category 'cleanup' `
                -WorkingDirectory (Join-Path $RepoRoot 'Distributed_DL') `
                -FilePath $DockerExe `
                -Arguments @('rm', '-f', $script:DistributedDlContainerName) `
                -TimeoutSec 300 `
                -Target 'Distributed_DL/.devcontainer/devcontainer.json' `
                -Note 'Stop and remove the Distributed DL devcontainer used for smoke tasks.') | Out-Null
    }

    if ($rayClusterStarted) {
        Invoke-Task (New-ProcessTask `
                -Id 'ray-cluster-down-final' `
                -Category 'cleanup' `
                -WorkingDirectory (Join-Path $RepoRoot 'Ray\1_cluster_setup') `
                -FilePath $DockerExe `
                -Arguments @('compose', 'down', '-v') `
                -TimeoutSec 900 `
                -Target 'Ray/1_cluster_setup/docker-compose.yml' `
                -Note 'Stop the Ray lesson cluster and remove its volumes.') | Out-Null
    }

    if (-not $SkipCleanIgnored) {
        Invoke-Task (New-ProcessTask `
                -Id 'clean-ignored' `
                -Category 'cleanup' `
                -WorkingDirectory $RepoRoot `
                -FilePath $PowerShellExe `
                -Arguments @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $CleanIgnoredScript) `
                -TimeoutSec 1800 `
                -Target 'repo_maintenance/clean_ignored.ps1' `
                -Note 'Remove ignored artifacts while preserving TLC data.') | Out-Null
    }

    Write-SummaryFiles
}

$failedCount = @($script:Results | Where-Object { $_.status -ne 'passed' }).Count
Write-Host ''
Write-Host "Summary written to $(Get-RelativeRepoPath $SummaryMdPath)"
Write-Host "Detailed JSON written to $(Get-RelativeRepoPath $SummaryJsonPath)"
Write-Host "Task logs written under $(Get-RelativeRepoPath $DumpRoot)"

if ($failedCount -gt 0) {
    exit 1
}

exit 0
