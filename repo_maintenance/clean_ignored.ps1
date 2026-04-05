<#
.SYNOPSIS
Deletes ignored artifact, data, and tracking paths from this repository while preserving TLC_data, runpod_output, colab_output, and test_logs.

.DESCRIPTION
Uses Git to enumerate ignored paths, removes them from the working tree, and refuses to
delete anything outside the repository root. Any path containing TLC_data, runpod_output,
colab_output, or test_logs is skipped on purpose.

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File .\repo_maintenance\clean_ignored.ps1 -WhatIf
Preview which ignored paths would be deleted without removing anything.

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File .\repo_maintenance\clean_ignored.ps1
Delete ignored paths in the repository while preserving TLC_data, runpod_output, colab_output, and test_logs.

.NOTES
You can also run: Get-Help .\repo_maintenance\clean_ignored.ps1 -Full
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-PathInsideRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [Parameter(Mandatory = $true)]
        [string]$Candidate
    )

    $rootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd("\", "/")
    $candidateFull = [System.IO.Path]::GetFullPath($Candidate)

    if ($candidateFull.Equals($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }

    $rootPrefix = $rootFull + [System.IO.Path]::DirectorySeparatorChar
    return $candidateFull.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (& git -C $scriptDir rev-parse --show-toplevel).Trim()

if (-not $repoRoot) {
    throw "Could not determine git repository root."
}

$statusLines = & git -C $repoRoot -c core.quotepath=off status --ignored --porcelain=v1
if ($LASTEXITCODE -ne 0) {
    throw "Failed to enumerate ignored paths from git."
}

$ignoredPaths = @(
    $statusLines |
        Where-Object { $_.StartsWith("!! ") } |
        ForEach-Object { $_.Substring(3).Trim() } |
        Where-Object { $_ }
)

if ($ignoredPaths.Count -eq 0) {
    Write-Host "No ignored paths found."
    return
}

$skipPattern = '(^|[\\/])(TLC_data|runpod_output|colab_output|test_logs)([\\/]|$)'
$uniquePaths = $ignoredPaths | Select-Object -Unique
$sortedPaths = $uniquePaths | Sort-Object { ($_ -split '[\\/]').Count } -Descending

$deletedCount = 0
$skippedCount = 0

foreach ($relativePath in $sortedPaths) {
    if ($relativePath -match $skipPattern) {
        Write-Host "Skipping preserved path: $relativePath"
        $skippedCount += 1
        continue
    }

    $trimmedRelativePath = $relativePath.TrimEnd("/", "\")
    $normalizedRelativePath = $trimmedRelativePath -replace "/", "\"
    $fullPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $normalizedRelativePath))

    if (-not (Test-PathInsideRoot -Root $repoRoot -Candidate $fullPath)) {
        throw "Refusing to delete path outside repo root: $relativePath"
    }

    if (-not (Test-Path -LiteralPath $fullPath)) {
        Write-Host "Already missing: $relativePath"
        continue
    }

    $action = "Delete ignored path"
    if ($PSCmdlet.ShouldProcess($fullPath, $action)) {
        if (Test-Path -LiteralPath $fullPath -PathType Container) {
            Remove-Item -LiteralPath $fullPath -Recurse -Force
        }
        else {
            Remove-Item -LiteralPath $fullPath -Force
        }

        Write-Host "Deleted: $relativePath"
        $deletedCount += 1
    }
}

Write-Host "Done. Deleted $deletedCount ignored paths; skipped $skippedCount preserved TLC_data/runpod_output/colab_output/test_logs paths."
