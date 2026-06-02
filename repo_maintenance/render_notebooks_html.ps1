<#
.SYNOPSIS
Renders repository notebooks to sibling single-file HTML payloads.

.DESCRIPTION
Finds tracked .ipynb files in this repository and converts each one to a same-name
.html file. The conversion is pinned to the GitHub notebook renderer package
versions reported by GitHub: nbformat 5.10.4 and nbconvert 7.17.0.

The script does not execute notebooks. It uses nbconvert's HTML exporter with
embedded images enabled so local Markdown image references are inlined into the
generated HTML file.

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File .\repo_maintenance\render_notebooks_html.ps1
Render every tracked notebook to a sibling HTML file, installing the pinned
renderer versions into the base Conda environment if needed.

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File .\repo_maintenance\render_notebooks_html.ps1 -WhatIf
Show which notebooks would be rendered without installing packages or writing HTML.

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File .\repo_maintenance\render_notebooks_html.ps1 -OutputRoot notebook_html
Render every tracked notebook under notebook_html while preserving repo-relative
subdirectories.

.NOTES
You can also run: Get-Help .\repo_maintenance\render_notebooks_html.ps1 -Full
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$CondaEnv = 'base',
    [string]$CondaExecutable = '',
    [string]$NotebookRoot = '',
    [string]$OutputRoot = '',
    [string]$NbFormatVersion = '5.10.4',
    [string]$NbConvertVersion = '7.17.0',
    [switch]$IncludeUntracked,
    [switch]$SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-CondaExecutable {
    param(
        [string]$ExplicitPath
    )

    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        if (-not (Test-Path -LiteralPath $ExplicitPath)) {
            throw "Conda executable not found: $ExplicitPath"
        }
        return (Resolve-Path -LiteralPath $ExplicitPath).Path
    }

    if ($env:CONDA_EXE -and (Test-Path -LiteralPath $env:CONDA_EXE)) {
        return (Resolve-Path -LiteralPath $env:CONDA_EXE).Path
    }

    $command = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    throw 'Could not find conda. Make sure conda is on PATH or pass -CondaExecutable.'
}

function Test-PathInsideRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [Parameter(Mandatory = $true)]
        [string]$Candidate
    )

    $rootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    $candidateFull = [System.IO.Path]::GetFullPath($Candidate)

    if ($candidateFull.Equals($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }

    $rootPrefix = $rootFull + [System.IO.Path]::DirectorySeparatorChar
    return $candidateFull.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-RelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $rootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    $pathFull = [System.IO.Path]::GetFullPath($Path)

    $rootUri = New-Object System.Uri(($rootFull + [System.IO.Path]::DirectorySeparatorChar))
    $pathUri = New-Object System.Uri($pathFull)
    $relativePath = $rootUri.MakeRelativeUri($pathUri).ToString()
    return [System.Uri]::UnescapeDataString($relativePath).Replace('/', [System.IO.Path]::DirectorySeparatorChar)
}

function Resolve-RepoPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Path))
}

function Invoke-CondaCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Conda,
        [Parameter(Mandatory = $true)]
        [string]$EnvironmentName,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$WorkingDirectory = ''
    )

    $condaArgs = @('run', '--no-capture-output', '-n', $EnvironmentName) + $Arguments
    if ([string]::IsNullOrWhiteSpace($WorkingDirectory)) {
        & $Conda @condaArgs
    }
    else {
        Push-Location -LiteralPath $WorkingDirectory
        try {
            & $Conda @condaArgs
        }
        finally {
            Pop-Location
        }
    }
}

function Get-RendererVersions {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Conda,
        [Parameter(Mandatory = $true)]
        [string]$EnvironmentName
    )

    $code = 'import json, nbconvert, nbformat; print(json.dumps(dict(nbconvert=nbconvert.__version__, nbformat=nbformat.__version__)))'
    $output = & $Conda @('run', '--no-capture-output', '-n', $EnvironmentName, 'python', '-c', $code)
    if ($LASTEXITCODE -ne 0) {
        return $null
    }

    return ($output -join "`n") | ConvertFrom-Json
}

function Get-NotebookRelativePaths {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [switch]$IncludeUntracked
    )

    $tracked = @(& git -C $RepoRoot -c core.quotepath=off ls-files -- '*.ipynb')
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to list tracked notebooks with git.'
    }

    $paths = New-Object System.Collections.Generic.List[string]
    foreach ($path in $tracked) {
        if (-not [string]::IsNullOrWhiteSpace($path)) {
            $null = $paths.Add($path)
        }
    }

    if ($IncludeUntracked) {
        $untracked = @(& git -C $RepoRoot -c core.quotepath=off ls-files --others --exclude-standard -- '*.ipynb')
        if ($LASTEXITCODE -ne 0) {
            throw 'Failed to list untracked notebooks with git.'
        }

        foreach ($path in $untracked) {
            if (-not [string]::IsNullOrWhiteSpace($path)) {
                $null = $paths.Add($path)
            }
        }
    }

    return @(
        $paths |
            Where-Object { $_ -notmatch '(^|/)\.ipynb_checkpoints/' } |
            Sort-Object -Unique
    )
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (& git -C $scriptDir rev-parse --show-toplevel).Trim()
if (-not $repoRoot) {
    throw 'Could not determine git repository root.'
}

$rendererScript = Join-Path $scriptDir 'scripts\render_notebook_html.py'
if (-not (Test-Path -LiteralPath $rendererScript)) {
    throw "Notebook HTML renderer helper not found: $rendererScript"
}

$notebookRootPath = if ([string]::IsNullOrWhiteSpace($NotebookRoot)) {
    $repoRoot
}
else {
    Resolve-RepoPath -RepoRoot $repoRoot -Path $NotebookRoot
}

if (-not (Test-Path -LiteralPath $notebookRootPath -PathType Container)) {
    throw "Notebook root does not exist: $notebookRootPath"
}

if (-not (Test-PathInsideRoot -Root $repoRoot -Candidate $notebookRootPath)) {
    throw "Notebook root must be inside the repository: $notebookRootPath"
}

$outputRootPath = $null
if (-not [string]::IsNullOrWhiteSpace($OutputRoot)) {
    $outputRootPath = Resolve-RepoPath -RepoRoot $repoRoot -Path $OutputRoot
    if (-not (Test-PathInsideRoot -Root $repoRoot -Candidate $outputRootPath)) {
        throw "Output root must be inside the repository: $outputRootPath"
    }
}

$conda = Get-CondaExecutable -ExplicitPath $CondaExecutable
$notebooks = @(
    Get-NotebookRelativePaths -RepoRoot $repoRoot -IncludeUntracked:$IncludeUntracked |
        ForEach-Object {
            $fullPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot ($_ -replace '/', '\')))
            if (Test-PathInsideRoot -Root $notebookRootPath -Candidate $fullPath) {
                [pscustomobject]@{
                    repo_relative = $_
                    full_path = $fullPath
                }
            }
        }
)

if ($notebooks.Count -eq 0) {
    Write-Host "No notebooks found under $(Get-RelativePath -Root $repoRoot -Path $notebookRootPath)."
    return
}

if (-not $PSCmdlet.ShouldProcess($CondaEnv, "Verify nbformat==$NbFormatVersion and nbconvert==$NbConvertVersion")) {
    Write-Host "Would verify renderer versions in Conda env '$CondaEnv'."
}
else {
    $versions = Get-RendererVersions -Conda $conda -EnvironmentName $CondaEnv
    $needsInstall = (
        $null -eq $versions -or
        [string]$versions.nbformat -ne $NbFormatVersion -or
        [string]$versions.nbconvert -ne $NbConvertVersion
    )

    if ($needsInstall) {
        if ($SkipInstall) {
            $found = if ($null -eq $versions) {
                'not installed'
            }
            else {
                "nbformat $($versions.nbformat), nbconvert $($versions.nbconvert)"
            }
            throw "Renderer versions are $found; expected nbformat $NbFormatVersion and nbconvert $NbConvertVersion."
        }

        if ($PSCmdlet.ShouldProcess($CondaEnv, "Install pinned notebook renderer packages")) {
            Invoke-CondaCommand `
                -Conda $conda `
                -EnvironmentName $CondaEnv `
                -Arguments @(
                    'python', '-m', 'pip', 'install',
                    '--disable-pip-version-check',
                    "nbformat==$NbFormatVersion",
                    "nbconvert==$NbConvertVersion"
                )
            if ($LASTEXITCODE -ne 0) {
                throw 'Failed to install pinned notebook renderer packages.'
            }
        }

        $versions = Get-RendererVersions -Conda $conda -EnvironmentName $CondaEnv
    }

    if (
        $null -eq $versions -or
        [string]$versions.nbformat -ne $NbFormatVersion -or
        [string]$versions.nbconvert -ne $NbConvertVersion
    ) {
        throw "Renderer version check failed after setup. Expected nbformat $NbFormatVersion and nbconvert $NbConvertVersion."
    }

    Write-Host "Using nbformat $($versions.nbformat) and nbconvert $($versions.nbconvert) in Conda env '$CondaEnv'."
}

$renderedCount = 0
foreach ($notebook in $notebooks) {
    $notebookPath = $notebook.full_path
    $notebookDir = Split-Path -Parent $notebookPath
    $notebookName = Split-Path -Leaf $notebookPath
    $htmlStem = [System.IO.Path]::GetFileNameWithoutExtension($notebookName)

    if ($null -eq $outputRootPath) {
        $htmlDir = $notebookDir
    }
    else {
        $relativeDir = Split-Path -Parent $notebook.repo_relative
        $htmlDir = if ([string]::IsNullOrWhiteSpace($relativeDir)) {
            $outputRootPath
        }
        else {
            Join-Path $outputRootPath ($relativeDir -replace '/', '\')
        }
    }

    $htmlPath = Join-Path $htmlDir ($htmlStem + '.html')
    $displayNotebook = $notebook.repo_relative
    $displayHtml = Get-RelativePath -Root $repoRoot -Path $htmlPath

    if ($PSCmdlet.ShouldProcess($displayHtml, "Render $displayNotebook")) {
        Invoke-CondaCommand `
            -Conda $conda `
            -EnvironmentName $CondaEnv `
            -Arguments @(
                'python', $rendererScript,
                '--notebook', $notebookPath,
                '--output-dir', $htmlDir,
                '--output-stem', $htmlStem
            )
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to render notebook: $displayNotebook"
        }

        $renderedCount += 1
    }
}

Write-Host "Done. Rendered $renderedCount notebook HTML files."
