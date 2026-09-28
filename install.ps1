<#
.SYNOPSIS
  Bootstrap: install the Qwen-Image-2.1 stack (ComfyUI, models, API, tunnel).

.EXAMPLE
  .\install.ps1
  .\install.ps1 -Profile mac-8gb
  .\install.ps1 -Profile nvidia-8gb -SkipModels
#>
param(
  [ValidateSet("auto", "nvidia-8gb", "mac-8gb", "mac-8gb-lite", "full")]
  [string]$Profile = "auto",
  [switch]$SkipModels,
  [switch]$Force,
  [switch]$RecreateEnv,
  [switch]$NoCli,
  [string]$ComfyPath = "",
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Extra = @()
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Find-Python {
  $candidates = @()
  if ($env:PYTHON) { $candidates += $env:PYTHON }
  $launcher = Get-Command py -ErrorAction SilentlyContinue
  if ($launcher) {
    $candidates += @("py", "py -3.12", "py -3.11", "py -3.10")
  }
  foreach ($name in @("python", "python3")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { $candidates += $name }
  }
  foreach ($candidate in $candidates) {
    $parts = $candidate -split " "
    $exe = $parts[0]
    $args = @()
    if ($parts.Count -gt 1) { $args = $parts[1..($parts.Count - 1)] }
    try {
      $version = & $exe @args -c "import sys;print('%d.%d' % sys.version_info[:2])" 2>$null
      if ($LASTEXITCODE -eq 0 -and $version -and [version]$version -ge [version]"3.10") {
        return @{ Exe = $exe; Args = $args }
      }
    } catch { }
  }
  return $null
}

$python = Find-Python
if (-not $python) {
  throw "Python 3.10+ was not found. Install it from python.org (tick 'Add python.exe to PATH') and re-run."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  throw "git is required (ComfyUI submodule + ComfyUI-GGUF)."
}

$version = & $python.Exe @($python.Args) -c "import platform;print(platform.python_version())"
Write-Host "python: $($python.Exe) ($version)"

if (-not $NoCli) {
  & $python.Exe @($python.Args) -m pip install --user --disable-pip-version-check -e . *> $null
  if ($LASTEXITCODE -eq 0) {
    Write-Host "installed the 'qwen21' command"
  } else {
    Write-Host "note: could not install the 'qwen21' command; use 'python -m qwen21 ...' instead"
  }
}

$arguments = @("install", "--profile", $Profile)
if ($SkipModels) { $arguments += "--skip-models" }
if ($Force) { $arguments += "--force" }
if ($RecreateEnv) { $arguments += "--recreate-env" }
if ($ComfyPath) { $arguments += @("--comfy-path", $ComfyPath) }
$arguments += $Extra

& $python.Exe @($python.Args) "-m" "qwen21" @arguments
exit $LASTEXITCODE
