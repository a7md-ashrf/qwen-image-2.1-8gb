param(
  [Parameter(Mandatory=$true)]
  [string]$ComfyUI,
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) {
  $Python = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $Python) {
  throw "Python was not found in PATH."
}

$ArgsList = @("qwen21.py", "install", "--comfy", $ComfyUI)
if ($Force) {
  $ArgsList += "--force"
}

& $Python.Source @ArgsList
exit $LASTEXITCODE
