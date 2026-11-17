# Gemma 4 E4B chat on this PC (run on Windows 11 with Windows PowerShell 5.1). chat.cmd runs this file.
#
#   chat.cmd [-Device auto|cuda|webgpu|cpu] [-Adapter <n>] [-Context 2048|4096|8192] [-ModelDir <folder>] [-Log <file>]
#            [-SetupOnly]
#
# First run: finds Python 3.12, reads the GPU and the free disk space, makes a Python environment next to this file
# (.venv-<device>, the pinned packages of requirements-<device>.txt), downloads the model package for the device
# (models\<package>, about 7.2 GB, with curl.exe from Windows; an interrupted download continues where it stopped) and
# checks every file's size and SHA-256 against models.json. Later runs skip what is done and open the chat at once.
#   -Device    auto (default) = CUDA on an NVIDIA GPU with 15,000 MiB of memory or more (a 16 GB card), else WebGPU;
#              or name one
#   -Adapter   with WebGPU only: which GPU, by its WebGPU number (default 0; on a PC with two GPUs try 1). The
#              window shows the number; Task Manager shows which GPU works
#   -Context   tokens one chat can hold, questions and answers together (default 4096)
#   -ModelDir  use a model folder that is already on this PC (nothing is downloaded)
#   -Log       add one JSON line per answer to this file: the question, the answer and the numbers the window showed
#   -SetupOnly set up, but do not open the chat
# Any failure prints one line that says what to do next and ends with a non-zero exit code.
param(
  [string]$Device = 'auto',
  [string]$Adapter = '0',
  [string]$Context = '4096',
  [string]$ModelDir = '',
  [string]$Log = '',
  [switch]$SetupOnly
)
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$env:ORT_DISABLE_TELEMETRY = '1'   # ONNX Runtime and ONNX Runtime GenAI send no usage telemetry from this chat
$env:PYTHONUTF8 = '1'              # Python's messages in UTF-8, read back as UTF-8 below
$here = $PSScriptRoot
# disk the Python environment takes, in MB: 1.3 x the size measured on Windows 11 x64 (cuda 1,546, webgpu 157,
# cpu 127)
$venvMb = @{ cuda = 2010; webgpu = 205; cpu = 166 }
$oldOut = $null
try { $oldOut = [Console]::OutputEncoding; [Console]::OutputEncoding = New-Object Text.UTF8Encoding($false) } catch { }

function Say([string]$Text) { [Console]::Out.WriteLine($Text) }

function Leave([int]$Code) {
  if ($oldOut) { try { [Console]::OutputEncoding = $oldOut } catch { } }
  exit $Code
}

function Started-FromExplorer {
  # double-clicked (explorer.exe -> cmd.exe -> this PowerShell): keep the window open after a failure
  try {
    $me = Get-CimInstance Win32_Process -Filter "ProcessId=$PID" -ErrorAction Stop
    $cmdProc = Get-CimInstance Win32_Process -Filter "ProcessId=$($me.ParentProcessId)" -ErrorAction Stop
    if ($cmdProc.Name -ne 'cmd.exe') { return $false }
    return ((Get-Process -Id $cmdProc.ParentProcessId -ErrorAction Stop).ProcessName -eq 'explorer')
  } catch { return $false }
}

function Fail([string]$Text, [int]$Code) {
  [Console]::Error.WriteLine($Text)
  if (Started-FromExplorer) { [void](Read-Host 'Press Enter to close this window') }
  Leave $Code
}

function Quote([string]$Text) {
  # one command-line argument in quotes; a trailing backslash is doubled so it does not escape the closing quote
  if ($Text.EndsWith('\')) { $Text += '\' }
  return '"' + $Text + '"'
}

function Find-Python312 {
  # the py launcher of python.org's installer, then python.exe on PATH: "<version>|<platform>|<interpreter>"
  $probe = "import sys, sysconfig; print('%d.%d|%s|%s' % (sys.version_info[0], sys.version_info[1], sysconfig.get_platform(), sys.executable))"
  foreach ($launcher in @(@('py.exe', '-3.12'), @('python.exe'))) {
    if (-not (Get-Command $launcher[0] -ErrorAction SilentlyContinue)) { continue }
    $cmdArgs = @($launcher | Select-Object -Skip 1) + @('-c', $probe)
    $out = @(& $launcher[0] @cmdArgs 2>$null)
    if ($LASTEXITCODE -ne 0 -or $out.Count -eq 0) { continue }
    $parts = ([string]$out[-1]).Trim() -split '\|', 3
    if ($parts.Count -eq 3 -and $parts[0] -eq '3.12' -and (Test-Path -LiteralPath $parts[2])) {
      return @{ platform = $parts[1]; exe = $parts[2] }
    }
  }
  return $null
}

function Free-MB([string]$Folder) {
  $root = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Folder))
  return [math]::Floor((New-Object IO.DriveInfo($root)).AvailableFreeSpace / 1MB)
}

# --- the options -----------------------------------------------------------------------------------------------------
if ($args.Count -gt 0) {
  Fail ('Unknown option: ' + ($args -join ' ') + '. The options are -Device, -Adapter, -Context, -ModelDir, -Log and -SetupOnly.') 2
}
$Device = $Device.ToLower()
if ($Device -notin @('auto', 'cuda', 'webgpu', 'cpu')) { Fail "-Device takes auto, cuda, webgpu or cpu, not '$Device'." 2 }
if ($Context -notin @('2048', '4096', '8192')) { Fail "-Context takes 2048, 4096 or 8192, not '$Context'." 2 }
$ctx = [int]$Context
$adapterN = 0
if (-not [int]::TryParse($Adapter, [ref]$adapterN) -or $adapterN -lt 0) { Fail "-Adapter takes a GPU number (0, 1, 2 ...), not '$Adapter'." 2 }

# --- Python 3.12 and the device ------------------------------------------------------------------------------------
$pyInfo = Find-Python312
if (-not $pyInfo) {
  Fail 'Python 3.12 was not found. Install Python 3.12 (64-bit) from https://www.python.org/downloads/windows/ and run chat.cmd again.' 2
}
if ($pyInfo.platform -ne 'win-amd64') {
  Fail "The Python 3.12 found is $($pyInfo.platform) ($($pyInfo.exe)); the chat needs 64-bit (x64) Python 3.12 from https://www.python.org/downloads/windows/." 2
}
$py312 = $pyInfo.exe
$manifestPath = Join-Path $here 'models.json'
try { $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json } catch { $manifest = $null }
if (-not $manifest) { Fail "models.json is missing or unreadable next to chat.ps1. Download this folder again." 2 }
$engineScript = Join-Path $here 'engine.py'
$dev = $Device
if ($dev -eq 'auto') {
  $dev = (@(& $py312 $engineScript --resolve-device auto 2>$null) | Select-Object -Last 1)
  if ($LASTEXITCODE -ne 0 -or -not $dev) { Fail 'Reading the GPU failed. Start the chat with chat.cmd -Device cpu.' 2 }
  $dev = $dev.Trim()
}
if ($PSBoundParameters.ContainsKey('Adapter') -and $dev -ne 'webgpu') { Say "-Adapter is used only with WebGPU; this chat runs on $dev." }
$label = (@(& $py312 $engineScript --label $dev --adapter $adapterN 2>$null) | Select-Object -Last 1)
if ($LASTEXITCODE -ne 0 -or -not $label) { $label = $dev } else { $label = $label.Trim() }
$pkg = $manifest.devices.$dev
$pkgInfo = $manifest.packages.$pkg
if ($ModelDir) {
  if (-not (Test-Path -LiteralPath $ModelDir -PathType Container)) { Fail "The model folder $ModelDir does not exist." 7 }
  $modelFolder = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $ModelDir).ProviderPath)
  if ($modelFolder.Length -gt 3) { $modelFolder = $modelFolder.TrimEnd('\') }
} else {
  $modelFolder = Join-Path $here (Join-Path 'models' $pkg)
}
Say "Gemma 4 E4B chat: $dev on $label, $ctx tokens per chat, model in $modelFolder"
$logFile = ''
if ($Log) {
  $logFile = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Log)
  if ($logFile.EndsWith('\')) { Fail "-Log takes a file name, not a folder: $Log" 2 }
  $logFolder = [IO.Path]::GetDirectoryName($logFile)
  if (-not (Test-Path -LiteralPath $logFolder -PathType Container)) { Fail "The folder for -Log does not exist: $logFolder" 2 }
}

# --- free disk space -------------------------------------------------------------------------------------------------
$venv = Join-Path $here ".venv-$dev"
$venvPy = Join-Path $venv 'Scripts\python.exe'
$requirements = Join-Path $here "requirements-$dev.txt"
$reqHash = (Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash
$readyFile = Join-Path $venv 'chat-ready.txt'
$venvReady = (Test-Path -LiteralPath $venvPy) -and (Test-Path -LiteralPath $readyFile) -and ((Get-Content -LiteralPath $readyFile -Raw).Trim() -eq $reqHash)
$needMb = 0
if (-not $venvReady) { $needMb += $venvMb[$dev] }
if (-not $ModelDir) {
  foreach ($f in $pkgInfo.files) {
    $p = Join-Path $modelFolder ($f.path -replace '/', '\')
    $have = 0
    if (Test-Path -LiteralPath $p) { $have = (Get-Item -LiteralPath $p).Length }
    if ($have -lt $f.size) { $needMb += [math]::Ceiling(($f.size - $have) / 1MB) }
  }
}
if ($needMb -gt 0) {
  $freeMb = Free-MB $here
  if ($freeMb -lt $needMb + 1024) {
    Fail ('Not enough free disk space: {0:N1} GB free, about {1:N1} GB needed. Free some space and run chat.cmd again.' -f ($freeMb / 1024), (($needMb + 1024) / 1024)) 3
  }
}
if (-not $venvReady -and $env:TEMP) {
  # pip downloads and unpacks the packages in the temporary folder first
  $tempRoot = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($env:TEMP))
  if ($tempRoot -ne [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($here))) {
    $tempFree = Free-MB $env:TEMP
    if ($tempFree -lt $venvMb[$dev] + 1024) {
      Fail ('Not enough free disk space on {0} for the temporary files of the setup: {1:N1} GB free, about {2:N1} GB needed. Free some space and run chat.cmd again.' -f $tempRoot, ($tempFree / 1024), (($venvMb[$dev] + 1024) / 1024)) 3
    }
  }
}

# --- the Python environment ------------------------------------------------------------------------------------------
if (-not $venvReady) {
  if (-not (Test-Path -LiteralPath $venvPy)) {
    Say "Making the Python environment $venv"
    & $py312 -m venv $venv
    if ($LASTEXITCODE -ne 0) { Fail "Making the Python environment failed (exit $LASTEXITCODE). Delete $venv and run chat.cmd again." 4 }
  }
  Say "Installing the pinned packages of requirements-$dev.txt"
  & $venvPy -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r $requirements
  if ($LASTEXITCODE -ne 0) { Fail "Installing the Python packages failed (pip exit $LASTEXITCODE): see pip's message above, then run chat.cmd again." 4 }
  Set-Content -LiteralPath $readyFile -Value $reqHash -Encoding ASCII
}

# --- the model -------------------------------------------------------------------------------------------------------
if ($ModelDir) {
  $out = @(& $venvPy $engineScript --check-model-dir $modelFolder 2>&1)
  if ($LASTEXITCODE -ne 0) { Fail (($out | Select-Object -Last 1) -as [string]) 7 }
} else {
  $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
  if (-not (Test-Path -LiteralPath $curl)) { Fail 'curl.exe was not found in Windows.' 5 }
  $marker = Join-Path $modelFolder 'chat-verified.json'
  $verified = [ordered]@{}
  if (Test-Path -LiteralPath $marker) {
    try {
      $m = Get-Content -LiteralPath $marker -Raw -Encoding UTF8 | ConvertFrom-Json
      if ($m.revision -eq $manifest.revision -and $m.package -eq $pkg) { foreach ($v in $m.files) { $verified[$v.path] = [long]$v.size } }
    } catch { $verified = [ordered]@{} }
  }
  $files = @($pkgInfo.files)
  $k = 0
  foreach ($f in $files) {
    $k++
    $dst = Join-Path $modelFolder ($f.path -replace '/', '\')
    $url = $manifest.url.Replace('{package}', $pkg).Replace('{path}', $f.path)
    for ($try = 1; $try -le 2; $try++) {
      $have = 0
      if (Test-Path -LiteralPath $dst) { $have = (Get-Item -LiteralPath $dst).Length }
      if ($have -eq $f.size -and $verified.Contains($f.path) -and $verified[$f.path] -eq $f.size) { break }
      if ($have -gt $f.size) { Remove-Item -LiteralPath $dst -Force; $have = 0 }
      if ($have -lt $f.size) {
        [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($dst))
        Say ('[{0}/{1}] {2} ({3:N0} MB)' -f $k, $files.Count, $f.path, ($f.size / 1MB))
        & $curl -L --fail --retry 3 --retry-delay 2 --connect-timeout 30 --speed-limit 10000 --speed-time 60 -C - --progress-bar -o $dst $url
        $rc = $LASTEXITCODE
        if ($rc -eq 28) {
          Fail "The download of $($f.path) stalled or timed out (curl exit 28). Check the internet connection and run chat.cmd again: it continues where it stopped." 5
        }
        if ($rc -in @(5, 6, 7, 35)) {
          # no proxy / no name / no connection / TLS handshake
          $hostName = ([Uri]$url).Host
          Fail "The download of $($f.path) could not reach $hostName (curl exit $rc). Check the internet connection and run chat.cmd again: it continues where it stopped." 5
        }
        if ($rc -ne 0) { Fail "The download of $($f.path) stopped (curl exit $rc). Run chat.cmd again: it continues where it stopped." 5 }
        $have = (Get-Item -LiteralPath $dst).Length
        if ($have -ne $f.size) { Fail "$($f.path) has $have bytes, not $($f.size). Run chat.cmd again: it continues where it stopped." 5 }
      }
      Say "Checking $($f.path)..."
      $hash = (Get-FileHash -LiteralPath $dst -Algorithm SHA256).Hash.ToLower()
      if ($hash -eq $f.sha256) {
        $verified[$f.path] = [long]$f.size
        # the files checked so far: a later run does not check them again
        $rows = @($verified.Keys | ForEach-Object { [ordered]@{ path = $_; size = [long]$verified[$_] } })
        $doc = [ordered]@{ revision = $manifest.revision; package = $pkg; files = $rows }
        [IO.File]::WriteAllText($marker, ($doc | ConvertTo-Json -Depth 4), (New-Object Text.UTF8Encoding($false)))
        break
      }
      Remove-Item -LiteralPath $dst -Force
      if ($try -eq 2) { Fail "$($f.path) did not match its SHA-256 twice. Run chat.cmd again later." 6 }
      Say "$($f.path) did not match its SHA-256: downloading it again"
    }
  }
}

if ($SetupOnly) {
  Say "Setup done: $dev on $label. Open the chat with chat.cmd."
  Leave 0
}

# --- the chat window -------------------------------------------------------------------------------------------------
$pythonw = Join-Path $venv 'Scripts\pythonw.exe'
$argList = @((Quote (Join-Path $here 'app.py')), '--device', $dev, '--context', "$ctx", '--model-dir', (Quote $modelFolder))
if ($Device -eq 'auto') { $argList += @('--auto-picked') }
if ($dev -eq 'webgpu') { $argList += @('--adapter', "$adapterN") }
if ($logFile) { $argList += @('--log', (Quote $logFile)) }
$app = Start-Process -FilePath $pythonw -ArgumentList $argList -WorkingDirectory $here -PassThru
$null = $app.Handle   # makes ExitCode readable after WaitForExit
if ($app.WaitForExit(4000)) {
  if ($app.ExitCode -eq 3) { Say 'Gemma 4 chat is already open: its window was brought to the front.'; Leave 0 }
  if ($app.ExitCode -eq 0) { Leave 0 }
  Fail "The chat window closed at once (exit $($app.ExitCode)). Run $venvPy app.py --device $dev to see why." 8
}
Say 'The chat window is open.'
Leave 0
