param(
    [switch]$SkipToolkit,
    [switch]$StartJupyter
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Write-Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Ok($Message) {
    Write-Host "OK  $Message" -ForegroundColor Green
}

function Write-Warn($Message) {
    Write-Host "WARN $Message" -ForegroundColor Yellow
}

function Require-Command($Name, $Hint) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "$Name was not found. $Hint"
    }
    return $cmd
}

function Get-PythonCommand {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        $versions = & py -0p 2>$null
        if ($versions -match '3\.10') { return @('py', '-3.10') }
        if ($versions -match '3\.11') { return @('py', '-3.11') }
    }

    throw "Python 3.10 or 3.11 was not found. Install Python 3.10/3.11 from python.org, enable 'Add python.exe to PATH', then rerun setup_5090.bat. Do not use Python 3.13/3.14 for this trainer because some training dependencies may need source builds."
}

function Invoke-CommandArray($CommandArray, $Arguments) {
    $exe = $CommandArray[0]
    $baseArgs = @()
    if ($CommandArray.Count -gt 1) {
        $baseArgs = $CommandArray[1..($CommandArray.Count - 1)]
    }
    & $exe @baseArgs @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $exe $($baseArgs -join ' ') $($Arguments -join ' ')"
    }
}

function Invoke-VenvPython($Arguments) {
    & $VenvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: python $($Arguments -join ' ')"
    }
}

Write-Host "DB9-Toolkit-Trainner RTX 5090 local setup" -ForegroundColor Magenta
Write-Host "Project: $ProjectRoot"

Write-Step "Checking Git and pulling latest code"
Require-Command git "Install Git for Windows: https://git-scm.com/download/win" | Out-Null
git pull origin main
if ($LASTEXITCODE -ne 0) { throw "git pull failed" }

Write-Step "Checking NVIDIA GPU"
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    $gpuInfo = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
    Write-Host $gpuInfo
    if ($gpuInfo -match '5090') {
        Write-Ok "RTX 5090 detected. Using CUDA 12.8 PyTorch wheels."
    } else {
        Write-Warn "RTX 5090 was not detected. The script will still install CUDA 12.8 PyTorch wheels."
    }
} else {
    Write-Warn "nvidia-smi was not found. Install/update NVIDIA drivers before training."
}

Write-Step "Checking Python"
$PythonCommand = Get-PythonCommand
Write-Host "Using Python command: $($PythonCommand -join ' ')"
Invoke-CommandArray $PythonCommand @('--version')

Write-Step "Creating virtual environment"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Invoke-CommandArray $PythonCommand @('-m', 'venv', '.venv')
} else {
    Write-Ok ".venv already exists"
}

$script:VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$venvVersion = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($venvVersion -notin @('3.10', '3.11')) {
    throw ".venv uses Python $venvVersion. Delete .venv and rerun setup_5090.bat after installing Python 3.10 or 3.11."
}
Write-Ok ".venv Python version: $venvVersion"

Write-Step "Upgrading pip"
Invoke-VenvPython @('-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel')

Write-Step "Installing PyTorch for RTX 5090 / CUDA 12.8"
Invoke-VenvPython @('-m', 'pip', 'install', '--upgrade', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu128')

Write-Step "Installing DB9 toolkit requirements"
Invoke-VenvPython @('-m', 'pip', 'install', '-r', 'requirements.txt')

Write-Step "Checking CUDA from Python"
$cudaCheck = @'
import torch
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA runtime:', torch.version.cuda)
    print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
else:
    raise SystemExit('CUDA is not available from PyTorch. Check NVIDIA driver and PyTorch CUDA wheel.')
'@
$cudaCheck | & $VenvPython -
if ($LASTEXITCODE -ne 0) { throw "CUDA check failed" }

Write-Step "Preparing folders"
New-Item -ItemType Directory -Force datasets\raw | Out-Null
New-Item -ItemType Directory -Force datasets\processed\img | Out-Null
New-Item -ItemType Directory -Force datasets\processed\captions | Out-Null
New-Item -ItemType Directory -Force configs | Out-Null
New-Item -ItemType Directory -Force outputs | Out-Null

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Warn "Created .env from .env.example. Open .env and set GEMINI_API_KEY/HF_TOKEN if needed."
    }
} else {
    Write-Ok ".env already exists"
}

if (-not $SkipToolkit) {
    Write-Step "Cloning/updating ai-toolkit"
    if (-not (Test-Path "ai-toolkit")) {
        git clone --recurse-submodules https://github.com/ostris/ai-toolkit.git ai-toolkit
        if ($LASTEXITCODE -ne 0) { throw "ai-toolkit clone failed" }
    } else {
        Write-Ok "ai-toolkit already exists"
    }

    git -C ai-toolkit submodule update --init --recursive
    if ($LASTEXITCODE -ne 0) { throw "ai-toolkit submodule update failed" }

    if (Test-Path "ai-toolkit\requirements.txt") {
        Write-Step "Installing ai-toolkit requirements"
        Invoke-VenvPython @('-m', 'pip', 'install', '-r', 'ai-toolkit\requirements.txt')
    } else {
        Write-Warn "ai-toolkit requirements.txt not found; skipped"
    }
}

Write-Step "Running project verification"
Invoke-VenvPython @('verify.py')

Write-Step "Setup complete"
Write-Ok "Put training images in: datasets\raw"
Write-Ok "Activate env with: .\.venv\Scripts\Activate.ps1"
Write-Ok "Start notebooks with: jupyter notebook"

if ($StartJupyter) {
    Write-Step "Starting Jupyter Notebook"
    & (Join-Path $ProjectRoot ".venv\Scripts\jupyter.exe") notebook
}
