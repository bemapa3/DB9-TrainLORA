Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$form = New-Object System.Windows.Forms.Form
$form.Text = "DB9-Toolkit-Trainner Launcher"
$form.Size = New-Object System.Drawing.Size(980, 720)
$form.StartPosition = "CenterScreen"
$form.Font = New-Object System.Drawing.Font("Segoe UI", 9)

$title = New-Object System.Windows.Forms.Label
$title.Text = "DB9-Toolkit-Trainner - Local LoRA Training Launcher"
$title.Font = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold)
$title.AutoSize = $true
$title.Location = New-Object System.Drawing.Point(16, 12)
$form.Controls.Add($title)

$status = New-Object System.Windows.Forms.Label
$status.Text = "Project: $ProjectRoot"
$status.AutoSize = $true
$status.Location = New-Object System.Drawing.Point(18, 45)
$form.Controls.Add($status)

$commandsBox = New-Object System.Windows.Forms.TextBox
$commandsBox.Multiline = $true
$commandsBox.ReadOnly = $true
$commandsBox.ScrollBars = "Vertical"
$commandsBox.Font = New-Object System.Drawing.Font("Consolas", 9)
$commandsBox.Location = New-Object System.Drawing.Point(18, 78)
$commandsBox.Size = New-Object System.Drawing.Size(455, 255)
$form.Controls.Add($commandsBox)

$logBox = New-Object System.Windows.Forms.TextBox
$logBox.Multiline = $true
$logBox.ReadOnly = $true
$logBox.ScrollBars = "Vertical"
$logBox.Font = New-Object System.Drawing.Font("Consolas", 9)
$logBox.Location = New-Object System.Drawing.Point(18, 350)
$logBox.Size = New-Object System.Drawing.Size(925, 315)
$form.Controls.Add($logBox)

function Append-Log($Text) {
    $logBox.AppendText("[$(Get-Date -Format HH:mm:ss)] $Text`r`n")
    $logBox.SelectionStart = $logBox.Text.Length
    $logBox.ScrollToCaret()
    [System.Windows.Forms.Application]::DoEvents()
}

function Set-Commands($Text) {
    $commandsBox.Text = $Text.Trim() -replace "`n", "`r`n"
}

function Run-Command($Title, $Command) {
    Append-Log "START: $Title"
    Append-Log "> $Command"
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "powershell.exe"
    $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -Command `$ErrorActionPreference='Stop'; cd '$ProjectRoot'; $Command"
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi
    [void]$p.Start()
    $stdout = $p.StandardOutput.ReadToEnd()
    $stderr = $p.StandardError.ReadToEnd()
    $p.WaitForExit()
    if ($stdout) { Append-Log $stdout.TrimEnd() }
    if ($stderr) { Append-Log $stderr.TrimEnd() }
    if ($p.ExitCode -eq 0) {
        Append-Log "DONE: $Title"
    } else {
        Append-Log "FAILED: $Title (exit $($p.ExitCode))"
    }
}

function Start-Detached($Title, $Command) {
    Append-Log "OPEN: $Title"
    Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -NoExit -Command cd '$ProjectRoot'; $Command"
}

$commands = @{
    FullSetup = @"
cd "$ProjectRoot"
git pull origin main
.\setup_5090.bat
"@
    Verify = @"
cd "$ProjectRoot"
.\.venv\Scripts\python.exe verify.py
"@
    Jupyter = @"
cd "$ProjectRoot"
.\.venv\Scripts\python.exe -m jupyter notebook
"@
    DataClean = @"
cd "$ProjectRoot"
.\.venv\Scripts\python.exe scripts\data_clean.py --input datasets\raw --output datasets\processed\img --min-res 512 --max-res 4096 --format jpg
"@
    Caption = @"
cd "$ProjectRoot"
.\.venv\Scripts\python.exe scripts\caption_gemini.py --input datasets\processed\img --output datasets\processed\captions --prompt "Mo ta chi tiet hinh anh nay"
"@
    Train = @"
cd "$ProjectRoot\ai-toolkit"
..\.venv\Scripts\python.exe run.py ..\configs\db9_toolkit_trainner.yaml
"@
}

$commandsBox.Text = @"
Recommended flow:
1. Full setup once
2. Put images into datasets\raw
3. Edit .env if using Gemini/Hugging Face
4. Open Jupyter and run notebooks 1 -> 5

Useful commands are shown here when you click a button.
"@

function Add-Button($Text, $X, $Y, $W, $Handler) {
    $btn = New-Object System.Windows.Forms.Button
    $btn.Text = $Text
    $btn.Location = New-Object System.Drawing.Point($X, $Y)
    $btn.Size = New-Object System.Drawing.Size($W, 34)
    $btn.Add_Click($Handler)
    $form.Controls.Add($btn)
    return $btn
}

Add-Button "1. Full Setup" 495 78 140 {
    Set-Commands $commands.FullSetup
    Start-Detached "Full setup" ".\setup_5090.bat"
} | Out-Null

Add-Button "Verify" 645 78 120 {
    Set-Commands $commands.Verify
    Run-Command "Verify project" ".\.venv\Scripts\python.exe verify.py"
} | Out-Null

Add-Button "Open Jupyter" 775 78 140 {
    Set-Commands $commands.Jupyter
    Start-Detached "Jupyter" ".\.venv\Scripts\python.exe -m jupyter notebook"
} | Out-Null

Add-Button "Open Data Folder" 495 122 140 {
    Set-Commands "explorer `"$ProjectRoot\datasets\raw`""
    New-Item -ItemType Directory -Force "$ProjectRoot\datasets\raw" | Out-Null
    Start-Process explorer.exe "$ProjectRoot\datasets\raw"
    Append-Log "Opened datasets\raw"
} | Out-Null

Add-Button "Open Output" 645 122 120 {
    Set-Commands "explorer `"$ProjectRoot\outputs`""
    New-Item -ItemType Directory -Force "$ProjectRoot\outputs" | Out-Null
    Start-Process explorer.exe "$ProjectRoot\outputs"
    Append-Log "Opened outputs"
} | Out-Null

Add-Button "Edit .env" 775 122 140 {
    Set-Commands "notepad `"$ProjectRoot\.env`""
    if (-not (Test-Path "$ProjectRoot\.env") -and (Test-Path "$ProjectRoot\.env.example")) {
        Copy-Item "$ProjectRoot\.env.example" "$ProjectRoot\.env"
    }
    Start-Process notepad.exe "$ProjectRoot\.env"
    Append-Log "Opened .env"
} | Out-Null

Add-Button "Clean Images" 495 166 140 {
    Set-Commands $commands.DataClean
    Start-Detached "Clean images" ".\.venv\Scripts\python.exe scripts\data_clean.py --input datasets\raw --output datasets\processed\img --min-res 512 --max-res 4096 --format jpg"
} | Out-Null

Add-Button "Gemini Caption" 645 166 120 {
    Set-Commands $commands.Caption
    Start-Detached "Gemini caption" ".\.venv\Scripts\python.exe scripts\caption_gemini.py --input datasets\processed\img --output datasets\processed\captions --prompt 'Mo ta chi tiet hinh anh nay'"
} | Out-Null

Add-Button "Train" 775 166 140 {
    Set-Commands $commands.Train
    Start-Detached "Train" "cd ai-toolkit; ..\.venv\Scripts\python.exe run.py ..\configs\db9_toolkit_trainner.yaml"
} | Out-Null

Add-Button "Open Notebooks" 495 210 140 {
    Set-Commands "explorer `"$ProjectRoot\notebooks`""
    Start-Process explorer.exe "$ProjectRoot\notebooks"
    Append-Log "Opened notebooks"
} | Out-Null

Add-Button "Git Pull" 645 210 120 {
    Set-Commands "cd `"$ProjectRoot`"`ngit pull origin main"
    Run-Command "Git pull" "git pull origin main"
} | Out-Null

Add-Button "CUDA Check" 775 210 140 {
    $cmd = ".\.venv\Scripts\python.exe -c `"import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')`""
    Set-Commands $cmd
    Run-Command "CUDA check" $cmd
} | Out-Null

$note = New-Object System.Windows.Forms.Label
$note.Text = "Tip: Full Setup, Jupyter, Clean, Caption, and Train open in separate consoles so long jobs keep running visibly."
$note.AutoSize = $true
$note.Location = New-Object System.Drawing.Point(495, 260)
$form.Controls.Add($note)

Append-Log "Launcher ready."
[void]$form.ShowDialog()
