<#
.SYNOPSIS
    Builds ntt_gpu.exe.

.DESCRIPTION
    By default the executable carries code for both Maxwell (sm_50) and Turing
    (sm_75), so one binary runs on an old GTX 750 Ti and on a GTX 16/20 series
    card. Use -Fast to compile only for the GPU that is currently installed,
    which is what you want while developing.

    The script finds Visual Studio and the CUDA toolkit itself; it does not
    need a Developer PowerShell prompt.

.EXAMPLE
    ./build.ps1
    ./build.ps1 -Fast
#>
[CmdletBinding()]
param(
    [switch]$Fast,
    [string]$Output = 'ntt_gpu.exe'
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$source = Join-Path $root 'ntt_gpu.cu'
if (-not (Test-Path -LiteralPath $source)) {
    throw "Source not found: $source"
}

function Find-VcVars {
    $installer = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $installer) {
        $install = & $installer -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($install) {
            $candidate = Join-Path $install.Trim() 'VC\Auxiliary\Build\vcvars64.bat'
            if (Test-Path -LiteralPath $candidate) {
                return $candidate
            }
        }
    }
    foreach ($version in @('2022', '2019')) {
        foreach ($edition in @('Enterprise', 'Professional', 'Community', 'BuildTools')) {
            $candidate = "${env:ProgramFiles}\Microsoft Visual Studio\$version\$edition\VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path -LiteralPath $candidate) {
                return $candidate
            }
        }
    }
    return $null
}

function Import-VcEnvironment([string]$vcvars) {
    $lines = & $env:ComSpec /c "`"$vcvars`" >nul 2>&1 && set"
    foreach ($line in $lines) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
}

function Find-Nvcc {
    $onPath = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($onPath) {
        return $onPath.Source
    }
    $roots = @()
    if ($env:CUDA_PATH) {
        $roots += $env:CUDA_PATH
    }
    $toolkit = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path -LiteralPath $toolkit) {
        $roots += Get-ChildItem -LiteralPath $toolkit -Directory |
            Sort-Object Name -Descending |
            Select-Object -ExpandProperty FullName
    }
    foreach ($candidate in $roots) {
        $nvcc = Join-Path $candidate 'bin\nvcc.exe'
        if (Test-Path -LiteralPath $nvcc) {
            return $nvcc
        }
    }
    return $null
}

function Get-NativeArch {
    $capability = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null |
        Select-Object -First 1)
    if (-not $capability) {
        return $null
    }
    $capability = $capability.Trim()
    if ($capability -notmatch '^(\d+)\.(\d+)$') {
        return $null
    }
    return "$($Matches[1])$($Matches[2])"
}

$vcvars = Find-VcVars
if (-not $vcvars) {
    throw 'Visual Studio C++ build tools not found. Install "Desktop development with C++" or run this from a Developer PowerShell prompt.'
}
Import-VcEnvironment $vcvars

$nvcc = Find-Nvcc
if (-not $nvcc) {
    throw 'nvcc not found. Install the CUDA Toolkit or run this from a Developer PowerShell prompt.'
}

if ($Fast) {
    $arch = Get-NativeArch
    if (-not $arch) {
        throw 'Cannot detect the local GPU compute capability; drop -Fast to build for every supported architecture.'
    }
    $gencode = @('-gencode', "arch=compute_$arch,code=sm_$arch")
    Write-Host "target: sm_$arch (local GPU only)"
}
else {
    $gencode = @(
        '-gencode', 'arch=compute_50,code=sm_50',
        '-gencode', 'arch=compute_75,code=sm_75'
    )
    Write-Host 'target: sm_50 (Maxwell) + sm_75 (Turing), one binary'
}

$target = Join-Path $root $Output
$nvccArguments = @($source, '-O2', '-Xcompiler=/O2', '-std=c++17') + $gencode + @('-o', $target)

Write-Host "nvcc $($nvccArguments -join ' ')"
& $nvcc @nvccArguments
if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE"
}

Write-Host ("built {0} ({1:N0} bytes)" -f $target, (Get-Item -LiteralPath $target).Length)
