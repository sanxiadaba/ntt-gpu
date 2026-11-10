<#
.SYNOPSIS
    编译 ntt_gpu.exe。

.DESCRIPTION
    默认按本机显卡的算力（nvidia-smi 报的 compute_cap）生成代码，所以编出来的
    二进制一定能在自己的卡上跑。没有 GPU 的机器（比如 CI）退回 sm_75。

    脚本自己找 Visual Studio 与 CUDA，不需要先开 Developer PowerShell。

.EXAMPLE
    ./build.ps1              # 按本机显卡编译
    ./build.ps1 -Arch 86     # 指定算力，编 sm_86
#>
[CmdletBinding()]
param(
    [string]$Arch = '',
    [string]$Output = 'ntt_gpu.exe'
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$source = Join-Path $root 'ntt_gpu.cu'
if (-not (Test-Path -LiteralPath $source)) {
    throw "找不到源文件：$source"
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

# 本机显卡的算力，形如 "75"；取不到就返回空。
function Get-LocalArch {
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
    throw '找不到 Visual Studio C++ 生成工具。请安装「使用 C++ 的桌面开发」工作负载。'
}
Import-VcEnvironment $vcvars

$nvcc = Find-Nvcc
if (-not $nvcc) {
    throw '找不到 nvcc。请安装 CUDA Toolkit。'
}

if ($Arch) {
    $arch = $Arch
}
else {
    $arch = Get-LocalArch
    if (-not $arch) {
        $arch = '75'
        Write-Host '未检测到本机显卡，退回 sm_75'
    }
}

# 先问 nvcc 支不支持这个算力，比让它自己报 depends 错误清楚得多。
$supported = & $nvcc --list-gpu-arch 2>$null | ForEach-Object { $_.Trim() }
if ($supported -and $supported -notcontains "compute_$arch") {
    $available = ($supported -join ', ')
    throw "当前 CUDA（$(& $nvcc --version | Select-String -Pattern 'release' | ForEach-Object { $_.ToString().Trim() })）不支持 sm_$arch。可用的算力：$available。若是较老的 Maxwell 显卡，需要 CUDA 12.x。"
}

Write-Host "target: sm_$arch"
$target = Join-Path $root $Output
# 源码是 UTF-8 带 BOM，nvcc 能正确读中文注释与中文字符串。注意不要加
# -Xcompiler=/utf-8：它会让 nvcc 自己的前端改写多字节字面量的判定，报
# "missing closing quote"。C4819 是中文代码页下读 CUDA 头文件的无害告警，直接关掉。
$nvccArguments = @(
    $source, '-O2', '-Xcompiler=/O2,/wd4819', '-std=c++17',
    '-gencode', "arch=compute_$arch,code=sm_$arch",
    '-o', $target
)

Write-Host "nvcc $($nvccArguments -join ' ')"
& $nvcc @nvccArguments
if ($LASTEXITCODE -ne 0) {
    throw "nvcc 失败，退出码 $LASTEXITCODE"
}

Write-Host ("built {0} ({1:N0} bytes)" -f $target, (Get-Item -LiteralPath $target).Length)
