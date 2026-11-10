<#
    编译 ntt_gpu.exe。
    自己找 Visual Studio 和 CUDA，按本机显卡生成代码，不需要先开 Developer PowerShell。
#>
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$source = Join-Path $root 'ntt_gpu.cu'
$target = Join-Path $root 'ntt_gpu.exe'

# cl.exe 需要的路径都在 vcvars64.bat 里
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$install = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $install) {
    throw '找不到 Visual Studio C++ 生成工具，请安装“使用 C++ 的桌面开发”工作负载。'
}
$vcvars = Join-Path $install.Trim() 'VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}

# nvcc：先看 PATH，再看 CUDA 安装目录里版本号最大的那个
$nvcc = (Get-Command nvcc.exe -ErrorAction SilentlyContinue).Source
if (-not $nvcc) {
    $toolkit = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
    $candidates = Get-ChildItem -LiteralPath $toolkit -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | ForEach-Object { Join-Path $_.FullName 'bin\nvcc.exe' }
    $nvcc = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $nvcc) {
    throw '找不到 nvcc，请安装 CUDA Toolkit。'
}

# -arch=native 按本机显卡生成代码，换显卡后重新编译一次即可。
# 源码要带 UTF-8 BOM，否则 nvcc 按系统代码页读中文注释会报 missing closing quote。
& $nvcc $source -O2 '-Xcompiler=/O2,/wd4819' -std=c++17 -arch=native -o $target
if ($LASTEXITCODE -ne 0) {
    throw "nvcc 失败，退出码 $LASTEXITCODE"
}
Write-Host "编译完成：$target"
