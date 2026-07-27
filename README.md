# ntt-gpu

使用 CUDA NTT 计算超大十进制整数乘法。

当前实现每个系数保存 4 位十进制数，并分别在两个 NTT 素数下计算，最后通过 CRT 还原精确卷积结果。相比最初的单模、单十进制位实现，它可以正确处理包含大量 `9` 的长输入，同时显著降低变换长度和显存占用。

## 环境

- NVIDIA GPU 与 CUDA Toolkit
- Windows 上需要 Visual Studio C++ Build Tools
- 可选：GMP，用于编译 CPU 对照程序 `gmp_cacl.cpp`

请先进入 Visual Studio x64 Developer PowerShell，再执行：

```powershell
nvcc -O2 -Xcompiler=/O2 ntt_gpu.cu -o ntt_gpu.exe
```

## 运行

`input.zip` 内含基准输入。解压后运行：

```powershell
tar -xf input.zip input.txt
./ntt_gpu.exe
```

也可以显式指定输入和输出路径：

```powershell
./ntt_gpu.exe input.txt output.txt
```

输入必须恰好包含两行非负十进制整数。程序会输出输入位数、变换长度以及输入、GPU、CRT/进位、输出和总耗时。

## 本机基准

测试环境：GTX 1660 SUPER 6 GB、CUDA 13.1，两行各 6000 万位输入。

| 实现 | NTT 长度 | 端到端耗时 |
|---|---:|---:|
| 原始单模、每系数 1 位 | `2^27` | 约 16-27 秒 |
| 双模 CRT、每系数 4 位 | `2^25` | 约 2.8-3.1 秒 |

优化版输出与原版基准输出的 SHA-256 完全一致：

```text
A095C60E4EB753E9AB0B61D0C18F9C70176970BF1340D42FFC156118D622833E
```

具体耗时会随 GPU 状态、磁盘缓存和系统负载变化。

## 实现说明

- NTT 素数：`2281701377`、`2013265921`
- 系数基数：`10000`
- 最大 NTT 长度：`2^27`
- CUDA API、内核启动、输入范围和 CRT 系数上界均有错误检查
- 输出采用 4 MiB 缓冲区批量写入

`毛啸：再探快速傅里叶变换.pdf` 是仓库原有的算法参考资料。
