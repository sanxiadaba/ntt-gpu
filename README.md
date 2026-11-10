# ntt-gpu

使用 CUDA NTT 计算超大十进制整数乘法。

每个系数保存 4 位十进制数，分别在两个 NTT 素数下计算，最后通过 CRT 还原精确卷积结果。相比单模、单十进制位实现，它能正确处理包含大量 `9` 的长输入，同时显著降低变换长度和显存占用。

## 环境

- Windows + Visual Studio 2022（需 “使用 C++ 的桌面开发” 工作负载）
- CUDA Toolkit **12.x**（实测 12.6）
- 可选：GMP，用于编译 CPU 对照程序 `gmp_calc.cpp`

> **CUDA 13 不再支持 Maxwell（sm_50）及更早的架构**，最低只到 sm_75。因此想继续支持 GTX 750 Ti 一类的老卡，必须停在 CUDA 12.x。仓库中的 `build.ps1` 会同时为 **sm_50** 与 **sm_75** 生成代码，一个可执行文件即可在两种卡上运行。

## 构建

```powershell
./build.ps1          # 单文件内含 sm_50 + sm_75
./build.ps1 -Fast    # 只为本机 GPU 编译，开发时用
```

`build.ps1` 会自动定位 Visual Studio 与 CUDA，不要求预先打开 Developer PowerShell。它执行的等价命令是：

```powershell
nvcc ntt_gpu.cu -O2 -Xcompiler=/O2 -std=c++17 `
     -gencode arch=compute_50,code=sm_50 `
     -gencode arch=compute_75,code=sm_75 `
     -o ntt_gpu.exe
```

> 历史上这里只写了 `nvcc -O2 -Xcompiler=/O2 ntt_gpu.cu -o ntt_gpu.exe`。这条命令生成的是 nvcc 的默认架构（12.6 为 sm_52），在 GTX 750 Ti（sm_50）上会直接报
> `no kernel image is available for execution on the device`。
> 构建时请显式给出 `-gencode`，或直接使用 `build.ps1`。

## 运行

```powershell
tar -xf input.zip input.txt     # 仓库自带的 6000 万位基准
./ntt_gpu.exe                   # 默认 ./input.txt -> ./output.txt
./ntt_gpu.exe input.txt output.txt
./ntt_gpu.exe --help
```

程序只往 stdout 打诊断信息，结果写入输出文件：

```text
left_digits=60000000
right_digits=60000000
result_digits=119999999
transform_size=33554432
input_ms=466.0
gpu_phase_ms=1290.7
crt_carry_ms=234.5
output_ms=432.4
total_ms=2485.6
```

退出码：`0` 成功，`1` 输入或 GPU 错误，`2` 参数错误。

## 输入格式

- **恰好两行**，每行只含十进制数字，允许前导零
- 兼容 UTF-8 BOM、CRLF、单独 CR、末行没有换行符
- 其它情况（三行、空行、字母、空格）一律 `exit 1` 并在 stderr 说明原因

## 规模上限

两个素数各自支持到 `2^27` 的变换长度，这是数学上限。实际可用长度还要看显存：每个点需要 3 个 `u32`（left / right / 旋转因子表），即 12 字节。

程序启动时用 `cudaMemGetInfo` 取当前空闲显存，按 75% 预算向下取 2 的幂，再与 `2^27` 取小：

| 显存 | 可用变换长度 | 大致可算的位数（两侧相同） |
|---:|---:|---:|
| 2 GB（GTX 750 Ti） | `2^26` | 约 1.34 亿位 |
| 6 GB（GTX 1660 SUPER） | `2^27` | 约 2.68 亿位 |

超出时会在启动阶段直接报 “需要 X MiB / 可用 Y MiB”，而不是崩在 `cudaMalloc` 里。

## 基准

测试环境：GTX 750 Ti 2 GB、CUDA 12.6、i5-10400F、`input.zip`（两行各 6000 万位）。

| 阶段 | 优化前 | 现在 |
|---|---:|---:|
| 读入 + 解析 | 2894–3755 ms | 400–500 ms |
| GPU（6 次 NTT + 点乘） | 2503–2825 ms | 1280–1290 ms |
| CRT + 进位 | 337–386 ms | 190–250 ms |
| 写输出 | 639–1036 ms* | 430–850 ms* |
| **合计** | **6798–7631 ms** | **2320–2640 ms** |

\* 写盘耗时取决于存储。本机 `D:` 是机械盘，脏页回写被打满时同一份代码的 `output_ms` 可以到 8 秒，因此这一项只能当作数量级参考。

输出与原实现的 SHA-256 完全一致：

```text
A095C60E4EB753E9AB0B61D0C18F9C70176970BF1340D42FFC156118D622833E
```

历史上的 “GTX 1660 SUPER 上端到端 2.8–3.1 秒” 这一行没有保留可复现的环境与代码，因此不再作为基准引用；需要对比时请在本机重新测量。

### 基准输入本身不够狠

`input.zip` 里的两行数字**完全相同**（等于自乘），且只由 `1`、`2`、`3` 组成——它一个 `9` 都没有，永远压不到最长的进位链。要覆盖真正的边界，用生成器：

```powershell
python tests/gen_input.py --digits 1000000 --pattern all9 --out nines.txt
python tests/gen_input.py --digits 60000000 --pattern digits123 --out input.txt
```

支持 `all9`、`all1`、`sparse9`、`random`、`digits123`，`--seed` 固定即可复现。

## 测试

黑盒测试，用 Python 自带的大整数当对照：

```powershell
python tests/run_tests.py                # 秒级，22 个用例
python tests/run_tests.py --full         # 追加 100 万位用例
python tests/run_tests.py --benchmark    # 跑 input.zip 并校验 SHA-256
python tests/run_tests.py --errors-only  # 无 GPU 的机器也能跑
```

覆盖：0、前导零、全 `9`、10 的幂、不平衡输入、BOM/CRLF/CR、以及各种非法输入。

## 实现说明

- NTT 素数：`2281701377`（原根 3）、`2013265921`（原根 31），两者都支持到 `2^27`
- 系数基数：`10000`；两素数乘积 < 2^63，CRT 用 64 位算术
- 卷积系数上界 `(10^4-1)^2 · min(len1, len2)` 会在启动时与模数乘积比较，不满足直接报错
- **正变换用 DIF、逆变换用 DIT**：前者自然序进、位倒序出，后者位倒序进、自然序出，中间的点乘乘完仍然对得上，因此整个流程不需要任何位反转置换（早期版本每侧要做 3 次全数组随机置换，是主要瓶颈之一）
- 逆变换的旋转因子用 `w^-j = -w^(h-j)` 现算，不需要第二张表
- **模乘用 Barrett**（预存 `floor(2^64/p)`）：两个因子都小于模数时商只会小 1，一次条件减法即可，替代了 64 位取模（后者会被编译成 `I2F` + `MUFU.RCP` 一长串指令）
- 小尺寸的 11 级蝶形在共享内存里一次做完，只需要一趟全局访存；大尺寸的级才回全局内存
- 输入按 8 位一组解析（一次产出两个 limb），输出用 4 字节查表拷贝，两者都在 4 MiB 缓冲上批量写

## 已知限制

完整清单、证据与优化路线见 [REVIEW.md](REVIEW.md)。最要紧的几条：

- 仍是 radix-2：`2^25` 变换有 14 趟全局访存，radix-4 可以再砍一半
- CRT + 进位在 CPU 上单线程，约 200 ms
- 不平衡输入（例如 1 位 × 100 万位）仍会跑完整的 6 次 NTT，没有退化成 schoolbook
- `gmp_calc.cpp` 只改了代码，本机没有 GMP，**未编译验证**

## CPU 对照程序

```powershell
# 需先安装 GMP（如 MSYS2: pacman -S mingw-w64-x86_64-gmp）
g++ -O2 gmp_calc.cpp -lgmp -o gmp_calc.exe
./gmp_calc.exe                    # 读 ./input.txt，写 ./output.txt
./gmp_calc.exe in.txt out.txt
```

## 参考资料

`毛啸：再探快速傅里叶变换.pdf` 是仓库原有的算法参考资料。
