# ntt-gpu

用 CUDA 的数论变换（NTT）乘两个超大十进制整数。

| 输入（每边位数） | 耗时 |
|---|---:|
| 6000 万 | 0.37 s |
| 2.68 亿 | 1.5 s |
| 5.37 亿（上限） | 3.3 s，结果 10.7 亿位 |

GTX 1660 SUPER 6 GB / CUDA 13.1 / i3-12100F，不含读写文件。

## 用法

```powershell
./build.ps1                      # 编译。脚本自己找 Visual Studio 和 CUDA，按本机显卡出代码
./ntt_gpu.exe                    # input.txt -> output.txt
./ntt_gpu.exe in.txt out.txt
./ntt_gpu.exe in.txt -           # 结果写到屏幕
type in.txt | ./ntt_gpu.exe - -  # 全走管道
./ntt_gpu.exe --bench 200000000  # 压测，不写文件
```

输入恰好两行十进制数，输出一行结果。退出码：0 成功，1 出错，2 参数错。

每次运行打印六行：

```
left_digits=60000000
right_digits=60000000
result_digits=119999999
transform_size=16777216
total_ms=375
stage_ms=input 27 init 127 gpu 117 crt 36 carry 46 output 20
```

## 上限

变换长度受素数限制。支持 `2^27` 次单位根的 32 位素数只有三个，所以 `2^27` 是天花板；
一个 limb 装八位十进制，单边最多 `4 × 2^27 = 5.37 亿位`。

显存每点占 3 个 `u32`（左右两边加旋转因子，共 12 字节），启动时按空闲显存的 75% 向下取到 2 的幂：

| 显存 | 最大变换 | 单边位数 |
|---:|---:|---:|
| 2 GB | 2^26 | 2.68 亿 |
| 6 GB | 2^27 | 5.37 亿 |

超出时启动阶段就报“显存不足：需要 X MiB，只有 Y MiB 可用”。

## 怎么算的

- 按 `10^8` 切 limb，三个素数下各做一次卷积，Garner 合成精确系数后进位成十进制。
- radix-4 蝶形。正变换 DIF、逆变换 DIT，中间不落回自然序，全程不需要位反转。
- 最内层 8192 点在共享内存里做完，只走一趟全局显存。
- 解析、格式化多线程；CUDA 初始化与读文件并行。
- 两行输入相同时只做一次正变换。

## 测试

```powershell
python test.py            # 22 个用例，秒级
python test.py --big      # 加 100 万位
python test.py --fuzz 200 # 加 200 组随机尺寸
python test.py --huge     # 加 5.37 亿位上限用例
```

标准答案是 Python 大整数，逐字节比较。

## 两个坑

- **源码是 UTF-8 带 BOM**。编辑时别把 BOM 去掉：nvcc 按系统代码页读源文件，没有 BOM 时中文注释会让它报 `missing closing quote`。
- **CUDA 13 起不支持 Maxwell（sm_50）及更早架构**，老卡得留在 CUDA 12.x。
