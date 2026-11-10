// ntt_gpu.cu —— 用 CUDA NTT 计算两个超大十进制整数的乘积
//
// 做法：十进制按 10^8 切成 limb（八位一个），在三个 32 位 NTT 素数下各做一次
// 卷积，再用 Garner 把每个卷积系数还原成精确整数，从低位向高位进位成十进制。
//
// 为什么是三个素数：基数 10^8 时卷积系数上界是 (10^8-1)^2 * min(limb 数)，
// 三个素数乘积约 1.79e28，够用到单边 5.37 亿位。系数最大约 6.8e23，超过 64 位，
// 用三个 32 位字表示（< 2^94）。
//
// 变换方向：正变换用 DIF（自然序进、倒序出），逆变换用 DIT（倒序进、自然序出），
// 两者配对后频域点乘仍落在倒序上，整个流程不需要位翻转置换。

#define _CRT_SECURE_NO_WARNINGS

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

namespace {

using u32 = uint32_t;
using u64 = uint64_t;
using Clock = std::chrono::steady_clock;

constexpr char kDefaultInput[] = "./input.txt";
constexpr char kDefaultOutput[] = "./output.txt";

// 基数 10^8：一个 limb 装八位十进制。
constexpr u32 kBase = 100000000u;
constexpr u32 kBaseDigits = 8;

constexpr u32 kBlockSize = 256;
constexpr u32 kTwiddleWorkers = 8192;
constexpr u32 kMaxTransform = 1u << 27;
constexpr u32 kMaxTileLog = 13;   // 共享内存 tile 深度：2^13 点 = 32 KB
constexpr u32 kMaxLevels = 16;

// 每个点占 3 个 u32：left、right、旋转因子表。
constexpr size_t kBytesPerPoint = 3 * sizeof(u32);

// 三个素数都支持 2^27 次单位根，这是 32 位算术下变换长度的天花板。
constexpr u32 kP1 = 2013265921u;  // 15 * 2^27 + 1，原根 31
constexpr u32 kG1 = 31u;
constexpr u32 kP2 = 2281701377u;  // 17 * 2^27 + 1，原根 3
constexpr u32 kG2 = 3u;
constexpr u32 kP3 = 3892314113u;  // 29 * 2^27 + 1，原根 3
constexpr u32 kG3 = 3u;

constexpr u64 kPrime12 = static_cast<u64>(kP1) * kP2;  // < 2^63

struct Mod {
    u32 value;
    u32 imaginary;   // 4 次单位根 i = root^((p-1)/4)，i^2 = -1
    u64 reciprocal;  // floor(2^64 / p)
};

// ---------------------------------------------------------------------------
// 模运算
// ---------------------------------------------------------------------------

void cuda_check(cudaError_t error, const char* expression, const char* file, int line) {
    if (error == cudaSuccess) {
        return;
    }
    throw std::runtime_error(
        std::string("CUDA 调用失败于 ") + file + ":" + std::to_string(line) +
        " 的 " + expression + "：" + cudaGetErrorString(error));
}

#define CUDA_CHECK(expression) cuda_check((expression), #expression, __FILE__, __LINE__)
#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

__device__ __forceinline__ u32 multiply_mod(u32 left, u32 right, const Mod& mod) {
    const u64 product = static_cast<u64>(left) * right;
    const u64 quotient = __umul64hi(product, mod.reciprocal);
    // 商最多小 1，余数可能落在 [p, 2p)；先按 64 位比较再截断，否则 2p > 2^32
    // 的素数会被截断坏。
    u64 rest = product - quotient * mod.value;
    if (rest >= mod.value) {
        rest -= mod.value;
    }
    return static_cast<u32>(rest);
}

// 两个操作数都小于 p：先跟 p-right 比一次，加法路径永远不会越过 p。
__device__ __forceinline__ u32 add_mod(u32 left, u32 right, u32 modulus) {
    const u32 gap = modulus - right;
    return left >= gap ? left - gap : left + right;
}

__device__ __forceinline__ u32 sub_mod(u32 left, u32 right, u32 modulus) {
    return left >= right ? left - right : modulus - (right - left);
}

__device__ u32 pow_mod_device(u32 base, u32 exponent, const Mod& mod) {
    u32 result = 1u;
    while (exponent != 0) {
        if ((exponent & 1u) != 0) {
            result = multiply_mod(result, base, mod);
        }
        base = multiply_mod(base, base, mod);
        exponent >>= 1;
    }
    return result;
}

u32 pow_mod_host(u32 base, u32 exponent, u32 modulus) {
    u64 result = 1;
    u64 current = base;
    while (exponent != 0) {
        if ((exponent & 1u) != 0) {
            result = result * current % modulus;
        }
        current = current * current % modulus;
        exponent >>= 1;
    }
    return static_cast<u32>(result);
}

Mod make_mod(u32 prime, u32 generator) {
    Mod mod{};
    mod.value = prime;
    mod.reciprocal = 0xFFFFFFFFFFFFFFFFull / prime;
    mod.imaginary = pow_mod_host(generator, (prime - 1) / 4, prime);
    return mod;
}

// ---------------------------------------------------------------------------
// 级计划
//
// 变换长度 n = 2^L 拆成一串「级」，每级 radix-4（吃 2 个二进制位）或 radix-2。
// log_stride 是该级的块大小对数：radix-4 按四分之一长度分块，radix-2 按一半。
//
// 正变换按级序走（大步长 -> 小步长），逆变换把同样的级反过来走，一条计划
// 描述两个方向，全程不需要位反转。
// ---------------------------------------------------------------------------

struct Level {
    u32 radix;          // 2 或 4
    u32 log_stride;
    u32 forward_table;
    u32 inverse_table;
};

struct Plan {
    std::array<Level, kMaxLevels> levels{};
    u32 count = 0;
    u32 tile_begin = 0;
    u32 tile_log = 0;
};

u32 integer_log2(u32 value) {
    u32 result = 0;
    while ((1u << result) < value) {
        ++result;
    }
    return result;
}

Plan make_plan(u32 log_n) {
    Plan plan;
    u32 remaining = log_n;
    while (remaining > 0) {
        if (plan.count >= kMaxLevels) {
            throw std::runtime_error("变换长度超出级表容量");
        }
        if (remaining >= 2) {
            plan.levels[plan.count++] = {4, remaining - 2, 0, 0};
            remaining -= 2;
        } else {
            plan.levels[plan.count++] = {2, remaining - 1, 0, 0};
            remaining -= 1;
        }
    }

    u32 units = 0;
    u32 index = plan.count;
    while (index > 0) {
        const u32 take = plan.levels[index - 1].radix == 4 ? 2u : 1u;
        if (units + take > kMaxTileLog) {
            break;
        }
        units += take;
        --index;
    }
    plan.tile_begin = index;
    plan.tile_log = units;

    u32 offset = 0;
    for (u32 i = 0; i < plan.count; ++i) {
        plan.levels[i].forward_table = offset;
        offset += 1u << plan.levels[i].log_stride;
    }
    for (u32 i = 0; i < plan.count; ++i) {
        plan.levels[i].inverse_table = offset;
        offset += 1u << plan.levels[i].log_stride;
    }
    return plan;
}

// ---------------------------------------------------------------------------
// 核函数
// ---------------------------------------------------------------------------

// 生成一级旋转因子表：tw[j] = root^j。每个 worker 先求一次幂再往后递推。
__global__ void generate_twiddles(
    u32* __restrict__ twiddles,
    u32 count,
    u32 root,
    Mod mod,
    u32 worker_count) {
    const u32 worker = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker >= worker_count) {
        return;
    }
    const u32 chunk = count / worker_count;
    const u32 begin = worker * chunk;
    u32 value = pow_mod_device(root, begin, mod);
    for (u32 offset = 0; offset < chunk; ++offset) {
        twiddles[begin + offset] = value;
        value = multiply_mod(value, root, mod);
    }
}

// radix-4 DIF 蝶形（正变换）。a0..a3 是相隔 stride 的四个点。
__device__ __forceinline__ void butterfly4_forward(
    u32& a0, u32& a1, u32& a2, u32& a3,
    u32 w1, u32 w2, u32 w3, const Mod& mod) {
    const u32 t0 = add_mod(a0, a2, mod.value);
    const u32 t1 = sub_mod(a0, a2, mod.value);
    const u32 t2 = add_mod(a1, a3, mod.value);
    const u32 t3 = sub_mod(a1, a3, mod.value);
    const u32 i3 = multiply_mod(t3, mod.imaginary, mod);
    a0 = add_mod(t0, t2, mod.value);
    a1 = multiply_mod(add_mod(t1, i3, mod.value), w1, mod);
    a2 = multiply_mod(sub_mod(t0, t2, mod.value), w2, mod);
    a3 = multiply_mod(sub_mod(t1, i3, mod.value), w3, mod);
}

// radix-4 DIT 蝶形（逆变换）：上面那个蝶形的逆运算，差 4 倍因子，由 1/n 补上。
__device__ __forceinline__ void butterfly4_inverse(
    u32& a0, u32& a1, u32& a2, u32& a3,
    u32 w1, u32 w2, u32 w3, u32 negative_imaginary, const Mod& mod) {
    const u32 s1 = multiply_mod(a1, w1, mod);
    const u32 s2 = multiply_mod(a2, w2, mod);
    const u32 s3 = multiply_mod(a3, w3, mod);
    const u32 t0 = add_mod(a0, s2, mod.value);
    const u32 t2 = sub_mod(a0, s2, mod.value);
    const u32 t1 = add_mod(s1, s3, mod.value);
    const u32 t3 = multiply_mod(sub_mod(s1, s3, mod.value), negative_imaginary, mod);
    a0 = add_mod(t0, t1, mod.value);
    a2 = sub_mod(t0, t1, mod.value);
    a1 = add_mod(t2, t3, mod.value);
    a3 = sub_mod(t2, t3, mod.value);
}

// 一整级 radix-4 蝶形。蝶形之间互不相干，直接铺在全局内存上做。
template <bool kForward>
__global__ void level4_kernel(
    u32* __restrict__ data,
    const u32* __restrict__ twiddles,
    u32 log_stride,
    Mod mod,
    u32 butterfly_count) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= butterfly_count) {
        return;
    }
    const u32 stride = 1u << log_stride;
    const u32 offset = index & (stride - 1u);
    // 组内第 offset 个蝶形，它的首点是 4*index - 3*offset。
    const u32 base = (index << 2) - 3u * offset;
    u32 a0 = data[base];
    u32 a1 = data[base + stride];
    u32 a2 = data[base + 2u * stride];
    u32 a3 = data[base + 3u * stride];

    // 只读 w^j，w^2j 与 w^3j 各用一次模乘推出：这里访存比算术贵。
    const u32 w1 = twiddles[offset];
    const u32 w2 = multiply_mod(w1, w1, mod);
    const u32 w3 = multiply_mod(w2, w1, mod);

    if constexpr (kForward) {
        butterfly4_forward(a0, a1, a2, a3, w1, w2, w3, mod);
    } else {
        butterfly4_inverse(a0, a1, a2, a3, w1, w2, w3, mod.value - mod.imaginary, mod);
    }

    data[base] = a0;
    data[base + stride] = a1;
    data[base + 2u * stride] = a2;
    data[base + 3u * stride] = a3;
}

// 一整级 radix-2 蝶形，只在变换长度对数为奇数时出现一次。
template <bool kForward>
__global__ void level2_kernel(
    u32* __restrict__ data,
    const u32* __restrict__ twiddles,
    u32 log_stride,
    Mod mod,
    u32 butterfly_count) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= butterfly_count) {
        return;
    }
    const u32 stride = 1u << log_stride;
    const u32 offset = index & (stride - 1u);
    const u32 left_index = (index << 1) - offset;
    const u32 right_index = left_index + stride;
    const u32 left = data[left_index];
    const u32 right = data[right_index];
    const u32 w = twiddles[offset];

    if constexpr (kForward) {
        data[left_index] = add_mod(left, right, mod.value);
        data[right_index] = multiply_mod(sub_mod(left, right, mod.value), w, mod);
    } else {
        const u32 scaled = multiply_mod(right, w, mod);
        data[left_index] = add_mod(left, scaled, mod.value);
        data[right_index] = sub_mod(left, scaled, mod.value);
    }
}

// 共享内存里要跑的那几级，级数很少（最多 6 级），按值传进核函数。
constexpr u32 kMaxTileLevels = kMaxLevels / 2;

struct TilePlan {
    u32 count;
    u32 log_size;
    u32 radix[kMaxTileLevels];
    u32 log_stride[kMaxTileLevels];
    u32 table[kMaxTileLevels];
};

// 最内层若干级：一次读进共享内存，全部做完再写回，只花一趟全局访存。
template <bool kForward>
__global__ void tile_kernel(
    u32* __restrict__ data,
    const u32* __restrict__ twiddles,
    TilePlan plan,
    Mod mod) {
    extern __shared__ u32 tile[];
    const u32 tile_size = 1u << plan.log_size;
    const u32 base_index = blockIdx.x * tile_size;

    for (u32 index = threadIdx.x; index < tile_size; index += blockDim.x) {
        tile[index] = data[base_index + index];
    }
    __syncthreads();

    for (u32 step = 0; step < plan.count; ++step) {
        const u32 radix = plan.radix[step];
        const u32 stride = 1u << plan.log_stride[step];
        const u32* table = twiddles + plan.table[step];
        const u32 butterfly_count = tile_size / radix;

        for (u32 index = threadIdx.x; index < butterfly_count; index += blockDim.x) {
            const u32 offset = index & (stride - 1u);
            const u32 position = index * radix - (radix - 1u) * offset;
            if (radix == 4) {
                u32 a0 = tile[position];
                u32 a1 = tile[position + stride];
                u32 a2 = tile[position + 2u * stride];
                u32 a3 = tile[position + 3u * stride];
                const u32 w1 = table[offset];
                const u32 w2 = multiply_mod(w1, w1, mod);
                const u32 w3 = multiply_mod(w2, w1, mod);
                if constexpr (kForward) {
                    butterfly4_forward(a0, a1, a2, a3, w1, w2, w3, mod);
                } else {
                    butterfly4_inverse(a0, a1, a2, a3, w1, w2, w3,
                                       mod.value - mod.imaginary, mod);
                }
                tile[position] = a0;
                tile[position + stride] = a1;
                tile[position + 2u * stride] = a2;
                tile[position + 3u * stride] = a3;
            } else {
                const u32 left = tile[position];
                const u32 right = tile[position + stride];
                const u32 w = table[offset];
                if constexpr (kForward) {
                    tile[position] = add_mod(left, right, mod.value);
                    tile[position + stride] =
                        multiply_mod(sub_mod(left, right, mod.value), w, mod);
                } else {
                    const u32 scaled = multiply_mod(right, w, mod);
                    tile[position] = add_mod(left, scaled, mod.value);
                    tile[position + stride] = sub_mod(left, scaled, mod.value);
                }
            }
        }
        __syncthreads();
    }

    for (u32 index = threadIdx.x; index < tile_size; index += blockDim.x) {
        data[base_index + index] = tile[index];
    }
}

// 频域点乘，顺带把逆变换的 1/n 缩放乘进去，省掉独立的一趟。
template <bool kSquare>
__global__ void pointwise_kernel(
    u32* __restrict__ left,
    const u32* __restrict__ right,
    u32 size,
    u32 scale,
    Mod mod) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }
    const u32 factor = kSquare ? left[index] : right[index];
    left[index] = multiply_mod(multiply_mod(left[index], factor, mod), scale, mod);
}

// ---------------------------------------------------------------------------
// 显存缓冲
// ---------------------------------------------------------------------------

class GpuBuffers {
public:
    explicit GpuBuffers(u32 size)
        : size_(size), bytes_(static_cast<size_t>(size) * sizeof(u32)) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const size_t required = 3 * bytes_;
        if (required > free_bytes) {
            throw std::runtime_error(
                "显存不足：需要 " + std::to_string(required >> 20) + " MiB，只有 " +
                std::to_string(free_bytes >> 20) + " MiB 可用，请缩短输入");
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&left_), bytes_));
        try {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&right_), bytes_));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&twiddles_), bytes_));
        } catch (...) {
            release();
            throw;
        }
    }

    GpuBuffers(const GpuBuffers&) = delete;
    GpuBuffers& operator=(const GpuBuffers&) = delete;

    ~GpuBuffers() { release(); }

    u32* left() const { return left_; }
    u32* right() const { return right_; }
    u32* twiddles() const { return twiddles_; }
    size_t bytes() const { return bytes_; }
    u32 size() const { return size_; }

private:
    void release() noexcept {
        if (left_ != nullptr) cudaFree(left_);
        if (right_ != nullptr) cudaFree(right_);
        if (twiddles_ != nullptr) cudaFree(twiddles_);
        left_ = nullptr;
        right_ = nullptr;
        twiddles_ = nullptr;
    }

    u32 size_;
    size_t bytes_;
    u32* left_ = nullptr;
    u32* right_ = nullptr;
    u32* twiddles_ = nullptr;
};

// 锁页（pinned）主机缓冲：分页内存做 D2H 只有 1.8 GB/s 左右，锁页能到 10 GB/s 以上。
class PinnedBuffer {
public:
    explicit PinnedBuffer(size_t count) : data_(nullptr) {
        if (count != 0) {
            CUDA_CHECK(cudaHostAlloc(
                reinterpret_cast<void**>(&data_), count * sizeof(u32), cudaHostAllocDefault));
        }
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    ~PinnedBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }

    u32* data() const { return data_; }

private:
    u32* data_;
};

u32 grid_size(u32 item_count) {
    return (item_count + kBlockSize - 1) / kBlockSize;
}

void prepare_twiddles(GpuBuffers& buffers, const Plan& plan, const Mod& mod, u32 generator) {
    for (u32 index = 0; index < plan.count; ++index) {
        const Level& level = plan.levels[index];
        const u32 count = 1u << level.log_stride;
        const u32 width = level.radix * count;  // 2*stride 或 4*stride
        const u32 root = pow_mod_host(generator, (mod.value - 1) / width, mod.value);
        const u32 inverse_root = pow_mod_host(root, mod.value - 2, mod.value);
        const u32 workers = std::min(count, kTwiddleWorkers);
        generate_twiddles<<<grid_size(workers), kBlockSize>>>(
            buffers.twiddles() + level.forward_table, count, root, mod, workers);
        CUDA_KERNEL_CHECK();
        generate_twiddles<<<grid_size(workers), kBlockSize>>>(
            buffers.twiddles() + level.inverse_table, count, inverse_root, mod, workers);
        CUDA_KERNEL_CHECK();
    }
}

TilePlan make_tile_plan(const Plan& plan, bool forward) {
    TilePlan tile{};
    tile.log_size = plan.tile_log;
    tile.count = plan.tile_begin < plan.count ? plan.count - plan.tile_begin : 0;
    for (u32 step = 0; step < tile.count; ++step) {
        // 正变换按原顺序做到底；逆变换要把级倒过来走。
        const u32 index = forward ? plan.tile_begin + step : plan.count - 1 - step;
        const Level& level = plan.levels[index];
        tile.radix[step] = level.radix;
        tile.log_stride[step] = level.log_stride;
        tile.table[step] = forward ? level.forward_table : level.inverse_table;
    }
    return tile;
}

// 正变换：先做 tile 之外的级（大步长，从前往后），再进共享内存做最内层。
void run_forward(u32* data, const GpuBuffers& buffers, const Plan& plan, const Mod& mod) {
    for (u32 index = 0; index < plan.tile_begin; ++index) {
        const Level& level = plan.levels[index];
        const u32* table = buffers.twiddles() + level.forward_table;
        if (level.radix == 4) {
            const u32 butterflies = buffers.size() / 4;
            level4_kernel<true><<<grid_size(butterflies), kBlockSize>>>(
                data, table, level.log_stride, mod, butterflies);
        } else {
            const u32 butterflies = buffers.size() / 2;
            level2_kernel<true><<<grid_size(butterflies), kBlockSize>>>(
                data, table, level.log_stride, mod, butterflies);
        }
        CUDA_KERNEL_CHECK();
    }
    if (plan.tile_log > 0) {
        const TilePlan tile = make_tile_plan(plan, true);
        tile_kernel<true><<<buffers.size() >> plan.tile_log, kBlockSize,
                            sizeof(u32) << plan.tile_log>>>(
            data, buffers.twiddles(), tile, mod);
        CUDA_KERNEL_CHECK();
    }
}

// 逆变换：先把最内层倒着做完，再做大步长的级（从后往前）。
void run_inverse(u32* data, const GpuBuffers& buffers, const Plan& plan, const Mod& mod) {
    if (plan.tile_log > 0) {
        const TilePlan tile = make_tile_plan(plan, false);
        tile_kernel<false><<<buffers.size() >> plan.tile_log, kBlockSize,
                             sizeof(u32) << plan.tile_log>>>(
            data, buffers.twiddles(), tile, mod);
        CUDA_KERNEL_CHECK();
    }
    for (u32 remaining = plan.tile_begin; remaining-- > 0;) {
        const Level& level = plan.levels[remaining];
        const u32* table = buffers.twiddles() + level.inverse_table;
        if (level.radix == 4) {
            const u32 butterflies = buffers.size() / 4;
            level4_kernel<false><<<grid_size(butterflies), kBlockSize>>>(
                data, table, level.log_stride, mod, butterflies);
        } else {
            const u32 butterflies = buffers.size() / 2;
            level2_kernel<false><<<grid_size(butterflies), kBlockSize>>>(
                data, table, level.log_stride, mod, butterflies);
        }
        CUDA_KERNEL_CHECK();
    }
}

// 把 limb 铺到显存：只清空 limb 之后的那一段，前面的会被 memcpy 覆盖。
void upload_limbs(u32* data, const std::vector<u32>& limbs, size_t bytes) {
    const size_t used = limbs.size() * sizeof(u32);
    if (bytes > used) {
        CUDA_CHECK(cudaMemset(reinterpret_cast<char*>(data) + used, 0, bytes - used));
    }
    CUDA_CHECK(cudaMemcpy(data, limbs.data(), used, cudaMemcpyHostToDevice));
}

// 一个模数下的完整卷积：两个正变换、频域点乘（含 1/n）、一个逆变换。
void run_modulus(
    GpuBuffers& buffers,
    const std::vector<u32>& left,
    const std::vector<u32>& right,
    const Mod& mod,
    u32 generator,
    bool square,
    const Plan& plan,
    size_t coefficient_count,
    u32* residues) {
    upload_limbs(buffers.left(), left, buffers.bytes());
    if (!square) {
        upload_limbs(buffers.right(), right, buffers.bytes());
    }

    prepare_twiddles(buffers, plan, mod, generator);

    if (plan.count > 0) {
        run_forward(buffers.left(), buffers, plan, mod);
        if (!square) {
            run_forward(buffers.right(), buffers, plan, mod);
        }
    }

    const u32 size = buffers.size();
    const u32 scale = pow_mod_host(size, mod.value - 2, mod.value);
    if (square) {
        pointwise_kernel<true><<<grid_size(size), kBlockSize>>>(
            buffers.left(), buffers.right(), size, scale, mod);
    } else {
        pointwise_kernel<false><<<grid_size(size), kBlockSize>>>(
            buffers.left(), buffers.right(), size, scale, mod);
    }
    CUDA_KERNEL_CHECK();

    if (plan.count > 0) {
        run_inverse(buffers.left(), buffers, plan, mod);
    }

    // 只要前 coefficient_count 个系数，后面的都是卷积回绕区，不必下载。
    CUDA_CHECK(cudaMemcpy(
        residues, buffers.left(), coefficient_count * sizeof(u32), cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// 并行工具
// ---------------------------------------------------------------------------

unsigned worker_count(size_t items, size_t min_per_worker) {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        hw = 4;
    }
    const size_t by_items = (items + min_per_worker - 1) / min_per_worker;
    const size_t cap = std::max<size_t>(1, std::min<size_t>(hw, by_items));
    return static_cast<unsigned>(cap);
}

// 把 [0, count) 切给几个线程跑；fn(begin, end) 必须能并发调用。
template <typename Fn>
void parallel_do(size_t count, size_t min_per_worker, Fn&& fn) {
    const unsigned workers = worker_count(count, min_per_worker);
    if (workers <= 1) {
        fn(size_t{0}, count);
        return;
    }
    const size_t chunk = (count + workers - 1) / workers;
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    for (unsigned worker = 1; worker < workers; ++worker) {
        const size_t begin = std::min(count, worker * chunk);
        const size_t end = std::min(count, begin + chunk);
        threads.emplace_back([&fn, begin, end] { fn(begin, end); });
    }
    fn(size_t{0}, std::min(count, chunk));
    for (std::thread& thread : threads) {
        thread.join();
    }
}

// ---------------------------------------------------------------------------
// 输入
// ---------------------------------------------------------------------------

u32 query_transform_limit() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    // 留四分之一给驱动、显示输出和其它进程。
    const size_t budget = free_bytes - free_bytes / 4;
    u32 limit = kMaxTransform;
    while (limit > 1 && static_cast<size_t>(limit) * kBytesPerPoint > budget) {
        limit >>= 1;
    }
    return limit;
}

u32 next_power_of_two(size_t required, u32 limit) {
    if (required <= 1) {
        return 1;
    }
    u32 result = 1;
    while (result < required) {
        if (result >= limit) {
            throw std::runtime_error(
                "输入需要的变换长度超过 2^" + std::to_string(integer_log2(limit)) +
                " 点，超过这块显卡的上限，请缩短输入");
        }
        result <<= 1;
    }
    return result;
}

std::string read_file_bytes(const char* path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(std::string("打不开输入文件：") + path);
    }
    input.seekg(0, std::ios::end);
    const std::streamoff length = input.tellg();
    input.seekg(0, std::ios::beg);
    if (length < 0) {
        throw std::runtime_error(std::string("读不了输入文件：") + path);
    }
    std::string text(static_cast<size_t>(length), '\0');
    if (length > 0) {
        input.read(text.data(), static_cast<std::streamsize>(length));
    }
    if (!input) {
        throw std::runtime_error(std::string("读取输入文件时出错：") + path);
    }
    return text;
}

// 去掉前导零，让 "000123" 与 "123" 走同一条路。
std::string_view trim_leading_zeroes(std::string_view value) {
    const size_t first_nonzero = value.find_first_not_of('0');
    if (first_nonzero == std::string_view::npos) {
        return value.substr(value.size() - 1);
    }
    return value.substr(first_nonzero);
}

// 十进制串转 base-10^8 limb（低位在前）。每八个字符正好一个 limb，校验并入
// 同一趟：先用 SWAR 检查高半字节全是 3、低半字节不超过 9，再用三次乘加拼值。
// 字节序依赖小端（x86-64），本程序的平台假设就是它。
std::vector<u32> parse_limbs(std::string_view value, const char* label) {
    const size_t digits = value.size();
    const size_t groups = digits / kBaseDigits;
    const size_t head = digits % kBaseDigits;
    const char* text = value.data();
    std::vector<u32> limbs(groups + (head != 0 ? 1 : 0));

    std::atomic<bool> invalid{false};
    parallel_do(groups, 1u << 13, [&](size_t begin, size_t end) {
        for (size_t group = begin; group < end; ++group) {
            const char* at = text + digits - kBaseDigits * (group + 1);
            u64 word;
            std::memcpy(&word, at, sizeof(word));
            if ((word & 0xF0F0F0F0F0F0F0F0ull) != 0x3030303030303030ull) {
                invalid = true;
                return;
            }
            const u64 low = word & 0x0F0F0F0F0F0F0F0Full;
            if (((low + 0x0606060606060606ull) & 0xF0F0F0F0F0F0F0F0ull) != 0) {
                invalid = true;
                return;
            }
            const u64 digit = word - 0x3030303030303030ull;
            u64 pair = (digit * 10 + (digit >> 8)) & 0x00FF00FF00FF00FFull;
            pair = (pair * 100 + (pair >> 16)) & 0x0000FFFF0000FFFFull;
            limbs[group] = static_cast<u32>(pair * 10000 + (pair >> 32));
        }
    });

    if (head != 0) {
        u32 chunk = 0;
        for (size_t offset = 0; offset < head; ++offset) {
            const u32 digit = static_cast<u32>(static_cast<unsigned char>(text[offset])) - '0';
            if (digit > 9) {
                invalid = true;
                break;
            }
            chunk = chunk * 10u + digit;
        }
        limbs[groups] = chunk;
    }

    if (invalid) {
        throw std::runtime_error(std::string(label) + "含非十进制字符");
    }
    return limbs;
}

struct TwoLines {
    std::string_view left;
    std::string_view right;
};

TwoLines split_input(std::string_view text) {
    size_t begin = 0;
    if (text.size() >= 3 &&
        static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBB &&
        static_cast<unsigned char>(text[2]) == 0xBF) {
        begin = 3;
    }

    std::string_view lines[2];
    u32 count = 0;
    size_t position = begin;
    while (position < text.size()) {
        size_t end = text.size();
        const size_t newline = text.find('\n', position);
        if (newline != std::string_view::npos) {
            end = newline;
        }
        const size_t carriage = text.find('\r', position);
        if (carriage != std::string_view::npos && carriage < end) {
            end = carriage;
        }
        const std::string_view line(text.data() + position, end - position);
        if (count < 2) {
            lines[count++] = line;
        } else if (!line.empty()) {
            throw std::runtime_error("输入多于两行");
        }
        if (end == text.size()) {
            break;
        }
        if (text[end] == '\r' && end + 1 < text.size() && text[end + 1] == '\n') {
            position = end + 2;
        } else {
            position = end + 1;
        }
    }
    if (count != 2) {
        throw std::runtime_error("输入必须是恰好两行十进制数");
    }
    if (lines[0].empty()) {
        throw std::runtime_error("第一个输入为空");
    }
    if (lines[1].empty()) {
        throw std::runtime_error("第二个输入为空");
    }
    return {lines[0], lines[1]};
}

// ---------------------------------------------------------------------------
// 三素数 CRT（Garner）与进位
//
// 系数 x = r1 + p1*t1 + p1*p2*t2，最大约 6.8e23，用三个 32 位字表示。
// 每个系数拆成 x = q*10^8 + r：拆法在并行段做，串行段只剩一次加法和一次
// 除以 10^8（编译器换成乘法），这样进位链虽然串行但很短。
// ---------------------------------------------------------------------------

struct CrtConstants {
    u32 inverse_p1_mod_p2;
    u32 inverse_p12_mod_p3;
};

CrtConstants make_crt_constants() {
    CrtConstants constants{};
    constants.inverse_p1_mod_p2 = pow_mod_host(kP1 % kP2, kP2 - 2, kP2);
    constants.inverse_p12_mod_p3 =
        pow_mod_host(static_cast<u32>(kPrime12 % kP3), kP3 - 2, kP3);
    return constants;
}

// 把三个余数合成 x，再拆成 x = quotient * 10^8 + remainder。
inline void crt_split(u32 residue1, u32 residue2, u32 residue3,
                      const CrtConstants& constants, u64& quotient, u32& remainder) {
    // r1 本来就小于 p1 < p2，不用再对 p2 取模。
    const u32 delta1 = residue2 >= residue1 ? residue2 - residue1
                                            : kP2 - (residue1 - residue2);
    const u32 t1 = static_cast<u32>(
        static_cast<u64>(delta1) * constants.inverse_p1_mod_p2 % kP2);
    const u64 partial = residue1 + static_cast<u64>(kP1) * t1;  // < p1*p2 < 2^63

    const u32 partial_mod_p3 = static_cast<u32>(partial % kP3);
    const u32 delta2 = residue3 >= partial_mod_p3
        ? residue3 - partial_mod_p3
        : kP3 - (partial_mod_p3 - residue3);
    const u32 t2 = static_cast<u32>(
        static_cast<u64>(delta2) * constants.inverse_p12_mod_p3 % kP3);

    // x = partial + p1*p2*t2，三个 32 位字。
    const u32 low_mul = static_cast<u32>(kPrime12);
    const u32 high_mul = static_cast<u32>(kPrime12 >> 32);
    const u64 low_product = static_cast<u64>(low_mul) * t2;
    u64 high_product = static_cast<u64>(high_mul) * t2 + (low_product >> 32);

    u32 word0 = static_cast<u32>(low_product);
    u32 word1 = static_cast<u32>(high_product);
    u32 word2 = static_cast<u32>(high_product >> 32);

    const u64 sum0 = static_cast<u64>(word0) + static_cast<u32>(partial);
    word0 = static_cast<u32>(sum0);
    const u64 sum1 = static_cast<u64>(word1) + static_cast<u32>(partial >> 32) + (sum0 >> 32);
    word1 = static_cast<u32>(sum1);
    word2 += static_cast<u32>(sum1 >> 32);

    // 除以 10^8：每一步的余数小于 10^8 < 2^27，拼上 32 位不会溢出 64 位。
    // 最高字不到 2^16（系数 < 6.8e23），本来就不够 10^8，第一步跳过。
    const u32 words[3] = {word0, word1, word2};
    u64 rest = words[2];
    u64 current = (rest << 32) | words[1];
    const u32 quotient_high = static_cast<u32>(current / kBase);
    rest = current % kBase;
    current = (rest << 32) | words[0];
    const u32 quotient_low = static_cast<u32>(current / kBase);
    rest = current % kBase;
    // 由输入尺寸保证商小于 2^64（单边 5.37 亿位时约 6.8e15）。
    quotient = static_cast<u64>(quotient_low) | (static_cast<u64>(quotient_high) << 32);
    remainder = static_cast<u32>(rest);
}

// 并行拆系数、串行推进位，分块做，内存占用与输入规模无关。
std::vector<u32> combine_residues(
    const u32* residue1,
    const u32* residue2,
    const u32* residue3,
    size_t count,
    double& crt_ms,
    double& carry_ms) {
    constexpr size_t kTile = 1u << 22;
    const CrtConstants constants = make_crt_constants();
    std::vector<u64> quotients(std::min(count, kTile) + 1);
    std::vector<u32> remainders(std::min(count, kTile) + 1);

    std::vector<u32> result;
    result.reserve(count + 7);
    u64 carry = 0;
    crt_ms = 0;
    carry_ms = 0;

    for (size_t base = 0; base < count; base += kTile) {
        const size_t tile = std::min(kTile, count - base);
        const auto crt_begin = Clock::now();
        parallel_do(tile, 1u << 13, [&](size_t begin, size_t end) {
            for (size_t index = begin; index < end; ++index) {
                crt_split(residue1[base + index], residue2[base + index],
                          residue3[base + index], constants,
                          quotients[index], remainders[index]);
            }
        });
        crt_ms += std::chrono::duration<double, std::milli>(Clock::now() - crt_begin).count();

        const auto carry_begin = Clock::now();
        for (size_t index = 0; index < tile; ++index) {
            const u64 sum = remainders[index] + carry;
            result.push_back(static_cast<u32>(sum % kBase));
            carry = quotients[index] + sum / kBase;
        }
        carry_ms += std::chrono::duration<double, std::milli>(Clock::now() - carry_begin).count();
    }

    while (carry != 0) {
        result.push_back(static_cast<u32>(carry % kBase));
        carry /= kBase;
    }
    while (result.size() > 1 && result.back() == 0) {
        result.pop_back();
    }
    if (result.empty()) {
        result.push_back(0);
    }
    return result;
}

// ---------------------------------------------------------------------------
// 输出
// ---------------------------------------------------------------------------

// 4 位十进制到字符的查表，输出热循环里只剩 4 字节拷贝。
const std::array<char, 40000>& decimal_table() {
    static const std::array<char, 40000> table = [] {
        std::array<char, 40000> digits{};
        for (u32 value = 0; value < 10000; ++value) {
            digits[value * 4 + 0] = static_cast<char>('0' + value / 1000);
            digits[value * 4 + 1] = static_cast<char>('0' + value / 100 % 10);
            digits[value * 4 + 2] = static_cast<char>('0' + value / 10 % 10);
            digits[value * 4 + 3] = static_cast<char>('0' + value % 10);
        }
        return digits;
    }();
    return table;
}

// 把 limb 写成十进制字符，返回位数；buffer 至少 8*limbs.size()+1 字节。
size_t format_limbs(const std::vector<u32>& limbs, char* buffer) {
    const std::array<char, 40000>& table = decimal_table();
    const std::string highest = std::to_string(limbs.back());
    std::memcpy(buffer, highest.data(), highest.size());
    const size_t head = highest.size();

    const size_t tail = limbs.size() - 1;
    parallel_do(tail, 1u << 13, [&](size_t begin, size_t end) {
        for (size_t index = begin; index < end; ++index) {
            const u32 value = limbs[index];
            char* target = buffer + head + (tail - 1 - index) * kBaseDigits;
            std::memcpy(target, table.data() + (value / 10000) * 4, 4);
            std::memcpy(target + 4, table.data() + (value % 10000) * 4, 4);
        }
    });
    return head + tail * kBaseDigits;
}

// ---------------------------------------------------------------------------
// 一次完整乘法
// ---------------------------------------------------------------------------

struct Timings {
    double parse = 0;
    double init = 0;
    double gpu = 0;
    double crt = 0;
    double carry = 0;
};

struct Product {
    std::vector<u32> limbs;
    Timings ms;
    size_t transform_size = 0;
    size_t left_digits = 0;
    size_t right_digits = 0;
};

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

Product multiply(std::string_view left_text, std::string_view right_text, double init_ms) {
    const auto total_begin = Clock::now();

    const std::string_view left = trim_leading_zeroes(left_text);
    const std::string_view right = trim_leading_zeroes(right_text);

    Product product;
    product.left_digits = left.size();
    product.right_digits = right.size();
    product.ms.init = init_ms;

    std::vector<u32> left_limbs = parse_limbs(left, "第一个输入");
    std::vector<u32> right_limbs = parse_limbs(right, "第二个输入");
    const bool square = left == right;
    product.ms.parse = milliseconds(total_begin, Clock::now());

    const size_t coefficient_count = left_limbs.size() + right_limbs.size() - 1;
    const u64 coefficient_bound =
        static_cast<u64>(kBase - 1) * (kBase - 1) *
        std::min(left_limbs.size(), right_limbs.size());
    // 三素数乘积约 1.79e28，装不进 u64，用除法比较。
    if (coefficient_bound / kPrime12 >= kP3) {
        throw std::runtime_error("输入太大，三个素数的乘积不够做 CRT 还原");
    }
    if (coefficient_bound / kBase >= (1ull << 62)) {
        throw std::runtime_error("输入太大，进位的中间结果会超出 64 位");
    }

    const u32 transform_size = next_power_of_two(coefficient_count, query_transform_limit());
    const Plan plan = make_plan(integer_log2(transform_size));
    product.transform_size = transform_size;

    const Mod moduli[3] = {make_mod(kP1, kG1), make_mod(kP2, kG2), make_mod(kP3, kG3)};
    const u32 generators[3] = {kG1, kG2, kG3};

    // 残留数组用锁页内存，D2H 拷贝从 1.8 GB/s 提到 10 GB/s 以上。
    PinnedBuffer residues1(coefficient_count);
    PinnedBuffer residues2(coefficient_count);
    PinnedBuffer residues3(coefficient_count);

    const auto gpu_begin = Clock::now();
    {
        GpuBuffers buffers(transform_size);
        for (u32 index = 0; index < 3; ++index) {
            u32* target = index == 0 ? residues1.data()
                        : index == 1 ? residues2.data()
                                     : residues3.data();
            run_modulus(buffers, left_limbs, right_limbs, moduli[index], generators[index],
                        square, plan, coefficient_count, target);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    product.ms.gpu = milliseconds(gpu_begin, Clock::now());

    product.limbs = combine_residues(residues1.data(), residues2.data(), residues3.data(),
                                     coefficient_count, product.ms.crt, product.ms.carry);
    return product;
}

// ---------------------------------------------------------------------------
// 命令行
// ---------------------------------------------------------------------------

void print_usage(std::ostream& stream, const char* program) {
    stream << "用法：" << program << " [输入文件] [输出文件]\n"
           << "      " << program << " --bench [位数]\n"
           << "\n"
           << "输入文件里恰好两行十进制整数，结果写入输出文件。任一路径写 - 表示\n"
           << "标准输入 / 标准输出。--bench 用内置随机数压测，位数缺省跑 100 万、\n"
           << "1000 万、6000 万三档。\n"
           << "退出码：0 成功，1 输入或 GPU 出错，2 参数错误。\n";
}

bool is_help_request(const char* argument) {
    const std::string_view value(argument);
    return value == "-h" || value == "--help";
}

// CUDA 上下文初始化要 170 ms 左右，和读文件、解析并行起来，单次运行省下的
// 就是这 170 ms。
class CudaWarmup {
public:
    CudaWarmup() {
        thread_ = std::thread([this] {
            const auto begin = Clock::now();
            cudaFree(nullptr);
            cudaFree(nullptr);
            milliseconds_.store(milliseconds(begin, Clock::now()));
        });
    }

    CudaWarmup(const CudaWarmup&) = delete;
    CudaWarmup& operator=(const CudaWarmup&) = delete;

    ~CudaWarmup() { wait(); }

    void wait() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    double milliseconds_taken() const { return milliseconds_.load(); }

private:
    std::thread thread_;
    std::atomic<double> milliseconds_{0};
};

void print_report(std::ostream& stream, const Product& product, size_t result_digits,
                  double output_ms) {
    const double total = product.ms.parse + product.ms.gpu + product.ms.crt +
                         product.ms.carry + product.ms.init + output_ms;
    stream << "left_digits=" << product.left_digits << '\n'
           << "right_digits=" << product.right_digits << '\n'
           << "result_digits=" << result_digits << '\n'
           << "transform_size=" << product.transform_size << '\n'
           << "input_ms=" << product.ms.parse << '\n'
           << "init_ms=" << product.ms.init << '\n'
           << "gpu_phase_ms=" << product.ms.gpu << '\n'
           << "crt_ms=" << product.ms.crt << '\n'
           << "carry_ms=" << product.ms.carry << '\n'
           << "output_ms=" << output_ms << '\n'
           << "total_ms=" << total << '\n';
}

std::string make_digits(size_t count, u32 seed) {
    std::string block(4096, '0');
    std::mt19937 rng(seed);
    for (char& digit : block) {
        digit = static_cast<char>('0' + rng() % 10);
    }
    std::string text(count, '0');
    for (size_t offset = 0; offset < count; offset += block.size()) {
        std::memcpy(text.data() + offset, block.data(),
                    std::min(block.size(), count - offset));
    }
    if (count > 0) {
        text[0] = static_cast<char>('1' + seed % 9);  // 避开前导零
    }
    return text;
}

size_t parse_count(const char* text) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || value == 0) {
        throw std::runtime_error(std::string("无法理解的位数：") + text);
    }
    return static_cast<size_t>(value);
}

constexpr int kBenchRounds = 3;

int run_bench(const char* count_text) {
    std::vector<size_t> sizes;
    if (count_text != nullptr) {
        sizes.push_back(parse_count(count_text));
    } else {
        sizes = {1000000, 10000000, 60000000};
    }

    CudaWarmup warmup;
    for (const size_t size : sizes) {
        const std::string left = make_digits(size, 1);
        const std::string right = make_digits(size, 2);
        warmup.wait();

        std::array<Timings, kBenchRounds> rounds{};
        std::array<double, kBenchRounds> output_ms{};
        Product product;
        size_t result_digits = 0;
        size_t transform_size = 0;

        for (int round = 0; round < kBenchRounds; ++round) {
            const double init = round == 0 ? warmup.milliseconds_taken() : 0.0;
            product = multiply(left, right, init);
            std::unique_ptr<char[]> buffer(new char[product.limbs.size() * kBaseDigits + 16]);
            const auto begin = Clock::now();
            result_digits = format_limbs(product.limbs, buffer.get());
            output_ms[round] = milliseconds(begin, Clock::now());
            rounds[round] = product.ms;
            transform_size = product.transform_size;
        }

        auto median = [](std::array<double, kBenchRounds> values) {
            std::sort(values.begin(), values.end());
            return values[kBenchRounds / 2];
        };
        std::array<double, kBenchRounds> gpu{}, crt{}, carry{}, parse{};
        for (int round = 0; round < kBenchRounds; ++round) {
            parse[round] = rounds[round].parse;
            gpu[round] = rounds[round].gpu;
            crt[round] = rounds[round].crt;
            carry[round] = rounds[round].carry;
        }
        const double total = median(parse) + median(gpu) + median(crt) + median(carry) +
                             median(output_ms) + rounds[0].init;

        std::cout << "--- " << size << " 位 × " << size << " 位，结果 " << result_digits
                  << " 位，变换 " << transform_size << " 点\n"
                  << "input_ms=" << median(parse) << '\n'
                  << "init_ms=" << rounds[0].init << '\n'
                  << "gpu_phase_ms=" << median(gpu) << '\n'
                  << "crt_ms=" << median(crt) << '\n'
                  << "carry_ms=" << median(carry) << '\n'
                  << "output_ms=" << median(output_ms) << '\n'
                  << "total_ms=" << total << '\n'
                  << "throughput_mdigit_s=" << (2.0 * static_cast<double>(size) /
                                                (total / 1000.0) / 1e6) << '\n';
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    if (argc > 3) {
        print_usage(std::cerr, argv[0]);
        return 2;
    }
    if (argc > 1 && is_help_request(argv[1])) {
        print_usage(std::cout, argv[0]);
        return 0;
    }
    if (argc > 1 && std::string_view(argv[1]) == "--bench") {
        try {
            return run_bench(argc > 2 ? argv[2] : nullptr);
        } catch (const std::exception& error) {
            std::cerr << "错误：" << error.what() << '\n';
            return 1;
        }
    }

    const char* input_path = argc > 1 ? argv[1] : kDefaultInput;
    const char* output_path = argc > 2 ? argv[2] : kDefaultOutput;
    const bool to_stdout = std::string_view(output_path) == "-";
    std::ostream& diagnostics = to_stdout ? std::cerr : std::cout;

    try {
        CudaWarmup warmup;

        std::string text;
        if (std::string_view(input_path) == "-") {
#ifdef _WIN32
            _setmode(_fileno(stdin), _O_BINARY);
#endif
            std::string chunk(1 << 20, '\0');
            while (std::cin) {
                std::cin.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
                text.append(chunk.data(), static_cast<size_t>(std::cin.gcount()));
            }
        } else {
            text = read_file_bytes(input_path);
        }
        const TwoLines lines = split_input(text);
        warmup.wait();

        Product product = multiply(lines.left, lines.right, warmup.milliseconds_taken());
        std::string().swap(text);  // 原文用完就还回去，上限用例能省 1 GB

        std::unique_ptr<char[]> buffer(new char[product.limbs.size() * kBaseDigits + 16]);
        const auto format_begin = Clock::now();
        const size_t length = format_limbs(product.limbs, buffer.get());
        buffer[length] = '\n';
        const double format_ms = milliseconds(format_begin, Clock::now());

        if (to_stdout) {
#ifdef _WIN32
            _setmode(_fileno(stdout), _O_BINARY);
#endif
            std::cout.write(buffer.get(), static_cast<std::streamsize>(length + 1));
            std::cout.flush();
        } else {
            std::ofstream output(output_path, std::ios::binary);
            if (!output) {
                throw std::runtime_error(std::string("打不开输出文件：") + output_path);
            }
            output.write(buffer.get(), static_cast<std::streamsize>(length + 1));
            if (!output) {
                throw std::runtime_error("写出结果时出错");
            }
        }

        print_report(diagnostics, product, length, format_ms);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "错误：" << error.what() << '\n';
        return 1;
    }
}
