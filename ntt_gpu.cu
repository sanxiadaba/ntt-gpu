// ntt_gpu.cu —— 用 CUDA NTT 计算两个超大十进制整数的乘积
//
// 做法：把十进制数按 10^4 切成 limb（4 位一个），在两个支持 2^k 次单位根的
// 素数下各做一次卷积，再用中国剩余定理把每个卷积系数精确还原成 64 位整数，
// 最后从低位向高位做进位，得到十进制结果。
//
// 为什么用两个素数：单个 32 位素数装不下卷积系数。基数 10^4 时系数上界是
// (10^4-1)^2 * min(limb 数)，6000 万位输入约 1.5e15；两个素数的乘积
// ≈ 4.59e18 够用，且乘积 < 2^63，CRT 全程 64 位算术不会溢出。
//
// 变换方向：正变换用 DIF（自然序进、倒序出），逆变换用 DIT（倒序进、自然序出），
// 两者配对后中间的频域点乘仍落在倒序上，于是整个流程不需要位反转置换。

#define _CRT_SECURE_NO_WARNINGS

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using u32 = uint32_t;
using u64 = uint64_t;
using Clock = std::chrono::steady_clock;

constexpr char kDefaultInput[] = "./input.txt";
constexpr char kDefaultOutput[] = "./output.txt";

constexpr u32 kBlockSize = 256;
constexpr u32 kTwiddleWorkers = 8192;
constexpr u32 kBase = 10000;
constexpr u32 kBaseDigits = 4;
constexpr u32 kMaxTransform = 1u << 27;

// 每个点占 3 个 u32：left、right、旋转因子表。
constexpr size_t kBytesPerPoint = 3 * sizeof(u32);

// 两个素数都支持到 2^27 次单位根。乘积 < 2^63，CRT 可用标准 64 位算术。
constexpr u32 kP1 = 2281701377u;  // = 17 * 2^27 + 1，原根 3
constexpr u32 kG1 = 3u;
constexpr u32 kP2 = 2013265921u;  // = 15 * 2^27 + 1，原根 31
constexpr u32 kG2 = 31u;
constexpr u64 kModulusProduct = static_cast<u64>(kP1) * kP2;

// 共享内存里一次做完的最内层级的对数上限：2^12 点 = 16 KB
constexpr u32 kMaxTileLog = 12;
constexpr u32 kMaxLevels = 16;

struct Modulus {
    u32 value;
    u32 root;
    u32 imaginary;  // 4 次单位根 i = root^((p-1)/4)，满足 i^2 = -1
};

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

// floor(2^64 / modulus)，Barrett 用的倒数。两个因子都小于模数时商最多小 1。
u64 barrett_reciprocal(u32 modulus) {
    return 0xFFFFFFFFFFFFFFFFull / modulus;
}

// ---------------------------------------------------------------------------
// 模运算
// ---------------------------------------------------------------------------

__device__ __forceinline__ u32 multiply_mod(u32 left, u32 right, u32 modulus, u64 reciprocal) {
    // 乘积 < 2^62，商估计最多小 1，一次条件减法即可，替代 64 位取模。
    const u64 product = static_cast<u64>(left) * right;
    const u64 quotient = __umul64hi(product, reciprocal);
    const u32 remainder = static_cast<u32>(product - quotient * modulus);
    return remainder >= modulus ? remainder - modulus : remainder;
}

// 两个操作数都小于 modulus（< 2^31），先跟 p-b 比一次，避免 a+b 溢出 32 位。
__device__ __forceinline__ u32 add_mod(u32 left, u32 right, u32 modulus) {
    const u32 gap = modulus - right;
    return left >= gap ? left - gap : left + right;
}

__device__ __forceinline__ u32 sub_mod(u32 left, u32 right, u32 modulus) {
    return left >= right ? left - right : modulus - (right - left);
}

__device__ u32 pow_mod_device(u32 base, u32 exponent, u32 modulus, u64 reciprocal) {
    u32 result = 1;
    while (exponent != 0) {
        if ((exponent & 1u) != 0) {
            result = multiply_mod(result, base, modulus, reciprocal);
        }
        base = multiply_mod(base, base, modulus, reciprocal);
        exponent >>= 1;
    }
    return result;
}

// ---------------------------------------------------------------------------
// 级计划
//
// 变换长度 n = 2^L 被拆成一串「级」，每级是 radix-4（吃掉 2 个二进制位）或
// radix-2（吃掉 1 位）。log_stride 是该级的块大小对数：radix-4 级按四分之一
// 长度分块，radix-2 级按一半长度分块。
//
// 正变换按级的顺序走（大步长 -> 小步长），逆变换把同样的级反过来走，
// 因此一条计划同时描述两个方向，且全程不需要位反转置换。
// ---------------------------------------------------------------------------

struct Level {
    u32 radix;          // 2 或 4
    u32 log_stride;     // radix-4: log2(四分之一长度)；radix-2: log2(一半长度)
    u32 forward_table;  // 前向旋转因子表在 twiddles 里的起点
    u32 inverse_table;  // 逆向旋转因子表起点
};

struct Plan {
    std::array<Level, kMaxLevels> levels{};
    u32 count = 0;
    u32 tile_begin = 0;  // [tile_begin, count) 这几级在共享内存里做
    u32 tile_log = 0;    // tile 点数 = 2^tile_log
};

u32 integer_log2(u32 value) {
    u32 result = 0;
    while ((1u << result) < value) {
        ++result;
    }
    return result;
}

// 把 log_n 拆成 radix-4 / radix-2 级，并把最内层若干级划给共享内存 tile。
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

    // 旋转因子表布局：所有前向表在前，所有逆向表在后，每级表长 = 块大小。
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

// 生成一级旋转因子表：tw[j] = root^j，j < count。每个 worker 先求一次幂再往后递推。
__global__ void generate_twiddles(
    u32* twiddles,
    u32 count,
    u32 root,
    u32 modulus,
    u64 reciprocal,
    u32 worker_count) {
    const u32 worker = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker >= worker_count) {
        return;
    }
    const u32 chunk = count / worker_count;
    const u32 begin = worker * chunk;
    u32 value = pow_mod_device(root, begin, modulus, reciprocal);
    for (u32 offset = 0; offset < chunk; ++offset) {
        twiddles[begin + offset] = value;
        value = multiply_mod(value, root, modulus, reciprocal);
    }
}

// radix-4 DIF 蝶形（正变换）。a0..a3 是相隔 stride 的四个点。
__device__ __forceinline__ void butterfly4_forward(
    u32& a0, u32& a1, u32& a2, u32& a3,
    u32 w1, u32 w2, u32 w3, u32 imaginary, u32 modulus, u64 reciprocal) {
    const u32 t0 = add_mod(a0, a2, modulus);
    const u32 t1 = sub_mod(a0, a2, modulus);
    const u32 t2 = add_mod(a1, a3, modulus);
    const u32 t3 = sub_mod(a1, a3, modulus);
    const u32 i3 = multiply_mod(t3, imaginary, modulus, reciprocal);
    a0 = add_mod(t0, t2, modulus);
    a1 = multiply_mod(add_mod(t1, i3, modulus), w1, modulus, reciprocal);
    a2 = multiply_mod(sub_mod(t0, t2, modulus), w2, modulus, reciprocal);
    a3 = multiply_mod(sub_mod(t1, i3, modulus), w3, modulus, reciprocal);
}

// radix-4 DIT 蝶形（逆变换），是上面那个蝶形的逆运算，差一个 4 倍因子，
// 由最后的 1/n 缩放统一补上。negative_imaginary = -i。
__device__ __forceinline__ void butterfly4_inverse(
    u32& a0, u32& a1, u32& a2, u32& a3,
    u32 w1, u32 w2, u32 w3, u32 negative_imaginary, u32 modulus, u64 reciprocal) {
    const u32 s1 = multiply_mod(a1, w1, modulus, reciprocal);
    const u32 s2 = multiply_mod(a2, w2, modulus, reciprocal);
    const u32 s3 = multiply_mod(a3, w3, modulus, reciprocal);
    const u32 t0 = add_mod(a0, s2, modulus);
    const u32 t2 = sub_mod(a0, s2, modulus);
    const u32 t1 = add_mod(s1, s3, modulus);
    const u32 t3 = multiply_mod(sub_mod(s1, s3, modulus), negative_imaginary, modulus, reciprocal);
    a0 = add_mod(t0, t1, modulus);
    a2 = sub_mod(t0, t1, modulus);
    a1 = add_mod(t2, t3, modulus);
    a3 = sub_mod(t2, t3, modulus);
}

// 一整级 radix-4 蝶形。蝶形之间互不相干，直接铺在全局内存上做。
template <bool kForward>
__global__ void level4_kernel(
    u32* data,
    const u32* twiddles,
    u32 log_stride,
    u32 modulus,
    u64 reciprocal,
    u32 imaginary,
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

    // 只读 w^j，w^2j 与 w^3j 各用一次模乘推出：这里访存远比算术贵。
    const u32 w1 = twiddles[offset];
    const u32 w2 = multiply_mod(w1, w1, modulus, reciprocal);
    const u32 w3 = multiply_mod(w2, w1, modulus, reciprocal);

    if constexpr (kForward) {
        butterfly4_forward(a0, a1, a2, a3, w1, w2, w3, imaginary, modulus, reciprocal);
    } else {
        butterfly4_inverse(a0, a1, a2, a3, w1, w2, w3, modulus - imaginary, modulus, reciprocal);
    }

    data[base] = a0;
    data[base + stride] = a1;
    data[base + 2u * stride] = a2;
    data[base + 3u * stride] = a3;
}

// 一整级 radix-2 蝶形，只在变换长度的对数为奇数时出现一次。
template <bool kForward>
__global__ void level2_kernel(
    u32* data,
    const u32* twiddles,
    u32 log_stride,
    u32 modulus,
    u64 reciprocal,
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
        data[left_index] = add_mod(left, right, modulus);
        data[right_index] = multiply_mod(sub_mod(left, right, modulus), w, modulus, reciprocal);
    } else {
        const u32 scaled = multiply_mod(right, w, modulus, reciprocal);
        data[left_index] = add_mod(left, scaled, modulus);
        data[right_index] = sub_mod(left, scaled, modulus);
    }
}

// 共享内存里要跑的那几级。级数很少（最多 6 级），直接按值传进核函数。
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
    u32* data,
    const u32* twiddles,
    TilePlan plan,
    u32 modulus,
    u64 reciprocal,
    u32 imaginary) {
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
                const u32 w2 = multiply_mod(w1, w1, modulus, reciprocal);
                const u32 w3 = multiply_mod(w2, w1, modulus, reciprocal);
                if constexpr (kForward) {
                    butterfly4_forward(a0, a1, a2, a3, w1, w2, w3, imaginary, modulus, reciprocal);
                } else {
                    butterfly4_inverse(a0, a1, a2, a3, w1, w2, w3, modulus - imaginary, modulus, reciprocal);
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
                    tile[position] = add_mod(left, right, modulus);
                    tile[position + stride] =
                        multiply_mod(sub_mod(left, right, modulus), w, modulus, reciprocal);
                } else {
                    const u32 scaled = multiply_mod(right, w, modulus, reciprocal);
                    tile[position] = add_mod(left, scaled, modulus);
                    tile[position + stride] = sub_mod(left, scaled, modulus);
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
    u32* left,
    const u32* right,
    u32 size,
    u32 scale,
    u32 modulus,
    u64 reciprocal) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }
    const u32 factor = kSquare ? left[index] : right[index];
    left[index] = multiply_mod(
        multiply_mod(left[index], factor, modulus, reciprocal), scale, modulus, reciprocal);
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

u32 grid_size(u32 item_count) {
    return (item_count + kBlockSize - 1) / kBlockSize;
}

// 锁页（pinned）主机缓冲。分页内存做 D2H 拷贝只有 1.8 GB/s 左右，锁页能到
// 10 GB/s 以上；128 MB 的残留数组拷两次，差出来的就是这个量级。
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

void prepare_twiddles(GpuBuffers& buffers, const Plan& plan, Modulus modulus) {
    const u64 reciprocal = barrett_reciprocal(modulus.value);
    for (u32 index = 0; index < plan.count; ++index) {
        const Level& level = plan.levels[index];
        const u32 count = 1u << level.log_stride;
        const u32 width = level.radix * count;  // 2*stride 或 4*stride
        const u32 root = pow_mod_host(
            modulus.root, (modulus.value - 1) / width, modulus.value);
        const u32 workers = std::min(count, kTwiddleWorkers);
        generate_twiddles<<<grid_size(workers), kBlockSize>>>(
            buffers.twiddles() + level.forward_table, count, root,
            modulus.value, reciprocal, workers);
        CUDA_KERNEL_CHECK();

        const u32 inverse_root = pow_mod_host(root, modulus.value - 2, modulus.value);
        generate_twiddles<<<grid_size(workers), kBlockSize>>>(
            buffers.twiddles() + level.inverse_table, count, inverse_root,
            modulus.value, reciprocal, workers);
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
void run_forward(u32* data, const GpuBuffers& buffers, const Plan& plan, Modulus modulus) {
    const u64 reciprocal = barrett_reciprocal(modulus.value);
    for (u32 index = 0; index < plan.tile_begin; ++index) {
        const Level& level = plan.levels[index];
        const u32* table = buffers.twiddles() + level.forward_table;
        if (level.radix == 4) {
            const u32 butterflies = buffers.size() / 4;
            level4_kernel<true><<<grid_size(butterflies), kBlockSize>>>(
                data, table, level.log_stride, modulus.value, reciprocal,
                modulus.imaginary, butterflies);
        } else {
            const u32 butterflies = buffers.size() / 2;
            level2_kernel<true><<<grid_size(butterflies), kBlockSize>>>(
                data, table, level.log_stride, modulus.value, reciprocal, butterflies);
        }
        CUDA_KERNEL_CHECK();
    }
    if (plan.tile_log > 0) {
        const TilePlan tile = make_tile_plan(plan, true);
        tile_kernel<true><<<buffers.size() >> plan.tile_log, kBlockSize,
                            sizeof(u32) << plan.tile_log>>>(
            data, buffers.twiddles(), tile, modulus.value, reciprocal, modulus.imaginary);
        CUDA_KERNEL_CHECK();
    }
}

// 逆变换：先把最内层倒着做完，再做大步长的级（从后往前）。
void run_inverse(u32* data, const GpuBuffers& buffers, const Plan& plan, Modulus modulus) {
    const u64 reciprocal = barrett_reciprocal(modulus.value);
    if (plan.tile_log > 0) {
        const TilePlan tile = make_tile_plan(plan, false);
        tile_kernel<false><<<buffers.size() >> plan.tile_log, kBlockSize,
                             sizeof(u32) << plan.tile_log>>>(
            data, buffers.twiddles(), tile, modulus.value, reciprocal, modulus.imaginary);
        CUDA_KERNEL_CHECK();
    }
    for (u32 remaining = plan.tile_begin; remaining-- > 0;) {
        const Level& level = plan.levels[remaining];
        const u32* table = buffers.twiddles() + level.inverse_table;
        if (level.radix == 4) {
            const u32 butterflies = buffers.size() / 4;
            level4_kernel<false><<<grid_size(butterflies), kBlockSize>>>(
                data, table, level.log_stride, modulus.value, reciprocal,
                modulus.imaginary, butterflies);
        } else {
            const u32 butterflies = buffers.size() / 2;
            level2_kernel<false><<<grid_size(butterflies), kBlockSize>>>(
                data, table, level.log_stride, modulus.value, reciprocal, butterflies);
        }
        CUDA_KERNEL_CHECK();
    }
}

// 一个模数下的完整卷积：两个正变换、频域点乘（含缩放）、一个逆变换。
void run_modulus(
    GpuBuffers& buffers,
    const std::vector<u32>& left,
    const std::vector<u32>& right,
    Modulus modulus,
    bool square,
    const Plan& plan,
    u32* residues) {
    const u64 reciprocal = barrett_reciprocal(modulus.value);

    // 只清空 limb 之后的那一段，前面的会被 memcpy 覆盖。两个操作数长度可以不同，
    // 必须各按各的长度来，不能共用 left 的长度。
    const size_t left_bytes = left.size() * sizeof(u32);
    if (buffers.bytes() > left_bytes) {
        CUDA_CHECK(cudaMemset(
            reinterpret_cast<char*>(buffers.left()) + left_bytes, 0,
            buffers.bytes() - left_bytes));
    }
    CUDA_CHECK(cudaMemcpy(buffers.left(), left.data(), left_bytes, cudaMemcpyHostToDevice));

    // 自乘时频域就是左侧自己，右侧数组用不到，不必上传。
    if (!square) {
        const size_t right_bytes = right.size() * sizeof(u32);
        if (buffers.bytes() > right_bytes) {
            CUDA_CHECK(cudaMemset(
                reinterpret_cast<char*>(buffers.right()) + right_bytes, 0,
                buffers.bytes() - right_bytes));
        }
        CUDA_CHECK(cudaMemcpy(
            buffers.right(), right.data(), right_bytes, cudaMemcpyHostToDevice));
    }

    prepare_twiddles(buffers, plan, modulus);

    if (plan.count > 0) {
        run_forward(buffers.left(), buffers, plan, modulus);
        if (!square) {
            run_forward(buffers.right(), buffers, plan, modulus);
        }
    }

    const u32 size = buffers.size();
    // 逆变换的 1/n 缩放直接乘进频域点乘，省掉独立的一趟全数组读写。
    const u32 scale = pow_mod_host(size, modulus.value - 2, modulus.value);
    if (square) {
        pointwise_kernel<true><<<grid_size(size), kBlockSize>>>(
            buffers.left(), buffers.right(), size, scale, modulus.value, reciprocal);
    } else {
        pointwise_kernel<false><<<grid_size(size), kBlockSize>>>(
            buffers.left(), buffers.right(), size, scale, modulus.value, reciprocal);
    }
    CUDA_KERNEL_CHECK();

    if (plan.count > 0) {
        run_inverse(buffers.left(), buffers, plan, modulus);
    }

    CUDA_CHECK(cudaMemcpy(residues, buffers.left(), buffers.bytes(), cudaMemcpyDeviceToHost));
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

// 十进制串转 base-10000 limb（低位在前）。一次处理 8 位，正好产出两个 limb，
// 数字校验并入同一趟。
//
// 从右往左按 8 位一组处理，每组正好落在两个 limb 边界上，直接写进输出区间。
std::vector<u32> parse_limbs(std::string_view value, const char* label) {
    const size_t digits = value.size();
    const size_t groups = digits / 8;
    const size_t head = digits % 8;
    const char* text = value.data();
    std::vector<u32> limbs((digits + kBaseDigits - 1) / kBaseDigits);

    // 一次吃掉 8 个字节：先用 SWAR 校验（高位半字节必须全是 3，低位半字节必须
    // <= 9），再用三步乘加把 8 个数字拼成数值。逐字节带分支的老写法在 6000 万位
    // 输入上要多花两倍时间。
    // 字节序依赖小端（x86-64），本程序的平台假设就是它。
    auto parse_group = [&](size_t group) {
        const char* at = text + digits - 8 * (group + 1);
        u64 word;
        std::memcpy(&word, at, sizeof(word));
        const u64 high_nibbles = word & 0xF0F0F0F0F0F0F0F0ull;
        const u64 low_nibbles = word & 0x0F0F0F0F0F0F0F0Full;
        if (high_nibbles != 0x3030303030303030ull ||
            ((low_nibbles + 0x0606060606060606ull) & 0xF0F0F0F0F0F0F0F0ull) != 0) {
            return false;
        }
        const u64 digits_u64 = word - 0x3030303030303030ull;  // 每字节一个数字
        u64 pair = (digits_u64 * 10 + (digits_u64 >> 8)) & 0x00FF00FF00FF00FFull;
        pair = (pair * 100 + (pair >> 16)) & 0x0000FFFF0000FFFFull;
        const u32 chunk =
            static_cast<u32>((pair * 10000 + (pair >> 32)) & 0xFFFFFFFFull);
        limbs[2 * group] = chunk % kBase;
        limbs[2 * group + 1] = chunk / kBase;
        return true;
    };

    // 最左边不足 8 位的那几位由主线程处理，产出最高的 1~2 个 limb。
    if (head != 0) {
        u32 chunk = 0;
        for (size_t offset = 0; offset < head; ++offset) {
            const u32 digit = static_cast<u32>(static_cast<unsigned char>(text[offset])) - '0';
            if (digit > 9) {
                throw std::runtime_error(std::string(label) + "含非十进制字符");
            }
            chunk = chunk * 10u + digit;
        }
        const size_t base = 2 * groups;
        if (head > kBaseDigits) {
            limbs[base] = chunk % kBase;
            limbs[base + 1] = chunk / kBase;
        } else {
            limbs[base] = chunk;
        }
    }

    for (size_t group = 0; group < groups; ++group) {
        if (!parse_group(group)) {
            throw std::runtime_error(std::string(label) + "含非十进制字符");
        }
    }
    return limbs;
}

struct InputData {
    std::vector<u32> left;
    std::vector<u32> right;
    size_t left_digits = 0;
    size_t right_digits = 0;
    bool identical = false;
};

InputData read_input(const char* path) {
    std::string text = read_file_bytes(path);

    size_t begin = 0;
    if (text.size() >= 3 &&
        static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBB &&
        static_cast<unsigned char>(text[2]) == 0xBF) {
        begin = 3;
    }

    std::string_view lines[2];
    u32 line_count = 0;
    size_t position = begin;
    while (position < text.size()) {
        size_t end = text.size();
        const size_t newline = text.find('\n', position);
        if (newline != std::string::npos) {
            end = newline;
        }
        const size_t carriage = text.find('\r', position);
        if (carriage != std::string::npos && carriage < end) {
            end = carriage;
        }
        const std::string_view line(text.data() + position, end - position);
        if (line_count < 2) {
            lines[line_count++] = line;
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
    if (line_count != 2) {
        throw std::runtime_error("输入必须是恰好两行十进制数");
    }
    if (lines[0].empty()) {
        throw std::runtime_error("第一个输入为空");
    }
    if (lines[1].empty()) {
        throw std::runtime_error("第二个输入为空");
    }

    const std::string_view left = trim_leading_zeroes(lines[0]);
    const std::string_view right = trim_leading_zeroes(lines[1]);

    InputData result;
    result.left_digits = left.size();
    result.right_digits = right.size();
    result.identical = left == right;
    result.left = parse_limbs(left, "第一个输入");
    result.right = parse_limbs(right, "第二个输入");
    return result;
}

// ---------------------------------------------------------------------------
// 小输入：CPU 竖式乘法，几微秒，省掉 GPU 约 190 ms 的上下文初始化
// ---------------------------------------------------------------------------

// 只有当较短的一侧很小、且总乘算次数可控时才走这条路。
bool worth_schoolbook(const InputData& input) {
    const size_t short_side = std::min(input.left.size(), input.right.size());
    const size_t long_side = std::max(input.left.size(), input.right.size());
    return short_side <= 64 && short_side * long_side <= (1u << 21);
}

std::vector<u32> multiply_schoolbook(const InputData& input) {
    const std::vector<u32>& shorter =
        input.left.size() <= input.right.size() ? input.left : input.right;
    const std::vector<u32>& longer =
        input.left.size() <= input.right.size() ? input.right : input.left;

    std::vector<u64> accumulator(longer.size() + shorter.size(), 0);
    for (size_t i = 0; i < shorter.size(); ++i) {
        const u64 digit = shorter[i];
        if (digit == 0) {
            continue;
        }
        for (size_t j = 0; j < longer.size(); ++j) {
            accumulator[i + j] += digit * longer[j];
        }
    }

    std::vector<u32> limbs;
    limbs.reserve(accumulator.size() + 1);
    u64 carry = 0;
    for (const u64 value : accumulator) {
        const u64 total = value + carry;
        limbs.push_back(static_cast<u32>(total % kBase));
        carry = total / kBase;
    }
    while (carry != 0) {
        limbs.push_back(static_cast<u32>(carry % kBase));
        carry /= kBase;
    }
    while (limbs.size() > 1 && limbs.back() == 0) {
        limbs.pop_back();
    }
    return limbs;
}

// ---------------------------------------------------------------------------
// CRT 与进位
// ---------------------------------------------------------------------------

std::vector<u32> reconstruct_and_carry(
    const u32* residues1,
    const u32* residues2,
    size_t coefficient_count) {
    const u32 p1_inverse_mod_p2 = pow_mod_host(kP1 % kP2, kP2 - 2, kP2);
    std::vector<u32> result(coefficient_count + 4, 0);
    u64 carry = 0;

    for (size_t index = 0; index < coefficient_count; ++index) {
        // kP1 < 2*kP2，所以 residue1 对 kP2 取模只是一次条件减法。
        const u32 residue1 = residues1[index] >= kP2 ? residues1[index] - kP2 : residues1[index];
        const u32 delta = residues2[index] >= residue1
            ? residues2[index] - residue1
            : kP2 - (residue1 - residues2[index]);
        const u32 multiplier = static_cast<u32>(
            static_cast<u64>(delta) * p1_inverse_mod_p2 % kP2);
        const u64 coefficient = residues1[index] + static_cast<u64>(kP1) * multiplier;
        const u64 value = coefficient + carry;
        result[index] = static_cast<u32>(value % kBase);
        carry = value / kBase;
    }

    size_t used = coefficient_count;
    while (carry != 0) {
        result[used++] = static_cast<u32>(carry % kBase);
        carry /= kBase;
    }
    if (used == 0) {
        used = 1;
    }
    result.resize(used);
    while (result.size() > 1 && result.back() == 0) {
        result.pop_back();
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
        for (u32 value = 0; value < kBase; ++value) {
            digits[value * 4 + 0] = static_cast<char>('0' + value / 1000);
            digits[value * 4 + 1] = static_cast<char>('0' + value / 100 % 10);
            digits[value * 4 + 2] = static_cast<char>('0' + value / 10 % 10);
            digits[value * 4 + 3] = static_cast<char>('0' + value % 10);
        }
        return digits;
    }();
    return table;
}

size_t write_output(const char* path, const std::vector<u32>& limbs) {
    // 二进制写：换行固定为 LF，不受平台影响，也不走文本模式的逐字符转换。
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error(std::string("打不开输出文件：") + path);
    }

    constexpr size_t kOutputBufferSize = 4 * 1024 * 1024;
    std::vector<char> buffer(kOutputBufferSize);
    size_t used = 0;
    const std::array<char, 40000>& table = decimal_table();

    auto reserve_bytes = [&](size_t count) {
        if (used + count > buffer.size()) {
            output.write(buffer.data(), static_cast<std::streamsize>(used));
            used = 0;
        }
    };

    const std::string highest = std::to_string(limbs.back());
    reserve_bytes(highest.size());
    std::memcpy(buffer.data() + used, highest.data(), highest.size());
    used += highest.size();

    // 两个 limb 一次搬 8 字节，比每 limb 一次 memcpy 少一半存储指令。
    size_t index = limbs.size() - 1;
    while (index >= 2) {
        reserve_bytes(kBaseDigits * 2);
        std::memcpy(buffer.data() + used, table.data() + limbs[index - 1] * kBaseDigits, kBaseDigits);
        std::memcpy(buffer.data() + used + kBaseDigits,
                    table.data() + limbs[index - 2] * kBaseDigits, kBaseDigits);
        used += kBaseDigits * 2;
        index -= 2;
    }
    if (index == 1) {
        reserve_bytes(kBaseDigits);
        std::memcpy(buffer.data() + used, table.data() + limbs[0] * kBaseDigits, kBaseDigits);
        used += kBaseDigits;
    }

    reserve_bytes(1);
    buffer[used++] = '\n';
    output.write(buffer.data(), static_cast<std::streamsize>(used));
    if (!output) {
        throw std::runtime_error("写出结果时出错");
    }
    return highest.size() + (limbs.size() - 1) * kBaseDigits;
}

// ---------------------------------------------------------------------------

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void print_usage(std::ostream& stream, const char* program) {
    stream << "用法：" << program << " [输入文件] [输出文件]\n"
           << "\n"
           << "把输入文件里恰好两行的十进制整数相乘，结果写入输出文件。两个路径都默认\n"
           << "指向当前目录。退出码 0 成功，1 输入或 GPU 出错，2 参数错误。\n";
}

bool is_help_request(const char* argument) {
    const std::string_view value(argument);
    return value == "-h" || value == "--help";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 3) {
        print_usage(std::cerr, argv[0]);
        return 2;
    }
    if (argc > 1 && is_help_request(argv[1])) {
        print_usage(std::cout, argv[0]);
        return 0;
    }

    const char* input_path = argc > 1 ? argv[1] : kDefaultInput;
    const char* output_path = argc > 2 ? argv[2] : kDefaultOutput;

    try {
        const auto total_begin = Clock::now();
        InputData input = read_input(input_path);
        const auto input_end = Clock::now();

        // 小输入直接在 CPU 上做竖式乘法，避免为几微秒的计算付出 190 ms 的
        // CUDA 上下文初始化。
        if (worth_schoolbook(input)) {
            std::vector<u32> result = multiply_schoolbook(input);
            const auto carry_end = Clock::now();
            const size_t result_digits = write_output(output_path, result);
            const auto output_end = Clock::now();

            std::cout << "left_digits=" << input.left_digits << '\n';
            std::cout << "right_digits=" << input.right_digits << '\n';
            std::cout << "result_digits=" << result_digits << '\n';
            std::cout << "transform_size=0\n";
            std::cout << "input_ms=" << milliseconds(total_begin, input_end) << '\n';
            std::cout << "gpu_phase_ms=0\n";
            std::cout << "crt_carry_ms=" << milliseconds(input_end, carry_end) << '\n';
            std::cout << "output_ms=" << milliseconds(carry_end, output_end) << '\n';
            std::cout << "total_ms=" << milliseconds(total_begin, output_end) << '\n';
            return 0;
        }

        const size_t coefficient_count = input.left.size() + input.right.size() - 1;
        const u64 coefficient_bound =
            static_cast<u64>(kBase - 1) * (kBase - 1) *
            std::min(input.left.size(), input.right.size());
        if (coefficient_bound >= kModulusProduct) {
            throw std::runtime_error("输入太大，两个素数的乘积不够做 CRT 还原");
        }
        const u32 transform_size =
            next_power_of_two(coefficient_count, query_transform_limit());
        const Plan plan = make_plan(integer_log2(transform_size));

        std::vector<u32> result;
        const Modulus moduli[2] = {
            {kP1, kG1, pow_mod_host(kG1, (kP1 - 1) / 4, kP1)},
            {kP2, kG2, pow_mod_host(kG2, (kP2 - 1) / 4, kP2)},
        };
        // 残留数组用锁页内存，D2H 拷贝从 1.8 GB/s 提到 10 GB/s 以上。
        PinnedBuffer residues1(transform_size);
        PinnedBuffer residues2(transform_size);
        const auto gpu_begin = Clock::now();
        {
            GpuBuffers buffers(transform_size);
            run_modulus(buffers, input.left, input.right, moduli[0], input.identical, plan,
                        residues1.data());
            run_modulus(buffers, input.left, input.right, moduli[1], input.identical, plan,
                        residues2.data());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        const auto gpu_end = Clock::now();

        result = reconstruct_and_carry(residues1.data(), residues2.data(), coefficient_count);
        const auto carry_end = Clock::now();

        const size_t result_digits = write_output(output_path, result);
        const auto output_end = Clock::now();

        std::cout << "left_digits=" << input.left_digits << '\n';
        std::cout << "right_digits=" << input.right_digits << '\n';
        std::cout << "result_digits=" << result_digits << '\n';
        std::cout << "transform_size=" << transform_size << '\n';
        std::cout << "input_ms=" << milliseconds(total_begin, input_end) << '\n';
        std::cout << "gpu_phase_ms=" << milliseconds(gpu_begin, gpu_end) << '\n';
        std::cout << "crt_carry_ms=" << milliseconds(gpu_end, carry_end) << '\n';
        std::cout << "output_ms=" << milliseconds(carry_end, output_end) << '\n';
        std::cout << "total_ms=" << milliseconds(total_begin, output_end) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "错误：" << error.what() << '\n';
        return 1;
    }
}
