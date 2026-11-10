// ntt.cu —— 显卡上做的那部分：把两个 limbs 数组卷积一遍。
//
// 这是整个程序里唯一难懂的文件。日常使用、改参数、看结果都不需要看它。
//
// 一句话：要对两个数列做卷积，正常做法是 O(n^2) 次乘加，太慢；把两个数列各做一次
// "快速傅里叶变换"（在这里是整数版的 NTT），变成频域后逐点相乘，再变回来，
// 就只要 O(n log n)。缺点是只能算长度 2 的幂，所以要把数列补零到 2 的幂。
//
// 几个词：
//   单位根 / 旋转因子   变换里反复用到的常数表，本文件叫 roots
//   蝶形               一次基本运算：取 2 个或 4 个数，组合出同样多的新数
//   级                 变换分若干轮，一轮叫一级，每级把所有数据扫一遍
//   模数               所有运算都在"除以 p 取余"的世界里做，p 就是模数

#include "ntt.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace {

// 每 256 个线程一个线程块，这是显卡调度的单位。
constexpr u32 kBlockSize = 256;
constexpr u32 kTwiddleWorkers = 8192;
constexpr u32 kMaxTransform = 1u << 27;
constexpr u32 kMaxTileLog = 13;  // 共享内存一次装 2^13 个数 = 32 KB
constexpr u32 kMaxLevels = 16;

// 每个点占 3 个 u32：left、right、roots。
constexpr size_t kBytesPerPoint = 3 * sizeof(u32);

void cuda_check(cudaError_t error, const char* expression, const char* file, int line) {
    if (error == cudaSuccess) {
        return;
    }
    throw std::runtime_error(
        std::string("CUDA call failed at ") + file + ":" + std::to_string(line) + " (" +
        expression + "): " + cudaGetErrorString(error));
}

#define CUDA_CHECK(expression) cuda_check((expression), #expression, __FILE__, __LINE__)
#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

// ---------------------------------------------------------------------------
// 模运算：加、减、乘，结果都对 p 取余。数都保持在 [0, p) 里。
// ---------------------------------------------------------------------------

__device__ __forceinline__ u32 multiply_mod(u32 left, u32 right, const Modulus& m) {
    const u64 product = static_cast<u64>(left) * right;
    const u64 quotient = __umul64hi(product, m.reciprocal);
    // 用倒数估的商最多小 1，所以余数可能落在 [p, 2p)；先按 64 位比较再截断，
    // 否则 2p > 2^32 的素数会被截断坏。
    u64 rest = product - quotient * m.prime;
    if (rest >= m.prime) {
        rest -= m.prime;
    }
    return static_cast<u32>(rest);
}

// 先跟 p - right 比一次，加法路径永远不会越过 p，不会溢出 32 位。
__device__ __forceinline__ u32 add_mod(u32 left, u32 right, u32 prime) {
    const u32 gap = prime - right;
    return left >= gap ? left - gap : left + right;
}

__device__ __forceinline__ u32 sub_mod(u32 left, u32 right, u32 prime) {
    return left >= right ? left - right : prime - (right - left);
}

__device__ u32 pow_mod_device(u32 base, u32 exponent, const Modulus& m) {
    u32 result = 1;
    while (exponent != 0) {
        if ((exponent & 1u) != 0) {
            result = multiply_mod(result, base, m);
        }
        base = multiply_mod(base, base, m);
        exponent >>= 1;
    }
    return result;
}

// ---------------------------------------------------------------------------
// 级计划：把长度 2^L 拆成若干轮，每轮一趟读写全局显存。
//
// 每级要么一次处理 4 个点（radix-4，吃 2 个二进制位），要么处理 2 个点
// （radix-2，吃 1 位）。最内层几级数据量小，交给共享内存一次做完。
// ---------------------------------------------------------------------------

struct Level {
    u32 radix;         // 2 或 4
    u32 log_stride;    // 块大小的对数
    u32 forward_roots; // 正向单位根表在 roots 里的起点
    u32 inverse_roots; // 反向表起点
};

struct Plan {
    std::array<Level, kMaxLevels> levels{};
    u32 count = 0;
    u32 tile_begin = 0;  // [tile_begin, count) 这几级在共享内存里做
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
            throw std::runtime_error("transform length exceeds the level table");
        }
        if (remaining >= 2) {
            plan.levels[plan.count++] = {4, remaining - 2, 0, 0};
            remaining -= 2;
        } else {
            plan.levels[plan.count++] = {2, remaining - 1, 0, 0};
            remaining -= 1;
        }
    }

    // 从最内层往外数，能塞进共享内存的级都塞进去。
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

    // 正向表在前，反向表在后，每级表长等于块大小。
    u32 offset = 0;
    for (u32 i = 0; i < plan.count; ++i) {
        plan.levels[i].forward_roots = offset;
        offset += 1u << plan.levels[i].log_stride;
    }
    for (u32 i = 0; i < plan.count; ++i) {
        plan.levels[i].inverse_roots = offset;
        offset += 1u << plan.levels[i].log_stride;
    }
    return plan;
}

// ---------------------------------------------------------------------------
// 核函数：真正在显卡上跑的代码
// ---------------------------------------------------------------------------

// 生成一级要用的单位根表：roots[j] = root^j。每个线程算一段，往后递推。
__global__ void generate_roots(
    u32* __restrict__ roots,
    u32 count,
    u32 root,
    Modulus m,
    u32 worker_count) {
    const u32 worker = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker >= worker_count) {
        return;
    }
    const u32 chunk = count / worker_count;
    const u32 begin = worker * chunk;
    u32 value = pow_mod_device(root, begin, m);
    for (u32 offset = 0; offset < chunk; ++offset) {
        roots[begin + offset] = value;
        value = multiply_mod(value, root, m);
    }
}

// 正向蝶形：4 个点进，4 个点出。a0..a3 是相隔 stride 的四个点。
__device__ __forceinline__ void butterfly4_forward(
    u32& a0, u32& a1, u32& a2, u32& a3,
    u32 w1, u32 w2, u32 w3, const Modulus& m) {
    const u32 t0 = add_mod(a0, a2, m.prime);
    const u32 t1 = sub_mod(a0, a2, m.prime);
    const u32 t2 = add_mod(a1, a3, m.prime);
    const u32 t3 = sub_mod(a1, a3, m.prime);
    const u32 i3 = multiply_mod(t3, m.imaginary, m);
    a0 = add_mod(t0, t2, m.prime);
    a1 = multiply_mod(add_mod(t1, i3, m.prime), w1, m);
    a2 = multiply_mod(sub_mod(t0, t2, m.prime), w2, m);
    a3 = multiply_mod(sub_mod(t1, i3, m.prime), w3, m);
}

// 反向蝶形：正好是上面那个的逆运算，差一个 4 倍因子，由最后的 1/n 统一补上。
__device__ __forceinline__ void butterfly4_inverse(
    u32& a0, u32& a1, u32& a2, u32& a3,
    u32 w1, u32 w2, u32 w3, u32 negative_imaginary, const Modulus& m) {
    const u32 s1 = multiply_mod(a1, w1, m);
    const u32 s2 = multiply_mod(a2, w2, m);
    const u32 s3 = multiply_mod(a3, w3, m);
    const u32 t0 = add_mod(a0, s2, m.prime);
    const u32 t2 = sub_mod(a0, s2, m.prime);
    const u32 t1 = add_mod(s1, s3, m.prime);
    const u32 t3 = multiply_mod(sub_mod(s1, s3, m.prime), negative_imaginary, m);
    a0 = add_mod(t0, t1, m.prime);
    a2 = sub_mod(t0, t1, m.prime);
    a1 = add_mod(t2, t3, m.prime);
    a3 = sub_mod(t2, t3, m.prime);
}

// 一整级 radix-4。各级之间互不相干，直接铺在全局显存上做。
template <bool kForward>
__global__ void level4_kernel(
    u32* __restrict__ data,
    const u32* __restrict__ roots,
    u32 log_stride,
    Modulus m,
    u32 butterfly_count) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= butterfly_count) {
        return;
    }
    const u32 stride = 1u << log_stride;
    const u32 offset = index & (stride - 1u);
    // 组内第 offset 个蝶形的首点就在 4*index - 3*offset。
    const u32 base = (index << 2) - 3u * offset;
    u32 a0 = data[base];
    u32 a1 = data[base + stride];
    u32 a2 = data[base + 2u * stride];
    u32 a3 = data[base + 3u * stride];

    // 只读 w^j，w^2j 和 w^3j 用两次乘法现推：这里读显存比做乘法贵。
    const u32 w1 = roots[offset];
    const u32 w2 = multiply_mod(w1, w1, m);
    const u32 w3 = multiply_mod(w2, w1, m);

    if constexpr (kForward) {
        butterfly4_forward(a0, a1, a2, a3, w1, w2, w3, m);
    } else {
        butterfly4_inverse(a0, a1, a2, a3, w1, w2, w3, m.prime - m.imaginary, m);
    }

    data[base] = a0;
    data[base + stride] = a1;
    data[base + 2u * stride] = a2;
    data[base + 3u * stride] = a3;
}

// 一整级 radix-2，只在长度是奇数次幂时出现一级。
template <bool kForward>
__global__ void level2_kernel(
    u32* __restrict__ data,
    const u32* __restrict__ roots,
    u32 log_stride,
    Modulus m,
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
    const u32 w = roots[offset];

    if constexpr (kForward) {
        data[left_index] = add_mod(left, right, m.prime);
        data[right_index] = multiply_mod(sub_mod(left, right, m.prime), w, m);
    } else {
        const u32 scaled = multiply_mod(right, w, m);
        data[left_index] = add_mod(left, scaled, m.prime);
        data[right_index] = sub_mod(left, scaled, m.prime);
    }
}

// 最内层几级的计划，级数很少（最多 6 级），按值传给核函数。
constexpr u32 kMaxTileLevels = kMaxLevels / 2;

struct TilePlan {
    u32 count;
    u32 log_size;
    u32 radix[kMaxTileLevels];
    u32 log_stride[kMaxTileLevels];
    u32 roots[kMaxTileLevels];
};

// 最内层几级：整块读进共享内存，全部做完再写回，只花一趟全局显存。
template <bool kForward>
__global__ void tile_kernel(
    u32* __restrict__ data,
    const u32* __restrict__ roots,
    TilePlan plan,
    Modulus m) {
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
        const u32* table = roots + plan.roots[step];
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
                const u32 w2 = multiply_mod(w1, w1, m);
                const u32 w3 = multiply_mod(w2, w1, m);
                if constexpr (kForward) {
                    butterfly4_forward(a0, a1, a2, a3, w1, w2, w3, m);
                } else {
                    butterfly4_inverse(a0, a1, a2, a3, w1, w2, w3,
                                       m.prime - m.imaginary, m);
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
                    tile[position] = add_mod(left, right, m.prime);
                    tile[position + stride] =
                        multiply_mod(sub_mod(left, right, m.prime), w, m);
                } else {
                    const u32 scaled = multiply_mod(right, w, m);
                    tile[position] = add_mod(left, scaled, m.prime);
                    tile[position + stride] = sub_mod(left, scaled, m.prime);
                }
            }
        }
        __syncthreads();
    }

    for (u32 index = threadIdx.x; index < tile_size; index += blockDim.x) {
        data[base_index + index] = tile[index];
    }
}

// 频域逐点相乘，顺带把逆变换的 1/n 也乘进去，省一趟全数组读写。
template <bool kSquare>
__global__ void pointwise_kernel(
    u32* __restrict__ left,
    const u32* __restrict__ right,
    u32 size,
    u32 scale,
    Modulus m) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }
    const u32 factor = kSquare ? left[index] : right[index];
    left[index] = multiply_mod(multiply_mod(left[index], factor, m), scale, m);
}

// ---------------------------------------------------------------------------
// 显存里的三块数组，和主机侧的锁页缓冲
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
                "not enough GPU memory: need " + std::to_string(required >> 20) + " MiB, only " +
                std::to_string(free_bytes >> 20) + " MiB free; shorten the input");
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&left_), bytes_));
        try {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&right_), bytes_));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&roots_), bytes_));
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
    u32* roots() const { return roots_; }
    size_t bytes() const { return bytes_; }
    u32 size() const { return size_; }

private:
    void release() noexcept {
        if (left_ != nullptr) cudaFree(left_);
        if (right_ != nullptr) cudaFree(right_);
        if (roots_ != nullptr) cudaFree(roots_);
        left_ = nullptr;
        right_ = nullptr;
        roots_ = nullptr;
    }

    u32 size_;
    size_t bytes_;
    u32* left_ = nullptr;
    u32* right_ = nullptr;
    u32* roots_ = nullptr;
};

u32 grid_size(u32 item_count) {
    return (item_count + kBlockSize - 1) / kBlockSize;
}

// 变换长度只能是 2 的幂，这里根据显存余量决定最大能到多少。
u32 transform_limit() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t budget = free_bytes - free_bytes / 4;  // 留四分之一给系统和显示
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
                "input needs a transform longer than 2^" + std::to_string(integer_log2(limit)) +
                " points, which this GPU cannot do; shorten the input");
        }
        result <<= 1;
    }
    return result;
}

void prepare_roots(GpuBuffers& buffers, const Plan& plan, const Modulus& m) {
    for (u32 index = 0; index < plan.count; ++index) {
        const Level& level = plan.levels[index];
        const u32 count = 1u << level.log_stride;
        const u32 width = level.radix * count;
        const u32 root = pow_mod(m.root, (m.prime - 1) / width, m.prime);
        const u32 inverse_root = pow_mod(root, m.prime - 2, m.prime);
        const u32 workers = std::min(count, kTwiddleWorkers);
        generate_roots<<<grid_size(workers), kBlockSize>>>(
            buffers.roots() + level.forward_roots, count, root, m, workers);
        CUDA_KERNEL_CHECK();
        generate_roots<<<grid_size(workers), kBlockSize>>>(
            buffers.roots() + level.inverse_roots, count, inverse_root, m, workers);
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
        tile.roots[step] = forward ? level.forward_roots : level.inverse_roots;
    }
    return tile;
}

// 正变换：先做大步长的级，再进共享内存做最内层。
void run_forward(u32* data, const GpuBuffers& buffers, const Plan& plan, const Modulus& m) {
    for (u32 index = 0; index < plan.tile_begin; ++index) {
        const Level& level = plan.levels[index];
        const u32* table = buffers.roots() + level.forward_roots;
        if (level.radix == 4) {
            const u32 butterflies = buffers.size() / 4;
            level4_kernel<true><<<grid_size(butterflies), kBlockSize>>>(
                data, table, level.log_stride, m, butterflies);
        } else {
            const u32 butterflies = buffers.size() / 2;
            level2_kernel<true><<<grid_size(butterflies), kBlockSize>>>(
                data, table, level.log_stride, m, butterflies);
        }
        CUDA_KERNEL_CHECK();
    }
    if (plan.tile_log > 0) {
        const TilePlan tile = make_tile_plan(plan, true);
        tile_kernel<true><<<buffers.size() >> plan.tile_log, kBlockSize,
                            sizeof(u32) << plan.tile_log>>>(
            data, buffers.roots(), tile, m);
        CUDA_KERNEL_CHECK();
    }
}

// 逆变换：反过来，先把最内层做完，再做大步长的级。
void run_inverse(u32* data, const GpuBuffers& buffers, const Plan& plan, const Modulus& m) {
    if (plan.tile_log > 0) {
        const TilePlan tile = make_tile_plan(plan, false);
        tile_kernel<false><<<buffers.size() >> plan.tile_log, kBlockSize,
                             sizeof(u32) << plan.tile_log>>>(
            data, buffers.roots(), tile, m);
        CUDA_KERNEL_CHECK();
    }
    for (u32 remaining = plan.tile_begin; remaining-- > 0;) {
        const Level& level = plan.levels[remaining];
        const u32* table = buffers.roots() + level.inverse_roots;
        if (level.radix == 4) {
            const u32 butterflies = buffers.size() / 4;
            level4_kernel<false><<<grid_size(butterflies), kBlockSize>>>(
                data, table, level.log_stride, m, butterflies);
        } else {
            const u32 butterflies = buffers.size() / 2;
            level2_kernel<false><<<grid_size(butterflies), kBlockSize>>>(
                data, table, level.log_stride, m, butterflies);
        }
        CUDA_KERNEL_CHECK();
    }
}

// 把 limbs 传进显存：只清空 limbs 之后那一段，前面的会被 memcpy 覆盖。
void upload_limbs(u32* data, const std::vector<u32>& limbs, size_t bytes) {
    const size_t used = limbs.size() * sizeof(u32);
    if (bytes > used) {
        CUDA_CHECK(cudaMemset(reinterpret_cast<char*>(data) + used, 0, bytes - used));
    }
    CUDA_CHECK(cudaMemcpy(data, limbs.data(), used, cudaMemcpyHostToDevice));
}

// 一个素数下的完整流程：正变换 ×1~2、逐点相乘、逆变换，然后把结果拷回来。
void run_modulus(
    GpuBuffers& buffers,
    const std::vector<u32>& left,
    const std::vector<u32>& right,
    const Modulus& m,
    bool square,
    const Plan& plan,
    size_t coefficient_count,
    u32* residues) {
    upload_limbs(buffers.left(), left, buffers.bytes());
    if (!square) {
        upload_limbs(buffers.right(), right, buffers.bytes());
    }

    prepare_roots(buffers, plan, m);

    if (plan.count > 0) {
        run_forward(buffers.left(), buffers, plan, m);
        if (!square) {
            run_forward(buffers.right(), buffers, plan, m);
        }
    }

    const u32 size = buffers.size();
    const u32 scale = pow_mod(size, m.prime - 2, m.prime);  // 1/n
    if (square) {
        pointwise_kernel<true><<<grid_size(size), kBlockSize>>>(
            buffers.left(), buffers.right(), size, scale, m);
    } else {
        pointwise_kernel<false><<<grid_size(size), kBlockSize>>>(
            buffers.left(), buffers.right(), size, scale, m);
    }
    CUDA_KERNEL_CHECK();

    if (plan.count > 0) {
        run_inverse(buffers.left(), buffers, plan, m);
    }

    // 只要前 coefficient_count 个系数，后面是补零区，不用下载。
    CUDA_CHECK(cudaMemcpy(
        residues, buffers.left(), coefficient_count * sizeof(u32), cudaMemcpyDeviceToHost));
}

}  // namespace

PinnedBuffer::PinnedBuffer(size_t count) {
    if (count != 0) {
        CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void**>(&data_), count * sizeof(u32), cudaHostAllocDefault));
    }
}

PinnedBuffer::~PinnedBuffer() {
    if (data_ != nullptr) {
        cudaFreeHost(data_);
    }
}

double convolve(const std::vector<u32>& left, const std::vector<u32>& right, bool square,
                size_t coefficient_count, u32* r1, u32* r2, u32* r3) {
    const u32 size = next_power_of_two(coefficient_count, transform_limit());
    const Plan plan = make_plan(integer_log2(size));

    const auto begin = Clock::now();
    {
        GpuBuffers buffers(size);
        u32* targets[3] = {r1, r2, r3};
        for (u32 index = 0; index < 3; ++index) {
            run_modulus(buffers, left, right, kModuli[index], square, plan,
                        coefficient_count, targets[index]);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    return milliseconds(begin, Clock::now());
}
