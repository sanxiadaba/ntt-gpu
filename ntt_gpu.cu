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
// Stages whose butterflies stay inside a tile of this many points run in shared
// memory, so the first ten stages cost one global pass instead of ten.
#ifndef NTT_TILE_LOG
#define NTT_TILE_LOG 11
#endif
constexpr u32 kTileLog = NTT_TILE_LOG;
constexpr u32 kTileSize = 1u << kTileLog;
constexpr u32 kBase = 10000;
constexpr u32 kBaseDigits = 4;
constexpr u32 kMaxTransform = 1u << 27;

// Each point costs one u32 in the left, right and twiddle arrays.
constexpr size_t kBytesPerPoint = 3 * sizeof(u32);

// Both primes support transforms through 2^27. Their product is below 2^63,
// which keeps CRT reconstruction in standard 64-bit arithmetic.
constexpr u32 kP1 = 2281701377u;
constexpr u32 kG1 = 3u;
constexpr u32 kP2 = 2013265921u;
constexpr u32 kG2 = 31u;
constexpr u64 kModulusProduct = static_cast<u64>(kP1) * kP2;

struct Modulus {
    u32 value;
    u32 primitive_root;
};

constexpr Modulus kModuli[] = {{kP1, kG1}, {kP2, kG2}};

void cuda_check(cudaError_t error, const char* expression, const char* file, int line) {
    if (error == cudaSuccess) {
        return;
    }
    throw std::runtime_error(
        std::string("CUDA error at ") + file + ":" + std::to_string(line) +
        " for " + expression + ": " + cudaGetErrorString(error));
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

// floor(2^64 / modulus), the Barrett reciprocal. The modulus is odd and larger
// than one, so this equals floor((2^64 - 1) / modulus) as well.
u64 barrett_reciprocal(u32 modulus) {
    return 0xFFFFFFFFFFFFFFFFull / modulus;
}

__device__ __forceinline__ u32 multiply_mod(
    u32 left,
    u32 right,
    u32 modulus,
    u64 reciprocal) {
    // Both factors are below the modulus, so the product is below 2^62 and the
    // quotient estimate can only be one too small. A single conditional
    // subtraction is enough; this replaces a 64-bit division with two
    // multiply-high instructions.
    const u64 product = static_cast<u64>(left) * right;
    const u64 quotient = __umul64hi(product, reciprocal);
    const u32 remainder = static_cast<u32>(product - quotient * modulus);
    return remainder >= modulus ? remainder - modulus : remainder;
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

__global__ void generate_twiddles(
    u32* twiddles,
    u32 half_stage,
    u32 root,
    u32 modulus,
    u64 reciprocal,
    u32 worker_count) {
    const u32 worker = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker >= worker_count) {
        return;
    }

    const u32 chunk = half_stage / worker_count;
    const u32 begin = worker * chunk;
    u32 value = pow_mod_device(root, begin, modulus, reciprocal);
    for (u32 offset = 0; offset < chunk; ++offset) {
        twiddles[begin + offset] = value;
        value = multiply_mod(value, root, modulus, reciprocal);
    }
}

__device__ __forceinline__ u32 add_mod(u32 left, u32 right, u32 modulus) {
    const u64 sum = static_cast<u64>(left) + right;
    return static_cast<u32>(sum >= modulus ? sum - modulus : sum);
}

__device__ __forceinline__ u32 subtract_mod(u32 left, u32 right, u32 modulus) {
    return left >= right ? left - right : modulus - (right - left);
}

// One radix-2 stage over the whole array.
//
// kDif = true is the decimation-in-frequency form used by the forward
// transform: natural input order, bit-reversed output order, larger stages
// first. kDif = false is decimation-in-time, used by the inverse transform:
// bit-reversed input order, natural output order, smaller stages first.
//
// Because both directions are paired like this, no bit-reversal permutation is
// ever needed: the pointwise product stays in bit-reversed order between them.
template <bool kDif>
__global__ void butterfly_stage(
    u32* data,
    const u32* twiddles,
    u32 log_half_stage,
    u32 butterfly_count,
    u32 modulus,
    u64 reciprocal) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= butterfly_count) {
        return;
    }

    // Shift and mask instead of division and modulo: the whole stage is a power
    // of two, so index = group * half + offset gives
    // left = 2 * index - offset and right = left + half.
    const u32 half_stage = 1u << log_half_stage;
    const u32 offset = index & (half_stage - 1u);
    const u32 left_index = (index << 1) - offset;
    const u32 right_index = left_index + half_stage;
    const u32 left = data[left_index];
    const u32 right = data[right_index];

    if constexpr (kDif) {
        // Forward: (u + v, (u - v) * w^j)
        const u32 twiddle = twiddles[offset];
        data[left_index] = add_mod(left, right, modulus);
        data[right_index] = multiply_mod(
            subtract_mod(left, right, modulus), twiddle, modulus, reciprocal);
    } else {
        // Inverse: (u + v * w^-j, u - v * w^-j). Because w^half = -1, the
        // inverse twiddle is -w^(half - j), so no second table is needed.
        const u32 twiddle = offset == 0 ? 1u : modulus - twiddles[half_stage - offset];
        const u32 scaled = multiply_mod(right, twiddle, modulus, reciprocal);
        data[left_index] = add_mod(left, scaled, modulus);
        data[right_index] = subtract_mod(left, scaled, modulus);
    }
}

// The stages whose butterflies stay inside a tile of kTileSize consecutive
// points all run in shared memory, so they cost one global pass instead of
// kTileLog passes. Large stages stay in global memory.
template <bool kDif>
__global__ void butterfly_tile_stages(
    u32* data,
    const u32* twiddles,
    u32 stage_count,
    u32 modulus,
    u64 reciprocal) {
    __shared__ u32 tile[kTileSize];

    const u32 base = blockIdx.x * kTileSize;
    for (u32 index = threadIdx.x; index < kTileSize; index += blockDim.x) {
        tile[index] = data[base + index];
    }
    __syncthreads();

    for (u32 step = 0; step < stage_count; ++step) {
        // kDif walks the stages from the largest down, kDit from the smallest up.
        const u32 log_half_stage = kDif ? stage_count - 1 - step : step;
        const u32 half_stage = 1u << log_half_stage;
        const u32 offset_mask = half_stage - 1u;
        // The twiddle table stores halves back to back: 1, 2, 4, ... points.
        const u32 table_offset = half_stage - 1u;
        for (u32 index = threadIdx.x; index < kTileSize / 2; index += blockDim.x) {
            const u32 offset = index & offset_mask;
            const u32 left_index = (index << 1) - offset;
            const u32 right_index = left_index + half_stage;
            const u32 left = tile[left_index];
            const u32 right = tile[right_index];

            if constexpr (kDif) {
                const u32 twiddle = twiddles[table_offset + offset];
                tile[left_index] = add_mod(left, right, modulus);
                tile[right_index] = multiply_mod(
                    subtract_mod(left, right, modulus), twiddle, modulus, reciprocal);
            } else {
                const u32 twiddle = offset == 0
                    ? 1u
                    : modulus - twiddles[table_offset + half_stage - offset];
                const u32 scaled = multiply_mod(right, twiddle, modulus, reciprocal);
                tile[left_index] = add_mod(left, scaled, modulus);
                tile[right_index] = subtract_mod(left, scaled, modulus);
            }
        }
        __syncthreads();
    }

    for (u32 index = threadIdx.x; index < kTileSize; index += blockDim.x) {
        data[base + index] = tile[index];
    }
}

__global__ void pointwise_multiply(
    u32* left,
    const u32* right,
    u32 size,
    u32 modulus,
    u64 reciprocal) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        left[index] = multiply_mod(left[index], right[index], modulus, reciprocal);
    }
}

__global__ void scale_inverse(
    u32* data,
    u32 size,
    u32 inverse_size,
    u32 modulus,
    u64 reciprocal) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        data[index] = multiply_mod(data[index], inverse_size, modulus, reciprocal);
    }
}

u32 grid_size(u32 item_count) {
    return (item_count + kBlockSize - 1) / kBlockSize;
}

u32 integer_log2(u32 value) {
    u32 result = 0;
    while ((1u << result) < value) {
        ++result;
    }
    return result;
}

u32 next_power_of_two(size_t required, u32 limit) {
    if (required <= 1) {
        return 1;
    }
    u32 result = 1;
    while (result < required) {
        if (result >= limit) {
            throw std::runtime_error(
                "Input requires an NTT longer than 2^" +
                std::to_string(integer_log2(limit)) +
                " points, which this GPU cannot hold; shorten the input");
        }
        result <<= 1;
    }
    return result;
}

// The four arrays must fit, so keep a quarter of the free memory for the
// driver, the display and anything else that may be running.
u32 query_transform_limit() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t budget = free_bytes - free_bytes / 4;
    u32 limit = kMaxTransform;
    while (limit > 1 && static_cast<size_t>(limit) * kBytesPerPoint > budget) {
        limit >>= 1;
    }
    return limit;
}

std::string read_file_bytes(const char* path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(std::string("Cannot open input file: ") + path);
    }

    input.seekg(0, std::ios::end);
    const std::streamoff length = input.tellg();
    input.seekg(0, std::ios::beg);
    if (length < 0) {
        throw std::runtime_error(std::string("Cannot read input file: ") + path);
    }

    std::string text(static_cast<size_t>(length), '\0');
    if (length > 0) {
        input.read(text.data(), static_cast<std::streamsize>(length));
    }
    if (!input) {
        throw std::runtime_error(std::string("Failed while reading input file: ") + path);
    }
    return text;
}

// Drops leading zeroes so that "000123" and "123" follow the same path.
std::string_view trim_leading_zeroes(std::string_view value) {
    const size_t first_nonzero = value.find_first_not_of('0');
    if (first_nonzero == std::string_view::npos) {
        return value.substr(value.size() - 1);
    }
    return value.substr(first_nonzero);
}

// Converts a run of decimal digits into base-10000 limbs, least significant
// first. Eight digits per iteration produce two limbs and keep the digit check
// on the same pass as the conversion, so the input is walked only once.
std::vector<u32> parse_limbs(std::string_view value, const char* label) {
    std::vector<u32> limbs;
    limbs.reserve((value.size() + kBaseDigits - 1) / kBaseDigits);

    const char* digits = value.data();
    size_t index = value.size();
    while (index >= 8) {
        index -= 8;
        u32 chunk = 0;
        for (size_t offset = 0; offset < 8; ++offset) {
            const u32 digit =
                static_cast<u32>(static_cast<unsigned char>(digits[index + offset])) - '0';
            if (digit > 9) {
                throw std::runtime_error(
                    std::string(label) + " contains a non-decimal character");
            }
            chunk = chunk * 10u + digit;
        }
        limbs.push_back(chunk % kBase);
        limbs.push_back(chunk / kBase);
    }

    if (index != 0) {
        u32 chunk = 0;
        for (size_t offset = 0; offset < index; ++offset) {
            const u32 digit =
                static_cast<u32>(static_cast<unsigned char>(digits[offset])) - '0';
            if (digit > 9) {
                throw std::runtime_error(
                    std::string(label) + " contains a non-decimal character");
            }
            chunk = chunk * 10u + digit;
        }
        if (index > kBaseDigits) {
            limbs.push_back(chunk % kBase);
            limbs.push_back(chunk / kBase);
        } else {
            limbs.push_back(chunk);
        }
    }
    return limbs;
}

struct InputData {
    std::vector<u32> left;
    std::vector<u32> right;
    size_t left_digits = 0;
    size_t right_digits = 0;
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
        // find() is a vectorised scan; a byte loop over 120 MB is measurable.
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
            throw std::runtime_error("Input contains more than two lines");
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
        throw std::runtime_error("Input must contain exactly two decimal lines");
    }
    if (lines[0].empty()) {
        throw std::runtime_error("First input is empty");
    }
    if (lines[1].empty()) {
        throw std::runtime_error("Second input is empty");
    }

    const std::string_view left = trim_leading_zeroes(lines[0]);
    const std::string_view right = trim_leading_zeroes(lines[1]);

    InputData result;
    result.left_digits = left.size();
    result.right_digits = right.size();
    result.left = parse_limbs(left, "First input");
    result.right = parse_limbs(right, "Second input");
    return result;
}

class GpuBuffers {
public:
    explicit GpuBuffers(u32 size) : size_(size), bytes_(static_cast<size_t>(size) * sizeof(u32)) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const size_t required = 3 * bytes_;
        if (required > free_bytes) {
            throw std::runtime_error(
                "Not enough GPU memory: need " + std::to_string(required >> 20) +
                " MiB, only " + std::to_string(free_bytes >> 20) +
                " MiB available; shorten the input");
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

    ~GpuBuffers() {
        release();
    }

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

void prepare_twiddles(GpuBuffers& buffers, Modulus modulus) {
    const u64 reciprocal = barrett_reciprocal(modulus.value);
    u32 table_offset = 0;
    for (u32 stage = 2; stage <= buffers.size(); stage <<= 1) {
        const u32 half_stage = stage >> 1;
        const u32 root = pow_mod_host(
            modulus.primitive_root,
            (modulus.value - 1) / stage,
            modulus.value);
        const u32 workers = std::min(half_stage, kTwiddleWorkers);
        generate_twiddles<<<grid_size(workers), kBlockSize>>>(
            buffers.twiddles() + table_offset,
            half_stage,
            root,
            modulus.value,
            reciprocal,
            workers);
        CUDA_KERNEL_CHECK();
        table_offset += half_stage;
    }
}

template <bool kDif>
void transform_ntt(u32* data, const GpuBuffers& buffers, Modulus modulus) {
    const u64 reciprocal = barrett_reciprocal(modulus.value);
    const u32 size = buffers.size();
    const u32 butterfly_count = size / 2;
    const u32 log_size = integer_log2(size);

    const bool use_tile = size >= kTileSize;
    const u32 tile_stages = use_tile ? kTileLog : 0u;
    const u32 global_stages = log_size - tile_stages;

    // kDif walks the large stages down towards the tile, kDit the other way.
    const auto run_tile = [&] {
        if (!use_tile) {
            return;
        }
        if constexpr (kDif) {
            butterfly_tile_stages<true><<<size / kTileSize, kBlockSize>>>(
                data, buffers.twiddles(), tile_stages, modulus.value, reciprocal);
        } else {
            butterfly_tile_stages<false><<<size / kTileSize, kBlockSize>>>(
                data, buffers.twiddles(), tile_stages, modulus.value, reciprocal);
        }
        CUDA_KERNEL_CHECK();
    };

    const auto run_global = [&] {
        for (u32 step = 0; step < global_stages; ++step) {
            const u32 log_half_stage = kDif ? log_size - 1 - step : tile_stages + step;
            const u32 half_stage = 1u << log_half_stage;
            const u32* table = buffers.twiddles() + (half_stage - 1u);
            if constexpr (kDif) {
                butterfly_stage<true><<<grid_size(butterfly_count), kBlockSize>>>(
                    data, table, log_half_stage, butterfly_count, modulus.value, reciprocal);
            } else {
                butterfly_stage<false><<<grid_size(butterfly_count), kBlockSize>>>(
                    data, table, log_half_stage, butterfly_count, modulus.value, reciprocal);
            }
            CUDA_KERNEL_CHECK();
        }
    };

    if constexpr (kDif) {
        run_global();
        run_tile();
    } else {
        run_tile();
        run_global();
    }
}

void run_modulus(
    GpuBuffers& buffers,
    const std::vector<u32>& left,
    const std::vector<u32>& right,
    Modulus modulus,
    std::vector<u32>& residues) {
    const u64 reciprocal = barrett_reciprocal(modulus.value);
    CUDA_CHECK(cudaMemset(buffers.left(), 0, buffers.bytes()));
    CUDA_CHECK(cudaMemset(buffers.right(), 0, buffers.bytes()));
    CUDA_CHECK(cudaMemcpy(
        buffers.left(), left.data(), left.size() * sizeof(u32), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        buffers.right(), right.data(), right.size() * sizeof(u32), cudaMemcpyHostToDevice));

    prepare_twiddles(buffers, modulus);

    // Forward is decimation-in-frequency (natural in, bit-reversed out), the
    // inverse is decimation-in-time (bit-reversed in, natural out), so the
    // pointwise product in between never needs a permutation.
    transform_ntt<true>(buffers.left(), buffers, modulus);
    transform_ntt<true>(buffers.right(), buffers, modulus);

    pointwise_multiply<<<grid_size(buffers.size()), kBlockSize>>>(
        buffers.left(), buffers.right(), buffers.size(), modulus.value, reciprocal);
    CUDA_KERNEL_CHECK();

    transform_ntt<false>(buffers.left(), buffers, modulus);

    const u32 inverse_size = pow_mod_host(buffers.size(), modulus.value - 2, modulus.value);
    scale_inverse<<<grid_size(buffers.size()), kBlockSize>>>(
        buffers.left(), buffers.size(), inverse_size, modulus.value, reciprocal);
    CUDA_KERNEL_CHECK();

    residues.resize(buffers.size());
    CUDA_CHECK(cudaMemcpy(
        residues.data(), buffers.left(), buffers.bytes(), cudaMemcpyDeviceToHost));
}

std::vector<u32> reconstruct_and_carry(
    const std::vector<u32>& residues1,
    const std::vector<u32>& residues2,
    size_t coefficient_count) {
    const u32 p1_inverse_mod_p2 = pow_mod_host(kP1 % kP2, kP2 - 2, kP2);
    std::vector<u32> result(coefficient_count + 4, 0);
    u64 carry = 0;

    for (size_t index = 0; index < coefficient_count; ++index) {
        const u32 residue1_mod_p2 = residues1[index] % kP2;
        const u32 delta = residues2[index] >= residue1_mod_p2
            ? residues2[index] - residue1_mod_p2
            : kP2 - (residue1_mod_p2 - residues2[index]);
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

// Four decimal digits per limb, taken from a precomputed table so that the hot
// loop is a four byte copy instead of three divisions.
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
    // Text mode keeps the platform newline, so the output bytes stay identical
    // on Windows (CRLF) and elsewhere (LF).
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error(std::string("Cannot open output file: ") + path);
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

    for (size_t index = limbs.size() - 1; index > 0; --index) {
        reserve_bytes(kBaseDigits);
        std::memcpy(buffer.data() + used, table.data() + limbs[index - 1] * kBaseDigits, kBaseDigits);
        used += kBaseDigits;
    }

    reserve_bytes(1);
    buffer[used++] = '\n';
    output.write(buffer.data(), static_cast<std::streamsize>(used));
    if (!output) {
        throw std::runtime_error("Failed while writing output");
    }
    return highest.size() + (limbs.size() - 1) * kBaseDigits;
}

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void print_usage(std::ostream& stream, const char* program) {
    stream << "usage: " << program << " [input.txt] [output.txt]\n"
           << "\n"
           << "Multiplies the two decimal integers in input.txt, one per line, and\n"
           << "writes the product to output.txt. Both paths default to the current\n"
           << "directory. Exit code is 0 on success, 1 on input or GPU errors and 2\n"
           << "on bad usage.\n";
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

        const size_t coefficient_count = input.left.size() + input.right.size() - 1;
        const u64 coefficient_bound =
            static_cast<u64>(kBase - 1) * (kBase - 1) *
            std::min(input.left.size(), input.right.size());
        if (coefficient_bound >= kModulusProduct) {
            throw std::runtime_error("CRT modulus product is too small for this input");
        }
        const u32 transform_size =
            next_power_of_two(coefficient_count, query_transform_limit());

        std::vector<u32> residues1;
        std::vector<u32> residues2;
        const auto gpu_begin = Clock::now();
        {
            GpuBuffers buffers(transform_size);
            run_modulus(buffers, input.left, input.right, kModuli[0], residues1);
            run_modulus(buffers, input.left, input.right, kModuli[1], residues2);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        const auto gpu_end = Clock::now();

        std::vector<u32> result = reconstruct_and_carry(
            residues1, residues2, coefficient_count);
        const auto carry_end = Clock::now();
        residues1.clear();
        residues2.clear();
        residues1.shrink_to_fit();
        residues2.shrink_to_fit();

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
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
