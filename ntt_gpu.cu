#define _CRT_SECURE_NO_WARNINGS

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
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

__device__ __forceinline__ u32 multiply_mod(u32 left, u32 right, u32 modulus) {
    return static_cast<u32>(static_cast<u64>(left) * right % modulus);
}

__device__ u32 pow_mod_device(u32 base, u32 exponent, u32 modulus) {
    u32 result = 1;
    while (exponent != 0) {
        if ((exponent & 1u) != 0) {
            result = multiply_mod(result, base, modulus);
        }
        base = multiply_mod(base, base, modulus);
        exponent >>= 1;
    }
    return result;
}

__global__ void build_bit_reverse(u32* reversed, u32 size, u32 log_size) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        reversed[index] = log_size == 0 ? 0 : __brev(index) >> (32u - log_size);
    }
}

__global__ void apply_bit_reverse(u32* data, const u32* reversed, u32 size) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        const u32 target = reversed[index];
        if (target < index) {
            const u32 value = data[index];
            data[index] = data[target];
            data[target] = value;
        }
    }
}

__global__ void generate_twiddles(
    u32* twiddles,
    u32 half_stage,
    u32 root,
    u32 modulus,
    u32 worker_count) {
    const u32 worker = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker >= worker_count) {
        return;
    }

    const u32 chunk = half_stage / worker_count;
    const u32 begin = worker * chunk;
    u32 value = pow_mod_device(root, begin, modulus);
    for (u32 offset = 0; offset < chunk; ++offset) {
        twiddles[begin + offset] = value;
        value = multiply_mod(value, root, modulus);
    }
}

__global__ void butterfly_stage(
    u32* data,
    const u32* twiddles,
    u32 half_stage,
    u32 butterfly_count,
    u32 modulus) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= butterfly_count) {
        return;
    }

    const u32 group = index / half_stage;
    const u32 offset = index % half_stage;
    const u32 left_index = group * (half_stage << 1) + offset;
    const u32 right_index = left_index + half_stage;
    const u32 left = data[left_index];
    const u32 right = multiply_mod(data[right_index], twiddles[offset], modulus);

    const u64 sum = static_cast<u64>(left) + right;
    data[left_index] = static_cast<u32>(sum >= modulus ? sum - modulus : sum);
    data[right_index] = left >= right ? left - right : modulus - (right - left);
}

__global__ void pointwise_multiply(u32* left, const u32* right, u32 size, u32 modulus) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        left[index] = multiply_mod(left[index], right[index], modulus);
    }
}

__global__ void reverse_for_inverse(u32* data, u32 size) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > 0 && index < size / 2) {
        const u32 value = data[index];
        data[index] = data[size - index];
        data[size - index] = value;
    }
}

__global__ void scale_inverse(u32* data, u32 size, u32 inverse_size, u32 modulus) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        data[index] = multiply_mod(data[index], inverse_size, modulus);
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

u32 next_power_of_two(size_t required) {
    if (required <= 1) {
        return 1;
    }
    u32 result = 1;
    while (result < required) {
        if (result >= kMaxTransform) {
            throw std::runtime_error("Input requires an NTT longer than 2^27");
        }
        result <<= 1;
    }
    return result;
}

void validate_decimal(const std::string& value, const char* label) {
    if (value.empty()) {
        throw std::runtime_error(std::string(label) + " is empty");
    }
    for (char character : value) {
        if (character < '0' || character > '9') {
            throw std::runtime_error(std::string(label) + " contains a non-decimal character");
        }
    }
}

void trim_leading_zeroes(std::string& value) {
    const size_t first_nonzero = value.find_first_not_of('0');
    if (first_nonzero == std::string::npos) {
        value.assign("0");
    } else if (first_nonzero != 0) {
        value.erase(0, first_nonzero);
    }
}

std::vector<u32> parse_limbs(const std::string& value) {
    std::vector<u32> limbs;
    limbs.reserve((value.size() + kBaseDigits - 1) / kBaseDigits);

    size_t end = value.size();
    while (end != 0) {
        const size_t begin = end > kBaseDigits ? end - kBaseDigits : 0;
        u32 limb = 0;
        for (size_t index = begin; index < end; ++index) {
            limb = limb * 10u + static_cast<u32>(value[index] - '0');
        }
        limbs.push_back(limb);
        end = begin;
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
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(std::string("Cannot open input file: ") + path);
    }

    std::string left_text;
    std::string right_text;
    std::string extra;
    if (!std::getline(input, left_text) || !std::getline(input, right_text)) {
        throw std::runtime_error("Input must contain exactly two decimal lines");
    }
    if (std::getline(input, extra)) {
        throw std::runtime_error("Input contains more than two lines");
    }

    validate_decimal(left_text, "First input");
    validate_decimal(right_text, "Second input");
    trim_leading_zeroes(left_text);
    trim_leading_zeroes(right_text);

    InputData result;
    result.left_digits = left_text.size();
    result.right_digits = right_text.size();
    result.left = parse_limbs(left_text);
    result.right = parse_limbs(right_text);
    return result;
}

class GpuBuffers {
public:
    explicit GpuBuffers(u32 size) : size_(size), bytes_(static_cast<size_t>(size) * sizeof(u32)) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&left_), bytes_));
        try {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&right_), bytes_));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&reversed_), bytes_));
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
    u32* reversed() const { return reversed_; }
    u32* twiddles() const { return twiddles_; }
    size_t bytes() const { return bytes_; }
    u32 size() const { return size_; }

private:
    void release() noexcept {
        if (left_ != nullptr) cudaFree(left_);
        if (right_ != nullptr) cudaFree(right_);
        if (reversed_ != nullptr) cudaFree(reversed_);
        if (twiddles_ != nullptr) cudaFree(twiddles_);
        left_ = nullptr;
        right_ = nullptr;
        reversed_ = nullptr;
        twiddles_ = nullptr;
    }

    u32 size_;
    size_t bytes_;
    u32* left_ = nullptr;
    u32* right_ = nullptr;
    u32* reversed_ = nullptr;
    u32* twiddles_ = nullptr;
};

void prepare_twiddles(GpuBuffers& buffers, Modulus modulus) {
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
            workers);
        CUDA_KERNEL_CHECK();
        table_offset += half_stage;
    }
}

void forward_ntt(u32* data, const GpuBuffers& buffers, Modulus modulus) {
    u32 table_offset = 0;
    const u32 butterfly_count = buffers.size() / 2;
    for (u32 stage = 2; stage <= buffers.size(); stage <<= 1) {
        const u32 half_stage = stage >> 1;
        butterfly_stage<<<grid_size(butterfly_count), kBlockSize>>>(
            data,
            buffers.twiddles() + table_offset,
            half_stage,
            butterfly_count,
            modulus.value);
        CUDA_KERNEL_CHECK();
        table_offset += half_stage;
    }
}

void run_modulus(
    GpuBuffers& buffers,
    const std::vector<u32>& left,
    const std::vector<u32>& right,
    Modulus modulus,
    std::vector<u32>& residues) {
    CUDA_CHECK(cudaMemset(buffers.left(), 0, buffers.bytes()));
    CUDA_CHECK(cudaMemset(buffers.right(), 0, buffers.bytes()));
    CUDA_CHECK(cudaMemcpy(
        buffers.left(), left.data(), left.size() * sizeof(u32), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        buffers.right(), right.data(), right.size() * sizeof(u32), cudaMemcpyHostToDevice));

    prepare_twiddles(buffers, modulus);

    apply_bit_reverse<<<grid_size(buffers.size()), kBlockSize>>>(
        buffers.left(), buffers.reversed(), buffers.size());
    CUDA_KERNEL_CHECK();
    apply_bit_reverse<<<grid_size(buffers.size()), kBlockSize>>>(
        buffers.right(), buffers.reversed(), buffers.size());
    CUDA_KERNEL_CHECK();

    forward_ntt(buffers.left(), buffers, modulus);
    forward_ntt(buffers.right(), buffers, modulus);

    pointwise_multiply<<<grid_size(buffers.size()), kBlockSize>>>(
        buffers.left(), buffers.right(), buffers.size(), modulus.value);
    CUDA_KERNEL_CHECK();

    apply_bit_reverse<<<grid_size(buffers.size()), kBlockSize>>>(
        buffers.left(), buffers.reversed(), buffers.size());
    CUDA_KERNEL_CHECK();
    forward_ntt(buffers.left(), buffers, modulus);

    reverse_for_inverse<<<grid_size(buffers.size()), kBlockSize>>>(
        buffers.left(), buffers.size());
    CUDA_KERNEL_CHECK();
    const u32 inverse_size = pow_mod_host(buffers.size(), modulus.value - 2, modulus.value);
    scale_inverse<<<grid_size(buffers.size()), kBlockSize>>>(
        buffers.left(), buffers.size(), inverse_size, modulus.value);
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

size_t write_output(const char* path, const std::vector<u32>& limbs) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error(std::string("Cannot open output file: ") + path);
    }

    constexpr size_t kOutputBufferSize = 4 * 1024 * 1024;
    std::vector<char> buffer(kOutputBufferSize);
    size_t used = 0;
    auto append = [&](char character) {
        if (used == buffer.size()) {
            output.write(buffer.data(), static_cast<std::streamsize>(used));
            used = 0;
        }
        buffer[used++] = character;
    };

    const std::string highest = std::to_string(limbs.back());
    for (char character : highest) {
        append(character);
    }
    for (size_t index = limbs.size() - 1; index > 0; --index) {
        const u32 value = limbs[index - 1];
        append(static_cast<char>('0' + value / 1000));
        append(static_cast<char>('0' + value / 100 % 10));
        append(static_cast<char>('0' + value / 10 % 10));
        append(static_cast<char>('0' + value % 10));
    }
    append('\n');
    output.write(buffer.data(), static_cast<std::streamsize>(used));
    if (!output) {
        throw std::runtime_error("Failed while writing output");
    }
    return highest.size() + (limbs.size() - 1) * kBaseDigits;
}

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

}  // namespace

int main(int argc, char** argv) {
    const char* input_path = argc > 1 ? argv[1] : kDefaultInput;
    const char* output_path = argc > 2 ? argv[2] : kDefaultOutput;

    try {
        const auto total_begin = Clock::now();
        InputData input = read_input(input_path);
        const auto input_end = Clock::now();

        const size_t coefficient_count = input.left.size() + input.right.size() - 1;
        const u32 transform_size = next_power_of_two(coefficient_count);
        const u64 coefficient_bound =
            static_cast<u64>(kBase - 1) * (kBase - 1) *
            std::min(input.left.size(), input.right.size());
        if (coefficient_bound >= kModulusProduct) {
            throw std::runtime_error("CRT modulus product is too small for this input");
        }

        std::vector<u32> residues1;
        std::vector<u32> residues2;
        const auto gpu_begin = Clock::now();
        {
            GpuBuffers buffers(transform_size);
            build_bit_reverse<<<grid_size(transform_size), kBlockSize>>>(
                buffers.reversed(), transform_size, integer_log2(transform_size));
            CUDA_KERNEL_CHECK();
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
