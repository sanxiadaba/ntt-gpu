// ntt.h —— 三个 .cu 文件共用的东西。
//
// 只需要先理解两个概念：
//   limbs    数字在程序里的样子：八位十进制一组，一组存一个 uint32，低位在前。
//            比如 123456789012 存成 {34567890, 1234}。
//   Modulus  一个素数，以及做模运算需要的几个常数。
#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

using u32 = uint32_t;
using u64 = uint64_t;
using Clock = std::chrono::steady_clock;

constexpr u32 kBase = 100000000u;  // 10^8，一组装八位十进制
constexpr u32 kBaseDigits = 8;

// 三个素数，都形如 奇数 × 2^27 + 1，所以都支持长度 2^27 的变换。
// 32 位整数能支持到 2^27 的素数只有这三个，长度上限就此定死。
constexpr u32 kP1 = 2013265921u;  // 15 * 2^27 + 1，原根 31
constexpr u32 kP2 = 2281701377u;  // 17 * 2^27 + 1，原根 3
constexpr u32 kP3 = 3892314113u;  // 29 * 2^27 + 1，原根 3
constexpr u64 kPrime12 = static_cast<u64>(kP1) * kP2;

struct Modulus {
    u32 prime;       // 素数本身
    u32 root;        // 原根
    u32 imaginary;   // 4 次单位根 i，满足 i^2 = -1
    u64 reciprocal;  // floor(2^64 / prime)，用来把除法换成乘法
};

// 编译期算好三个素数需要的常数。
constexpr u32 pow_mod(u32 base, u32 exponent, u32 modulus) {
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

constexpr Modulus make_modulus(u32 prime, u32 generator) {
    return {prime, generator, pow_mod(generator, (prime - 1) / 4, prime),
            0xFFFFFFFFFFFFFFFFull / prime};
}

inline constexpr Modulus kModuli[3] = {
    make_modulus(kP1, 31),
    make_modulus(kP2, 3),
    make_modulus(kP3, 3),
};

inline double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

// 锁页内存。显卡往这里拷数据比往普通内存快五倍以上。
class PinnedBuffer {
public:
    explicit PinnedBuffer(size_t count);
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    ~PinnedBuffer();

    u32* data() const { return data_; }

private:
    u32* data_ = nullptr;
};

// ntt.cu：在显卡上把 left 和 right 卷积一遍。
// 三个素数各算一次，结果分别写进 r1 / r2 / r3 的前 coefficient_count 个元素。
// 返回显卡这一段的耗时（毫秒）。
double convolve(const std::vector<u32>& left, const std::vector<u32>& right, bool square,
                size_t coefficient_count, u32* r1, u32* r2, u32* r3);

// decimal.cu：十进制字符串和 limbs 互转。
std::vector<u32> parse_decimal(std::string_view text, const char* label);
size_t format_decimal(const std::vector<u32>& limbs, char* buffer);

// decimal.cu：把三个素数算出的余数拼回精确系数，再进位成十进制。
std::vector<u32> combine_residues(const u32* r1, const u32* r2, const u32* r3, size_t count,
                                  double& merge_ms, double& carry_ms);
