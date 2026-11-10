// decimal.cu —— 十进制字符串和内部表示（limbs）之间的来回转换。
//
// limbs：八位十进制一组，一组装进一个 uint32，低位在前。
// 例如 "123456789012" 存成 {34567890, 1234}。
//
// 这里的活都在 CPU 上，而且天然能切成互不相干的块（第三块算错了不影响第一块），
// 所以切块分给几个线程同时做。

#include "ntt.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace {

// 把 [0, count) 切给几个线程，fn(begin, end) 会被并发调用。
template <typename Fn>
void parallel_do(size_t count, size_t min_per_worker, Fn&& fn) {
    unsigned workers = std::thread::hardware_concurrency();
    if (workers == 0) {
        workers = 4;
    }
    workers = static_cast<unsigned>(
        std::max<size_t>(1, std::min<size_t>(workers, (count + min_per_worker - 1) / min_per_worker)));
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

// 4 位十进制到字符的查表，格式化时一次拷贝 4 个字节。
const std::array<char, 40000>& digit_table() {
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

// 三素数合成要用的两个倒数。
struct CrtConstants {
    u32 inverse_p1_mod_p2;
    u32 inverse_p12_mod_p3;
};

CrtConstants make_crt_constants() {
    CrtConstants constants{};
    constants.inverse_p1_mod_p2 = pow_mod(kP1 % kP2, kP2 - 2, kP2);
    constants.inverse_p12_mod_p3 = pow_mod(static_cast<u32>(kPrime12 % kP3), kP3 - 2, kP3);
    return constants;
}

// 三个余数拼出精确系数 x（最大约 6.8e23，超过 64 位，用三个 32 位字），
// 再拆成 x = quotient * 10^8 + remainder。
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

    // x = partial + p1*p2*t2，用三个 32 位字装。
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

    // 除以 10^8。最高字不到 2^16（系数 < 6.8e23），本来就不够 10^8，第一步跳过。
    u64 rest = word2;
    u64 current = (rest << 32) | word1;
    const u32 quotient_high = static_cast<u32>(current / kBase);
    rest = current % kBase;
    current = (rest << 32) | word0;
    const u32 quotient_low = static_cast<u32>(current / kBase);
    rest = current % kBase;

    // 由输入尺寸保证商小于 2^64（单边 5.37 亿位时约 6.8e15）。
    quotient = static_cast<u64>(quotient_low) | (static_cast<u64>(quotient_high) << 32);
    remainder = static_cast<u32>(rest);
}

}  // namespace

// 十进制字符串转 limbs。每八个字符正好一个 limb，校验并入同一趟：
// 先用位运算一次检查八个字节是不是都是数字，再用三次乘加拼出数值。
std::vector<u32> parse_decimal(std::string_view text, const char* label) {
    const size_t digits = text.size();
    const size_t groups = digits / kBaseDigits;
    const size_t head = digits % kBaseDigits;
    const char* data = text.data();
    std::vector<u32> limbs(groups + (head != 0 ? 1 : 0));

    std::atomic<bool> invalid{false};
    parallel_do(groups, 1u << 13, [&](size_t begin, size_t end) {
        for (size_t group = begin; group < end; ++group) {
            const char* at = data + digits - kBaseDigits * (group + 1);
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

    // 最左边不足八位的那几个字符。
    if (head != 0) {
        u32 chunk = 0;
        for (size_t offset = 0; offset < head; ++offset) {
            const u32 digit = static_cast<u32>(static_cast<unsigned char>(data[offset])) - '0';
            if (digit > 9) {
                invalid = true;
                break;
            }
            chunk = chunk * 10u + digit;
        }
        limbs[groups] = chunk;
    }

    if (invalid) {
        throw std::runtime_error(std::string(label) + " contains a non-digit character");
    }
    return limbs;
}

// limbs 转十进制字符，返回位数。buffer 至少要 8 * limbs.size() + 1 字节。
size_t format_decimal(const std::vector<u32>& limbs, char* buffer) {
    const std::array<char, 40000>& table = digit_table();

    // 最高一组不用补前导零，长度不定，先单独写。
    const std::string highest = std::to_string(limbs.back());
    std::memcpy(buffer, highest.data(), highest.size());
    const size_t head = highest.size();

    // 其余每组固定八位，位置能直接算出来，所以可以并行。
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

// 从三个素数算出的余数拼回精确值，再从低位往高位进位，得到十进制 limbs。
//
// 拼一个系数要几步大数运算，很慢，但每个系数互不相干，所以并行做；
// 进位天生只能一个接一个，所以串行做。
// 为了内存不随输入暴涨，按块处理：一块并行拼、紧接着串行进位。
std::vector<u32> combine_residues(const u32* r1, const u32* r2, const u32* r3, size_t count,
                                  double& merge_ms, double& carry_ms) {
    constexpr size_t kTile = 1u << 22;  // 一块 419 万个系数，约 50 MB
    const CrtConstants constants = make_crt_constants();
    std::vector<u64> quotients(std::min(count, kTile) + 1);
    std::vector<u32> remainders(std::min(count, kTile) + 1);

    std::vector<u32> result;
    result.reserve(count + 7);
    u64 carry = 0;
    merge_ms = 0;
    carry_ms = 0;

    for (size_t base = 0; base < count; base += kTile) {
        const size_t tile = std::min(kTile, count - base);

        const auto begin = Clock::now();
        parallel_do(tile, 1u << 13, [&](size_t block_begin, size_t block_end) {
            for (size_t index = block_begin; index < block_end; ++index) {
                crt_split(r1[base + index], r2[base + index], r3[base + index], constants,
                          quotients[index], remainders[index]);
            }
        });
        merge_ms += milliseconds(begin, Clock::now());

        const auto carry_begin = Clock::now();
        for (size_t index = 0; index < tile; ++index) {
            const u64 sum = remainders[index] + carry;
            result.push_back(static_cast<u32>(sum % kBase));
            carry = quotients[index] + sum / kBase;
        }
        carry_ms += milliseconds(carry_begin, Clock::now());
    }

    // 最后剩下的进位就是结果的最高几位。
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
