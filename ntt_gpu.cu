// ntt_gpu.cu —— 程序入口。整个流程只有四步：
//
//   1. 读出两行十进制数
//   2. 转成内部表示（decimal.cu 的 parse_decimal）
//   3. 交给显卡做卷积（ntt.cu 的 convolve）
//   4. 把三份结果合成最终答案，转回十进制，写出去（decimal.cu 和本文件）
//
// 显卡那部分在 ntt.cu 里，看不懂可以整个跳过。

#include "ntt.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

constexpr char kDefaultInput[] = "./input.txt";
constexpr char kDefaultOutput[] = "./output.txt";

// 各阶段耗时，单位毫秒。
struct Measurements {
    double read = 0;     // 读文件
    double startup = 0;  // 显卡初始化，和读文件并行
    double parse = 0;    // 十进制转内部表示
    double gpu = 0;      // 显卡上的卷积
    double merge = 0;    // 三份结果合成
    double carry = 0;    // 进位
    double write = 0;    // 转回十进制并写出
};

std::string read_all(const char* path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(std::string("cannot open input file: ") + path);
    }
    input.seekg(0, std::ios::end);
    const std::streamoff length = input.tellg();
    input.seekg(0, std::ios::beg);
    if (length < 0) {
        throw std::runtime_error(std::string("cannot read input file: ") + path);
    }
    std::string text(static_cast<size_t>(length), '\0');
    if (length > 0) {
        input.read(text.data(), static_cast<std::streamsize>(length));
    }
    if (!input) {
        throw std::runtime_error(std::string("error reading input file: ") + path);
    }
    return text;
}

std::string read_stdin() {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    std::string text;
    std::string chunk(1 << 20, '\0');
    while (std::cin) {
        std::cin.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        text.append(chunk.data(), static_cast<size_t>(std::cin.gcount()));
    }
    return text;
}

// 去掉前导零，"000123" 和 "123" 走同一条路。
std::string_view trim_leading_zeroes(std::string_view value) {
    const size_t first = value.find_first_not_of('0');
    if (first == std::string_view::npos) {
        return value.substr(value.size() - 1);
    }
    return value.substr(first);
}

struct TwoLines {
    std::string_view left;
    std::string_view right;
};

// 切出两行。允许 UTF-8 BOM、CRLF、单独 CR、末行没有换行。
TwoLines split_lines(std::string_view text) {
    size_t position = 0;
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBB &&
        static_cast<unsigned char>(text[2]) == 0xBF) {
        position = 3;
    }

    std::string_view lines[2];
    u32 count = 0;
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
            throw std::runtime_error("input has more than two lines");
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
        throw std::runtime_error("input must be exactly two lines of decimal digits");
    }
    if (lines[0].empty()) {
        throw std::runtime_error("first input is empty");
    }
    if (lines[1].empty()) {
        throw std::runtime_error("second input is empty");
    }
    return {lines[0], lines[1]};
}

// 显卡初始化要一百多毫秒，扔到后台和读文件、转换并行。
std::future<double> warm_up_cuda() {
    return std::async(std::launch::async, [] {
        const auto begin = Clock::now();
        cudaFree(nullptr);
        return milliseconds(begin, Clock::now());
    });
}

void write_output(const char* path, const char* buffer, size_t length) {
    if (std::string_view(path) == "-") {
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        std::cout.write(buffer, static_cast<std::streamsize>(length));
        std::cout.flush();
        return;
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error(std::string("cannot open output file: ") + path);
    }
    output.write(buffer, static_cast<std::streamsize>(length));
    if (!output) {
        throw std::runtime_error("error writing output");
    }
}

void print_report(std::ostream& stream, size_t left_digits, size_t right_digits,
                  size_t result_digits, double total_ms, const Measurements& ms) {
    // startup 和 input 是并行跑的，两项相加会略大于 total_ms。
    stream << "left_digits=" << left_digits << '\n'
           << "right_digits=" << right_digits << '\n'
           << "result_digits=" << result_digits << '\n'
           << "total_ms=" << total_ms << '\n'
           << "stage_ms=input " << (ms.read + ms.parse) << " startup " << ms.startup
           << " gpu " << ms.gpu << " merge " << ms.merge << " carry " << ms.carry
           << " write " << ms.write << '\n';
}

// ---------------------------------------------------------------------------
// 压测：内部造两个随机数，跑三轮报最快的一轮
// ---------------------------------------------------------------------------

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
        throw std::runtime_error(std::string("bad digit count: ") + text);
    }
    return static_cast<size_t>(value);
}

void run_bench(const char* count_text) {
    const std::vector<size_t> sizes = count_text != nullptr
        ? std::vector<size_t>{parse_count(count_text)}
        : std::vector<size_t>{1000000, 10000000, 60000000};

    auto warm = warm_up_cuda();
    double startup = 0;
    for (const size_t size : sizes) {
        const std::string left_text = make_digits(size, 1);
        const std::string right_text = make_digits(size, 2);
        if (startup == 0) {
            startup = warm.get();
        }

        double best_total = 0;
        Measurements best;
        size_t best_digits = 0;
        for (int round = 0; round < 3; ++round) {
            Measurements ms;
            ms.startup = round == 0 ? startup : 0;

            const auto parse_begin = Clock::now();
            const std::string_view left = trim_leading_zeroes(left_text);
            const std::string_view right = trim_leading_zeroes(right_text);
            const std::vector<u32> left_limbs = parse_decimal(left, "first input");
            const std::vector<u32> right_limbs = parse_decimal(right, "second input");
            ms.parse = milliseconds(parse_begin, Clock::now());

            const size_t coefficient_count = left_limbs.size() + right_limbs.size() - 1;
            PinnedBuffer r1(coefficient_count);
            PinnedBuffer r2(coefficient_count);
            PinnedBuffer r3(coefficient_count);
            ms.gpu = convolve(left_limbs, right_limbs, left == right, coefficient_count,
                              r1.data(), r2.data(), r3.data());

            const std::vector<u32> result =
                combine_residues(r1.data(), r2.data(), r3.data(), coefficient_count,
                                 ms.merge, ms.carry);

            std::unique_ptr<char[]> buffer(new char[result.size() * kBaseDigits + 16]);
            const auto write_begin = Clock::now();
            const size_t length = format_decimal(result, buffer.get());
            ms.write = milliseconds(write_begin, Clock::now());

            const double total = ms.startup + ms.parse + ms.gpu + ms.merge + ms.carry + ms.write;
            if (round == 0 || total < best_total) {
                best_total = total;
                best = ms;
                best_digits = length;
            }
        }
        // 压测不读文件，read 记为 0。
        best.read = 0;
        print_report(std::cout, size, size, best_digits, best_total, best);
    }
}

void print_usage(std::ostream& stream, const char* program) {
    stream << "usage:\n"
           << "  " << program << "                     read input.txt, write output.txt\n"
           << "  " << program << " in.txt out.txt        use your own file names\n"
           << "  " << program << " in.txt -              write the result to the screen\n"
           << "  " << program << " --bench 200000000     benchmark, no files involved\n"
           << "  " << program << " --help\n"
           << "\n"
           << "The input file must be exactly two lines of decimal digits.\n"
           << "Exit code: 0 ok, 1 error, 2 bad arguments.\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 3) {
        print_usage(std::cerr, argv[0]);
        return 2;
    }
    if (argc > 1 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
        print_usage(std::cout, argv[0]);
        return 0;
    }

    const auto total_begin = Clock::now();
    try {
        if (argc > 1 && std::string_view(argv[1]) == "--bench") {
            run_bench(argc > 2 ? argv[2] : nullptr);
            return 0;
        }

        const char* input_path = argc > 1 ? argv[1] : kDefaultInput;
        const char* output_path = argc > 2 ? argv[2] : kDefaultOutput;
        const bool to_screen = std::string_view(output_path) == "-";
        std::ostream& diagnostics = to_screen ? std::cerr : std::cout;

        Measurements ms;
        auto warm = warm_up_cuda();

        const auto read_begin = Clock::now();
        std::string text = std::string_view(input_path) == "-" ? read_stdin() : read_all(input_path);
        const TwoLines lines = split_lines(text);
        ms.read = milliseconds(read_begin, Clock::now());
        ms.startup = warm.get();

        const auto parse_begin = Clock::now();
        const std::string_view left = trim_leading_zeroes(lines.left);
        const std::string_view right = trim_leading_zeroes(lines.right);
        const std::vector<u32> left_limbs = parse_decimal(left, "first input");
        const std::vector<u32> right_limbs = parse_decimal(right, "second input");
        ms.parse = milliseconds(parse_begin, Clock::now());

        const size_t coefficient_count = left_limbs.size() + right_limbs.size() - 1;
        // 三个素数的乘积约 1.79e28，装不进 u64，用除法比较。
        const u64 bound = static_cast<u64>(kBase - 1) * (kBase - 1) *
                          std::min(left_limbs.size(), right_limbs.size());
        if (bound / kPrime12 >= kP3) {
            throw std::runtime_error("input too large: three primes are not enough");
        }

        PinnedBuffer r1(coefficient_count);
        PinnedBuffer r2(coefficient_count);
        PinnedBuffer r3(coefficient_count);
        ms.gpu = convolve(left_limbs, right_limbs, left == right, coefficient_count,
                          r1.data(), r2.data(), r3.data());

        const std::vector<u32> result = combine_residues(
            r1.data(), r2.data(), r3.data(), coefficient_count, ms.merge, ms.carry);
        std::string().swap(text);  // 原文用完就还回去，上限用例能省 1 GB

        std::unique_ptr<char[]> buffer(new char[result.size() * kBaseDigits + 16]);
        const auto write_begin = Clock::now();
        const size_t length = format_decimal(result, buffer.get());
        buffer[length] = '\n';
        write_output(output_path, buffer.get(), length + 1);
        ms.write = milliseconds(write_begin, Clock::now());

        print_report(diagnostics, left.size(), right.size(), length,
                     milliseconds(total_begin, Clock::now()), ms);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
