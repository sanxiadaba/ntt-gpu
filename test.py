#!/usr/bin/env python3
"""用 Python 自带的大整数当标准答案，对账 ntt_gpu.exe。

    python test.py            # 秒级
    python test.py --big      # 追加 100 万位
    python test.py --fuzz 200 # 追加 200 组随机尺寸
    python test.py --huge     # 追加 5.37 亿位（单边上限，约 4 GB 内存）

逐字节比较乘积。覆盖 0、前导零、全 9（最长进位链）、10 的幂、不平衡输入、
BOM / CRLF / 单独 CR / 末行无换行，以及各种非法输入的错误信息。
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile
import time

sys.set_int_max_str_digits(0)

EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ntt_gpu.exe")
MAX_DIGITS = 1 << 29  # 4 * 2^27，单边位数上限


def decode(data):
    """程序按控制台代码页输出提示信息，两种编码都试一下。"""
    for encoding in ("utf-8", "gbk"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def digits(count, seed, alphabet="0123456789"):
    rng = random.Random(seed)
    return "".join(rng.choice(alphabet) for _ in range(count)).lstrip("0") or "7"


def run(payload, expect_error=None):
    """跑一次，成功返回输出文本，失败返回 False。"""
    with tempfile.TemporaryDirectory() as workdir:
        source = os.path.join(workdir, "in.txt")
        target = os.path.join(workdir, "out.txt")
        with open(source, "wb") as handle:
            handle.write(payload)
        result = subprocess.run([EXE, source, target], capture_output=True)
        if expect_error is not None:
            return result.returncode == 1 and expect_error in decode(result.stderr)
        if result.returncode != 0:
            print(f"    退出码 {result.returncode}: {decode(result.stderr).strip()}")
            return False
        with open(target, "r", encoding="ascii") as handle:
            return handle.read().strip()


def check(name, left, right):
    expected = str(int(left) * int(right))
    got = run((left + "\n" + right + "\n").encode())
    ok = got == expected
    print(f"[{'OK' if ok else 'FAIL'}] {name:<26} {len(expected)} 位")
    if not ok and isinstance(got, str):
        print(f"       期望 {expected[:40]}...  得到 {got[:40]}...")
    return ok


def fuzz(count):
    """随机尺寸对账，重点照顾 limb 边界（8 位一个 limb）。"""
    sizes = [1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 129,
             255, 256, 1000, 4096, 8191, 8192, 20000]
    passes = 0
    started = time.time()
    for index in range(count):
        if index % 3 == 0:
            left_size = sizes[index % len(sizes)]
            right_size = sizes[(index * 7 + 3) % len(sizes)]
        else:
            rng = random.Random(index)
            left_size = rng.randrange(1, 20000)
            right_size = rng.randrange(1, 20000)
        alphabet = "0123456789" if index % 4 else "09"
        left = digits(left_size, index, alphabet)
        right = digits(right_size, index + 1000, alphabet)
        if index % 5 == 0:
            right = left
        expected = str(int(left) * int(right))
        got = run((left + "\n" + right + "\n").encode())
        if got != expected:
            print(f"[FAIL] fuzz #{index} {left_size}×{right_size}")
            print(f"       期望 {expected[:40]}...  得到 {str(got)[:40]}...")
            return passes, count
        passes += 1
    print(f"[OK] fuzz {passes} 组，{time.time() - started:.1f} s")
    return passes, count


def huge(path):
    """单边上限：全 9 平方，答案有闭式，不必用 Python 硬乘。"""
    length = MAX_DIGITS
    block = "9" * (1 << 22)
    with open(path, "wb") as handle:
        for _ in range(2):
            written = 0
            while written < length:
                take = min(len(block), length - written)
                handle.write(block[:take].encode())
                written += take
            handle.write(b"\n")
    print(f"[..] 已写入 {os.path.getsize(path) / 2**30:.2f} GiB 输入")

    target = path + ".out"
    started = time.time()
    result = subprocess.run([EXE, path, target], capture_output=True)
    if result.returncode != 0:
        print(f"[FAIL] 上限用例退出码 {result.returncode}: {decode(result.stderr).strip()}")
        return False
    print(f"[..] 用时 {time.time() - started:.1f} s")

    # (10^k - 1)^2 = 10^2k - 2*10^k + 1：(k-1) 个 9、一个 8、(k-1) 个 0、一个 1
    ok = True
    position = 0
    with open(target, "rb") as handle:
        while ok:
            chunk = handle.read(1 << 24)
            if not chunk:
                break
            for offset, byte in enumerate(chunk):
                at = position + offset
                if at < length - 1:
                    want = 0x39  # '9'
                elif at == length - 1:
                    want = 0x38  # '8'
                elif at < 2 * length - 1:
                    want = 0x30  # '0'
                elif at == 2 * length - 1:
                    want = 0x31  # '1'
                elif at == 2 * length:
                    want = 0x0A  # 换行
                else:
                    ok = False
                    break
                if byte != want:
                    ok = False
                    print(f"       位置 {at} 期望 {chr(want)!r} 得到 {chr(byte)!r}")
                    break
            position += len(chunk)
    if position != 2 * length + 1:
        ok = False
    print(f"[{'OK' if ok else 'FAIL'}] 上限 {length} 位 × {length} 位")
    os.remove(target)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--big", action="store_true", help="追加 100 万位用例")
    parser.add_argument("--fuzz", type=int, default=0, metavar="N", help="追加 N 组随机用例")
    parser.add_argument("--huge", action="store_true", help="追加 5.37 亿位上限用例")
    options = parser.parse_args()

    passes = 0
    total = 0

    pairs = [
        ("零", "0", "0"),
        ("前导零", "00000", "0000000"),
        ("单个数字", "7", "9"),
        ("全 9 · 131 位", "9" * 131, "9" * 131),
        ("全 9 · 4096 位", "9" * 4096, "9" * 4096),
        ("随机 · 5000 位", digits(5000, 1), digits(5000, 2)),
        ("全 9 × 全 1 · 6 万位", "9" * 60000, "1" * 60000),
        ("10 的幂", "1" + "0" * 99999, "1" + "0" * 99999),
        ("不平衡 · 1 × 10 万位", "7", "9" * 100000),
        ("随机 · 30 万位", digits(300000, 3), digits(300000, 4)),
    ]
    if options.big:
        pairs.append(("全 9 · 100 万位", "9" * 1000000, "9" * 1000000))
        pairs.append(("随机 · 100 万位", digits(1000000, 11), digits(1000000, 12)))

    for name, left, right in pairs:
        total += 1
        passes += check(name, left, right)

    expected = str(int("123456789") * int("987654321"))
    variants = {
        "CRLF": b"123456789\r\n987654321\r\n",
        "末行无换行": b"123456789\n987654321",
        "末尾空行": b"123456789\n987654321\n\n",
        "单独 CR": b"123456789\r987654321\r",
        "UTF-8 BOM": b"\xef\xbb\xbf123456789\n987654321\n",
    }
    for name, payload in variants.items():
        total += 1
        ok = run(payload) == expected
        print(f"[{'OK' if ok else 'FAIL'}] 换行 · {name}")
        passes += ok

    errors = {
        "三行": (b"1\n2\n3\n", "输入多于两行"),
        "只有一行": (b"123\n", "恰好两行"),
        "空文件": (b"", "恰好两行"),
        "第二行为空": (b"123\n\n", "第二个输入为空"),
        "含字母": (b"12a\n34\n", "非十进制字符"),
        "含空格": (b"12 \n34\n", "非十进制字符"),
        "中文": ("12中\n34\n".encode(), "非十进制字符"),
    }
    for name, (payload, message) in errors.items():
        total += 1
        ok = run(payload, expect_error=message)
        print(f"[{'OK' if ok else 'FAIL'}] 非法 · {name}")
        passes += ok

    if options.fuzz:
        good, count = fuzz(options.fuzz)
        passes += good
        total += count

    if options.huge:
        total += 1
        with tempfile.TemporaryDirectory() as workdir:
            passes += huge(os.path.join(workdir, "huge.txt"))

    print(f"\n{passes}/{total} 通过")
    return 0 if passes == total else 1


if __name__ == "__main__":
    sys.exit(main())
