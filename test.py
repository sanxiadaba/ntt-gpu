#!/usr/bin/env python3
"""用 Python 自带的大整数当标准答案，对账 ntt_gpu.exe。

    python test.py          # 秒级
    python test.py --big    # 追加 100 万位用例

对照是逐字节比较乘积，覆盖 0、前导零、全 9（最长进位链）、10 的幂、
不平衡输入，以及 BOM / CRLF / 单独 CR / 末行无换行。
"""

import os
import random
import subprocess
import sys
import tempfile

sys.set_int_max_str_digits(0)

EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ntt_gpu.exe")


def decode(data):
    """程序按系统代码页（中文 Windows 是 GBK）输出提示信息，两种编码都试一下。"""
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


def main():
    big = "--big" in sys.argv

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
    if big:
        pairs.append(("全 9 · 100 万位", "9" * 1000000, "9" * 1000000))
        pairs.append(("随机 · 100 万位", digits(1000000, 11), digits(1000000, 12)))

    failures = 0
    for name, left, right in pairs:
        expected = str(int(left) * int(right))
        got = run((left + "\n" + right + "\n").encode())
        ok = got == expected
        print(f"[{'OK' if ok else 'FAIL'}] {name:<24} {len(expected)} 位")
        if not ok and isinstance(got, str):
            print(f"       期望 {expected[:40]}...  得到 {got[:40]}...")
        failures += 0 if ok else 1

    # 换行与 BOM 的各种写法都应得到同一结果
    expected = str(int("123456789") * int("987654321"))
    variants = {
        "CRLF": b"123456789\r\n987654321\r\n",
        "末行无换行": b"123456789\n987654321",
        "末尾空行": b"123456789\n987654321\n\n",
        "单独 CR": b"123456789\r987654321\r",
        "UTF-8 BOM": b"\xef\xbb\xbf123456789\n987654321\n",
    }
    for name, payload in variants.items():
        ok = run(payload) == expected
        print(f"[{'OK' if ok else 'FAIL'}] 换行 · {name}")
        failures += 0 if ok else 1

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
        ok = run(payload, expect_error=message)
        print(f"[{'OK' if ok else 'FAIL'}] 非法 · {name}")
        failures += 0 if ok else 1

    total = len(pairs) + len(variants) + len(errors)
    print(f"\n{total - failures}/{total} 通过")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
