#!/usr/bin/env python3
"""Black box tests for ntt_gpu.exe.

The oracle is Python's own big integers, which is why this file needs nothing
but the standard library. Run it from anywhere:

    python tests/run_tests.py                # quick suite, a few seconds
    python tests/run_tests.py --full         # adds 1,000,000 digit cases
    python tests/run_tests.py --benchmark    # also replays input.zip end to end

``--benchmark`` expects ``input.txt`` next to the executable (unzip input.zip
first) and checks the SHA-256 published in the README.
"""

import argparse
import hashlib
import os
import random
import subprocess
import sys
import tempfile
import time

sys.set_int_max_str_digits(0)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_SHA256 = "A095C60E4EB753E9AB0B61D0C18F9C70176970BF1340D42FFC156118D622833E"


def digits(count, alphabet="0123456789", seed=7):
    rng = random.Random(seed)
    return "".join(rng.choice(alphabet) for _ in range(count))


def run_case(exe, workdir, name, payload, expected=None, expect_error=None, timeout=900):
    input_path = os.path.join(workdir, "case_in.txt")
    output_path = os.path.join(workdir, "case_out.txt")
    with open(input_path, "wb") as handle:
        handle.write(payload)
    if os.path.exists(output_path):
        os.remove(output_path)

    started = time.time()
    process = subprocess.run([exe, input_path, output_path], capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - started

    if expect_error is not None:
        ok = process.returncode == 1 and expect_error in process.stderr
        detail = f"rc={process.returncode} stderr={process.stderr.strip()!r}"
        print(f"[{'OK' if ok else 'FAIL'}] {name:<28} {elapsed:6.2f}s  {'' if ok else detail}")
        return ok

    if process.returncode != 0:
        print(f"[FAIL] {name:<28} {elapsed:6.2f}s  rc={process.returncode} {process.stderr.strip()}")
        return False

    with open(output_path, "r", encoding="ascii") as handle:
        got = handle.read().strip()
    ok = got == expected
    print(f"[{'OK' if ok else 'FAIL'}] {name:<28} {elapsed:6.2f}s  {len(expected)} digits")
    if not ok:
        print(f"       expected {expected[:60]}... ({len(expected)} digits)")
        print(f"       got      {got[:60]}... ({len(got)} digits)")
        for index, (left, right) in enumerate(zip(expected, got)):
            if left != right:
                print(f"       first difference at {index}: expected {expected[index:index + 12]!r} got {got[index:index + 12]!r}")
                break
    return ok


def product(left, right):
    return str(int(left) * int(right))


def quick_cases():
    cases = []
    cases.append(("zero", b"0\n0\n", product("0", "0")))
    cases.append(("zero padded", b"00000\n0000\n", product("0", "0")))
    cases.append(("one", b"1\n1\n", "1"))
    cases.append(("all nines 131", (b"9" * 131 + b"\n" + b"9" * 131 + b"\n"), product("9" * 131, "9" * 131)))
    cases.append(("all nines 4096", (b"9" * 4096 + b"\n" + b"9" * 4096 + b"\n"), product("9" * 4096, "9" * 4096)))
    cases.append(("random 5k", (digits(5000, seed=1) + "\n" + digits(5000, seed=2) + "\n").encode(),
                  product(digits(5000, seed=1), digits(5000, seed=2))))
    cases.append(("all nines 50k", (b"9" * 50000 + b"\n" + b"9" * 50000 + b"\n"),
                  product("9" * 50000, "9" * 50000)))
    cases.append(("ones x nines 60k", (b"1" * 60000 + b"\n" + b"9" * 60000 + b"\n"),
                  product("1" * 60000, "9" * 60000)))
    cases.append(("power of ten", (b"1" + b"0" * 99999 + b"\n" + b"1" + b"0" * 99999 + b"\n"),
                  product("1" + "0" * 99999, "1" + "0" * 99999)))
    cases.append(("unbalanced 1 x 100k", (b"7\n" + b"9" * 100000 + b"\n"),
                  product("7", "9" * 100000)))

    bom = b"\xef\xbb\xbf" + b"123456789\n987654321\n"
    cases.append(("utf-8 bom", bom, product("123456789", "987654321")))
    cases.append(("crlf", b"123456789\r\n987654321\r\n", product("123456789", "987654321")))
    cases.append(("no trailing newline", b"123456789\n987654321", product("123456789", "987654321")))
    cases.append(("trailing blank line", b"123456789\n987654321\n\n", product("123456789", "987654321")))
    cases.append(("cr only", b"123456789\r987654321\r", product("123456789", "987654321")))
    return cases


def error_cases():
    return [
        ("three lines", b"1\n2\n3\n", "more than two lines"),
        ("one line", b"123\n", "exactly two decimal lines"),
        ("empty file", b"", "exactly two decimal lines"),
        ("blank second line", b"123\n\n", "Second input is empty"),
        ("letters", b"12a\n34\n", "non-decimal character"),
        ("trailing spaces", b"12 \n34\n", "non-decimal character"),
        ("mixed newline junk", b"12\n34\n56", "more than two lines"),
    ]


def full_cases():
    cases = []
    cases.append(("all nines 1M", (b"9" * 1000000 + b"\n" + b"9" * 1000000 + b"\n"),
                  product("9" * 1000000, "9" * 1000000)))
    cases.append(("random 1M x 1M", (digits(1000000, seed=11) + "\n" + digits(1000000, seed=12) + "\n").encode(),
                  product(digits(1000000, seed=11), digits(1000000, seed=12))))
    return cases


def benchmark(exe):
    input_path = os.path.join(REPO_ROOT, "input.txt")
    if not os.path.exists(input_path):
        print("[SKIP] benchmark: input.txt not found, run: tar -xf input.zip input.txt")
        return True
    output_path = os.path.join(tempfile.gettempdir(), "ntt_gpu_benchmark.txt")
    started = time.time()
    process = subprocess.run([exe, input_path, output_path], capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - started
    if process.returncode != 0:
        print(f"[FAIL] benchmark rc={process.returncode} {process.stderr.strip()}")
        return False
    digest = hashlib.sha256(open(output_path, "rb").read()).hexdigest().upper()
    ok = digest == BENCHMARK_SHA256
    print(f"[{'OK' if ok else 'FAIL'}] benchmark 60M x 60M      {elapsed:6.2f}s  sha256 {digest}")
    if not ok:
        print(f"       expected {BENCHMARK_SHA256}")
    for line in process.stdout.splitlines():
        print(f"       {line}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", default=os.path.join(REPO_ROOT, "ntt_gpu.exe"), help="path to ntt_gpu.exe")
    parser.add_argument("--full", action="store_true", help="add the 1,000,000 digit cases")
    parser.add_argument("--benchmark", action="store_true", help="also replay input.zip and check its SHA-256")
    parser.add_argument("--only", default=None, help="substring filter on the case name")
    parser.add_argument("--errors-only", action="store_true",
                        help="only the invalid-input cases; they never touch the GPU, so this runs on a CPU-only CI machine")
    args = parser.parse_args()

    if not os.path.exists(args.exe):
        print(f"error: {args.exe} not found, build it first (./build.ps1)")
        return 2

    cases = quick_cases() + (full_cases() if args.full else [])
    errors = error_cases()
    if args.only:
        cases = [case for case in cases if args.only in case[0]]
        errors = [case for case in errors if args.only in case[0]]
    if args.errors_only:
        cases = []

    passed = failed = 0
    with tempfile.TemporaryDirectory(prefix="ntt_gpu_tests_") as workdir:
        for name, payload, expected in cases:
            if run_case(args.exe, workdir, name, payload, expected=expected):
                passed += 1
            else:
                failed += 1
        for name, payload, message in errors:
            if run_case(args.exe, workdir, name, payload, expect_error=message):
                passed += 1
            else:
                failed += 1

        if args.benchmark and not args.errors_only and not args.only:
            if benchmark(args.exe):
                passed += 1
            else:
                failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
