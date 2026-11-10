#!/usr/bin/env python3
"""Generate a reproducible two-line decimal input for ntt_gpu.

The repository ships ``input.zip`` with a 60,000,000 digit benchmark, but that
number only uses the digits 1, 2 and 3 and both lines are identical, so it
never exercises a long carry chain. Use this generator to build the cases that
actually stress the implementation.

Examples::

    python tests/gen_input.py --digits 60000000 --pattern digits123 --out input.txt
    python tests/gen_input.py --digits 1000000 --pattern all9 --out nines.txt
    python tests/gen_input.py --digits 200000 --pattern random --seed 7 --out rnd.txt
"""

import argparse
import random
import sys

PATTERNS = ("all9", "all1", "random", "digits123", "sparse9")


def generate(digits: int, pattern: str, seed: int) -> str:
    if digits < 1:
        raise ValueError("--digits must be at least 1")
    if pattern == "all9":
        return "9" * digits
    if pattern == "all1":
        return "1" * digits
    if pattern == "sparse9":
        # Nine out of ten digits are 9, the rest breaks the runs.
        return "".join("1" if index % 10 == 9 else "9" for index in range(digits))
    if pattern in ("random", "digits123"):
        alphabet = "0123456789" if pattern == "random" else "123"
        rng = random.Random(seed)
        return "".join(rng.choice(alphabet) for _ in range(digits))
    raise ValueError(f"unknown pattern: {pattern}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--digits", type=int, required=True, help="digits per operand")
    parser.add_argument("--pattern", choices=PATTERNS, default="random", help="digit pattern of the left operand")
    parser.add_argument("--second", choices=PATTERNS, default=None, help="digit pattern of the right operand (default: same as --pattern)")
    parser.add_argument("--seed", type=int, default=20241205, help="random seed, keeps the output reproducible")
    parser.add_argument("--out", default="input.txt", help="output file (default: input.txt)")
    parser.add_argument("--newline", default="\n", help=r"line separator, use '\r\n' to mimic a Windows file")
    args = parser.parse_args()

    left = generate(args.digits, args.pattern, args.seed)
    right = generate(args.digits, args.second or args.pattern, args.seed)
    with open(args.out, "w", newline="", encoding="ascii") as handle:
        handle.write(left + args.newline + right + args.newline)

    print(f"wrote {args.out}: {len(left)} x {len(right)} digits ({args.pattern} / {args.second or args.pattern}, seed {args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
