"""Full pipeline entry point. Run with: python main.py"""
import argparse

from config import SEED, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Lab03 statistical arbitrage pipeline")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="fast run that never overwrites artifacts/ or docs/",
    )
    return parser.parse_args()


def main(quick=False):
    set_seed()
    print(f"Seed fixed at {SEED}")
    print("Mode: quick" if quick else "Mode: full")


if __name__ == "__main__":
    args = parse_args()
    main(quick=args.quick)
