import sys
import os
import argparse

# When running `python live/cli.py` ensure project root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from live.engine import LiveEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="모의 실행")
    args = parser.parse_args()

    engine = LiveEngine(mock=args.mock)
    engine.run_once()


if __name__ == "__main__":
    main()
