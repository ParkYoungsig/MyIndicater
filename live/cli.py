import argparse

from live.engine import LiveEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="모의 실행")
    args = parser.parse_args()

    engine = LiveEngine(mock=args.mock)
    engine.run_once()


if __name__ == "__main__":
    main()
