"""
Thin wrapper that delegates to `openai_embed.cli:main`.
"""

from openai_embed.cli import main as _cli_main


def main() -> None:
    _cli_main()


if __name__ == "__main__":
    main()
