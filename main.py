#!/usr/bin/env python3

from bootstrap import ensure_src_on_path

ensure_src_on_path()

from agent_multi_cr.cli import main


if __name__ == "__main__":
    main()
