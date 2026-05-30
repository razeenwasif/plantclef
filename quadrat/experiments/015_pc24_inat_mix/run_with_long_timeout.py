"""
015 — Wrapper that bumps DDP NCCL watchdog timeout to 2h, then invokes 010's
train.py without modifying shared code.

Why this exists:
    Epoch 1 of 015 trained fine but the validation pass at the end took >10 min
    (val set is ~272k rows after iNat is mixed in, ~2× the 014 anchor val set).
    The default ProcessGroupNCCL watchdog timeout in PyTorch is 10 min, so the
    rank-1 process killed the group on a 1-element ALLREDUCE while rank 0 was
    still validating. Training crashed *after* best.pt + last.pt saved cleanly.

Fix:
    Monkey-patch torch.distributed.init_process_group at import time so any
    call without an explicit `timeout=` kwarg gets `timedelta(hours=2)`. Then
    runpy.run_path 010's train.py in __main__ scope. CLI args go through
    untouched via sys.argv.

Sandbox:
    Edit-in-place of 010/train.py is blocked. This wrapper sits in 015's dir
    and changes nothing under 010/.
"""
from __future__ import annotations

import os
import runpy
import sys
from datetime import timedelta

import torch.distributed as dist


_LONG_TIMEOUT = timedelta(hours=2)
_orig_init_pg = dist.init_process_group


def _init_pg_with_long_timeout(*args, **kwargs):
    if "timeout" not in kwargs:
        kwargs["timeout"] = _LONG_TIMEOUT
    return _orig_init_pg(*args, **kwargs)


dist.init_process_group = _init_pg_with_long_timeout


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: run_with_long_timeout.py <path_to_train.py> [args...]", file=sys.stderr)
        return 2
    target = sys.argv[1]
    if not os.path.isfile(target):
        print(f"missing target script: {target}", file=sys.stderr)
        return 2
    sys.argv = [target] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
    runpy.run_path(target, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
