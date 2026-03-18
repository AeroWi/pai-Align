# CS336 Spring 2025 Assignment 5: Alignment
第一步：切除“毒瘤” flash-attn
在左侧文件树找到并双击打开 pyproject.toml。
找到 dependencies 列表里 flash-attn 那一行。
直接删掉这一行（或者在前面加个 # 注释掉），保存文件 (Ctrl + S)。
=================================================
请立刻在左边编辑器打开 scripts/train_ei.py：
找到 attn_implementation="flash_attention_2"。
把它改成 attn_implementation="sdpa"。

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

