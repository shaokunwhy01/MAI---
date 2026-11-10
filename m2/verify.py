"""m2/verify.py —— 完整性自检（可重复运行）

只验**机制**，不验效果。效果（方向性）需要 S3 数据量，这里不碰。

    python -m m2.verify
    m2ctl verify

产出 `m2_state/verify.json`，每条判据一行 PASS/FAIL。
"""

from __future__ import annotations

import argparse
import json
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .bootstrap import STATE_DIR, load_bootstrap
from .generate import generate
from .memory import BiasConfig, Memory
from .readout import FrozenReadout

EXPECTED_KEYS = {"i", "id", "tok", "z", "p", "a", "reb", "bias", "c", "r", "phi_T", "R_p", "ver"}
PROMPTS = [
    "什么是递归",
    "解释一下梯度下降",
    "写一个快速排序的思路",
    "为什么天空是蓝色的",
    "什么是贝叶斯定理",
]


class Report:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: Any = "") -> bool:
        self.rows.append({"check": name, "pass": bool(ok), "detail": detail})
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f"   {detail}" if detail != "" else ""), flush=True)
        return ok

    def summary(self) -> dict[str, Any]:
        n = len(self.rows)
        p = sum(1 for r in self.rows if r["pass"])
        return {"total": n, "pass": p, "fail": n - p, "rows": self.rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--max-new-tokens", type=int, default=8)
    ap.add_argument("--out", default=str(STATE_DIR / "verify.json"))
    args = ap.parse_args()

    rep = Report()
    n = args.max_new_tokens
    foo = PROMPTS[0]

    print("[1/8] 加载底座与记忆 ...", flush=True)
    readout = FrozenReadout()
    bs = load_bootstrap()
    if bs is None:
        print("缺少 bootstrap.json，先跑 m2ctl bootstrap", flush=True)
        raise SystemExit(2)
    mem = Memory(readout, bs)
    cfg1 = BiasConfig(mu=1.0)
    cfg0 = BiasConfig(mu=0.0)
    print(f"      事件 {len(mem.events)} 条，κ_d={bs['kappa_d']:.2f}, "
          f"‖μ0‖={bs['mu0_norm']:.1f}, ‖μ_W‖={bs['mu_w_norm']:.3f}", flush=True)

    print("[2/8] 启动判据：μ=0 是否逐位等于纯底座", flush=True)
    base = generate(readout, foo, max_new_tokens=n, record=False)
    zero = generate(readout, foo, max_new_tokens=n, record=False,
                    hook=lambda s, z, st: mem.bias(s, z, st, cfg0))
    rep.check("parity: μ=0 == 纯底座", base.ids == zero.ids,
              f"bias_absmax={zero.run_stats.get('bias_absmax')}")

    print("[3/8] 确定性：同版本重跑逐位相同", flush=True)
    again = generate(readout, foo, max_new_tokens=n, record=False)
    rep.check("determinism: 重跑一致", again.ids == base.ids, f"n={n}")

    print("[4/8] 注入：μ=1 时 bias 非零且有界", flush=True)
    one = generate(readout, foo, max_new_tokens=n, record=True,
                   hook=lambda s, z, st: mem.bias(s, z, st, cfg1))
    bmax = float(one.run_stats.get("bias_absmax") or 0.0)
    bound = 1.0 * (1.0 + cfg1.rho + cfg1.nu)
    rep.check("bias 非零", bmax > 1e-6, f"max|bias|={bmax:.4f}")
    rep.check("bias 有界", bmax <= bound + 1e-4, f"上界 {bound:.2f}")

    print("[5/8] 线性：μ 翻倍，|bias| 应近似翻倍", flush=True)
    half = generate(readout, foo, max_new_tokens=n, record=True,
                    hook=lambda s, z, st: mem.bias(s, z, st, BiasConfig(mu=0.5)))
    b_half = float(half.run_stats.get("bias_absmax") or 0.0)
    ratio = (bmax / b_half) if b_half > 1e-9 else float("inf")
    rep.check("μ 线性", 1.6 <= ratio <= 2.4, f"ratio={ratio:.2f}")

    print("[6/8] 账本 schema：13 个字段齐全", flush=True)
    keys = set(one.ledger[0].keys()) if one.ledger else set()
    rep.check("schema 完整", keys == EXPECTED_KEYS,
              f"缺失 {sorted(EXPECTED_KEYS - keys)} / 多余 {sorted(keys - EXPECTED_KEYS)}")

    print("[7/8] 多提示词稳健性", flush=True)
    ok_all, finite_all = True, True
    for p in PROMPTS:
        try:
            r = generate(readout, p, max_new_tokens=n, record=True,
                         hook=lambda s, z, st: mem.bias(s, z, st, cfg1))
            finite_all &= all(np.isfinite(v) for v in
                              (r.ledger[0]["z"], r.ledger[0]["bias"], r.ledger[0]["a"]))
            ok_all &= len(r.ids) > 0
        except Exception:
            ok_all = False
            traceback.print_exc()
    rep.check("多提示词不崩", ok_all, f"{len(PROMPTS)} 条")
    rep.check("数值有限", finite_all, "")

    print("[8/8] 空记忆 + 快照重载", flush=True)
    with tempfile.TemporaryDirectory() as td:
        empty = Memory(readout, bs, root=Path(td))
        e_base = generate(readout, foo, max_new_tokens=n, record=False)
        e_hook = generate(readout, foo, max_new_tokens=n, record=False,
                          hook=lambda s, z, st: empty.bias(s, z, st, cfg1))
        rep.check("空记忆 == 纯底座", e_base.ids == e_hook.ids,
                  f"bias_absmax={e_hook.run_stats.get('bias_absmax')}")

    mem2 = Memory(readout, bs)
    same = (len(mem2.events) == len(mem.events)
            and mem2.meta.get("m2_version") == mem.meta.get("m2_version")
            and abs((mem2.meta.get("R_plus") or 0) - (mem.meta.get("R_plus") or 0)) < 1e-9)
    rep.check("快照重载一致", same, f"m2={mem2.meta.get('m2_version')}")

    s = rep.summary()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**s, "n_tokens": n}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{s['pass']}/{s['total']} 通过   写入 {out}")
    raise SystemExit(0 if s["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
