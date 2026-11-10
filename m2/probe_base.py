"""m2/probe_base.py —— 底座选型探针（P1 / P2 / P4）

回答一个问题：M2 该坐在 base 上还是 instruct 上。

    P1 输出形态  —— 生成出来的东西是"回答"还是"续写"
    P2 参考系    —— T_p 指向内容词还是虚词/格式，R_p 有多大
    P4 迁移成本  —— 两个模型词嵌入的一致性

用法：
    python -m m2.probe_base --models base,instruct
    python -m m2.probe_base --models base        # 只跑 base
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .readout import WS, FrozenReadout, resolve_model_path
from .generate import generate

STATE_DIR = WS / "m2_state"

MODELS = {
    "base": "Qwen/Qwen2.5-0.5B",
    "instruct": "Qwen/Qwen2.5-0.5B-Instruct",
}

PROMPTS = [
    "黎曼猜想是什么？",
    "什么是梯度下降",
    "写一个快速排序",
    "解释一下什么是过拟合",
    "如何判断一个数是不是质数",
    "为什么天空是蓝色的",
    "用一句话概括相对论",
    "什么是区块链",
    "写一段 Python 读取 CSV",
    "解释牛顿第三定律",
    "什么是递归",
    "如何提高睡眠质量",
    "解释什么是熵",
    "写一个二分查找",
    "什么是贝叶斯定理",
]

FUNC_WORDS = set(
    "的 了 是 在 和 与 或 也 就 都 而 及 其 这 那 有 我 你 他 它 们 个 一 不 要 会 可 以 "
    "对 于 为 从 到 把 被 让 给 但 还 又 再 只 很 更 最 等 中 上 下 时 里 说 做 用 能 需 "
    "该 将 已 未 着 过 地 得 吗 呢 吧 啊 所 因 此 如 若 则 即 且 并 由 向 往 之 者".split()
)
LIST_RE = re.compile(r"^\s*(\d+[.、)]|[-*•#]|第[一二三四五六七八九十])")


def is_format(tok: str) -> bool:
    s = tok.strip()
    if s == "" or "\ufffd" in s:
        return True
    if all(c in string.punctuation + string.digits + " \t\n" for c in s):
        return True
    return s in FUNC_WORDS


def metrics(text: str, prompt: str) -> dict[str, Any]:
    t = text.strip()
    n4 = max(len(t) - 3, 0)
    grams = [t[i:i + 4] for i in range(n4)]
    distinct = len(set(grams)) / max(len(grams), 1)

    lcp = 0
    for a, b in zip(text, prompt):
        if a != b:
            break
        lcp += 1

    return {
        "chars": len(t),
        "starts_list": bool(LIST_RE.match(text)),
        "ends_terminal": t.endswith(("。", "！", "？", ".", "!", "?")),
        "distinct_4gram": round(distinct, 3),
        "prompt_echo": round(lcp / max(len(prompt), 1), 3),
    }


def p2_for(readout: FrozenReadout, prompts: list[str], ids_of) -> list[dict[str, Any]]:
    """一步前向，量 T_p 的可靠性与语义。"""
    W = readout.W.float().cpu().numpy()
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)

    rows = []
    for prompt in prompts:
        ids = ids_of(readout, prompt)
        out = readout.forward(ids)
        z = out["z"].cpu().numpy()
        p = torch.softmax(torch.from_numpy(z), dim=-1).numpy()

        v = p @ Wn                     # 未归一化的均值向量
        r_p = float(np.linalg.norm(v))  # = R_p
        t = v / (np.linalg.norm(v) + 1e-9)
        sim = Wn @ t
        top = np.argsort(-sim)[:20]
        toks = [readout.tok.decode([int(i)]) for i in top]
        n_bad = sum(1 for x in toks if is_format(x))

        rows.append({
            "prompt": prompt,
            "R_p": round(r_p, 4),
            "entropy": round(float(-(p * np.log(p + 1e-12)).sum()), 3),
            "top1_p": round(float(p.max()), 4),
            "format_ratio_top20": round(n_bad / 20, 3),
            "top": toks[:10],
        })
    return rows


def p1_for(readout: FrozenReadout, prompts: list[str], ids_of, n: int) -> list[dict[str, Any]]:
    rows = []
    for prompt in prompts:
        pid = ids_of(readout, prompt)
        res = generate(readout, prompt, max_new_tokens=n, prompt_ids=pid, record=False)
        rows.append({"prompt": prompt, "text": res.text,
                     **metrics(res.text, prompt)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="base,instruct")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=str(STATE_DIR / "base_choice.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    names = [m.strip() for m in args.models.split(",") if m.strip()]
    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS
    report: dict[str, Any] = {"prompts": len(prompts), "n_tokens": args.n, "models": {}}
    w_samples: dict[str, np.ndarray] = {}

    def raw_ids(readout, prompt):
        return readout.encode_ids(prompt)

    def chat_ids(readout, prompt):
        msgs = [{"role": "user", "content": prompt}]
        text = readout.tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        return readout.tok(text, return_tensors="pt").input_ids.to(readout.device)

    for name in names:
        mid = MODELS.get(name)
        if mid is None:
            print(f"[skip] 未知模型 {name}")
            continue
        path = resolve_model_path(mid)
        if not Path(path).exists() and "/" in path:
            print(f"[skip] {name}: 本地无快照")
            continue

        print(f"\n===== {name} ({mid}) =====", flush=True)
        readout = FrozenReadout(model_id=mid)
        ids_of = chat_ids if name == "instruct" else raw_ids

        print("  P2 参考系 ...", flush=True)
        p2 = p2_for(readout, prompts, ids_of)
        rp = np.array([r["R_p"] for r in p2])
        fr = np.array([r["format_ratio_top20"] for r in p2])
        print(f"      R_p  中位数 {np.median(rp):.3f}   p10 {np.percentile(rp, 10):.3f}")
        print(f"      格式类占比中位数 {np.median(fr):.3f}")

        print(f"  P1 输出形态（{len(prompts)}×{args.n} token）...", flush=True)
        p1 = p1_for(readout, prompts, ids_of, args.n)
        sl = sum(r["starts_list"] for r in p1) / len(p1)
        et = sum(r["ends_terminal"] for r in p1) / len(p1)
        dg = np.mean([r["distinct_4gram"] for r in p1])
        print(f"      列表开头 {sl:.0%}   句末标点 {et:.0%}   4-gram 去重 {dg:.2f}")

        idx = torch.linspace(0, readout.V - 1, 5000).long()
        w_samples[name] = readout.W[idx].float().cpu().numpy().copy()
        report["models"][name] = {
            "model_id": mid, "path": path,
            "R_p_median": round(float(np.median(rp)), 4),
            "R_p_p10": round(float(np.percentile(rp, 10)), 4),
            "format_ratio_median": round(float(np.median(fr)), 3),
            "p1": p1, "p2": p2,
        }
        del readout
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # P4 词嵌入一致性
    if len(w_samples) == 2:
        a, b = w_samples["base"], w_samples["instruct"]
        n = min(len(a), len(b))
        an = a[:n] / (np.linalg.norm(a[:n], axis=1, keepdims=True) + 1e-9)
        bn = b[:n] / (np.linalg.norm(b[:n], axis=1, keepdims=True) + 1e-9)
        cos = (an * bn).sum(axis=1)
        report["p4"] = {
            "rows": int(n),
            "cos_mean": round(float(cos.mean()), 4),
            "cos_median": round(float(np.median(cos)), 4),
            "cos_p5": round(float(np.percentile(cos, 5)), 4),
            "cos_min": round(float(cos.min()), 4),
        }
        print(f"\nP4 词嵌入一致性: mean {cos.mean():.4f} median {np.median(cos):.4f} "
              f"p5 {np.percentile(cos, 5):.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out}")

    # 打印样本，供人工判形态
    for name, m in report["models"].items():
        print(f"\n--- {name} P1 样本 ---")
        for r in m["p1"][:6]:
            print(f"  {r['prompt'][:12]:<14} | {r['text'][:60]!r}")


if __name__ == "__main__":
    main()
