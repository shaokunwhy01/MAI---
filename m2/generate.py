"""m2/generate.py —— 手写解码循环（beta0.2 §11、§20）

不用 `model.generate()`。原因是记忆需要在**每一步**改写 logits，
而 `generate()` 没有暴露这个钩子。这里自己走 prefill + step，钩子签名：

    hook(step: int, z: Tensor[V], state: dict) -> (z_new: Tensor[V], extra: dict)

`extra` 会并进该步的诊断记录。空记忆时传 `None`，钩子不生效，
输出与纯底座逐位相同（§13.1 启动判据）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from .readout import FrozenReadout

CONFIG_VERSION = "beta0.2.1"
DECODE_VERSION = "d1"

Hook = Callable[[int, torch.Tensor, dict], tuple[torch.Tensor, dict]]


@dataclass
class RunResult:
    text: str
    prompt: str
    ids: list[int]
    prompt_ids: list[int]
    ledger: list[dict[str, Any]]
    versions: dict[str, str]
    elapsed: float
    stopped: str = "length"
    run_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "prompt": self.prompt,
            "n_tokens": len(self.ids),
            "ids": self.ids,
            "versions": self.versions,
            "elapsed_s": round(self.elapsed, 2),
            "bias_absmax": self.run_stats.get("bias_absmax"),
            "bias_nonzero_max": self.run_stats.get("bias_nonzero_max"),
            "stopped": self.stopped,
            "ledger": self.ledger,
        }


def _softmax(z: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature 必须 > 0")
    return torch.softmax(z / temperature, dim=-1)


def generate(
    readout: FrozenReadout,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.0,
    top_k: int | None = None,
    seed: int = 0,
    hook: Hook | None = None,
    record: bool = True,
    versions: dict[str, str] | None = None,
    prompt_ids: torch.Tensor | None = None,
) -> RunResult:
    """生成一段文本，并记录每个生成 token 的诊断行（§20.3）。"""
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    ver = versions or {
        "base": "b1",
        "e": "e0",
        "m2": "m0",
        "decode": DECODE_VERSION,
        "config": CONFIG_VERSION,
    }
    ver_tag = "/".join(ver[k] for k in ("base", "e", "m2", "decode", "config"))

    t0 = time.time()
    if prompt_ids is None:
        prompt_ids = readout.encode_ids(prompt)
    out = readout.forward(prompt_ids)

    # state 里维护前缀的隐状态累积和，供后续 â_t（§6.1）使用。
    # 必须排除特殊 token：BOS 的范数是内容 token 的约 80 倍（§6.1）。
    pids = prompt_ids[0].tolist()
    keep = readout.content_mask(pids)
    h0 = out["h_t"][torch.as_tensor(keep, device=out["h_t"].device)]
    state: dict[str, Any] = {
        "h_t_sum": h0.sum(dim=0),
        "n_pos": int(h0.shape[0]),
        "prefix_ids": prompt_ids[0].tolist(),
        "prompt_len": int(prompt_ids.shape[1]),
        "step": 0,
    }

    past = out["past"]
    z = out["z"]
    ids: list[int] = []
    ledger: list[dict[str, Any]] = []
    stopped = "length"
    prev_clean = ""
    absmax: list[float] = []
    nz: list[int] = []

    for step in range(max_new_tokens):
        z_base = z

        # —— 记忆注入点 ——
        extra: dict[str, Any] = {}
        if hook is not None:
            state["step"] = step
            z, extra = hook(step, z, state)

        # —— 采样 ——
        if temperature <= 0:
            token = int(torch.argmax(z))
        else:
            logits = z
            if top_k:
                k = min(int(top_k), logits.shape[-1])
                thresh = torch.topk(logits, k).values[-1]
                logits = torch.where(logits < thresh,
                                     torch.full_like(logits, -float("inf")), logits)
            probs = _softmax(logits, temperature)
            token = int(torch.multinomial(probs, num_samples=1))

        # 字节级 BPE 的单个 token 可能是半个汉字，单独解码会得到替换字符。
        # 这里解码"截至本步的整个序列"，去掉末尾未完成的字节，再取增量。
        cur_clean = readout.decode(ids + [token]).rstrip("\ufffd")
        delta = cur_clean[len(prev_clean):] if cur_clean.startswith(prev_clean) else ""
        prev_clean = cur_clean

        logp = torch.log_softmax(z, dim=-1)
        row = {
            "i": step,
            "id": token,
            "tok": delta,
            "z": round(float(z[token]), 4),
            "p": round(float(torch.exp(logp[token])), 6),
            "a": 0.0,
            "reb": 0.0,
            "bias": round(float(z[token] - z_base[token]), 6),
            "c": 1.0,
            "r": 0.0,
            "phi_T": None,
            "R_p": None,
            "ver": ver_tag,
        }
        for k, v in extra.items():
            if k == "_absmax":
                absmax.append(float(v))
                continue
            if k == "_nz":
                nz.append(int(v))
                continue
            if k.startswith("_"):
                continue
            if isinstance(v, np.ndarray) and v.ndim == 1 and v.shape[0] > token:
                row[k] = round(float(v[token]), 6)
            elif isinstance(v, (float, np.floating)):
                row[k] = round(float(v), 6)
            else:
                row[k] = v
        if record:
            ledger.append(row)

        ids.append(token)
        if token == readout.eos_id:
            stopped = "eos"
            break

        # —— 下一步 ——
        nxt = torch.as_tensor([[token]], device=readout.device)
        out = readout.forward(nxt, past=past)
        past = out["past"]
        z = out["z"]
        state["h_t_sum"] = state["h_t_sum"] + out["h_t"][0]
        state["n_pos"] += 1
        state["prefix_ids"].append(token)

    text = readout.decode(ids)
    stats: dict[str, Any] = {
        "bias_absmax": round(max(absmax), 4) if absmax else 0.0,
        "bias_nonzero_max": max(nz) if nz else 0,
    }
    return RunResult(
        text=text,
        prompt=prompt,
        ids=ids,
        prompt_ids=prompt_ids[0].tolist(),
        ledger=ledger,
        versions=ver,
        elapsed=time.time() - t0,
        stopped=stopped,
        run_stats=stats,
    )


def pure_base_parity(readout: FrozenReadout, prompt: str, n: int = 12) -> dict[str, Any]:
    """启动判据（§13.1）：空记忆时必须与纯底座逐位相同。"""
    a = generate(readout, prompt, max_new_tokens=n, record=False)
    b = generate(readout, prompt, max_new_tokens=n, record=False,
                 hook=lambda s, z, st: (z, {}))
    return {
        "identical": a.ids == b.ids,
        "ids_a": a.ids,
        "ids_b": b.ids,
        "text": a.text,
    }
