"""m2/bootstrap.py —— 底座侧常数（beta0.2 §14.1）

产出两个量，都**不依赖任何事件**，可以在记忆为空时算好：

    κ_d  = 1 / std(cos)      词嵌入两两余弦的离散度 → 幅度核宽
    μ0   = 平均池化隐状态的语料均值 → vec16 的去中心向量

用法：
    python -m m2.bootstrap                 # 写到 m2_state/bootstrap.json
    python -m m2.bootstrap --corpus f.txt  # 追加自定义参考语料
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .readout import WS, FrozenReadout, pool_hidden

STATE_DIR = WS / "m2_state"
BOOTSTRAP_PATH = STATE_DIR / "bootstrap.json"

# 默认参考语料：只用于估计"语料均值方向"，不参与任何训练。
# 长度与领域都无关紧要，重要的是固定、可复现、可审计。
DEFAULT_CORPUS = """\
今天天气不错，适合出门散步。
请把这段话翻译成英文。
这个函数返回一个列表，元素是整数。
如果输入为空，直接返回 None。
这道题的答案是多少？
先化简，再代入求值。
程序报错说索引越界，需要检查边界条件。
我们把问题拆成三个小步骤来解。
数据库中存的是用户的历史订单。
模型输出了一段不太通顺的文字。
请在运行前先安装依赖。
这一步的目的是去掉重复项。
他昨天去了图书馆，借了两本书。
这个公式描述的是增长速率与当前量的关系。
训练集和验证集不能重叠。
把结果保存成 JSON 文件。
我想知道这句话是什么意思。
这个接口需要传入两个参数。
当样本量增大时，估计会变得更稳定。
先把数据清洗干净，再做统计。
这个结论只在特定条件下成立。
计算过程中要注意单位换算。
模型对长文本的处理效果一般。
请检查一下这个假设是否合理。
在二维平面上，两条平行线不会相交。
这个版本的性能比上一版好一些。
需要把配置项抽出来单独管理。
先跑一个小实验验证想法。
这段代码有一个隐蔽的并发问题。
结果与预期不符，需要重新核对。
请给出具体的执行步骤。
这个问题的关键在于如何定义损失。
数据分布偏移会导致模型失效。
我们把超参数固定下来再比较。
输入格式不对会直接抛异常。
总结一下刚才讨论的要点。
这个现象在真实场景中很常见。
先确认需求，再动手实现。
缓冲区满了以后需要刷盘。
这两组数据之间没有显著差异。
"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def estimate_kappa_d(
    W: np.ndarray, n: int = 2048, seed: int = 0, clip: tuple[float, float] = (1.0, 200.0)
) -> dict[str, float]:
    """κ_d = 1/std(cos)，clip 到 [1,200]。只需词嵌入矩阵。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(W.shape[0], size=min(n, W.shape[0]), replace=False)
    X = W[idx].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    C = X @ X.T
    iu = np.triu_indices_from(C, k=1)
    off = C[iu]
    std = float(off.std())
    kappa = 1.0 / std if std > 1e-9 else clip[1]
    return {
        "kappa_d": float(np.clip(kappa, *clip)),
        "w_cos_std": std,
        "w_cos_mean": float(off.mean()),
        "n_sample": int(X.shape[0]),
    }


def estimate_mu_w(W: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """μ_W = 归一化词嵌入的均值（方向聚合的去中心向量，§14.1）。"""
    V, d = W.shape
    acc = np.zeros(d, dtype=np.float64)
    for a in range(0, V, chunk):
        X = W[a : a + chunk].astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        acc += X.sum(axis=0, dtype=np.float64)
    return (acc / V).astype(np.float32)


def estimate_mu0(readout: FrozenReadout, texts: list[str], verbose: bool = True) -> np.ndarray:
    """μ0 = 参考语料上平均池化隐状态的均值（原始空间，未归一化）。"""
    vecs = []
    t0 = time.time()
    for i, text in enumerate(texts):
        vecs.append(readout.pooled(text))
        if verbose and (i + 1) % 10 == 0:
            print(f"  μ0 {i + 1}/{len(texts)}  {time.time() - t0:.1f}s", flush=True)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)


def build(
    model_id: str = "Qwen/Qwen2.5-0.5B",
    layer: int = 16,
    corpus: list[str] | None = None,
    out_path: Path = BOOTSTRAP_PATH,
    verbose: bool = True,
) -> dict[str, Any]:
    texts = list(corpus or DEFAULT_CORPUS.strip().splitlines())
    texts = [t.strip() for t in texts if t.strip()]
    if len(texts) < 50 and verbose:
        print(f"[warn] 参考语料只有 {len(texts)} 条，μ0 的估计会比较粗。"
              f"进入 S1 前建议补到 200 条以上同类文本。", flush=True)

    if verbose:
        print(f"[1/3] 加载底座 {model_id} ...", flush=True)
    readout = FrozenReadout(model_id=model_id, layer=layer)

    if verbose:
        print(f"[2/3] 估计 κ_d（词嵌入 {readout.V} 行抽样）...", flush=True)
    kappa = estimate_kappa_d(readout.w_numpy())
    W_np = readout.w_numpy()
    mu_w = estimate_mu_w(W_np)

    if verbose:
        print(f"[3/3] 估计 μ0（{len(texts)} 条参考语料）...", flush=True)
    mu0 = estimate_mu0(readout, texts, verbose=verbose)

    payload = {
        "version": "beta0.2",
        "created": datetime.now().isoformat(timespec="seconds"),
        "base": readout.describe(),
        "kappa_d": kappa["kappa_d"],
        "kappa_d_detail": kappa,
        "mu0": [round(float(x), 6) for x in mu0],
        "mu0_norm": float(np.linalg.norm(mu0)),
        "mu_w": [round(float(x), 6) for x in mu_w],
        "mu_w_norm": float(np.linalg.norm(mu_w)),
        "corpus": {
            "n_texts": len(texts),
            "sha256_16": _sha256("\n".join(texts)),
            "note": "仅用于估计语料均值方向；不属于记忆，不随事件增长。",
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"\n写入 {out_path}")
        print(f"  κ_d   = {payload['kappa_d']:.3f}   (std(cos)={kappa['w_cos_std']:.4f})")
        print(f"  μ0    = {len(mu0)} 维，范数 {payload['mu0_norm']:.3f}")
        print(f"  μ_W   = {len(mu_w)} 维，范数 {payload['mu_w_norm']:.3f}")
    return payload


def load_bootstrap(path: Path = BOOTSTRAP_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mu0_of(bootstrap: dict[str, Any] | None) -> np.ndarray | None:
    if not bootstrap:
        return None
    return np.asarray(bootstrap["mu0"], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="估计底座侧常数 κ_d 与 μ0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--corpus", default=None, help="追加的参考语料（每行一句）")
    ap.add_argument("--out", default=str(BOOTSTRAP_PATH))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    texts = DEFAULT_CORPUS.strip().splitlines()
    if args.corpus:
        extra = Path(args.corpus).read_text(encoding="utf-8").splitlines()
        texts = list(texts) + extra

    build(
        model_id=args.model,
        layer=args.layer,
        corpus=texts,
        out_path=Path(args.out),
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
