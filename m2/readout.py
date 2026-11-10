"""m2/readout.py —— 冻结底座读出层（beta0.2 §4）

只提供两样东西：

    h_t  —— 第 16 层逐位置隐状态，形状 (T, d)
    z_t  —— 末层隐状态经 RMSNorm 后与词嵌入矩阵相乘，形状 (V,)

不改权重、不训练、不加载多余模块。Qwen2.5-0.5B 的词嵌入与输出头绑定
（tie_word_embeddings=True），所以 `embed_tokens.weight` 就是 W，不需要
单独加载 lm_head。
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

WS = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER_COORD = 16


def resolve_model_path(model_id: str) -> str:
    """优先用工作区内已下载的快照，避免联网与缓存目录问题。"""
    if "/" not in model_id:
        return model_id
    org, name = model_id.split("/", 1)

    # 1) 手工放置的本地目录（镜像直连下载用）
    local = WS / ".hf-cache" / "local" / name
    if (local / "config.json").exists():
        return str(local)

    # 2) huggingface_hub 缓存快照
    pattern = WS / ".hf-cache" / "hub" / f"models--{org}--{name}" / "snapshots" / "*"
    for cand in sorted(glob.glob(str(pattern))):
        if (Path(cand) / "config.json").exists():
            return cand
    return model_id


class FrozenReadout:
    """冻结底座的唯一入口。"""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        layer: int = LAYER_COORD,
        device: str = "cpu",
        dtype: str = "float32",
        model_path: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.layer = int(layer)
        self.device = torch.device(device)
        self.torch_dtype = getattr(torch, dtype)
        self.path = model_path or resolve_model_path(model_id)

        self.tok = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModel.from_pretrained(
            self.path, trust_remote_code=True, torch_dtype=self.torch_dtype
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

        # AutoModel 返回的是 Qwen2Model 本体；AutoModelForCausalLM 才是外层包装。
        self.core = getattr(self.model, "model", self.model)

        # 词嵌入矩阵同时是输出头
        self.W = self.core.embed_tokens.weight.detach()   # (V, d)
        self.V, self.d = int(self.W.shape[0]), int(self.W.shape[1])
        self.norm = self.core.norm
        self.special: set[int] = set(int(x) for x in (self.tok.all_special_ids or []))
        self.n_layers = int(self.model.config.num_hidden_layers)

        if self.layer < 0 or self.layer > self.n_layers:
            raise ValueError(f"layer={self.layer} 越界（0={embedding}，{self.n_layers}=末层）")

    # ------------------------------------------------------------ 基础

    @property
    def eos_id(self) -> int:
        return int(self.tok.eos_token_id)

    def encode_ids(self, text: str) -> torch.Tensor:
        enc = self.tok(text, return_tensors="pt", add_special_tokens=True)
        return enc.input_ids.to(self.device)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def n_params(self) -> float:
        return sum(p.numel() for p in self.model.parameters()) / 1e6

    # ------------------------------------------------------------ 前向

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, past: Any = None) -> dict[str, Any]:
        """一次前向。

        ids: (1, T)  —— 有 cache 时 T=1
        past: 上一步返回的 past_key_values

        返回 h_t (T,d) / h_L (T,d) / z (V,) / past
        """
        out = self.model(
            input_ids=ids,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
        )
        hs = out.hidden_states              # 长度 = n_layers + 1
        h_t = hs[self.layer][0].float()     # (T, d)
        h_L = hs[-1][0].float()             # (T, d)

        # 读出：末层 → 最终 RMSNorm → W
        hL_last = self.norm(hs[-1])[:, -1]  # (1, d)
        z = F.linear(hL_last, self.W)[0].float()   # (V,)

        return {"h_t": h_t, "h_L": h_L, "z": z, "past": out.past_key_values}

    @torch.no_grad()
    def forward_text(self, text: str) -> dict[str, Any]:
        return self.forward(self.encode_ids(text))

    def content_mask(self, ids: list[int]) -> np.ndarray:
        """可池化的位置：排除第 0 位与特殊 token。

        第 0 位是注意力汇：实测 Qwen2.5-0.5B 第 16 层上它的隐状态范数约
        1700，而内容 token 只有 ~21。不排除的话均值 93–99% 由它决定，
        池化向量的模长还会随序列长度剧烈变化。
        """
        m = np.array([i not in self.special for i in ids], dtype=bool)
        if len(m) > 1:
            m[0] = False
        return m if m.any() else np.ones(len(ids), dtype=bool)

    @torch.no_grad()
    def pooled(self, text: str) -> np.ndarray:
        """vec16 的池化基元：跳过第 0 位与特殊 token 后取均值。"""
        ids = self.tok(text, add_special_tokens=True).input_ids
        out = self.forward(torch.as_tensor([ids], device=self.device))
        h = out["h_t"].cpu().numpy()
        m = self.content_mask(ids)
        return h[m].mean(axis=0)

    # ------------------------------------------------------------ 工具

    def embed_norm(self, ids: list[int]) -> np.ndarray:
        """token id → 归一化词向量（记忆侧用，W 空间）。"""
        x = self.W[torch.as_tensor(ids, device=self.device)].float()
        x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return x.cpu().numpy()

    def w_numpy(self) -> np.ndarray:
        return self.W.detach().float().cpu().numpy()

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "path": self.path,
            "layer_coord": self.layer,
            "n_layers": self.n_layers,
            "hidden_dim": self.d,
            "vocab": self.V,
            "dtype": str(self.torch_dtype).replace("torch.", ""),
            "device": str(self.device),
            "params_M": round(self.n_params(), 1),
            "tied_output_head": True,
        }


def pool_hidden(h: np.ndarray) -> np.ndarray:
    """(T, d) → (d,) 均值池化。"""
    return h.mean(axis=0)


if __name__ == "__main__":
    import json

    r = FrozenReadout()
    print(json.dumps(r.describe(), ensure_ascii=False, indent=2))
    out = r.forward_text("黎曼猜想是什么？")
    print("h_t", tuple(out["h_t"].shape), "h_L", tuple(out["h_L"].shape), "z", tuple(out["z"].shape))
    top = torch.topk(out["z"], 5)
    print("top-5:", [(r.tok.decode([int(i)]), round(float(v), 2))
                     for v, i in zip(top.values, top.indices)])
