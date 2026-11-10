"""m2/memory.py —— 事件记忆层（beta0.2.1 §5、§7、§13）

M1 范围：写入、快照构建、统计体检。**不含**逐 token 偏置（那是 M2）。

    append(text, score, mark)  → 追加一条事件
    rebuild()                  → 重建快照（df / idf / S± / R± / 闸门 / T_m）
    stats()                    → §14.4 的可用性判定所需全部数值

核心约定（§7.3）：
  * 方向聚合一律用中心化词向量 `x̃ = Ŵ − μ_W`，分母用 `Σ w‖x̃‖`
  * `a_e` 的核用未中心化的原始词嵌入
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .readout import FrozenReadout

S0 = 80.0            # 中性线
C_R = 3.0            # 可靠性闸门
LAMBDA_B = 1.0       # 坏侧放大
THETA_STOP = 0.5     # 停用阈值 df/N
ETA = 0.5            # IDF 指数
M_SUPPORT = 200      # 稀疏支撑大小
CHUNK = 16384        # 词表分块


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)


@dataclass
class BiasConfig:
    """注入参数（§11、§9）。全部是推理时标量，不进权重。"""
    mu: float = 1.0          # 全局话语权
    nu: float = 0.3          # 相量配比
    rho: float = 0.5         # 好坏不对称
    alpha: float = 3.0       # 好侧曲线
    beta: float = 0.9        # 坏侧曲线
    c_p: float = 0.35        # 参考系闸门
    kappa_e: float = 10.0    # 检索核宽（N_e<20 时的默认值）
    enabled: bool = True


@dataclass
class Event:
    id: str
    text: str
    score: float
    mark: str
    ts: str
    n_tok: int


class Memory:
    def __init__(
        self,
        readout: FrozenReadout,
        bootstrap: dict[str, Any],
        root: Path | None = None,
    ) -> None:
        self.r = readout
        self.boot = bootstrap
        self.mu0 = np.asarray(bootstrap["mu0"], dtype=np.float32)
        self.mu_w = np.asarray(bootstrap["mu_w"], dtype=np.float32)
        self.kappa_d = float(bootstrap["kappa_d"])
        self.V = readout.V
        self.d = readout.d

        self.root = Path(root or (Path(__file__).resolve().parent.parent / "m2_state" / "snapshot"))
        self.root.mkdir(parents=True, exist_ok=True)

        self.events: list[Event] = []
        self.cases: np.ndarray | None = None        # (N, d)
        self.sup_ids: list[np.ndarray] = []
        self.sup_D: list[np.ndarray] = []
        self.meta: dict[str, Any] = {
            "m2_version": "m0",
            "s0": S0, "c_r": C_R, "lambda_b": LAMBDA_B,
            "theta_stop": THETA_STOP, "eta": ETA, "m": M_SUPPORT,
            "D_ref": None, "tau_d": None, "n_pairs": 0,
            "R_plus": 0.0, "R_minus": 0.0, "N_plus": 0, "N_minus": 0,
            "gate_plus": False, "gate_minus": False,
            "rho_T": None, "norm_D": 0.0, "T_m_on": False,
        }
        self.trend: dict[str, np.ndarray | None] = {
            "T_plus": None, "T_minus": None, "T_m": None,
        }
        self._load()

    # ------------------------------------------------------------ 持久化

    def _load(self) -> None:
        ev = self.root / "events.jsonl"
        if ev.exists():
            self.events = [Event(**json.loads(l)) for l in ev.read_text(encoding="utf-8").splitlines() if l.strip()]
        cp = self.root / "cases.npy"
        if cp.exists():
            self.cases = np.load(cp)
        sp = self.root / "support.npz"
        if sp.exists():
            z = np.load(sp)
            off = z["offsets"]
            for i in range(len(off) - 1):
                self.sup_ids.append(z["tok_ids"][off[i]:off[i + 1]])
                self.sup_D.append(z["D"][off[i]:off[i + 1]])
        mp = self.root / "meta.json"
        if mp.exists():
            self.meta.update(json.loads(mp.read_text(encoding="utf-8")))
        tp = self.root / "trend.npz"
        if tp.exists():
            z = np.load(tp)
            for k in self.trend:
                arr = z[k] if k in z else None
                self.trend[k] = arr if (arr is not None and arr.size == self.d) else None

    def _save(self) -> None:
        with (self.root / "events.jsonl").open("w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
        if self.cases is not None and len(self.cases):
            np.save(self.root / "cases.npy", self.cases)
        if self.sup_ids:
            off = np.cumsum([0] + [len(x) for x in self.sup_ids]).astype(np.int64)
            np.savez_compressed(
                self.root / "support.npz",
                tok_ids=np.concatenate(self.sup_ids).astype(np.int32),
                D=np.concatenate(self.sup_D).astype(np.float32),
                offsets=off,
            )
        np.savez_compressed(
            self.root / "trend.npz",
            **{k: (v if v is not None else np.zeros(0, np.float32)) for k, v in self.trend.items()},
        )
        (self.root / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------ 基础量

    def _tokens(self, text: str) -> list[int]:
        return self.r.tok(text, add_special_tokens=False).input_ids

    def _unit_embed(self, ids: list[int]) -> np.ndarray:
        X = self.r.W[np.asarray(ids)].float().cpu().numpy()
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    def case_vector(self, text: str) -> np.ndarray:
        """ĉ_e = normalize(pool(H) − μ0)，L16。池化跳过特殊 token。"""
        return _unit(self.r.pooled(text) - self.mu0).astype(np.float32)

    def recompute_cases(self) -> int:
        """重算所有事件的 ĉ_e（池化口径变化后必须调用）。"""
        if not self.events:
            return 0
        self.cases = np.stack([self.case_vector(e.text) for e in self.events])
        self._save()
        return len(self.events)

    def _d_over_v(self, ids: list[int]) -> np.ndarray:
        """D_e(y) 对全词表（§5.2），未中心化核。"""
        X = self._unit_embed(ids)                  # (n, d)
        n = len(ids)
        out = np.empty(self.V, dtype=np.float32)
        W = self.r.W
        for a in range(0, self.V, CHUNK):
            b = min(a + CHUNK, self.V)
            Wc = W[a:b].float().cpu().numpy()
            Wc = Wc / (np.linalg.norm(Wc, axis=1, keepdims=True) + 1e-12)
            S = (Wc @ X.T) * self.kappa_d          # (c, n)
            m = S.max(axis=1)
            out[a:b] = m + np.log(np.exp(S - m[:, None]).sum(axis=1) + 1e-30) - np.log(n)
        return out

    # ------------------------------------------------------------ 写入

    def append(self, text: str, score: float, mark: str = "") -> dict[str, Any]:
        ids = self._tokens(text)
        if not ids:
            raise ValueError("事件文本为空")

        D = self._d_over_v(ids)
        k = min(M_SUPPORT, self.V)
        idx = np.argpartition(-D, k - 1)[:k]
        idx = idx[np.argsort(-D[idx])]

        # 首个事件引导 §14.2
        if self.meta["D_ref"] is None:
            self.meta["D_ref"] = float(np.median(D))
            self.meta["tau_d"] = float(np.std(D) / 2.0)
            self.meta["n_pairs"] = int(self.V)

        ev = Event(
            id=f"ev_{len(self.events) + 1:04d}",
            text=text,
            score=float(score),
            mark=mark,
            ts=datetime.now().isoformat(timespec="seconds"),
            n_tok=len(ids),
        )
        self.events.append(ev)

        case = self.case_vector(text)[None, :]
        self.cases = case if self.cases is None else np.vstack([self.cases, case])
        self.sup_ids.append(idx.astype(np.int32))
        self.sup_D.append(D[idx].astype(np.float32))

        self.rebuild()
        return asdict(ev)

    def rescore(self, ev_id: str, score: float) -> dict[str, Any]:
        """人工改分：只动分数，文本与支撑集不变（§20.2 签名之外唯一的写入口）。"""
        for e in self.events:
            if e.id == ev_id:
                e.score = float(score)
                self.rebuild()
                return asdict(e)
        raise KeyError(f"没有这条事件：{ev_id}")

    def list_events(self, only_unscored: bool = False,
                    limit: int | None = None) -> list[dict[str, Any]]:
        """列出事件（只读）。批量的给分工作流靠它导出待评条目。"""
        rows = self.unscored() if only_unscored else [asdict(e) for e in self.events]
        return rows[:limit] if limit else rows

    def unscored(self) -> list[dict[str, Any]]:
        """尚未人工评分的条目：score 停在中性线 S0，w=0，不进好/坏堆。"""
        return [asdict(e) for e in self.events if float(e.score) == S0]

    # ------------------------------------------------------------ 快照

    def rebuild(self) -> dict[str, Any]:
        self._tt = None          # 趋势变了，缓存作废
        n_ev = len(self.events)
        self.meta["m2_version"] = f"m{n_ev}"
        if n_ev == 0:
            self._save()
            return self.meta

        toks_per_event = [self._tokens(e.text) for e in self.events]

        # df / idf / w_t
        df: Counter[int] = Counter()
        for ids in toks_per_event:
            df.update(set(ids))
        w_t: dict[int, float] = {}
        for t, c in df.items():
            ratio = c / n_ev
            if ratio > THETA_STOP:
                w_t[t] = 0.0
            else:
                w_t[t] = (np.log((1 + n_ev) / (1 + c)) + 1.0) ** ETA

        Wn = self.r.W
        S = {"plus": np.zeros(self.d, np.float64), "minus": np.zeros(self.d, np.float64)}
        Wsum = {"plus": 0.0, "minus": 0.0}
        N = {"plus": 0, "minus": 0}

        for ev, ids in zip(self.events, toks_per_event):
            w = ev.score - S0
            if w > 0:
                side, gw = "plus", w
            elif w < 0:
                side, gw = "minus", -w
            else:
                continue
            N[side] += 1
            if gw <= 0:
                continue
            emb = Wn[np.asarray(ids)].float().cpu().numpy()
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
            x = emb - self.mu_w                       # 中心化
            nrm = np.linalg.norm(x, axis=1)
            wt = np.array([w_t.get(t, 0.0) for t in ids], dtype=np.float64)
            S[side] += gw * (wt[:, None] * x).sum(axis=0)
            Wsum[side] += gw * float((wt * nrm).sum())

        out: dict[str, Any] = {}
        for side, key in (("plus", "T_plus"), ("minus", "T_minus")):
            s = S[side]
            ns = np.linalg.norm(s)
            R = float(ns / Wsum[side]) if Wsum[side] > 0 else 0.0
            T = (s / ns).astype(np.float32) if ns > 0 else None
            self.trend[key] = T
            r_gated = R if R * np.sqrt(max(N[side], 1)) >= C_R else 0.0
            out[side] = {"R": R, "R_gated": r_gated, "N": N[side], "T_norm": float(ns)}
            self.meta[f"R_{side}"] = R
            self.meta[f"N_{side}"] = N[side]
            self.meta[f"gate_{side}"] = bool(r_gated > 0)
            self.meta[f"R_gated_{side}"] = r_gated

        # 合成
        Tp, Tm = self.trend["T_plus"], self.trend["T_minus"]
        rp = self.meta["R_gated_plus"]
        rm = self.meta["R_gated_minus"]
        if Tp is not None and Tm is not None:
            self.meta["rho_T"] = float(np.dot(Tp, Tm))
        else:
            self.meta["rho_T"] = None

        dvec = np.zeros(self.d, np.float64)
        if Tp is not None:
            dvec += rp * Tp
        if Tm is not None:
            dvec -= LAMBDA_B * rm * Tm
        nd = float(np.linalg.norm(dvec))
        self.meta["norm_D"] = nd
        if nd > 1e-9:
            self.trend["T_m"] = (dvec / nd).astype(np.float32)
            self.meta["T_m_on"] = True
        else:
            self.trend["T_m"] = None
            self.meta["T_m_on"] = False

        # 分布统计
        self.meta["n_tokens"] = int(sum(len(t) for t in toks_per_event))
        self.meta["n_distinct"] = int(len(df))
        self.meta["stop_ratio"] = float(sum(1 for t in df if w_t.get(t, 0) == 0.0) / max(len(df), 1))

        self._save()
        return self.meta

    # ------------------------------------------------------------ 注入（M2）

    def _ensure_tables(self) -> None:
        if getattr(self, "Wn", None) is not None:
            return
        W = self.r.W.float().cpu().numpy()
        nrm = np.linalg.norm(W, axis=1, keepdims=True)
        self.Wn = (W / np.maximum(nrm, 1e-12)).astype(np.float32)
        self.xn = np.linalg.norm(self.Wn - self.mu_w, axis=1).astype(np.float32)

    def _dir_from_state(self, state: dict) -> np.ndarray:
        """â_t = normalize(pool(H_{≤t}) − μ0)。第 0 步时它就是 q̂。"""
        mean = state["h_t_sum"] / max(int(state["n_pos"]), 1)
        v = mean.float().cpu().numpy() - self.mu0
        return v / (np.linalg.norm(v) + 1e-12)

    def _trend_tables(self) -> dict:
        """T_m / T± 不随生成变化，缓存一次。"""
        if getattr(self, "_tt", None) is not None:
            return self._tt
        self._ensure_tables()
        V = self.V
        Tp_, Tm_ = self.trend.get("T_plus"), self.trend.get("T_minus")
        Tm = self.trend.get("T_m")
        if Tm is not None and Tm.size != V:
            Tm = None
        t = {
            "T_m": Tm,
            "m_m": self.Wn @ Tm if Tm is not None else np.zeros(V, np.float32),
            "m_plus": self.Wn @ Tp_ if Tp_ is not None else np.zeros(V, np.float32),
            "m_minus": self.Wn @ Tm_ if Tm_ is not None else np.zeros(V, np.float32),
        }
        self._tt = t
        return t

    def bias(self, step: int, z, state: dict, cfg: BiasConfig | None = None):
        """返回 (z_new, extra)。extra 里是逐 token 数组或标量，供账本取用。"""
        cfg = cfg or BiasConfig()
        V = self.V
        extra: dict[str, Any] = {
            "a": np.zeros(V, np.float32),
            "reb": np.zeros(V, np.float32),
            "c": np.ones(V, np.float32),
            "r": np.zeros(V, np.float32),
            "phi_T": None,
            "R_p": None,
        }
        if not cfg.enabled or cfg.mu == 0.0 or not self.events:
            return z, extra

        self._ensure_tables()
        if "q" not in state:
            state["q"] = self._dir_from_state(state)       # 第 0 步 = q̂
        q = state["q"]
        a_t = self._dir_from_state(state)

        # —— 检索 γ 与相位 φ_e ——
        cos_q = np.clip(self.cases @ q, -1, 1)
        K = np.exp(cfg.kappa_e * (cos_q - 1.0))
        sc = np.array([e.score for e in self.events], dtype=np.float64)
        w = sc * K
        tot = w.sum()
        if tot <= 0:
            return z, extra
        gamma = w / tot
        phi_e = np.arccos(np.clip(self.cases @ a_t, -1, 1))
        gain = gamma * np.cos(phi_e)

        # —— 相量通道（稀疏）——
        reb = extra["reb"]
        d_ref, tau = self.meta.get("D_ref"), max(self.meta.get("tau_d") or 1e-6, 1e-6)
        if d_ref is not None:
            for i, (ids, D) in enumerate(zip(self.sup_ids, self.sup_D)):
                g = float(gain[i])
                if abs(g) < 1e-9:
                    continue
                a_e = (1.0 / (1.0 + np.exp(-(D - d_ref) / tau))).astype(np.float32)
                np.add.at(reb, ids, g * a_e)

        # —— 趋势通道 ——
        t = self._trend_tables()
        Tm = t["T_m"]
        rez = cfg.nu * reb
        if Tm is not None:
            p = torch.softmax(z.detach(), dim=-1).cpu().numpy().astype(np.float32)
            Vp = (p @ self.Wn) - self.mu_w                  # = Σ p·(Ŵ − μ_W)
            R_p = float(np.linalg.norm(Vp) / max(float(p @ self.xn), 1e-9))
            extra["R_p"] = R_p
            if R_p >= cfg.c_p and np.linalg.norm(Vp) > 1e-9:
                Tp = (Vp / np.linalg.norm(Vp)).astype(np.float32)
                phi_T = float(np.arccos(np.clip(float(Tp @ Tm), -1.0, 1.0)))
                extra["phi_T"] = phi_T
                m_p = self.Wn @ Tp
                r = t["m_m"] - np.cos(phi_T) * m_p
            else:
                r = t["m_m"]
            xp = np.clip((1 + t["m_plus"]) / 2, 0, 1)
            xm = np.clip((1 + t["m_minus"]) / 2, 0, 1)
            Ip = np.log1p(cfg.alpha * xp) / np.log1p(cfg.alpha)
            denom = np.log(1.0 / max(1.0 - cfg.beta, 1e-9))
            Ib = np.log(1.0 / np.maximum(1.0 - cfg.beta * xm, 1e-9)) / denom
            c = 1.0 + cfg.rho * (Ib - Ip)
            a = (c * r).astype(np.float32)
            extra["r"], extra["c"], extra["a"] = r.astype(np.float32), c.astype(np.float32), a
            rez = a + cfg.nu * reb

        extra["_absmax"] = float(cfg.mu * np.abs(rez).max())   # 实际加到 logits 上的量
        extra["_nz"] = int((np.abs(rez) > 1e-6).sum())
        z_new = z + cfg.mu * torch.from_numpy(rez).to(z.device, z.dtype)
        return z_new, extra

    # ------------------------------------------------------------ 体检

    def saturation(self) -> dict[str, float] | None:
        return _saturation(self.sup_D, self.meta)

    def stats(self) -> dict[str, Any]:
        return _build_stats(self.meta, len(self.events), self.saturation())


def _saturation(sup_D, meta) -> dict[str, float] | None:
    if not sup_D or meta.get("D_ref") is None:
        return None
    d_ref, tau = float(meta["D_ref"]), max(float(meta["tau_d"]), 1e-6)
    a = np.concatenate([1.0 / (1.0 + np.exp(-(D - d_ref) / tau)) for D in sup_D])
    return {
        "mean": round(float(a.mean()), 3),
        "in_band": round(float(((a > 0.05) & (a < 0.95)).mean()), 3),
        "frac_hi": round(float((a >= 0.95).mean()), 3),
        "frac_lo": round(float((a <= 0.05).mean()), 3),
    }


def _build_stats(meta: dict, n_events: int, sat: dict | None) -> dict[str, Any]:
    g = meta.get
    st: dict[str, Any] = {
        "m2_version": g("m2_version"),
        "n_events": n_events,
        "n_good": g("N_plus", 0), "n_bad": g("N_minus", 0),
        "R_plus": round(g("R_plus") or 0.0, 4),
        "R_minus": round(g("R_minus") or 0.0, 4),
        "gate_plus": bool(g("gate_plus")),
        "gate_minus": bool(g("gate_minus")),
        "rho_T": None if g("rho_T") is None else round(g("rho_T"), 4),
        "norm_D": round(g("norm_D") or 0.0, 4),
        "T_m_on": bool(g("T_m_on")),
        "D_ref": None if g("D_ref") is None else round(g("D_ref"), 4),
        "tau_d": None if g("tau_d") is None else round(g("tau_d"), 4),
        "n_distinct": g("n_distinct"),
        "stop_ratio": None if g("stop_ratio") is None else round(g("stop_ratio"), 3),
        "saturation": sat,
    }
    if n_events == 0:
        st.update(stage="S0", trend="off", phasor="off")
    elif not st["gate_plus"] and not st["gate_minus"]:
        st.update(stage="S1/S2（闸门未过，趋势关）", trend="gate-fail", phasor="on")
    elif st["rho_T"] is not None and st["rho_T"] >= 0.9:
        st.update(stage="S2（好坏同质）", trend="rho-fail", phasor="on")
    else:
        st.update(stage="S3（趋势可用）", trend="on", phasor="on")
    return st


def read_stats(root: Path | None = None) -> dict[str, Any]:
    """不加载底座，直接读快照出体检结果。"""
    root = Path(root or (Path(__file__).resolve().parent.parent / "m2_state" / "snapshot"))
    mp, sp = root / "meta.json", root / "support.npz"
    if not mp.exists():
        return _build_stats({}, 0, None)
    meta = json.loads(mp.read_text(encoding="utf-8"))
    n = 0
    sup_D = []
    if (root / "events.jsonl").exists():
        n = len([l for l in (root / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()])
    if sp.exists():
        z = np.load(sp)
        off = z["offsets"]
        sup_D = [z["D"][off[i]:off[i + 1]] for i in range(len(off) - 1)]
    return _build_stats(meta, n, _saturation(sup_D, meta))
