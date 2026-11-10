"""m2/apitest.py —— 对本地 M2 服务跑一套测试（HTTP 黑盒）

    ./m2test                只读测试（A + B + C）
    ./m2test --write        额外做一次写入，验证快照热更新

判读分组：
  A 契约      —— 任何数据量都必须全绿，红一条就是 bug
  B 生成能力  —— 只记录，不判优劣（这是底座属性）
  C 记忆效应  —— 12 条事件时无判读意义，只留作 S3 对比基线
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

WS = Path(__file__).resolve().parent.parent
S0 = 80.0


# ---------------------------------------------------------------- HTTP


def _post(base: str, path: str, body: dict, timeout: int = 300) -> tuple[int, Any]:
    req = urllib.request.Request(
        base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:  # noqa: BLE001
            return e.code, {}


def _get(base: str, path: str, timeout: int = 10) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}


def _stream(base: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = {"raw_lines": 0, "deltas": 0, "text": "", "done": False}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            s = line.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            out["raw_lines"] += 1
            payload = s[5:].strip()
            if payload == "[DONE]":
                out["done"] = True
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("choices"):
                delta = obj["choices"][0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    out["deltas"] += 1
                    out["text"] += piece
    return out


# ---------------------------------------------------------------- 指标


def metrics(text: str) -> dict:
    t = (text or "").strip()
    g4 = [t[i:i + 4] for i in range(max(len(t) - 3, 0))]
    g3 = [t[i:i + 3] for i in range(max(len(t) - 2, 0))]
    rep = 1
    if g3:
        from collections import Counter
        rep = max(Counter(g3).values())
    ascii_alpha = sum(1 for c in t if c.isascii() and c.isalpha())
    return {
        "chars": len(t),
        "ends_terminal": t.endswith(("。", "！", "？", ".", "!", "?")),
        "distinct_4gram": round(len(set(g4)) / max(len(g4), 1), 3),
        "max_3gram_repeat": int(rep),
        "ascii_ratio": round(ascii_alpha / max(len(t), 1), 3),
    }


# ---------------------------------------------------------------- 报告

ROWS: list[dict] = []


def rec(group: str, name: str, ok: Any, detail: str = "") -> None:
    ROWS.append({"group": group, "name": name,
                 "status": "PASS" if ok is True else ("FAIL" if ok is False else "INFO"),
                 "detail": detail})
    mark = {True: "PASS", False: "FAIL"}.get(ok, "INFO")
    print(f"  [{mark}] {name}" + (f"   {detail}" if detail else ""), flush=True)


# ---------------------------------------------------------------- A 组


def group_a(base: str, model: str, thr: dict) -> None:
    print("\n[A] 契约 / 接口", flush=True)

    st, d = _get(base, "/v1/models")
    ids = [m.get("id") for m in d.get("data", [])] if st == 200 else []
    rec("A", "GET /v1/models", st == 200 and model in ids, f"{ids}")

    st, d = _post(base, "/v1/chat/completions",
                  {"model": model, "messages": [{"role": "user", "content": "你好"}],
                   "max_tokens": 8})
    c = (d.get("choices") or [{}])[0].get("message", {}).get("content", "")
    rec("A", "chat 非流式", st == 200 and bool(c), f"{len(c)} chars")

    s = _stream(base, {"model": model, "messages": [{"role": "user", "content": "你好"}],
                       "max_tokens": 8, "stream": True})
    rec("A", "chat 流式", s["done"] and s["deltas"] >= 1,
        f"deltas={s['deltas']} done={s['done']}")

    st, _ = _post(base, "/v1/chat/completions", {"model": model, "messages": []})
    rec("A", "空 messages 拒绝", st == 400, f"status={st}")

    long_sys = "You are a helpful assistant. " * 900
    st, d = _post(base, "/v1/chat/completions",
                  {"model": model, "max_tokens": 4,
                   "messages": [{"role": "system", "content": long_sys},
                                {"role": "user", "content": "hi"}]})
    pt = (d.get("usage") or {}).get("prompt_tokens")
    rec("A", "超长输入被截断到 4096", st == 200 and pt == 4096, f"prompt_tokens={pt}")

    st, d = _post(base, "/v1/chat/completions",
                  {"model": model, "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 0})
    ct = (d.get("usage") or {}).get("completion_tokens")
    rec("A", "max_tokens=0 被夹紧", st == 200 and (ct or 0) >= 1, f"completion_tokens={ct}")

    body = {"model": model, "messages": [{"role": "user", "content": "什么是递归"}],
            "max_tokens": 12}
    _, d1 = _post(base, "/v1/chat/completions", body)
    _, d2 = _post(base, "/v1/chat/completions", body)
    c1 = (d1.get("choices") or [{}])[0].get("message", {}).get("content")
    c2 = (d2.get("choices") or [{}])[0].get("message", {}).get("content")
    rec("A", "确定性（同请求两次一致）", c1 is not None and c1 == c2, f"{len(c1 or '')} chars")

    results: list[int] = []

    def one() -> None:
        code, _ = _post(base, "/v1/chat/completions",
                        {"model": model, "messages": [{"role": "user", "content": "并发测试"}],
                         "max_tokens": 6})
        results.append(code)

    ts = [threading.Thread(target=one) for _ in range(3)]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    rec("A", "并发 3 请求", results and all(x == 200 for x in results),
        f"codes={results} {time.time() - t0:.1f}s")


# ---------------------------------------------------------------- B 组


def group_b(base: str, model: str, prompts: list[str], thr: dict) -> None:
    print("\n[B] 生成能力（只记录）", flush=True)
    n = int(thr.get("chat_max_tokens", 32))
    agg = []
    for p in prompts:
        _, d = _post(base, "/v1/chat/completions",
                     {"model": model, "messages": [{"role": "user", "content": p}],
                      "max_tokens": n})
        text = (d.get("choices") or [{}])[0].get("message", {}).get("content", "")
        m = metrics(text)
        agg.append(m)
        print(f"      {p[:14]:<16} {text[:40]!r}", flush=True)
    ok_rep = max(x["max_3gram_repeat"] for x in agg) <= thr["max_3gram_repeat_max"]
    ok_div = statistics.mean(x["distinct_4gram"] for x in agg) >= thr["distinct_4gram_min"]
    ok_ascii = statistics.mean(x["ascii_ratio"] for x in agg) <= thr["ascii_ratio_max"]
    rec("B", "无 3-gram 复读", ok_rep,
        f"max={max(x['max_3gram_repeat'] for x in agg)}")
    rec("B", "4-gram 多样性", ok_div,
        f"mean={statistics.mean(x['distinct_4gram'] for x in agg):.3f}")
    rec("B", "中文输出（ASCII 比例）", ok_ascii,
        f"mean={statistics.mean(x['ascii_ratio'] for x in agg):.3f}")
    rec("B", "句末收尾率", None,
        f"{sum(1 for x in agg if x['ends_terminal'])}/{len(agg)}")


# ---------------------------------------------------------------- C 组


def _load_support() -> tuple[set[int], set[int], dict]:
    root = WS / "m2_state" / "snapshot"
    events = [json.loads(l) for l in (root / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    z = __import__("numpy").load(root / "support.npz")
    off = z["offsets"]
    good: set[int] = set()
    bad: set[int] = set()
    for i, ev in enumerate(events):
        seg = set(int(t) for t in z["tok_ids"][off[i]:off[i + 1]])
        (good if ev["score"] > S0 else bad).update(seg)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    return good, bad, meta


def group_c(base: str, prompt: str, mu_values: list[float],
            rho_values: list[float]) -> None:
    print("\n[C] 记忆效应（12 条事件时无判读意义，仅留基线）", flush=True)

    try:
        good, bad, meta = _load_support()
    except Exception as e:  # noqa: BLE001
        rec("C", "读取快照", False, str(e))
        return
    rec("C", "快照可读", True,
        f"m2={meta.get('m2_version')} R+={meta.get('R_plus'):.3f} 闸门={meta.get('gate_plus')}")

    for mu in mu_values:
        _, d = _post(base, "/generate",
                     {"prompt": prompt, "max_new_tokens": 12, "mu": mu})
        ids = d.get("ids") or []
        bmax = d.get("bias_absmax")
        gr = sum(1 for t in ids if t in good) / max(len(ids), 1)
        br = sum(1 for t in ids if t in bad) / max(len(ids), 1)
        rec("C", f"μ={mu}", None,
            f"max|bias|={bmax}  好词率={gr:.2f} 坏词率={br:.2f}  {d.get('text','')[:20]!r}")

    for rho in rho_values:
        _, d = _post(base, "/generate",
                     {"prompt": prompt, "max_new_tokens": 12, "rho": rho})
        ids = d.get("ids") or []
        gr = sum(1 for t in ids if t in good) / max(len(ids), 1)
        br = sum(1 for t in ids if t in bad) / max(len(ids), 1)
        rec("C", f"ρ={rho}", None, f"好词率={gr:.2f} 坏词率={br:.2f}")


def group_c_write(base: str) -> None:
    print("\n[C+] 写入与热更新", flush=True)
    _, before = _get(base, "/inspect")
    _, d = _post(base, "/event",
                 {"text": "这是一条测试写入的事件，用于验证快照热更新。", "score": 81,
                  "mark": "selftest"})
    _, after = _get(base, "/inspect")
    ok = (d.get("added") is not None
          and after.get("m2_version") != before.get("m2_version")
          and after.get("n_events") == (before.get("n_events") or 0) + 1)
    rec("C", "写入并重建快照", ok,
        f"{before.get('m2_version')} -> {after.get('m2_version')}")


# ---------------------------------------------------------------- 主流程


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--suite", default=str(WS / "tests" / "suite.json"))
    ap.add_argument("--write", action="store_true", help="额外做一次事件写入")
    ap.add_argument("--out", default=str(WS / "m2_state" / "test_report"))
    args = ap.parse_args()

    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    base = (args.base or suite["base_url"]).rstrip("/")
    model = suite["model_id"]

    try:
        st, _ = _get(base, "/health")
    except Exception as e:  # noqa: BLE001
        print(f"连不上 {base}: {e}\n先跑 ./m2serve start", flush=True)
        raise SystemExit(2)
    if st != 200:
        print(f"{base}/health 返回 {st}，先跑 ./m2serve start", flush=True)
        raise SystemExit(2)

    print(f"目标 {base}  模型 {model}", flush=True)
    t0 = time.time()
    group_a(base, model, suite["thresholds"])
    group_b(base, model, suite["prompts_qa"], suite["thresholds"])
    group_c(base, suite["c_prompt"], suite["c_mu_values"], suite["c_rho_values"])
    if args.write:
        group_c_write(base)

    a = [r for r in ROWS if r["group"] == "A"]
    failed = [r for r in a if r["status"] == "FAIL"]
    total = len(ROWS)

    lines = ["# M2 测试报告", "",
             f"- 目标：`{base}`　模型：`{model}`",
             f"- 耗时：{time.time() - t0:.1f}s",
             f"- A 组契约：{len(a) - len(failed)}/{len(a)} 通过", "",
             "| 组 | 用例 | 结果 | 详情 |", "|---|---|---|---|"]
    for r in ROWS:
        lines.append(f"| {r['group']} | {r['name']} | {r['status']} | {r['detail']} |")
    if failed:
        lines += ["", "## 失败项", ""] + [f"- **{r['name']}**：{r['detail']}" for r in failed]
    lines += ["", "> C 组在每堆事件数 < 100 时不构成判据，只作 S3 对比基线。"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"rows": ROWS, "elapsed_s": round(time.time() - t0, 1)},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nA 组 {len(a) - len(failed)}/{len(a)} 通过   共 {total} 条", flush=True)
    print(f"报告 {out.with_suffix('.md')}", flush=True)
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()
