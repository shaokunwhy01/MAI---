"""m2.cli —— m2ctl 命令行入口。

设计原则（§20）：
  * 每条命令都支持 --json，给 agent 一个稳定的解析面
  * 默认只输出人读摘要，账本按需落盘
  * 模型只在需要的命令里加载（health 不加载权重）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from . import bootstrap as boot
from .readout import DEFAULT_MODEL, FrozenReadout, resolve_model_path

STATE_DIR = boot.STATE_DIR
RUNS_DIR = STATE_DIR / "runs"
SNAPSHOT_DIR = STATE_DIR / "snapshot"


# ------------------------------------------------------------------ 工具


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _versions() -> dict[str, str]:
    b = boot.load_bootstrap()
    return {
        "base": "b1",
        "e": "e1" if b else "e0",
        "m2": "m0",
        "decode": "d1",
        "config": "beta0.2",
    }


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for k, v in payload.items():
            if isinstance(v, (dict, list)):
                print(f"{k}: {json.dumps(v, ensure_ascii=False)}")
            else:
                print(f"{k}: {v}")


def _write_ledger(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


# ------------------------------------------------------------------ health


def cmd_health(args: argparse.Namespace) -> int:
    import torch
    import transformers

    path = resolve_model_path(DEFAULT_MODEL)
    cfg_path = Path(path) / "config.json"
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": "cpu",
        "mps": bool(torch.backends.mps.is_available()),
    }

    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        info["base"] = DEFAULT_MODEL
        info["base_path"] = path
        info["layers"] = cfg.get("num_hidden_layers")
        info["hidden_dim"] = cfg.get("hidden_size")
        info["vocab"] = cfg.get("vocab_size")
        info["tied_head"] = bool(cfg.get("tie_word_embeddings", False))
    else:
        info["base"] = f"{DEFAULT_MODEL}（未找到本地快照，将尝试联网）"

    bs = boot.load_bootstrap()
    if bs:
        info["bootstrap"] = {
            "path": str(boot.BOOTSTRAP_PATH),
            "kappa_d": bs["kappa_d"],
            "mu0_dim": len(bs["mu0"]),
            "corpus_texts": bs["corpus"]["n_texts"],
        }
    else:
        info["bootstrap"] = None

    n_events = 0
    if SNAPSHOT_DIR.exists():
        for f in SNAPSHOT_DIR.glob("*.jsonl"):
            n_events += sum(1 for _ in f.open(encoding="utf-8"))
    info["events"] = n_events
    info["stage"] = "S0（空记忆）" if n_events == 0 else f"S1/S2（{n_events} 条事件）"
    info["ready"] = bool(cfg_path.exists())

    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print(f"torch        {info['torch']}   transformers {info['transformers']}   cpu")
        print(f"base         {info.get('base')}")
        if cfg_path.exists():
            print(f"             {info['layers']} 层 / hidden {info['hidden_dim']} / vocab {info['vocab']}"
                  f" / tied={info['tied_head']}")
            print(f"path         {path}")
        print(f"bootstrap    " + (f"κ_d={bs['kappa_d']:.2f}, μ0={len(bs['mu0'])} 维"
                                 if bs else "（无）→ 先跑 m2ctl bootstrap"))
        print(f"events       {n_events}")
        print(f"stage        {info['stage']}")
    return 0 if info["ready"] else 2


# ------------------------------------------------------------------ bootstrap


def cmd_bootstrap(args: argparse.Namespace) -> int:
    corpus = None
    if args.corpus:
        corpus = (boot.DEFAULT_CORPUS.strip().splitlines()
                  + Path(args.corpus).read_text(encoding="utf-8").splitlines())
    payload = boot.build(model_id=args.model, layer=args.layer, corpus=corpus,
                         out_path=Path(args.out), verbose=True)
    if args.json:
        print(json.dumps({k: payload[k] for k in ("version", "kappa_d", "kappa_d_detail", "corpus")},
                         ensure_ascii=False, indent=2))
    return 0


# ------------------------------------------------------------------ run


def _load(args: argparse.Namespace) -> FrozenReadout:
    return FrozenReadout(model_id=args.model, layer=args.layer, device=args.device)


def cmd_run(args: argparse.Namespace) -> int:
    from .generate import generate
    from .memory import BiasConfig, Memory

    readout = _load(args)
    ver = _versions()
    bs = boot.load_bootstrap()
    hook = None
    mem = None
    if bs is None:
        print("[warn] 无 bootstrap.json，按纯底座运行。", file=sys.stderr)
    elif not args.no_memory and any(SNAPSHOT_DIR.glob("events.jsonl")):
        mem = Memory(readout, bs)
        cfg = BiasConfig(mu=args.mu, nu=args.nu, rho=args.rho,
                         alpha=args.alpha, beta=args.beta)
        ver["m2"] = mem.meta.get("m2_version", "m0")
        hook = lambda step, z, st: mem.bias(step, z, st, cfg)  # noqa: E731

    res = generate(
        readout,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        record=not args.no_record,
        versions=ver,
        hook=hook,
    )

    ledger_path = None
    if args.ledger:
        ledger_path = _write_ledger(res.ledger, Path(args.ledger))

    if args.json:
        out = res.to_dict()
        if ledger_path:
            out["ledger_path"] = str(ledger_path)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print(f"[run] {res.versions}")
    print(f"[run] max_new={args.max_new_tokens} temp={args.temperature} "
          f"seed={args.seed}  {res.elapsed:.1f}s  stop={res.stopped}")
    if mem is not None:
        print(f"[run] μ={args.mu} ν={args.nu} ρ={args.rho}  "
              f"max|bias|={res.run_stats.get('bias_absmax')}  "
              f"受影响 token≤{res.run_stats.get('bias_nonzero_max')}")
    print("-" * 60)
    print(res.text)
    print("-" * 60)
    if res.ledger:
        print(f"{'i':>3} {'tok':<10} {'z':>8} {'p':>8} {'a':>7} {'reb':>7} {'bias':>7}")
        for r in res.ledger[: args.show]:
            tok = r["tok"].replace("\n", "\\n")[:10]
            print(f"{r['i']:>3} {tok:<10} {r['z']:>8.2f} {r['p']:>8.4f} "
                  f"{r['a']:>7.3f} {r['reb']:>7.3f} {r['bias']:>7.3f}")
        if len(res.ledger) > args.show:
            print(f"... {len(res.ledger) - args.show} more")
    if ledger_path:
        print(f"ledger → {ledger_path}")
    return 0


def cmd_parity(args: argparse.Namespace) -> int:
    from .generate import pure_base_parity

    readout = _load(args)
    res = pure_base_parity(readout, args.prompt, n=args.max_new_tokens)
    out = {
        "identical": res["identical"],
        "n": args.max_new_tokens,
        "prompt": args.prompt,
        "text": res["text"],
        "ids": res["ids_a"],
        "pass": res["identical"],
    }
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"prompt: {args.prompt}")
        print(f"n={args.max_new_tokens}  ids={res['ids_a']}")
        print(f"text: {res['text']!r}")
        print("PASS 空记忆与纯底座逐位相同" if out["pass"]
              else "FAIL 逐位不同——启动判据不成立")
    return 0 if out["pass"] else 1


# ------------------------------------------------------------------ compare


def _read_ledger(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def cmd_compare(args: argparse.Namespace) -> int:
    a = _read_ledger(Path(args.a))
    b = _read_ledger(Path(args.b))
    n = min(len(a), len(b))
    ids_a = [r["id"] for r in a]
    ids_b = [r["id"] for r in b]

    first = next((i for i in range(n) if ids_a[i] != ids_b[i]), None)
    same = sum(1 for i in range(n) if ids_a[i] == ids_b[i])
    # 简单编辑距离（token 级）
    dp = list(range(len(ids_b) + 1))
    for i in range(1, len(ids_a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(ids_b) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (0 if ids_a[i - 1] == ids_b[j - 1] else 1))
            prev = cur

    out = {
        "len_a": len(ids_a), "len_b": len(ids_b),
        "same_ratio": round(same / n, 4) if n else None,
        "first_divergence": first,
        "edit_distance": dp[-1],
    }
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"A {len(ids_a)} tokens | B {len(ids_b)} tokens")
        print(f"一致率 {out['same_ratio']}")
        if first is None:
            print("前缀完全一致")
        else:
            print(f"首次分歧 step={first}: {a[first]['tok']!r} vs {b[first]['tok']!r}")
        print(f"token 编辑距离 {dp[-1]}")
    return 0


# ------------------------------------------------------------------ inspect


def cmd_inspect(args: argparse.Namespace) -> int:
    from .memory import read_stats

    st = read_stats()
    if args.json:
        print(json.dumps(st, ensure_ascii=False, indent=2))
        return 0

    print(f"事件      {st['n_events']} 条（好 {st['n_good']} / 坏 {st['n_bad']}）")
    print(f"版本      {st['m2_version']}")
    print(f"R⁺/R⁻     {st['R_plus']:.3f} / {st['R_minus']:.3f}"
          f"   闸门 {'✓' if st['gate_plus'] else '✗'} / {'✓' if st['gate_minus'] else '✗'}")
    print(f"rho_T     {st['rho_T']}    ‖D‖ {st['norm_D']:.3f}    T_m {'on' if st['T_m_on'] else 'off'}")
    print(f"D_ref     {st['D_ref']}    tau_d {st['tau_d']}")
    if st["saturation"]:
        a = st["saturation"]
        print(f"a_e 饱和  均值 {a['mean']:.3f}  在带内 {a['in_band']:.1%}  "
              f"饱和(高) {a['frac_hi']:.1%}")
    print(f"词表      去重 {st['n_distinct']}   停用占比 {st['stop_ratio']}")
    print(f"阶段      {st['stage']}   趋势 {st['trend']} / 相量 {st['phasor']}")
    return 0


# ------------------------------------------------------------------ event


def cmd_event(args: argparse.Namespace) -> int:
    from .memory import Memory

    bs = boot.load_bootstrap()
    if bs is None or "mu_w" not in bs:
        print("缺少 bootstrap（且需含 μ_W）。先跑：m2ctl bootstrap", file=sys.stderr)
        return 2

    items: list[dict[str, Any]] = []
    if args.file:
        for line in Path(args.file).read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(json.loads(line))
    if args.text:
        items.append({"text": args.text, "score": args.score, "mark": args.mark})
    if not items:
        print("没有要写入的事件", file=sys.stderr)
        return 2

    readout = _load(args)
    mem = Memory(readout, bs)
    rows = []
    for it in items:
        ev = mem.append(it["text"], float(it.get("score", 80.0)), it.get("mark", ""))
        rows.append(ev)
        print(f"  + {ev['id']}  s={ev['score']:.0f}  {ev['n_tok']}tok  {ev['text'][:34]!r}",
              flush=True)

    st = mem.stats()
    if args.json:
        print(json.dumps({"added": rows, "stats": st}, ensure_ascii=False, indent=2))
    else:
        print(f"\n共 {st['n_events']} 条   R⁺={st['R_plus']:.3f} R⁻={st['R_minus']:.3f}   "
              f"闸门 {st['gate_plus']}/{st['gate_minus']}   {st['stage']}")
    return 0


# ------------------------------------------------------------------ score


def cmd_score(args: argparse.Namespace) -> int:
    from .memory import S0, Memory

    bs = boot.load_bootstrap()
    if bs is None or "mu_w" not in bs:
        print("缺少 bootstrap（且需含 μ_W）。先跑：m2ctl bootstrap", file=sys.stderr)
        return 2

    readout = _load(args)
    mem = Memory(readout, bs)

    if args.list:
        rows = mem.unscored()
        if args.json:
            print(json.dumps({"unscored": rows, "neutral": S0}, ensure_ascii=False, indent=2))
        elif not rows:
            print(f"没有待评分的事件（中性线 {S0:.0f}）")
        else:
            print(f"待评分 {len(rows)} 条（score={S0:.0f}，w=0，不进好/坏堆）：")
            for ev in rows:
                print(f"  {ev['id']}  {ev['n_tok']:>3}tok  {ev['mark'] or '-':<10} {ev['text'][:44]!r}")
        return 0

    if not args.id:
        print("要么给 --id <ev_xxxx>（配合 --score），要么给 --list", file=sys.stderr)
        return 2
    if args.score is None:
        print("--score 必填：>80 进好堆，<80 进坏堆，=80 中性（w=0，不生效）", file=sys.stderr)
        return 2

    try:
        ev = mem.rescore(args.id, args.score)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 2

    st = mem.stats()
    if args.json:
        print(json.dumps({"updated": ev, "stats": st}, ensure_ascii=False, indent=2))
    else:
        side = "好" if ev["score"] > S0 else ("坏" if ev["score"] < S0 else "中性(不生效)")
        print(f"  ~ {ev['id']}  s={ev['score']:.0f} → {side}")
        print(f"共 {st['n_events']} 条   R⁺={st['R_plus']:.3f} R⁻={st['R_minus']:.3f}   "
              f"闸门 {st['gate_plus']}/{st['gate_minus']}   {st['stage']}")
        print("注意：常驻服务持有自己的内存副本，改分后请重启 m2serve 或改用 HTTP "
              "POST /score。", file=sys.stderr)
    return 0


# ------------------------------------------------------------------ sweep


def cmd_sweep(args: argparse.Namespace) -> int:
    from .generate import generate

    if args.param in ("mu", "nu", "rho"):
        msg = {"implemented": False,
               "reason": f"{args.param} 只在有记忆时才有效；当前为 S0 空记忆。",
               "next": "等 memory.py 上线后再扫。"}
        print(json.dumps(msg, ensure_ascii=False, indent=2) if args.json else msg["reason"])
        return 0

    readout = _load(args)
    values = [float(v) for v in args.values.split(",")]
    ver = _versions()
    base_ids: list[int] | None = None
    rows: list[dict[str, Any]] = []

    for v in values:
        kw: dict[str, Any] = {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        }
        if args.param == "temperature":
            kw["temperature"] = v
        elif args.param == "top_k":
            kw["top_k"] = int(v) if v > 0 else None
        elif args.param == "max_new_tokens":
            kw["max_new_tokens"] = int(v)
        elif args.param == "seed":
            kw["seed"] = int(v)

        res = generate(readout, args.prompt, record=False, versions=ver, **kw)
        if base_ids is None:
            base_ids = res.ids
        same = sum(1 for x, y in zip(base_ids, res.ids) if x == y)
        rows.append({
            "value": v,
            "len": len(res.ids),
            "same_prefix": same,
            "text": res.text[:80],
            "elapsed": round(res.elapsed, 1),
        })
        print(f"[{args.param}={v}] {res.elapsed:.1f}s  {res.text[:60]!r}", flush=True)

    if args.json:
        print(json.dumps({"param": args.param, "rows": rows}, ensure_ascii=False, indent=2))
    return 0


# ------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="m2ctl", description="M2 调制架构调试入口（beta0.2）")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", default=DEFAULT_MODEL)
        sp.add_argument("--layer", type=int, default=16)
        sp.add_argument("--device", default="cpu")

    h = sub.add_parser("health", help="环境与状态自检")
    h.add_argument("--json", action="store_true")
    h.set_defaults(func=cmd_health)

    b = sub.add_parser("bootstrap", help="估计 κ_d 与 μ0")
    common(b)
    b.add_argument("--corpus", default=None)
    b.add_argument("--out", default=str(boot.BOOTSTRAP_PATH))
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_bootstrap)

    r = sub.add_parser("run", help="生成一段文本")
    common(r)
    r.add_argument("--prompt", required=True)
    r.add_argument("-n", "--max-new-tokens", type=int, default=32)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--top-k", type=int, default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--ledger", default=None, help="账本落盘路径（jsonl）")
    r.add_argument("--no-record", action="store_true")
    r.add_argument("--show", type=int, default=15)
    r.add_argument("--mu", type=float, default=1.0)
    r.add_argument("--nu", type=float, default=0.3)
    r.add_argument("--rho", type=float, default=0.5)
    r.add_argument("--alpha", type=float, default=3.0)
    r.add_argument("--beta", type=float, default=0.9)
    r.add_argument("--no-memory", action="store_true")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_run)

    pa = sub.add_parser("parity", help="启动判据：空记忆 == 纯底座（§13.1）")
    common(pa)
    pa.add_argument("--prompt", required=True)
    pa.add_argument("-n", "--max-new-tokens", type=int, default=12)
    pa.add_argument("--json", action="store_true")
    pa.set_defaults(func=cmd_parity)

    c = sub.add_parser("compare", help="比对两份账本")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_compare)

    i = sub.add_parser("inspect", help="快照体检")
    i.add_argument("--json", action="store_true")
    i.set_defaults(func=cmd_inspect)

    e = sub.add_parser("event", help="写入事件")
    common(e)
    e.add_argument("--text", default=None)
    e.add_argument("--score", type=float, default=80.0)
    e.add_argument("--mark", default="")
    e.add_argument("--file", default=None, help="批量：jsonl，每行 {text,score,mark}")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_event)

    sc = sub.add_parser("score", help="人工评分（改分/看待评）")
    common(sc)
    sc.add_argument("--id", default=None, help="事件 id，如 ev_0014")
    sc.add_argument("--score", type=float, default=None, help="新分数；>80 进好堆，<80 进坏堆")
    sc.add_argument("--list", action="store_true", help="列出待评分的条目（score=中性线）")
    sc.add_argument("--json", action="store_true")
    sc.set_defaults(func=cmd_score)

    sv = sub.add_parser("serve", help="常驻 HTTP 服务")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--max-tokens", type=int, default=128)
    sv.add_argument("--no-warmup", action="store_true")
    sv.add_argument("--max-len", type=int, default=4096)
    sv.add_argument("--model-id", default="m2-0.2.1")
    sv.add_argument("--mu", type=float, default=1.0)
    sv.set_defaults(func=lambda a: _run_serve(a))

    v = sub.add_parser("verify", help="完整性自检（可重复运行）")
    v.add_argument("-n", "--max-new-tokens", type=int, default=8)
    v.add_argument("--out", default=str(STATE_DIR / "verify.json"))
    v.set_defaults(func=lambda a: (_run_verify(a)))

    s = sub.add_parser("sweep", help="参数扫描")
    common(s)
    s.add_argument("--prompt", required=True)
    s.add_argument("--param", default="temperature",
                   choices=["temperature", "top_k", "max_new_tokens", "seed", "mu", "nu", "rho"])
    s.add_argument("--values", default="0,0.7")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--top-k", type=int, default=None)
    s.add_argument("--max-new-tokens", type=int, default=24)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_sweep)

    return p


def _run_serve(args: argparse.Namespace) -> int:
    from .serve import serve

    serve(host=args.host, port=args.port, warmup=not args.no_warmup,
          max_tokens=args.max_tokens, max_len=args.max_len,
          model_id=args.model_id, mu=args.mu)
    return 0


def _run_verify(args: argparse.Namespace) -> int:
    from . import verify as vmod

    import sys as _sys
    argv = ["-n", str(args.max_new_tokens), "--out", args.out]
    old = _sys.argv
    _sys.argv = ["m2.verify"] + argv
    try:
        vmod.main()
    except SystemExit as e:
        return int(e.code or 0)
    finally:
        _sys.argv = old
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
