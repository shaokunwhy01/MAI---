"""m2/serve.py —— 常驻 HTTP 服务（模型只加载一次）

    m2ctl serve --port 8765

接口（全部 JSON）：

    GET  /health                    服务与记忆状态
    GET  /inspect                   快照体检（不生成）
    POST /generate                  {prompt, max_new_tokens, temperature, mu, nu, rho, ...}
    POST /event                     {text, score, mark}  追加一条事件并重建快照

单进程、单锁：CPU 上推理本来就是串行的，并发只会互相拖慢。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch

from .bootstrap import load_bootstrap
from .generate import CONFIG_VERSION, DECODE_VERSION, generate
from .memory import S0, BiasConfig, Memory, read_stats
from .readout import FrozenReadout

_LOCK = threading.Lock()
_S: dict[str, Any] = {}


def _json(h: BaseHTTPRequestHandler, obj: Any, code: int = 200) -> None:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    h.send_response(code)
    h.send_header("Content-Type", "application/json; charset=utf-8")
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    h.wfile.write(body)


# DSH 把运行时上下文以 **user 角色**的快照追加在用户问题之后
# （dsh-system-prompt：ordered dynamic contexts → sourced user-role snapshots）。
# 0.5B 底座会顺着最后一段英文续写，中文问题被这层"环境播报"淹没。
# 这里只把**尾部连续**的元信息快照并回 system 段，历史顺序不动，
# 让最后一条消息始终是真正的问题。
_META_MARKERS = (
    "Current runtime context",
    "Time sampled while preparing",
    "Browser time zone for this request",
    "Elapsed since the preceding",
    "Current working directory:",
    "Current DSH file policy:",
    "Approval prompts are enabled",
    "Current DeepSeek Harness implementation checkout:",
    "Current DeepSeek Harness Web GUI URL:",
    "A skill is a reusable set of task-specific instructions",
    "<system-reminder>",
)


def _text_of(msg: dict) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def _is_meta(msg: dict) -> bool:
    if msg.get("role") != "user":
        return False
    t = _text_of(msg).lstrip()
    return any(t.startswith(m) for m in _META_MARKERS)


def _demote_trailing_meta(msgs: list) -> tuple[list, int]:
    """把尾部连续的运行时上下文快照并进 system，返回 (新消息列表, 移动条数)。"""
    cut = len(msgs)
    while cut > 0 and _is_meta(msgs[cut - 1]):
        cut -= 1
    if cut == len(msgs):
        return msgs, 0
    body = "\n\n".join(_text_of(m).strip() for m in msgs[cut:])
    out = list(msgs[:cut])
    if out and out[0].get("role") == "system":
        out[0] = {**out[0], "content": _text_of(out[0]).rstrip() + "\n\n" + body}
    else:
        out.insert(0, {"role": "system", "content": body})
    return out, len(msgs) - cut


def _versions(mem: Memory | None) -> dict[str, str]:
    return {
        "base": "b1",
        "e": "e1",
        "m2": (mem.meta.get("m2_version") if mem else "m0") or "m0",
        "decode": DECODE_VERSION,
        "config": CONFIG_VERSION,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "m2/0.2.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        ts = datetime.now().strftime("%m-%d %H:%M:%S")
        ua = (self.headers.get("User-Agent") or "-")[:40]
        sys.stderr.write("%s  %s  %-10s  %s\n" % (ts, self.address_string(), ua, fmt % args))
        sys.stderr.flush()

    # ---------------------------------------------------------- 路由

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            if path == "/":
                _json(self, {
                    "name": "m2", "version": CONFIG_VERSION,
                    "endpoints": {
                        "GET /health": "服务与记忆状态",
                        "GET /inspect": "快照体检",
                        "GET /events": "事件列表 ?only_unscored=1&limit=N",
                        "POST /generate": "{prompt, max_new_tokens, temperature, mu, nu, rho}",
                        "POST /event": "{text, score, mark}",
                        "POST /score": "{id, score} 人工改分",
                    },
                })
            elif path == "/health":
                mem = _S.get("mem")
                _json(self, {
                    "ok": True,
                    "uptime_s": round(time.time() - _S["t0"], 1),
                    "base": _S["readout"].describe(),
                    "events": len(mem.events) if mem else 0,
                    "m2_version": (mem.meta.get("m2_version") if mem else None),
                    "versions": _versions(mem),
                })
            elif path == "/inspect":
                _json(self, read_stats())
            elif path == "/events":
                q = parse_qs(urlparse(self.path).query)
                only = q.get("only_unscored", ["0"])[0] not in ("0", "false", "no")
                limit = int(q.get("limit", ["0"])[0]) or None
                mem = _S.get("mem")
                if mem is None:
                    raise RuntimeError("服务未挂载记忆层（缺 bootstrap）")
                _json(self, {
                    "total": len(mem.events),
                    "pending": len(mem.unscored()),
                    "neutral": S0,
                    "items": mem.list_events(only, limit),
                })
            elif path == "/v1/models":
                _json(self, {
                    "object": "list",
                    "data": [{
                        "id": _S["model_id"], "object": "model",
                        "created": int(_S["t0"]), "owned_by": "local-m2",
                    }],
                })
            else:
                _json(self, {"error": "not found", "path": path}, 404)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            _json(self, {"error": type(e).__name__, "detail": str(e)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError as e:
            _json(self, {"error": "bad json", "detail": str(e)}, 400)
            return

        try:
            if path == "/generate":
                _json(self, self._generate(body))
            elif path == "/event":
                _json(self, self._event(body))
            elif path == "/score":
                _json(self, self._score(body))
            elif path == "/v1/chat/completions":
                self._chat(body)
            else:
                _json(self, {"error": "not found", "path": path}, 404)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            _json(self, {"error": type(e).__name__, "detail": str(e)}, 500)

    # ---------------------------------------------------------- 实现

    def _generate(self, body: dict) -> dict:
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt 必填")
        n = int(body.get("max_new_tokens", 32))
        if not (1 <= n <= _S["max_tokens"]):
            raise ValueError(f"max_new_tokens 必须在 1..{_S['max_tokens']}")

        with _LOCK:
            mem: Memory | None = _S.get("mem")
            hook = None
            if mem is not None and len(mem.events) > 0:
                cfg = BiasConfig(
                    mu=float(body.get("mu", 1.0)),
                    nu=float(body.get("nu", 0.3)),
                    rho=float(body.get("rho", 0.5)),
                    alpha=float(body.get("alpha", 3.0)),
                    beta=float(body.get("beta", 0.9)),
                )
                hook = lambda step, z, st: mem.bias(step, z, st, cfg)  # noqa: E731

            res = generate(
                _S["readout"], prompt,
                max_new_tokens=n,
                temperature=float(body.get("temperature", 0.0)),
                top_k=body.get("top_k"),
                seed=int(body.get("seed", 0)),
                record=bool(body.get("record", True)),
                versions=_versions(mem),
                hook=hook,
            )
            # 调试接口默认不入账（跑实验别污染账本），显式 "remember": true 才写。
            new_ev = None
            if mem is not None and body.get("remember") and res.text.strip():
                new_ev = mem.append(res.text, float(body.get("score", S0)),
                                    str(body.get("mark") or "api"))
                sys.stderr.write("%s  MEM  +%s  s=%.0f  %dtok  %r\n"
                                 % (datetime.now().strftime("%m-%d %H:%M:%S"), new_ev["id"],
                                    new_ev["score"], new_ev["n_tok"], new_ev["text"][:28]))
                sys.stderr.flush()
        out = res.to_dict()
        if new_ev is not None:
            out["event"] = new_ev
        return out

    def _chat(self, body: dict) -> None:
        """OpenAI 兼容的 chat/completions。流式只发一个增量块再到 [DONE]。"""
        msgs = body.get("messages") or []
        if not msgs:
            _json(self, {"error": {"message": "messages 必填"}}, 400)
            return

        ua = (self.headers.get("User-Agent") or "-")[:40]
        msgs, moved = _demote_trailing_meta(msgs)
        sys.stderr.write("%s  CHAT  ua=%s  msgs=%d  tools=%d  stream=%s  max_tokens=%s  meta_moved=%d\n"
                         % (datetime.now().strftime("%m-%d %H:%M:%S"), ua, len(msgs),
                            len(body.get("tools") or []), body.get("stream"),
                            body.get("max_tokens"), moved))
        sys.stderr.flush()

        readout = _S["readout"]
        text = readout.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        ids = readout.tok(text, return_tensors="pt").input_ids.to(readout.device)
        L = _S["max_len"]
        if ids.shape[1] > L:
            keep_head = int(L * 0.4)               # 系统提示在开头
            keep_tail = L - keep_head              # 生成提示在结尾
            ids = torch.cat([ids[:, :keep_head], ids[:, -keep_tail:]], dim=1)

        n = int(body.get("max_tokens") or 64)
        n = max(1, min(n, _S["max_tokens"]))
        t0 = time.time()

        with _LOCK:
            mem: Memory | None = _S.get("mem")
            hook = None
            if mem is not None and len(mem.events) > 0:
                cfg = BiasConfig(mu=float(body.get("mu", _S.get("mu", 1.0))))
                hook = lambda step, z, st: mem.bias(step, z, st, cfg)  # noqa: E731
            res = generate(
                readout, "", max_new_tokens=n,
                temperature=float(body.get("temperature") or 0.0),
                record=False, versions=_versions(mem),
                hook=hook, prompt_ids=ids,
            )

            # §20.2：一次生成结束追加一条 event。分数不在这里决定——
            # 默认落在中性线 S0（w=0，不进好/坏堆），等人工 /score 改分。
            # 带 tools 的请求算代理轮；标题/摘要这类 aux 调用不入账。
            remember = body.get("remember")
            if remember is None:
                remember = bool(body.get("tools"))
            new_ev = None
            if mem is not None and remember and res.text.strip():
                new_ev = mem.append(res.text, float(body.get("score", S0)),
                                    str(body.get("mark") or "chat"))
                sys.stderr.write("%s  MEM  +%s  s=%.0f  %dtok  %r\n"
                                 % (datetime.now().strftime("%m-%d %H:%M:%S"), new_ev["id"],
                                    new_ev["score"], new_ev["n_tok"], new_ev["text"][:28]))
                sys.stderr.flush()

        cid = f"chatcmpl-{int(t0 * 1000)}"
        created = int(t0)
        prompt_tokens = int(ids.shape[1])
        comp_tokens = len(res.ids)

        if body.get("stream"):
            # SSE 无法预知长度：必须 chunked 或关闭连接，否则客户端不知道
            # 响应已结束（curl/requests 会一直等到超时）。这里直接关连接。
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def ev(obj: dict) -> None:
                self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
                self.wfile.flush()

            base = {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": body.get("model") or _S["model_id"]}
            ev({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": res.text},
                                     "finish_reason": None}]})
            ev({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": comp_tokens,
                          "total_tokens": prompt_tokens + comp_tokens},
                **({"m2_event": new_ev} if new_ev else {})})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        _json(self, {
            "id": cid, "object": "chat.completion", "created": created,
            "model": body.get("model") or _S["model_id"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": res.text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": comp_tokens,
                      "total_tokens": prompt_tokens + comp_tokens},
            **({"m2_event": new_ev} if new_ev else {}),
        })

    def _event(self, body: dict) -> dict:
        text = body.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("text 必填")
        with _LOCK:
            mem: Memory | None = _S.get("mem")
            if mem is None:
                raise RuntimeError("服务未挂载记忆层（缺 bootstrap）")
            ev = mem.append(text, float(body.get("score", S0)), body.get("mark", ""))
            return {"added": ev, "stats": mem.stats()}

    def _score(self, body: dict) -> dict:
        """人工评分：只改分数，文本不动；改完按新分数重建好/坏堆。"""
        ev_id = body.get("id")
        if not isinstance(ev_id, str) or not ev_id:
            raise ValueError("id 必填（如 ev_0014）")
        if "score" not in body:
            raise ValueError("score 必填")
        with _LOCK:
            mem: Memory | None = _S.get("mem")
            if mem is None:
                raise RuntimeError("服务未挂载记忆层（缺 bootstrap）")
            ev = mem.rescore(ev_id, float(body["score"]))
            return {"updated": ev, "stats": mem.stats()}


def serve(host: str = "127.0.0.1", port: int = 8765, warmup: bool = True,
          max_tokens: int = 128, max_len: int = 4096, model_id: str = "m2-0.2.1",
          mu: float = 1.0) -> None:
    t0 = time.time()
    print(f"[serve] 加载底座 {CONFIG_VERSION} ...", flush=True)
    readout = FrozenReadout()

    bs = load_bootstrap()
    mem: Memory | None = None
    if bs is None:
        print("[serve] 无 bootstrap.json：只提供纯底座生成", flush=True)
    else:
        print("[serve] 挂载记忆层 ...", flush=True)
        mem = Memory(readout, bs)
        if len(mem.events) > 0:
            mem._ensure_tables()          # 预计算 Wn / xn，避免首个请求慢
            mem._trend_tables()
        print(f"[serve] 事件 {len(mem.events)} 条，{mem.meta.get('m2_version')}", flush=True)

    _S.update({"readout": readout, "mem": mem, "t0": t0,
               "max_tokens": max_tokens, "max_len": max_len,
               "model_id": model_id, "mu": mu})

    if warmup:
        print("[serve] 预热 ...", flush=True)
        generate(readout, "预热", max_new_tokens=1, record=False)

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    print(f"[serve] 就绪 http://{host}:{port}  pid={os.getpid()}  "
          f"(启动 {time.time() - t0:.1f}s, 上限 {max_tokens} token, ctx {max_len})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] 关闭", flush=True)
    finally:
        httpd.server_close()
