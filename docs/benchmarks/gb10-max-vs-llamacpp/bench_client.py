#!/usr/bin/env python3
"""Dependency-free streaming benchmark client for an OpenAI-compatible server.
Measures TTFT and single-stream generation tok/s (comparable to llama.cpp tg128).
"""
import json, time, sys, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "model"
WARMUP, TIMED = 2, 5
MAX_TOKENS = 128
PROMPT = ("Write a detailed technical explanation of how memory bandwidth limits "
          "large language model inference throughput on unified-memory systems. ")

def one_request():
    body = json.dumps({"model": MODEL, "prompt": PROMPT, "max_tokens": MAX_TOKENS,
                       "temperature": 0.0, "stream": True}).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; ntok = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            txt = chunk.get("choices", [{}])[0].get("text", "")
            if txt:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                ntok += 1
    total = time.perf_counter() - t0
    gen_tps = (ntok - 1) / (total - ttft) if (ttft and ntok > 1 and total > ttft) else 0.0
    return {"ttft_s": ttft, "total_s": total, "out_tokens": ntok, "gen_tok_s": gen_tps}

def main():
    for _ in range(WARMUP):
        one_request()
    runs = [one_request() for _ in range(TIMED)]
    ttfts = [r["ttft_s"] for r in runs if r["ttft_s"]]
    tps = [r["gen_tok_s"] for r in runs if r["gen_tok_s"]]
    summary = {
        "model": MODEL, "warmup": WARMUP, "timed": TIMED, "max_tokens": MAX_TOKENS,
        "ttft_s_mean": round(sum(ttfts)/len(ttfts), 4) if ttfts else None,
        "gen_tok_s_mean": round(sum(tps)/len(tps), 2) if tps else None,
        "gen_tok_s_all": [round(x, 2) for x in tps],
        "runs": runs,
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
