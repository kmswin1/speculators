#!/usr/bin/env python3
"""Reasoning-heavy on-policy regen: forces A.X-K2 to think by prefilling the assistant
turn with '<think>\n' and using the /v1/completions endpoint, so the model produces a
long CoT + answer. Reads SFT prompts (messages), renders im_start manually. Resumable."""
import json, ast, os, asyncio, argparse, httpx

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)          # http://node:8202/v1/completions
    ap.add_argument("--model", default="A.X-K2")
    ap.add_argument("--prompts", required=True)           # jsonl of {messages:[...]}
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    return ap.parse_args()

def extract(c):
    if isinstance(c, list): parts = c
    elif isinstance(c, str) and c.lstrip().startswith("["):
        try: parts = ast.literal_eval(c)
        except Exception: return c
    else: return c
    if isinstance(parts, list):
        return " ".join(p.get("value","") for p in parts if isinstance(p, dict) and p.get("type")=="text")
    return str(parts)

def build_prompt(msgs):
    parts = []
    for m in msgs:
        if m.get("role") == "assistant": break   # only context up to first assistant
        c = extract(m.get("content", m.get("value","")))
        if c and str(c).strip():
            parts.append(f"<|im_start|>{m.get('role','user')}\n{c}<|im_end|>")
    if not parts: return None
    # ★ force thinking
    return "\n".join(parts) + "\n<|im_start|>assistant\n<think>\n"

async def one(client, sem, args, idx, msgs, out_f, lock, done):
    if idx in done: return
    prompt = build_prompt(msgs)
    if not prompt: return
    payload = {"model": args.model, "prompt": prompt, "temperature": args.temperature,
               "top_p": args.top_p, "max_tokens": args.max_tokens}
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(args.endpoint, json=payload, timeout=1800)
                r.raise_for_status()
                gen = r.json()["choices"][0]["text"]
                if not gen or not gen.strip(): return
                content = "<think>\n" + gen   # reconstruct full assistant turn
                user_turns = [m for m in msgs if m.get("role") != "assistant"]
                # keep only up to first assistant context
                ctx = []
                for m in msgs:
                    if m.get("role") == "assistant": break
                    c = extract(m.get("content", m.get("value","")))
                    if c and str(c).strip(): ctx.append({"role": m.get("role","user"), "content": c})
                conv = ctx + [{"role": "assistant", "content": content}]
                async with lock:
                    out_f.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n"); out_f.flush()
                return
            except Exception as e:
                if attempt == 3: print(f"[rr] idx={idx} FAIL {str(e)[:80]}", flush=True)
                await asyncio.sleep(3*(attempt+1))

async def main():
    args = parse()
    prompts = []
    for line in open(args.prompts, encoding="utf-8"):
        try:
            r = json.loads(line); m = r.get("messages") or r.get("conversations")
            if m: prompts.append(m)
        except Exception: pass
    done = set()
    if os.path.exists(args.out):
        done = set(range(sum(1 for _ in open(args.out, encoding="utf-8"))))  # rough resume by count
    print(f"[rr] {len(prompts)} prompts, endpoint={args.endpoint}", flush=True)
    sem = asyncio.Semaphore(args.concurrency); lock = asyncio.Lock()
    out_f = open(args.out, "a", encoding="utf-8")
    lim = httpx.Limits(max_connections=args.concurrency+8, max_keepalive_connections=args.concurrency+8)
    async with httpx.AsyncClient(limits=lim) as client:
        tasks = [one(client, sem, args, i, m, out_f, lock, set()) for i, m in enumerate(prompts)]
        n = 0
        for fut in asyncio.as_completed(tasks):
            await fut; n += 1
            if n % 100 == 0: print(f"[rr] {n}/{len(tasks)}", flush=True)
    out_f.close(); print("RR_COMPLETE", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
