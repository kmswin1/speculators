#!/usr/bin/env python3
"""On-policy response regeneration for speculator training.
Reads prompts.jsonl ({"messages":[{"role":"user","content":...}]}), calls the FP8
A.X-K2 vLLM chat endpoint, and writes on-policy conversations (user + A.X-K2 assistant
response, incl. its reasoning) to conversations.jsonl. Resumable."""
import json, os, asyncio, argparse
import httpx

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://compute-gpu-st-distributed-ml-13:8202/v1/chat/completions")
    ap.add_argument("--model", default="A.X-K2")
    ap.add_argument("--prompts", default="/t1data/users/kmswin7/speculators/data/prompts.jsonl")
    ap.add_argument("--out", default="/t1data/users/kmswin7/speculators/data/conversations.jsonl")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    return ap.parse_args()

async def one(client, sem, args, idx, msgs, out_f, lock, done):
    if idx in done:
        return
    payload = {"model": args.model, "messages": msgs,
               "temperature": args.temperature, "top_p": args.top_p,
               "max_tokens": args.max_tokens}
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(args.endpoint, json=payload, timeout=1200)
                r.raise_for_status()
                d = r.json()
                content = d["choices"][0]["message"]["content"]
                rc = d["choices"][0]["message"].get("reasoning_content")
                if not content or not content.strip():
                    return
                asst = {"role": "assistant", "content": content}
                if rc:
                    asst["reasoning_content"] = rc
                row = {"idx": idx, "messages": msgs + [asst]}
                async with lock:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n"); out_f.flush()
                return
            except Exception as e:
                if attempt == 3:
                    print(f"[regen] idx={idx} FAIL {str(e)[:80]}", flush=True)
                await asyncio.sleep(3 * (attempt + 1))

async def main():
    args = parse()
    prompts = []
    for line in open(args.prompts, encoding="utf-8"):
        try: prompts.append(json.loads(line)["messages"])
        except Exception: pass
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try: done.add(json.loads(line)["idx"])
            except Exception: pass
    print(f"[regen] {len(prompts)} prompts, {len(done)} already done, {len(prompts)-len(done)} to do", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    out_f = open(args.out, "a", encoding="utf-8")
    limits = httpx.Limits(max_connections=args.concurrency + 8, max_keepalive_connections=args.concurrency + 8)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [one(client, sem, args, i, m, out_f, lock, done) for i, m in enumerate(prompts)]
        n = 0
        for fut in asyncio.as_completed(tasks):
            await fut; n += 1
            if n % 100 == 0: print(f"[regen] {n}/{len(tasks)}", flush=True)
    out_f.close()
    print("REGEN_COMPLETE", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
