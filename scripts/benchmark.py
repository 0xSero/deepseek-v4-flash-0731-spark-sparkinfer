#!/usr/bin/env python3
"""Measure C1/C2/C4 TTFT, prefill, and decode throughput over streaming SSE."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def request(
    base_url: str,
    model: str,
    prompt: str,
    output_tokens: int,
) -> dict[str, float | int]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": output_tokens,
        }
    ).encode()
    started = time.perf_counter()
    first_content_at: float | None = None
    usage: dict[str, int] = {}
    with urllib.request.urlopen(
        urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        ),
        timeout=1800,
    ) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if first_content_at is None and (
                    delta.get("content")
                    or delta.get("reasoning")
                    or delta.get("reasoning_content")
                ):
                    first_content_at = time.perf_counter()
    ended = time.perf_counter()
    if first_content_at is None:
        raise RuntimeError("stream ended without a content token")
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    ttft = first_content_at - started
    decode_s = max(ended - first_content_at, 1e-9)
    return {
        "started_at": started,
        "first_at": first_content_at,
        "ended_at": ended,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "prefill_tok_s": prompt_tokens / ttft,
        "decode_tok_s": max(completion_tokens - 1, 0) / decode_s,
        "elapsed_s": ended - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-flash-0731-spark")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--prompt-repetitions", type=int, default=2048)
    args = parser.parse_args()
    prompt = (
        "Study this repeated calibration phrase, then give one concise CUDA tip. "
        * args.prompt_repetitions
    )

    results = []
    for concurrency in (1, 2, 4):
        wall_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
            futures = [
                pool.submit(
                    request,
                    args.base_url,
                    args.model,
                    prompt,
                    args.output_tokens,
                )
                for _ in range(concurrency)
            ]
            runs = [future.result() for future in futures]
        wall_s = time.perf_counter() - wall_start
        aggregate_decode_window = max(
            max(float(run["ended_at"]) for run in runs)
            - min(float(run["first_at"]) for run in runs),
            1e-9,
        )
        aggregate_decode_tokens = sum(
            max(int(run["completion_tokens"]) - 1, 0) for run in runs
        )
        for run in runs:
            run.pop("started_at")
            run.pop("first_at")
            run.pop("ended_at")
        ttfts = [float(run["ttft_s"]) for run in runs]
        results.append(
            {
                "concurrency": concurrency,
                "wall_s": wall_s,
                "aggregate_decode_tok_s": (
                    aggregate_decode_tokens / aggregate_decode_window
                ),
                "mean_prefill_tok_s": statistics.fmean(
                    float(run["prefill_tok_s"]) for run in runs
                ),
                "mean_decode_tok_s_per_request": statistics.fmean(
                    float(run["decode_tok_s"]) for run in runs
                ),
                "ttft_p50_s": percentile(ttfts, 0.50),
                "ttft_p95_s": percentile(ttfts, 0.95),
                "requests": runs,
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
