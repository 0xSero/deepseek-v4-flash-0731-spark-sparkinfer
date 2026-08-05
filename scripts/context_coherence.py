#!/usr/bin/env python3
"""Validate exact fact retention across several character-length contexts."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from typing import Any


DEFAULT_LENGTHS = (500, 2000, 5000, 10000, 20000)


def filler(length: int) -> str:
    unit = " amber cedar quartz river lattice orbit "
    return (unit * ((length // len(unit)) + 1))[:length]


def make_case(length: int) -> tuple[str, dict[str, Any]]:
    expected = {
        "begin": f"B-{length}-ALPHA",
        "middle": f"M-{length}-BRAVO",
        "end": f"E-{length}-CHARLIE",
        "checksum": length + 323,
    }
    prefix = (
        "Read the entire record and retain its exact facts.\n"
        f"BEGIN_FACT={expected['begin']}\n"
    )
    middle = f"\nMIDDLE_FACT={expected['middle']}\n"
    suffix = (
        f"\nEND_FACT={expected['end']}\n"
        f"CHECKSUM={expected['checksum']}\n"
        "Return only the schema object containing all four facts."
    )
    remaining = length - len(prefix) - len(middle) - len(suffix)
    if remaining < 0:
        raise ValueError(f"context length {length} is too small")
    left = remaining // 2
    prompt = prefix + filler(left) + middle + filler(remaining - left) + suffix
    if len(prompt) != length:
        raise AssertionError((len(prompt), length))
    return prompt, expected


def run_case(base_url: str, model: str, length: int) -> dict[str, Any]:
    prompt, expected = make_case(length)
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": f"coherence_{length}",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "begin": {"type": "string", "const": expected["begin"]},
                    "middle": {"type": "string", "const": expected["middle"]},
                    "end": {"type": "string", "const": expected["end"]},
                    "checksum": {
                        "type": "integer",
                        "const": expected["checksum"],
                    },
                },
                "required": ["begin", "middle", "end", "checksum"],
                "additionalProperties": False,
            },
        },
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_completion_tokens": 128,
        "chat_template_kwargs": {"thinking": False},
        "response_format": schema,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - started
    choice = body["choices"][0]
    parsed = json.loads(choice["message"]["content"])
    if parsed != expected:
        raise RuntimeError(f"{length}-character coherence failed: {parsed!r}")
    if choice["finish_reason"] != "stop":
        raise RuntimeError(f"{length}-character finish reason: {choice['finish_reason']}")
    usage = body["usage"]
    return {
        "characters": length,
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": int(usage["completion_tokens"]),
        "elapsed_s": elapsed,
        "finish_reason": choice["finish_reason"],
        "parsed": parsed,
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-flash-0731-spark")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_LENGTHS),
    )
    args = parser.parse_args()
    results = [
        run_case(args.base_url, args.model, length) for length in args.lengths
    ]
    print(json.dumps({"model": args.model, "results": results}, indent=2))


if __name__ == "__main__":
    main()
