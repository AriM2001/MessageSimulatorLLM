
# adapters.py
import hashlib, json, os, time, math, random, re
from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class Decision:
    action: str               # "ALLOW" | "COMPRESS" | "AGGREGATE" | "SKIP"
    param: Optional[int]      # ms for AGGREGATE else None
    size_bytes: Optional[int] # bytes for {ALLOW,COMPRESS} else None

VALID_ACTIONS = {"ALLOW","COMPRESS","AGGREGATE","SKIP"}

class LRUCache:
    def __init__(self, max_items=10000):
        self.store = {}
        self.order = []
        self.max_items = max_items

    def get(self, k):
        v = self.store.get(k)
        if v is not None:
            # simple recency bump
            self.order.remove(k); self.order.append(k)
        return v

    def put(self, k, v):
        if k in self.store:
            self.order.remove(k)
        self.store[k] = v
        self.order.append(k)
        if len(self.order) > self.max_items:
            old = self.order.pop(0)
            self.store.pop(old, None)

def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _extract_first_json_object(text: str) -> str:
    """
    Returns the first top-level {...} JSON object substring from text.
    Strips typical code fences if present. Raises ValueError if not found.
    """
    # strip common code fences
    t = text.strip()
    t = re.sub(r'^\s*```(?:json)?\s*', '', t, flags=re.I)
    t = re.sub(r'\s*```\s*$', '', t)

    # fast path: looks like pure JSON already
    if t.startswith('{') and t.endswith('}'):
        return t

    # scan for first balanced {...}
    depth = 0
    start = None
    for i, ch in enumerate(t):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return t[start:i+1]
    raise ValueError("No JSON object found in text.")
    
def _validate(dec: dict, *, full_size: int, compress_size: int, agg_ms_choices: tuple[int,...]) -> Decision:
    # basic schema & constraints
    a = dec.get("action")
    if a not in VALID_ACTIONS:
        raise ValueError(f"bad action {a}")
    p = dec.get("param", None)
    sz = dec.get("size_bytes", None)

    if a == "AGGREGATE":
        if p not in agg_ms_choices or sz is not None:
            raise ValueError("AGGREGATE requires param∈choices and size_bytes=null")
    elif a == "COMPRESS":
        if p is not None or sz != compress_size:
            raise ValueError("COMPRESS requires size_bytes=compress_size and param=null")
    elif a == "ALLOW":
        if p is not None or sz != full_size:
            raise ValueError("ALLOW requires size_bytes=full_size and param=null")
    else:  # SKIP
        if p is not None or sz is not None:
            raise ValueError("SKIP requires both null")

    return Decision(a, p, sz)

class TokenBucket:
    """Simple per-process rate limiter."""
    def __init__(self, rate_per_sec: float, burst: int):
        self.rate = rate_per_sec
        self.burst = burst
        self.tokens = burst
        self.last = time.time()

    def take(self, n=1):
        now = time.time()
        self.tokens = min(self.burst, self.tokens + (now - self.last)*self.rate)
        self.last = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

def make_call_model_fn(
    *,
    llm_call: Callable[[str], str],
    full_size: int,
    compress_size: int,
    agg_ms_choices: tuple[int,...],
    rps: float = 10.0,
    burst: int = 20,
):
    cache = LRUCache(50_000)
    bucket = TokenBucket(rps, burst)

    def _call(prompt: str) -> dict:
        key = _hash_prompt(prompt)
        cached = cache.get(key)
        if cached:
            return cached

        # crude, local rate-limit
        while not bucket.take(1):
            time.sleep(0.02)

        # call provider
        raw = llm_call(prompt)
        try:
            json_str = _extract_first_json_object(str(raw))
            obj = json.loads(json_str)
        except Exception as e:
            raise ValueError(f"LLM returned non-JSON: {raw[:160]}...") from e

        dec = _validate(obj, full_size=full_size, compress_size=compress_size, agg_ms_choices=agg_ms_choices)
        out = {"action": dec.action, "param": dec.param, "size_bytes": dec.size_bytes}
        cache.put(key, out)
        return out

    return _call
 