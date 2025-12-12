# message_opt_sim.py
import json
import os
import simpy, math, random, heapq, hashlib
import ollama
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Callable
from collections import defaultdict
from math import isfinite

from adapters import make_call_model_fn
from providers import ollama_json

BITS_PER_BYTE = 8

os.environ["OLLAMA_MODEL"] = "phi3:mini"   # "llama3.1" was a previous model but it ran too slow "qwen2.5:7b-instruct", etc.

# -------------------- Link (network channel) --------------------
"""
    Simulated *directed* network link between two nodes 

    The link is represented as a SimPy process that:
      -Accepts messages via `send()` (they enter an internal queue).
      -For each queued message, waits for:
       serialization_time = size_bytes * 8 / bandwidth_bps
       plus a fixed propagation delay.
      -Delivers the message to the destination node's inbox.
"""
class Link:
    def __init__(self, env, u, v, bandwidth_bps, prop_delay_ms, capacity=32):   #udpate for smaller queue depth if needed
        # Simulation environment
        self.env = env
        # Source and destination node
        self.u, self.v = u, v
        # Link bandwidth
        self.bps = bandwidth_bps
        # Propagation delay
        self.prop = prop_delay_ms / 1000.0
        # Internal queue of messages
        self.q = simpy.Store(env, capacity=capacity) 
        # Metrics that will be oberserved
        self.bytes_sent = 0
        self.messages_sent = 0
        self.max_q_depth = 0
        env.process(self.run())

    # Enqueue message to be sent over this link
    def send(self, size_bytes, payload):
        # put() blocks if queue is full -> natural backpressure
        self.bytes_sent += size_bytes
        self.messages_sent += 1
        self.max_q_depth = max(self.max_q_depth, len(self.q.items) + 1)
        return self.q.put((size_bytes, payload))

    def run(self):
        while True:
            # Wait for message in queue
            size_bytes, payload = (yield self.q.get())
            # Push size_bytes through pipe with bits per second
            ser = (size_bytes * BITS_PER_BYTE) / self.bps
            # Simulate time
            yield self.env.timeout(ser + self.prop)
            payload["dst"].inbox.put(payload)

# -------------------- Queue-aware Rule Policy --------------------
class RulePolicy:
    """
    Simple rule-based policy for decision making in the message system
    
    Realistic queue-aware policy:
      - If tiny improvement and link is busy -> SKIP (shed low-value load)
      - If queue has backlog -> AGGREGATE (batch for a short window)
      - If small improvement -> COMPRESS (smaller payload)
      - Else -> ALLOW
    """
    def __init__(self, eps_small=0.05, eps_med=0.20, q_hi=1, agg_ms=20):
        # TINY Improvement Threshold (SKIP)
        self.eps_small = eps_small
        # SMALL Improvement threshold (COMPRESS)
        self.eps_med = eps_med
        # HIGH Queue Depth
        self.q_hi = q_hi
        self.agg_ms = agg_ms

    def decide(self, *, delta, queue_depth, util=None):
        # Low value message with busy queue
        if delta < self.eps_small and queue_depth > 0:
            return "SKIP", None, None
        # If high queue we encourage batching
        if queue_depth >= self.q_hi:
            return "AGGREGATE", self.agg_ms, None
        # Small improvements threshold send compressed message
        if delta < self.eps_med:
            return "COMPRESS", None, 48  # bytes
        # Messages that require full message
        return "ALLOW", None, 96         # bytes

# -------------------- Node (distributed process) --------------------

class Node:
    """
    Simulated node in a distributed shortest-path

    Each Node:
      - Maintains an estimate dist[s] = best known distance from source s to this node.
      - Sends "RELAX" messages to neighbors when it improves its estimate.
      - Receives messages on an inbox (SimPy Store) and processes them in recv_loop().
      - Optionally uses a RulePolicy to decide whether to send, compress, aggregate,
        or skip outgoing messages, based on their incremental value and link load.

    The global Controller (self.ctrl):
      - Owns all nodes.
      - Tracks `inflight` message count for instrumentation.
      - Exposes a size_model(kind) callback to estimate message sizes in bytes.
    """
    def __init__(self, env, node_id, controller, policy):
        self.env = env
        self.id = node_id
        self.ctrl = controller
        self.policy = policy                    # may be None for baseline
        self.neighbors = {}                     # v -> (link, weight)
        self.inbox = simpy.Store(env)
        self.dist = defaultdict(lambda: math.inf)
        self.last_sent = defaultdict(lambda: math.inf)  # best value last sent per (v,s)
        self.pending = {}                       # aggregation buffers
        self.action_counts = defaultdict(int)   # instrumentation
        env.process(self.recv_loop())

    def add_neighbor(self, v, link, w):
        self.neighbors[v] = (link, w)

    def init_source(self, s):
        if self.id == s:
            self.dist[s] = 0.0
            for v,(link,w) in self.neighbors.items():
                self._maybe_send(v, s, self.dist[s] + w)

    def _enqueue_send(self, link, payload, size_bytes):
        # Help to enqueue a payload on a link
        self.ctrl.inflight += 1
        link.send(size_bytes, payload)

    def _maybe_send(self, v, s, d):
        link,_ = self.neighbors[v]

        # Baseline: always send full message
        if self.policy is None:
            msg = {"kind":"RELAX","src":self,"dst":self.ctrl.nodes[v],"s":s,"d":d}
            self.last_sent[(v,s)] = d
            self._enqueue_send(link, msg, self.ctrl.size_model("RELAX"))
            return

        # Behavior for when it is aware of a policy
        old = self.last_sent[(v,s)]
        delta = abs(d - old)
        queue_depth = len(link.q.items)

        action, param, size_bytes = self.policy.decide(
        delta=delta, queue_depth=queue_depth, util=None
        )

        # TEMP sanity: don’t allow silent coercion to ALLOW
        assert action in ("ALLOW","COMPRESS","AGGREGATE","SKIP"), f"bad action {action}"

        
        self.action_counts[action] += 1
        if action == "SKIP":
            return

        # AGGREGATE Action -> buffer value and schedule timer to send later
        if action == "AGGREGATE":
            key = (v,s)
            self.pending[key] = d
            if ("timer", key) not in self.pending:
                self.pending[("timer", key)] = True
                self.env.process(self._aggregate_after(param, v, s, link))
            return

        # ALLOW/COMPRESS -> send immediately
        send_size = size_bytes if size_bytes is not None else self.ctrl.size_model("RELAX")
        msg = {"kind":"RELAX","src":self,"dst":self.ctrl.nodes[v],"s":s,"d":d}
        self.last_sent[(v,s)] = d
        self._enqueue_send(link, msg, send_size)

    def _aggregate_after(self, delay_ms, v, s, link):
        # For sending latest pending value after an aggregation delay
        yield self.env.timeout(delay_ms / 1000.0)
        key = (v,s)
        if key in self.pending:
            d = self.pending.pop(key)
            _ = self.pending.pop(("timer", key), None)
            send_size = self.ctrl.size_model("RELAX")
            msg = {"kind":"RELAX","src":self,"dst":self.ctrl.nodes[v],"s":s,"d":d}
            self.last_sent[(v,s)] = d
            self._enqueue_send(link, msg, send_size)
        else:
            _ = self.pending.pop(("timer", key), None)

    def recv_loop(self):
        # Loop for receiving for this node
        while True:
            msg = (yield self.inbox.get())
            self.ctrl.inflight -= 1
            if msg["kind"] == "RELAX":
                s, d = msg["s"], msg["d"]
                # A standard shortest-path relaxation
                if d < self.dist[s]:
                    self.dist[s] = d
                    for v,(link,w) in self.neighbors.items():
                        self._maybe_send(v, s, self.dist[s] + w)



#-------------------------------------------------------------------------------------------------------------------------------------------------#


class LLMPolicy:
    """
    Real-LLM policy stub:
    - Bins features (delta, queue_depth) to reduce distinct prompts.
    - Calls `call_model_fn(prompt: str) -> str` that returns a JSON decision.
    - Returns (action, param, size_bytes) like other policies.
    """
    def __init__(self,
                 call_model_fn,
                 bins_delta=(0.02, 0.1, 0.5, 1.0),
                 bins_q=(0, 1, 2, 4, 8),
                 agg_ms_choices=(5, 10, 20),
                 compress_size=48,
                 full_size=96,
                 use_cache=True):
        self.call_model_fn = call_model_fn
        self.bins_delta = tuple(sorted(bins_delta))
        self.bins_q = tuple(sorted(bins_q))
        self.agg_ms_choices = agg_ms_choices
        self.compress_size = compress_size
        self.full_size = full_size
        self.use_cache = use_cache
        self._cache = {}  # (delta_bin, q_bin) -> (action, param, size_bytes)
        
    # ---- helpers ----
    def _bin(self, x, edges):
        # place x into a discrete bin index: 0..len(edges)
        for i, e in enumerate(edges):
            if x < e:
                return i
        return len(edges)

    # Inside class LLMPolicy ...
    def decide(self, delta, queue_depth, util=None):
        """
        Return a tuple: (action, param, size_bytes)
          action ∈ {"ALLOW","COMPRESS","AGGREGATE","SKIP"}
          param  = ms for AGGREGATE, else None
          size_bytes = 96 for ALLOW, 48 for COMPRESS, else None
        """
    
        # --- bin the inputs (assumes self._bin exists) ---
        dbin = self._bin(delta, self.bins_delta)
        qbin = self._bin(queue_depth, self.bins_q)
    
        # --- build concise, bin-based prompt (bin IDs only) ---
        prompt = (
            f"delta_bin={dbin}; queue_bin={qbin}. "
            "You are a network policy agent controlling message sends in a distributed system. "
            "Choose one of these actions:\n"
            f'  - ALLOW: send full {self.full_size} bytes now\n'
            f'  - COMPRESS: send {self.compress_size} bytes now\n'
            f'  - AGGREGATE: wait {list(self.agg_ms_choices)} ms to combine messages\n'
            f'  - SKIP: send nothing now\n'
            "Tiny deltas or high queue bins should prefer COMPRESS / AGGREGATE / SKIP. "
            "Large deltas or empty queues should prefer ALLOW. "
            'Return ONLY JSON: {"action":"...", "param":..., "size_bytes":...}'
        )
    
        # ---- call the adapter (already does JSON extraction/validation) ----
        try:
            raw = self.call_model_fn(prompt)  # expected to return a dict with keys shown above
        except Exception as e:
            # Safe fallback if provider hiccups
            raw = {"action": "COMPRESS", "param": None, "size_bytes": self.compress_size}
    
        # --- unpack & return (updated to have some guardrails on the prompt outputs) ---
        
        

        action = raw.get("action")
        param = raw.get("param")
        size_bytes = raw.get("size_bytes")
        
        # ===== Deterministic guardrails ("policy clamps") =====
        # Use bins already computed above: dbin, qbin
        # dbin range 0..len(bins_delta), qbin 0..len(bins_q)
        
        # 1) Huge change? —send full.
        if dbin >= 3:
            action, param, size_bytes = "ALLOW", None, self.full_size
        
        # 2) Very tiny change and queues light? Skip.
        elif dbin == 0 and qbin <= 1:
            action, param, size_bytes = "SKIP", None, None
        
        # 3) Small change, queues light? Aggregate briefly; else compress.
        elif dbin == 1:
            if qbin <= 1:
                action, param, size_bytes = "AGGREGATE", min(self.agg_ms_choices), None
            else:
                action, param, size_bytes = "COMPRESS", None, self.compress_size
        
        # 4) Medium change? Trust the LLM *if* it picked one of the three; otherwise compress.
        elif dbin == 2:
            if action not in ("ALLOW", "COMPRESS", "AGGREGATE"):
                action, param, size_bytes = "COMPRESS", None, self.compress_size
            # If AGGREGATE, normalize bad/missing param
            if action == "AGGREGATE" and param not in self.agg_ms_choices:
                action, param, size_bytes = "AGGREGATE", min(self.agg_ms_choices), None
        
        # 5) Queue very high? Avoid delay; prefer compress (or skip if delta tiny).
        if qbin >= 3:
            if dbin <= 1:
                action, param, size_bytes = "SKIP", None, None
            else:
                action, param, size_bytes = "COMPRESS", None, self.compress_size
        
        # Final sanity for sizes/params
        if action == "ALLOW":
            param, size_bytes = None, self.full_size
        elif action == "COMPRESS":
            param, size_bytes = None, self.compress_size
        elif action == "AGGREGATE":
            size_bytes = None
        elif action == "SKIP":
            param = size_bytes = None
        
        return action, param, size_bytes



    def _parse_and_validate(self, raw):
        # default safe fallback: ALLOW full size
        fallback = ("ALLOW", None, self.full_size)

        try:
            obj = json.loads(raw.strip())
        except Exception:
            return fallback

        action = obj.get("action")
        param = obj.get("param", None)
        size  = obj.get("size_bytes", None)

        if action not in ("ALLOW", "COMPRESS", "AGGREGATE", "SKIP"):
            return fallback

        # enforce schema
        if action == "ALLOW":
            return ("ALLOW", None, self.full_size)
        if action == "COMPRESS":
            return ("COMPRESS", None, self.compress_size)
        if action == "AGGREGATE":
            if param in self.agg_ms_choices:
                return ("AGGREGATE", param, None)
            # snap to nearest allowed choice if model gave a number
            if isinstance(param, (int, float)):
                closest = min(self.agg_ms_choices, key=lambda x: abs(x - param))
                return ("AGGREGATE", closest, None)
            return fallback
        if action == "SKIP":
            return ("SKIP", None, None)

        return fallback

# -------------------- Controller (orchestrator) --------------------
class Controller:
    def __init__(self, env, graph, base_bps=3e5, base_prop_ms=2.0, policy=None):
        self.env = env
        self.links = {}
        self.inflight = 0
        self.policy = policy  # None => baseline
        self.nodes = {u: Node(env, u, self, self.policy) for u in graph["nodes"]}

        # Heterogeneous links (bandwidth/latency vary per direction)
        for (u, v, w) in graph["edges"]:
            bps_uv = random.uniform(0.5, 1.0) * base_bps
            bps_vu = random.uniform(0.5, 1.0) * base_bps
            prop_uv = random.uniform(0.2, 2.0) * base_prop_ms
            prop_vu = random.uniform(0.2, 2.0) * base_prop_ms

            l_uv = Link(self.env, u, v, bps_uv, prop_uv, capacity=64)
            l_vu = Link(self.env, v, u, bps_vu, prop_vu, capacity=64)
            self.links[(u, v)] = l_uv
            self.links[(v, u)] = l_vu
            self.nodes[u].add_neighbor(v, l_uv, w)
            self.nodes[v].add_neighbor(u, l_vu, w)

    def size_model(self, kind):
        """
        Return a random message size in bytes, depending on kind.

        Uses log-normal distributions so most messages are small but
        occasionally you get large ones — closer to real traffic.
        """
        if kind == "RELAX":
            # many small messages, rare bigger ones
            base = random.lognormvariate(mu=4.0, sigma=0.5)   # ~exp(4) ~ 55 bytes typical
        elif kind == "AGGREGATE":
            # aggregations tend to be bigger
            base = random.lognormvariate(mu=5.0, sigma=0.6)   # ~exp(5) ~ 148 bytes typical
        else:
            base = random.lognormvariate(mu=4.5, sigma=0.5)

        # clamp and convert to int
        size = int(base)
        return max(32, min(size, 4096))

# -------------------- Graph + exact Dijkstra (for MAE) --------------------
def make_grid(n_side=25, w=1.0, jitter=0.1):
    nodes = [(i, j) for i in range(n_side) for j in range(n_side)]
    idx = {nodes[k]: k for k in range(len(nodes))}
    edges = []
    for i in range(n_side):
        for j in range(n_side):
            u = idx[(i, j)]
            if i + 1 < n_side:
                v = idx[(i + 1, j)]
                wij = random.uniform(w - jitter, w + jitter)
                edges.append((u, v, wij))
            if j + 1 < n_side:
                v = idx[(i, j + 1)]
                wij = random.uniform(w - jitter, w + jitter)
                edges.append((u, v, wij))
    return {"nodes": list(range(len(nodes))), "edges": edges}

def dijkstra_cpu(graph, source):
    adj = {u: [] for u in graph["nodes"]}
    for u, v, w in graph["edges"]:
        adj[u].append((v, w))
        adj[v].append((u, w))
    dist = {u: math.inf for u in graph["nodes"]}
    dist[source] = 0.0
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def random_traffic_source(env, ctrl, rate_per_sec=20):
    """
    SimPy process: generates random messages between random nodes.
    """
    # assumes ctrl.nodes is a dict of node_id -> Node
    node_ids = list(ctrl.nodes.keys())

    while True:
        # exponential inter-arrival times 
        interarrival = random.expovariate(rate_per_sec)
        yield env.timeout(interarrival)

        # pick random src/dst
        src = random.choice(node_ids)
        dst = random.choice(node_ids)
        if dst == src:
            continue

        kind = "APP"
        size_bytes = ctrl.size_model(kind)
        payload = make_random_payload(kind)

        payload.update({"src": src, "dst": dst})

        # may not be needed from API
        node = ctrl.nodes[src]

        # pick some neighbor link
        node.send_to(dst, size_bytes, payload)


# -------------------- Optional: dynamic edge jitter during run --------------------
def jitter_edges_periodically(ctrl, graph, period_ms=120, scale=0.02):
    def loop():
        while True:
            for _ in range(10):  # tweak how many edges per tick
                (u, v, w) = random.choice(graph["edges"])
                new_w = max(0.05, w * random.uniform(1 - scale, 1 + scale))
                # update both directions' stored weights on nodes
                if v in ctrl.nodes[u].neighbors:
                    link, _old = ctrl.nodes[u].neighbors[v]
                    ctrl.nodes[u].neighbors[v] = (link, new_w)
                if u in ctrl.nodes[v].neighbors:
                    link, _old = ctrl.nodes[v].neighbors[u]
                    ctrl.nodes[v].neighbors[u] = (link, new_w)
            yield ctrl.env.timeout(period_ms / 1000.0)
    ctrl.env.process(loop())

# -------------------- Run once (baseline or policy) --------------------


def make_random_payload(kind):
    """
    Create a random payload dict. You can adjust the fields
    to match what the rest of your simulator expects.
    """
    msg_id = os.urandom(8).hex()
    user_id = random.randint(1, 1_000_000)
    content_len = random.randint(10, 200)
    # you don't actually need to store the content bytes if you're only
    # simulating sizes; using length + checksum still gives diversity
    random_bytes = os.urandom(content_len)
    checksum = hashlib.sha256(random_bytes).hexdigest()
    
    return {
        "id": msg_id,
        "user_id": user_id,
        "kind": kind,
        "content_len": content_len,
        "checksum": checksum,
    }
        
def run_once(seed=None, policy_kind="rule", random_traffic_rate=None):
    """
    Run one simulation.

    seed:
        - None  -> don't reseed; results vary run-to-run
        - int   -> deterministic run for reproducible experiments

    policy_kind: "none", "rule", "mockllm", or "llm"

    random_traffic_rate:
        - None or 0 -> no extra synthetic random traffic
        - >0        -> messages generated by random_traffic_source at this rate (msg/sec)
    """
    # Optional reproducible randomness
    if seed is not None:
        random.seed(seed)

    # Create environment and graph
    env = simpy.Environment()
    G = make_grid(n_side=25, w=1.0, jitter=0.1)

    # Pick the policy based on policy_kind
    if policy_kind == "none":
        policy = None

    elif policy_kind == "rule":
        policy = RulePolicy(
            eps_small=0.05,
            eps_med=0.20,
            q_hi=1,
            agg_ms=20,
        )

    elif policy_kind == "mockllm":
        policy = LLMPolicy(
            bins_delta=(0.01, 0.05, 0.20, 0.80),
            q_hi=1,
            agg_ms_choices=(10, 15, 20),
        )

    elif policy_kind == "llm":
        policy = LLMPolicy(
            call_model_fn=make_call_model_fn(
                llm_call=ollama_json,       # This is where the LLM is chosen
                full_size=96,
                compress_size=48,
                agg_ms_choices=(5, 10, 20),
                rps=5.0,
                burst=10,
            ),
            bins_delta=(0.01, 0.05, 0.2, 0.8),
            bins_q=(0, 1, 2, 4, 8),
            agg_ms_choices=(5, 10, 20),
        )

    else:
        raise ValueError(f"unknown policy_kind: {policy_kind}")

    # Build controller with the chosen policy
    ctrl = Controller(env, G, base_bps=8e5, base_prop_ms=1.0, policy=policy)

    # Optional background random traffic (if you've implemented random_traffic_source)
    if random_traffic_rate:
        env.process(random_traffic_source(env, ctrl, rate_per_sec=random_traffic_rate))

    # Multiple staggered sources -> overlapping waves
    N = len(G["nodes"])
    sources = [0, 150, 300, 450, 600]

    def start_source_later(s, delay_ms):
        yield env.timeout(delay_ms / 1000.0)
        ctrl.nodes[s].init_source(s)

    for idx, s in enumerate(sources):
        env.process(start_source_later(s, delay_ms=idx * 25))  # 0, 25, 50, ...

    # Stop when quiet or time cap reached
    def stopper(timeout_s=15.0, quiet_ms=80):
        quiet = 0.0
        while env.now < timeout_s:
            yield env.timeout(0.005)
            if ctrl.inflight == 0:
                quiet += 0.005
                if quiet >= (quiet_ms / 1000.0):
                    break
            else:
                quiet = 0.0

    env.process(stopper())
    env.run()

    # Metrics
    total_bytes = sum(l.bytes_sent for l in ctrl.links.values())
    total_msgs  = sum(l.messages_sent for l in ctrl.links.values())
    max_q = max((l.max_q_depth for l in ctrl.links.values()), default=0)

    # Policy action summary (if enabled)
    action_totals = {}
    if ctrl.policy is not None:
        agg = defaultdict(int)
        for node in ctrl.nodes.values():
            for k, v in getattr(node, "action_counts", {}).items():
                agg[k] += v
        action_totals = dict(agg)
        print("Action summary:", action_totals)
        print("Max queue depth observed:", max_q)

    # Accuracy vs exact (per source), report average MAE
    maes = []
    for s in sources:
        exact = dijkstra_cpu(G, s)
        approx = {u: ctrl.nodes[u].dist.get(s, math.inf) for u in G["nodes"]}
        mae = sum(abs(approx[u] - exact[u]) for u in G["nodes"]) / len(G["nodes"])
        maes.append(mae)
    avg_mae = sum(maes) / len(maes)

    return {
        "bytes": total_bytes,
        "msgs": total_msgs,
        "time": env.now,
        "avg_mae": avg_mae,
        "max_q": max_q,
        "actions": action_totals,
    }

#ADDED STUFF TO CHECK


####ADDED MORE STUFF TO CHECK
# -------------------- Main: compare baseline vs policy --------------------
if __name__ == "__main__":
    baseline = run_once(policy_kind="none")
    print("Baseline (no policy):", {k:v for k,v in baseline.items() if k!='actions'})

    import time
    t0=time.perf_counter()
    r = ollama.chat(model="phi3:mini", messages=[{"role":"user","content":'return only {"ok":true}'}],
                    options={"temperature":0.0, "num_predict":16})
    print("latency:", round(time.perf_counter()-t0,2), "s")
    print("reply:", r["message"]["content"])
    print(ollama.list())
    
    ruled = run_once(policy_kind="rule")
    print("RulePolicy:", {k:v for k,v in ruled.items() if k!='actions'})

    mocked = run_once(policy_kind="llm")#testing 
    print("LLMPolicy:", {k:v for k,v in mocked.items() if k!='actions'})



NUM_RUNS = 30
policies = ["none", "rule", "llm"]

results_by_policy = {p: [] for p in policies}

PRINT_FIRST = 5   # how many runs to print at the beginning
PRINT_LAST  = 1   # how many runs to print at the end

for p in policies:
    print(f"\n=== Running policy_kind='{p}' for {NUM_RUNS} runs ===")
    
    for i in range(NUM_RUNS):
        res = run_once(seed=i, policy_kind=p)
        results_by_policy[p].append(res)

        # Conditional printing:
        if i < PRINT_FIRST:
            # print the first few
            print(
                f"  run {i+1}/{NUM_RUNS}: "
                f"bytes={res['bytes']}, msgs={res['msgs']}, "
                f"avg_mae={res['avg_mae']:.4f}, max_q={res['max_q']}"
            )
        
        # Print the last runs
        elif i >= NUM_RUNS - PRINT_LAST:
            print(
                f"  run {i+1}/{NUM_RUNS} (last): "
                f"bytes={res['bytes']}, msgs={res['msgs']}, "
                f"avg_mae={res['avg_mae']:.4f}, max_q={res['max_q']}"
            )

    print(f"=== Completed {NUM_RUNS} runs for policy '{p}' ===")