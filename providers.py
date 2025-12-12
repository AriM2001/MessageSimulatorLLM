
# providers.py
import json, os

#----------------------------------------------------------------------------------------------------------------------------------------------#
def ollama_json(prompt: str) -> str:
    """
    Use a local Ollama model and return a JSON string.
    Requires the Ollama server running locally and a pulled model.
      - Start server: `ollama serve`
      - Pull model:  `ollama pull llama3`   (or any chat-capable model)
    Optional: set OLLAMA_MODEL env var; defaults to 'llama3'.
    """
    import json
    try:
        import ollama
    except Exception as e:
        raise RuntimeError("Python package 'ollama' not installed. Run: pip install ollama") from e

    model = os.environ.get("OLLAMA_MODEL", "llama3")

    system = (
        "You are a network policy agent. "
        "Return ONLY a single minified JSON object with keys: action, param, size_bytes. "
        'Examples: {"action":"ALLOW","param":null,"size_bytes":96}   '
        '{"action":"COMPRESS","param":null,"size_bytes":48}   '
        '{"action":"AGGREGATE","param":10,"size_bytes":null}   '
        '{"action":"SKIP","param":null,"size_bytes":null}'
    )

    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.1,
                # You can nudge shorter outputs if desired:
                "num_predict": 24,
            },
        )
    except Exception as e:
        # Typical causes: server not running or model not pulled
        raise RuntimeError(
            "Could not reach Ollama. Make sure the server is running (`ollama serve`) "
            "and the model is pulled (e.g., `ollama pull llama3`)."
        ) from e

    # Ollama returns a dict: {"message": {"role": "...", "content": "..."} , ...}
    content = resp["message"]["content"]
    # The adapters validator will JSON-parse and verify schema. Just return the raw string.
    return content

# -------- OpenAI (Responses JSON mode) --------
def openai_json(prompt: str) -> str:
    """
    Returns a JSON string. Expects env vars:
      OPENAI_API_KEY, OPENAI_MODEL (e.g., `gpt-4.1-mini`)
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

    rsp = client.responses.create(
        model=model,
        input=[
            {"role":"system","content":"You are a network policy agent. Reply ONLY with a single JSON object."},
            {"role":"user","content":prompt},
        ],
        response_format={"type":"json_object"},
        temperature=0.2,
        max_output_tokens=64,
    )
    # For Responses API, the text is in .output_text
    return rsp.output_text

# -------- Anthropic (Claude JSON) --------
def anthropic_json(prompt: str) -> str:
    """
    Requires ANTHROPIC_API_KEY and optional ANTHROPIC_MODEL (e.g., `claude-3-5-sonnet-latest`).
    """
    import anthropic, os
    client = anthropic.Client(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    msg = client.messages.create(
        model=model,
        max_tokens=64,
        temperature=0.2,
        system="You are a network policy agent. Reply ONLY with a single JSON object.",
        messages=[{"role":"user","content":prompt}],
        extra_headers={"anthropic-beta":"prompt-caching-2024-07-31"},  # harmless if unsupported
        # If JSON mode is available in your SDK version, enable it; otherwise instruct via system.
    )
    return msg.content[0].text  # should be the JSON string

# -------- Local vLLM (OpenAI-compatible HTTP) --------
def vllm_http_json(prompt: str) -> str:
    """
    For a local OpenAI-compatible server at VLLM_BASE_URL with model name in VLLM_MODEL.
    """
    import requests, os
    url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1/chat/completions")
    model = os.environ.get("VLLM_MODEL", "Qwen2.5-7B-Instruct")
    j = {
        "model": model,
        "messages":[
            {"role":"system","content":"You are a network policy agent. Reply ONLY with a single JSON object."},
            {"role":"user","content":prompt},
        ],
        "temperature":0.2,
        "max_tokens":64,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(url, json=j, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]
 