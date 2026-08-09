"""The Gemini backend - Google's free-tier API, driven with one plain HTTP call.

HOW IT CONNECTS
---------------
One POST to the Gemini REST API per turn. No SDK to install: the `requests`
library and an API key are the entire dependency list.

The key is found by checking, in order (first match wins):
    1. environment variable GEMINI_API_KEY
    2. environment variable GOOGLE_API_KEY
    3. environment variable GEMINI_KEY
    4. a .env file in the REPO ROOT containing a line like  GEMINI_KEY=...

Free-tier notes: requests are capped per minute and per day (caps vary by
model). One agent run is ~5-9 calls, comfortably inside the caps; if runs
back-to-back trip the per-minute cap, the single retry below absorbs it.

THE CONTRACT
------------
Same as every backend (see llm_dummy.py for the full statement):

    gemini_llm(messages) -> {"thinking","tool","args"}  or  {"thinking","final_spec"}

Two Gemini-specific perks worth knowing:
  - `responseMimeType: application/json` makes the API GUARANTEE syntactically
    valid JSON, so no fence-stripping regex is needed here.
  - Temperature IS controllable here -- which matters later
    for the derivation-variance experiments.
"""

import json
import os
import re
import time

import requests

from build_system_prompt import build_prompt

#: Which model to call -- PINNED, deliberately. We tried the "gemini-flash-latest"
#: alias first; it silently resolved to gemini-3.6-flash, whose free tier allows
#: only 20 requests per DAY (~3 agent runs), and a day's quota vanished mid-run.
#: Free-tier quotas are PER MODEL, and the -lite models have much larger buckets.
#: To see what your key offers: GET /v1beta/models with your key.
GEMINI_MODEL = "gemini-3.5-flash-lite"

#: The names we accept for the key, in the order we look for them.
KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY")


def load_gemini_key():
    """Return the API key from the environment or the .env file next to this script."""
    for name in KEY_NAMES:
        if os.environ.get(name):
            return os.environ[name].strip()

    # .env stays at the REPO ROOT -- one level up from this agent folder. Anchoring
    # to __file__ (not the working directory) means every orchestration finds the
    # same key file no matter where it was launched from.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(repo_root, ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, _, value = line.partition("=")
                if name.strip() in KEY_NAMES:
                    return value.strip()

    raise RuntimeError(
        "No Gemini key found. Either set the GEMINI_API_KEY environment variable, "
        f"or put a line like GEMINI_KEY=<your key> into {env_path} "
        "(get a free key at https://aistudio.google.com/apikey). "
        "Keep .env in .gitignore -- a key must never be committed."
    )


def gemini_llm(messages):
    """Ask Gemini for the next move. See the module docstring for the contract."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/"
           f"models/{GEMINI_MODEL}:generateContent")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": build_prompt(messages)}]}],
        "generationConfig": {
            "responseMimeType": "application/json",   # API-enforced valid JSON
            # "temperature": 0.0,   # uncomment to make derivation maximally stable
        },
    }

    # The free tier caps requests PER MINUTE, and an agent loop fires faster than
    # that -- so mid-run 429s are normal, not exceptional. Google's error body
    # says how long to wait ("retryDelay": "22s"); honour it, with headroom.
    for attempt in range(1, 5):
        response = requests.post(
            url, json=payload, timeout=120,
            headers={"x-goog-api-key": load_gemini_key()},
        )
        if response.status_code == 429 and attempt < 4:
            hinted = re.search(r'"retryDelay":\s*"(\d+)s"', response.text)
            wait_seconds = int(hinted.group(1)) + 3 if hinted else 35
            print(f"        (free-tier rate limit -- waiting {wait_seconds}s, "
                  f"retry {attempt}/3)")
            time.sleep(wait_seconds)
            continue
        break

    if response.status_code != 200:
        # Show enough of the body to see WHICH quota was exceeded (per-minute
        # caps clear on their own; the per-day cap means stop until tomorrow).
        raise RuntimeError(
            f"Gemini API error {response.status_code}: {response.text[:600]}"
        )

    body = response.json()
    try:
        reply_text = body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        # Usually a safety block or an empty candidate -- show what came back.
        raise RuntimeError(f"Unexpected Gemini response shape: {json.dumps(body)[:300]}")

    return json.loads(reply_text)
