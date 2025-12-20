import json
import re
from typing import Optional, Any, Dict


def extract_json(text: str) -> Optional[Any]:
    """
    Extract a JSON value from an LLM response.

    Handles:
    - raw JSON
    - ```json fenced blocks```
    - ``` fenced blocks```
    - stray text before/after
    - object or array payloads
    """
    if not text:
        return None

    s = text.strip()

    # 1) If there's a fenced block anywhere, prefer the first one.
    #    This is more robust than only checking startswith("```")
    fence = re.search(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        s,
        flags=re.IGNORECASE
    )
    if fence:
        s = fence.group(1).strip()

    # 2) Try direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 3) Fallback: try object bounds
    obj_start = s.find("{")
    obj_end = s.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidate = s[obj_start:obj_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 4) Fallback: try array bounds
    arr_start = s.find("[")
    arr_end = s.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        candidate = s[arr_start:arr_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


import json, re

def extract_json_object(text: str) -> dict:
    if not text:
        raise ValueError("empty model response")

    # ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return json.loads(m.group(1))

    # first {...} block
    i = text.find("{")
    j = text.rfind("}")
    if i != -1 and j != -1 and j > i:
        return json.loads(text[i:j+1])

    raise ValueError("no JSON object found in model response")

