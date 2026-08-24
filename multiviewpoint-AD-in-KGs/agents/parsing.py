"""Getting a structured answer back out of whatever the model actually said.

These two functions are the whole of what stands between an LLM's final message
and the run file. Both agents rely on them, which is why they live apart from
either one.
"""
import json
import re


def first_json_object(text: str):
    """First balanced {...} in the text, parsed. None if there isn't one.

    Models fence their JSON, prefix it with prose, or both. Scanning for a
    balanced object is more forgiving than a regex and cheaper than a retry.
    """
    if not text:
        return None
    for start in (m.start() for m in re.finditer(r"\{", text)):
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def last_text(callback_context) -> str:
    """The final text this agent produced in this invocation.

    Filtered by invocation AND by author: session events carry every agent's
    output, so without both filters a viewpoint would read the root's answer,
    or a previous run's.
    """
    out = []
    for event in callback_context.session.events:
        if event.invocation_id != callback_context.invocation_id:
            continue
        if event.author != callback_context.agent_name:
            continue
        content = getattr(event, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                out.append(part.text)
    return out[-1] if out else ""
