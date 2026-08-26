"""Tool: the observer's judgement, one verdict per served candidate.

The only place an observer's opinion becomes part of the run. Verdicts refer
to candidate ids from find_suspects, and an id never served to THIS caller
is refused -- no judging facts you were not shown, no judging the other
observer's stack.

Verdicts remove and explain; they never score, rank, or reorder. The kept
order downstream is the scanners' order -- an LLM output is never a
ranking key.
"""
import json

from tools._observers import OBSERVER_NAMES, principal_of, state_key

VERDICTS = ("anomaly", "ok", "out_of_scope", "unsure")


def submit_verdicts(verdicts: list[dict], tool_context=None) -> str:
    """Judge served candidates against YOUR norms. Batch as you like.

    One entry per candidate: {"id": "c3", "verdict": "...", "why": "..."}.

      anomaly       violates your norms -- which, depending on who you are,
                    can include facts that are literally true
      ok            your norms accept it
      out_of_scope  a candidate your norms are silent on
      unsure        you genuinely cannot tell; say what is missing

    The why is one sentence, in terms of YOUR norms, not anyone else's. You
    are done when every candidate you were served has a verdict.

    Args:
        verdicts: a list of {"id", "verdict", "why"} objects.
    """
    # A reviewer is the same observer returning for the second-opinion phase;
    # its verdicts land in its principal's store like any others.
    agent = principal_of(tool_context.agent_name)
    if agent not in OBSERVER_NAMES:
        return f"ERROR: only an observer judges; '{agent}' is not one."

    served = json.loads(tool_context.state.get(state_key("served", agent)) or "{}")
    if not served:
        return "ERROR: nothing has been served to you yet. find_suspects first."

    if not verdicts:
        return "ERROR: an empty batch judges nothing."

    judged_key = state_key("verdicts", agent)
    judged = json.loads(tool_context.state.get(judged_key) or "{}")

    accepted = []
    for entry in verdicts:
        candidate_id = (entry.get("id") or "").strip()
        verdict = (entry.get("verdict") or "").strip().lower()
        why = (entry.get("why") or "").strip()

        if candidate_id not in served:
            return (f"ERROR: '{candidate_id}' was never served to you. "
                    f"Your candidates are {', '.join(sorted(served))}. "
                    f"Nothing from this batch was recorded.")
        if verdict not in VERDICTS:
            return (f"ERROR: '{verdict}' is not a verdict. "
                    f"One of: {', '.join(VERDICTS)}. "
                    f"Nothing from this batch was recorded.")
        if not why:
            return (f"ERROR: no why for {candidate_id}. One sentence, in "
                    f"terms of your norms. Nothing from this batch was "
                    f"recorded.")
        accepted.append((candidate_id, verdict, why))

    # All-or-nothing per batch, applied only after every entry validated.
    for candidate_id, verdict, why in accepted:
        judged[candidate_id] = {"verdict": verdict, "why": why,
                                "triple": served[candidate_id]["triple"],
                                "text": served[candidate_id].get("text", ""),
                                "scanner": served[candidate_id]["scanner"]}
    tool_context.state[judged_key] = json.dumps(judged)

    remaining = [cid for cid in served if cid not in judged]
    if remaining:
        return (f"Recorded {len(accepted)} verdict(s). "
                f"{len(remaining)} candidate(s) still unjudged: "
                f"{', '.join(sorted(remaining))}.")
    return (f"Recorded {len(accepted)} verdict(s). Every candidate you were "
            f"served is judged -- you are done, stop here.")
