"""Custom agentic orchestration - the hand-written agent loop, from the sketch.

    user sets goal
          |
          v
       AGENT (an LLM backend -- dummy, Claude or Gemini)
          |   ^
          |   |  "loop: evaluate and ask again"
          v   |
       TOOLS (plain python functions)
          1. list_columns()      what data exists
          2. describe_column()   how one column is distributed
          3. run_lof()           the statistical component: LOF anomaly scores
          |
          v
       FINAL SPEC (a plain dict: which columns, which rows, and why)
          |
          v
       runs/run_<timestamp>_<backend>.json  (spec + full agent trace)

THE FILES
---------
    orchestrator_custom.py       this file: the agent loop, saving runs, the CLI
    LLM/build_system_prompt.py   the system prompt + prompt builder the real backends share
    LLM/llm_dummy.py             backend 1: scripted, offline, deterministic (regression test)
    LLM/llm_claude.py            backend 2: your Claude subscription via the Claude Code CLI
    LLM/llm_gemini.py            backend 3: Google Gemini via API key (free tier), plain HTTP
    tools/registry.py            the tool index: name -> function. Read this to see the tools
    tools/<name>.py              one file per tool; run_lof.py also owns the dataset

TWO ORCHESTRATIONS, ONE SET OF PARTS
------------------------------------
This file is the hand-written loop. A second orchestration built on LangChain
will sit beside it as orchestrator_langchain.py and import the SAME backends from
LLM/ and the SAME tools from tools/registry.py -- so the two can be compared as
experimental conditions rather than as two different systems. Nothing in LLM/ or
tools/ imports an orchestrator, which is what keeps adding one a pure addition.

All three backends implement one identical contract -- llm(messages) -> one of two
dict shapes -- documented in LLM/llm_dummy.py. The loop below neither knows nor
cares which one it is talking to. To add another backend (Ollama, OpenAI, ...),
write a new LLM/llm_<name>.py with one function and wire it into the __main__ block.

Run it:
    python orchestrator_custom.py             offline, scripted
    python orchestrator_custom.py --claude    live, uses your Claude login
"""

import json
import os
import sys
from datetime import datetime

from LLM.llm_claude import CLAUDE_CLI_MODEL, claude_llm
from LLM.llm_dummy import dummy_llm
from LLM.llm_gemini import GEMINI_MODEL, gemini_llm
from tools.registry import TOOLS

# =============================================================================
# THE AGENT LOOP - this is the part that IS the agent. Read it top to bottom:
# ask the model, run the tool it asked for, feed the result back, repeat.
# =============================================================================


def derive_viewpoint(goal, llm=dummy_llm, max_steps=12):
    """Turn one goal into one viewpoint spec, by looping the LLM against the tools.

    Returns (final_spec, trace). The trace is a list with one entry per step --
    thinking, tool, args, and the FULL tool result (the console display truncates
    long results, the trace never does). The trace is the evidence of how the
    agent reached its spec, so it gets saved alongside the spec itself.

    max_steps is the HARD limit; the "at most 10 tool calls" line in
    LLM/build_system_prompt.py is only a request the model usually honours.
    Keep max_steps at least 2 above the prompt's number: the finalising reply
    consumes a step too, and an error-recovery retry costs another.
    """
    messages = [{"role": "user", "content": f"Goal: {goal}"}]
    trace = []

    for step in range(1, max_steps + 1):
        # If the backend dies (network, rate limit, bad JSON), keep the trace:
        # a half-finished run is still evidence of what the agent was doing.
        try:
            reply = llm(messages)
        except Exception as error:
            print(f"\nSTEP {step}  BACKEND FAILED: {error}")
            trace.append({"step": step, "error": str(error)})
            return None, trace

        print(f"\nSTEP {step}  agent thinks: {reply.get('thinking', '(no thinking given)')}")

        if "final_spec" in reply:                       # the agent decided it is done
            trace.append({"step": step,
                          "thinking": reply.get("thinking", ""),
                          "final_spec": reply["final_spec"]})
            return reply["final_spec"], trace

        tool_name, tool_args = reply["tool"], reply.get("args", {})
        print(f"        agent calls : {tool_name}({tool_args})")

        # A real LLM sometimes invents a tool name or passes wrong arguments.
        # Answer with ERROR text instead of crashing -- the model reads it and
        # corrects itself, exactly like the wrong-column-name case.
        if tool_name not in TOOLS:
            result = f"ERROR: unknown tool '{tool_name}'. Available tools: {list(TOOLS)}."
        else:
            try:
                result = TOOLS[tool_name](**tool_args)  # run the actual function
            except TypeError as error:
                result = f"ERROR: bad arguments for {tool_name}: {error}"
        print(f"        tool returns: {result[:400]}"
              + ("  [...truncated for display]" if len(result) > 400 else ""))

        trace.append({"step": step,
                      "thinking": reply.get("thinking", ""),
                      "tool": tool_name,
                      "args": tool_args,
                      "result": result})                # full text, never truncated

        # Feed both halves of the exchange back into the conversation, so the
        # next call sees everything that has happened. THIS is the loop from
        # the sketch: evaluate, then ask again.
        messages.append({"role": "assistant", "content": f"{tool_name}({tool_args})"})
        messages.append({"role": "tool_result", "content": result})

    # Ran out of steps without a final_spec. Return the trace anyway -- a run
    # that failed to finish is DATA, not garbage: how an agent burns its budget
    # without converging is exactly the kind of behaviour worth studying.
    print(f"\nAgent did NOT finalise within {max_steps} steps.")
    return None, trace


def save_run(goal, backend_name, spec, trace):
    """Write one run -- spec plus the full agent trace -- to its own file.

    Files land in runs/, named by timestamp and backend, e.g.
        runs/run_20260808_141507_claude.json

    One file per run (never overwritten) is what makes the variance experiment
    possible later: run the same goal five times, then compare the five files
    to see how differently the agent explored and what it settled on.
    """
    os.makedirs("runs", exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("runs", f"run_{run_id}_{backend_name}.json")

    with open(path, "w") as f:
        json.dump({
            "run_id": run_id,
            "backend": backend_name,
            "goal": goal,
            "status": "completed" if spec else "exhausted",   # did the agent finalise,
            "steps_taken": len(trace),                        # or run out of steps?
            "final_spec": spec,          # None if exhausted; else duplicated from the
            "trace": trace,              # last trace entry so it is easy to grab
        }, f, indent=2)
    return path


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Pick a backend with a flag; everything that is not a flag becomes the goal
    # (quotes optional -- the words are joined back together):
    #     python orchestrator_custom.py
    #     python orchestrator_custom.py --claude "find neighbourhoods that do not fit their region"
    #     python orchestrator_custom.py --gemini find blocks whose housing looks impossible
    # NOTE: VS Code's Run button passes NO arguments -- use a terminal for these.
    BACKENDS = {
        "--claude": ("claude", claude_llm, f"Claude via CLI, model '{CLAUDE_CLI_MODEL}'"),
        "--gemini": ("gemini", gemini_llm, f"Gemini API, model '{GEMINI_MODEL}'"),
    }
    # Split argv into flags (anything starting with "-") and goal words. Doing it
    # by prefix means a typo like "-claude" is caught as a bad flag instead of
    # silently becoming part of the goal text.
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    goal_words = [a for a in sys.argv[1:] if not a.startswith("-")]

    unrecognised = [a for a in flags if a not in BACKENDS]
    if unrecognised:
        sys.exit(f"Unrecognised flag(s): {unrecognised}. "
                 f"Valid flags: {sorted(BACKENDS)} -- or none for the offline dummy.")
    if len(flags) > 1:
        sys.exit(f"Pick ONE backend, not several: {flags}")

    if flags:
        backend_name, llm, banner = BACKENDS[flags[0]]
    else:
        backend_name, llm, banner = "dummy", dummy_llm, "dummy LLM -- scripted, fully offline"

    print("=" * 76)
    print(f"OBSERVER AGENT -- custom orchestration  ({banner})")
    print("=" * 76)

    goal = " ".join(goal_words) or "find census rows that cannot describe a real place"
    print(f"\nUser sets goal: {goal!r}")

    if backend_name == "dummy" and goal_words:
        print("NOTE: the dummy backend replays a fixed script written for the default")
        print("      goal -- it cannot react to yours. Use --claude or --gemini.")

    spec, trace = derive_viewpoint(goal, llm=llm)

    print("\n" + "=" * 76)
    if spec is None:
        print("NO FINAL SPEC -- the agent exhausted its steps. Trace saved for study.")
    else:
        print("FINAL SPEC (the agent's viewpoint, as a plain dict)")
    print("=" * 76)
    print(json.dumps(spec, indent=2))

    run_path = save_run(goal, backend_name, spec, trace)
    print(f"\nRun saved to {run_path}  ({len(trace)} steps, full untruncated trace)")
    print("Replaying the viewpoint later needs no LLM at all:")
    print("    from tools.run_lof import run_lof")
    print(f"    spec = json.load(open({run_path!r}))['final_spec']")
    print("    run_lof(spec['columns'], spec['row_filter'])")


# =============================================================================
# OTHER BACKENDS
# -----------------------------------------------------------------------------
# Any chat model works, because the contract is just "return JSON in one of the
# two shapes". Examples:
#
#   Ollama (free, offline):  POST http://localhost:11434/api/chat with
#       {"model": "qwen2.5:7b", "format": "json", ...} and json.loads the reply.
#
#   Anthropic / OpenAI SDK (paid API key): use their native tool-use API, then
#       translate the response into our two shapes.
#
# In every case: derive_viewpoint(goal, llm=your_function) -- the loop, the
# tools and the spec never change. And keep dummy_llm forever: it is the
# regression test that the loop itself still works, offline and deterministic.
# =============================================================================
