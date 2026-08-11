"""Custom agentic orchestration - the hand-written agent loop, from the sketch.

    user sets goal
          |
          v
       AGENT (an LLM backend -- dummy or Gemini)
          |   ^
          |   |  "loop: evaluate and ask again"
          v   |
       TOOLS (plain python functions)
          1. list_columns()      what data exists
          2. describe_column()   how one column is distributed
          3. run_lof_per_viewpoint()           the statistical component: LOF anomaly scores
          |
          v
       FINAL SPEC (a plain dict: which columns, which rows, and why)
          |
          v
       runs/run_<timestamp>_<backend>.json  (spec + full agent trace)

THIS FOLDER (agent_custom_single/) -- everything only the custom loop uses
--------------------------------------------------------------------------
    orchestrator_custom.py       this file: the agent loop, the CLI, saving runs
    build_system_prompt.py       the system prompt + prompt builder the backends share
    llm_dummy.py                 backend 1: scripted, offline, deterministic (regression test)
    llm_gemini.py                backend 2: Google Gemini via API key (free tier), plain HTTP

SHARED WITH THE OTHER AGENTS (one level up, in the project root)
---------------------------------------------------------------
    tools/registry.py            the tool index: name -> function. Read this to see the tools
    tools/<name>.py              one file per tool, and nothing else
    data/california_housing.py   the frame + the column vocabulary; a leaf, loaded once
    utils/save_run.py            writes runs/*.json -- shared by EVERY agent

Import direction is one way only:  agent -> tools/ -> data/
Nothing in tools/, data/ or utils/ imports an agent, which is what keeps adding
an agent a pure addition rather than a refactor.

ONE FOLDER PER AGENT
--------------------
Each agent gets its own folder and owns whatever only it needs. This loop owns
its LLM backends because only it uses them: the ADK agents next door
(agent_adk_single/, agent_adk_multiple/) let ADK talk to Gemini natively and
never touch these files. What every agent shares -- the tools, the dataset, the
run writer -- stays in the project root, so all agents are compared on identical
parts and only their ORCHESTRATION differs.

Both backends implement one identical contract -- llm(messages) -> one of two
dict shapes -- documented in llm_dummy.py. The loop below neither knows nor
cares which one it is talking to. To add another backend (Ollama, OpenAI, ...),
write a new llm_<name>.py here with one function and wire it into __main__.

(A Claude backend driven through the Claude Code CLI existed and was removed on
9 Aug 2026. It needed no API key, but frameworks have no equivalent for it --
ChatAnthropic requires a paid key -- so keeping it would have left the
orchestrations with different backend sets and made them incomparable.)

Run it from the PROJECT ROOT:
    python agent_custom_single/orchestrator_custom.py "<goal>"   live Gemini -- the DEFAULT
    python agent_custom_single/orchestrator_custom.py --dummy    offline scripted regression test

Gemini is the default because that is what you actually want to run; --dummy is
the free, offline, deterministic check that the LOOP itself still works. If the
dummy run breaks, the loop broke -- not the model, not your key, not the network.
"""

import json
import pathlib
import sys

# The shared tools/, data/ and utils/ packages live one level up, in the project
# root. Running this file puts THIS folder on sys.path, not the root, so add the
# root explicitly -- exactly as agent_adk_single/agent.py does. That is what lets
# every agent call the SAME tool functions rather than keeping copies.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_dummy import dummy_llm  # noqa: E402
from llm_gemini import GEMINI_MODEL, gemini_llm  # noqa: E402
from data.active import NAME as DATASET_NAME  # noqa: E402
from tools.registry import TOOLS  # noqa: E402
from utils.save_run import save_run  # noqa: E402

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
    build_system_prompt.py is only a request the model usually honours.
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


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Gemini is the DEFAULT -- just give a goal. Everything that is not a flag
    # becomes the goal (quotes optional; the words are joined back together):
    #     python agent_custom_single/orchestrator_custom.py "find blocks that do not fit their region"
    #     python agent_custom_single/orchestrator_custom.py find blocks whose housing looks impossible
    #     python agent_custom_single/orchestrator_custom.py --dummy    offline regression test
    #
    # NOTE: running with no arguments now CALLS THE API (against your free-tier
    # quota) on the default goal. Use --dummy for the free, offline, deterministic
    # run -- that is the one to use when checking that the LOOP still works.
    # VS Code's Run button passes no arguments, so it will hit the API too.
    DEFAULT_BACKEND = "--gemini"
    BACKENDS = {
        "--gemini": ("gemini", gemini_llm, f"Gemini API, model '{GEMINI_MODEL}'"),
        "--dummy": ("dummy", dummy_llm, "dummy LLM -- scripted, offline, deterministic"),
    }
    # Split argv into flags (anything starting with "-") and goal words. Doing it
    # by prefix means a typo like "-dummy" is caught as a bad flag instead of
    # silently becoming part of the goal text.
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    goal_words = [a for a in sys.argv[1:] if not a.startswith("-")]

    unrecognised = [a for a in flags if a not in BACKENDS]
    if unrecognised:
        sys.exit(f"Unrecognised flag(s): {unrecognised}. "
                 f"Valid flags: {sorted(BACKENDS)} -- or none for the default "
                 f"({DEFAULT_BACKEND}).")
    if len(flags) > 1:
        sys.exit(f"Pick ONE backend, not several: {flags}")

    backend_name, llm, banner = BACKENDS[flags[0] if flags else DEFAULT_BACKEND]

    print("=" * 76)
    print(f"OBSERVER AGENT -- custom orchestration  ({banner})")
    print("=" * 76)

    goal = " ".join(goal_words) or "find census rows that cannot describe a real place"
    print(f"\nUser sets goal: {goal!r}")

    if backend_name == "dummy" and goal_words:
        print("NOTE: the dummy backend replays a fixed script written for the default")
        print("      goal -- it cannot react to yours. Drop --dummy to use Gemini.")

    spec, trace = derive_viewpoint(goal, llm=llm)

    print("\n" + "=" * 76)
    if spec is None:
        print("NO FINAL SPEC -- the agent exhausted its steps. Trace saved for study.")
    else:
        print("FINAL SPEC (the agent's viewpoint, as a plain dict)")
    print("=" * 76)
    print(json.dumps(spec, indent=2))

    run_path = save_run(goal, backend_name, spec, trace, orchestrator="custom",
                        dataset=DATASET_NAME)
    # `goal` is this loop's user_prompt: here the user types a goal directly,
    # whereas the ADK agent is asked a broad question and derives its own.
    print(f"\nRun saved to {run_path}  ({len(trace)} steps, full untruncated trace)")
    print("Replaying the viewpoint later needs no LLM at all:")
    print("    from tools.run_lof_per_viewpoint import run_lof_per_viewpoint")
    print(f"    spec = json.load(open({run_path!r}))['final_specs'][0]")
    print("    run_lof_per_viewpoint(spec['columns'], spec['row_filter'])")


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
