"""
Math assistant powered by Microsoft AutoGen.
Improved orchestration with explicit PASS/FAIL/INCONCLUSIVE verification flow.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Optional
from dotenv import load_dotenv
import autogen
from autogen import AssistantAgent, UserProxyAgent
from trace_utils import collect_trace, render_trace, write_trace_json
from autogen.agentchat import register_function as ag_register_function  # noqa: F401

import sympy as sp
import mpmath as mp
import multiprocessing as mp_proc
import traceback

# Time limit for verification code in the separate process
TIME_LIMIT_SECONDS = 10


def _build_config_list(model: str, api_key: Optional[str]) -> List[Dict[str, str]]:
    """Create the config_list expected by AutoGen from the provided model/key."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Set OPENAI_API_KEY in your environment or pass api_key to build_math_assistant()."
        )
    return [{"model": model, "api_key": key}]

def _verification_worker(code: str, out_queue: "mp_proc.Queue") -> None:
    """Run the user-provided verification code in an isolated process."""
    env = {
        "sympy": sp,
        "sp": sp,
        "mpmath": mp,
        "mp": mp,
    }
    try:
        # Use the same dict for globals and locals
        exec(code, env, env)

        if "verification_passed" in env:
            passed = bool(env["verification_passed"])
            msg = env.get("verification_message", "")
            if passed:
                out_queue.put(f"Verification successful. {msg}")
            else:
                out_queue.put(f"Verification failed. {msg}")
        else:
            verification_result = env.get(
                "verification_output",
                "Code executed, but no explicit verification flag was provided.",
            )
            out_queue.put(
                f"Verification successful (no explicit flag). "
                f"Result: {verification_result}"
            )
    except AssertionError as e:
        out_queue.put(f"Verification failed. AssertionError: {e}")
    except Exception as e:
        tb = traceback.format_exc()
        out_queue.put(
            f"Verification failed. Error during execution: {e}\nTraceback:\n{tb}"
        )



def verify_solution(proposed_solution_code: str) -> str:
    """
    Execute proposed verification code in an isolated process with a time limit.

    The user code must set either:
      - verification_passed = True/False and optionally verification_message, OR
      - verification_output for a generic result.

    This function returns a structured, parseable string:

        VERDICT:PASS
        MESSAGE:<human-readable message>

        VERDICT:FAIL
        MESSAGE:<human-readable message>

        VERDICT:INCONCLUSIVE
        MESSAGE:<human-readable message>

    Any exception or timeout is treated as INCONCLUSIVE (tooling / environment issue),
    not as a mathematical FAIL.
    """
    q: mp_proc.Queue = mp_proc.Queue()
    p = mp_proc.Process(
        target=_verification_worker,
        args=(proposed_solution_code, q),
    )

    p.start()
    p.join(TIME_LIMIT_SECONDS)

    if p.is_alive():
        # Time limit exceeded
        p.terminate()
        p.join()
        return (
            "VERDICT:INCONCLUSIVE\n"
            f"MESSAGE:Verification failed. Error during execution: timeout after {TIME_LIMIT_SECONDS} seconds."
        )

    try:
        raw = q.get_nowait()
    except Exception:
        return (
            "VERDICT:INCONCLUSIVE\n"
            "MESSAGE:Verification failed. No result returned from verification process."
        )

    msg = raw.strip()

    # Map worker messages to structured verdicts
    if msg.startswith("Verification successful."):
        return f"VERDICT:PASS\nMESSAGE:{msg}"

    if msg.startswith("Verification successful (no explicit flag)."):
        return f"VERDICT:PASS\nMESSAGE:{msg}"

    if msg.startswith("Verification failed. AssertionError"):
        # Explicit assertion means the mathematical check failed
        return f"VERDICT:FAIL\nMESSAGE:{msg}"

    if msg.startswith("Verification failed. Error during execution"):
        # SymPy / numeric / environment issues → inconclusive
        return f"VERDICT:INCONCLUSIVE\nMESSAGE:{msg}"

    if msg.startswith("Verification failed."):
        # Covers verification_passed = False with custom message
        return f"VERDICT:FAIL\nMESSAGE:{msg}"

    # Fallback if message is unexpected
    return f"VERDICT:INCONCLUSIVE\nMESSAGE:{msg}"


def build_math_assistant(
    *,
    model: str = "gpt-5.1",
    api_key: Optional[str] = None,
    human_input_mode: str = "ALWAYS",
    max_consecutive_auto_reply: int = 5,
) -> tuple[AssistantAgent, UserProxyAgent]:
    """
    Construct the AutoGen math assistant and its user proxy.
    """
    researcher_llm_config = {
        "config_list": _build_config_list(model, api_key),
        "temperature": 0.2,
    }
    verifier_llm_config = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "verify_solution",
                    "description": (
                        "Executes Python code using SymPy and numeric estimates "
                        "to verify a mathematical solution. Use this whenever the "
                        "Researcher provides a solution or a verifiable step."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "proposed_solution_code": {
                                "type": "string",
                                "description": (
                                    "Python code, using the appropriate libraries, "
                                    "containing an assertion to verify the Researcher's claims."
                                ),
                            }
                        },
                        "required": ["proposed_solution_code"],
                    },
                },
            }
        ],
        "config_list": _build_config_list(model, api_key),
        "temperature": 0.0,
    }

    user_proxy = UserProxyAgent(
        name="math_user",
        human_input_mode=human_input_mode,
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        code_execution_config=False,
    )

    # Researcher Agent: Generates mathematical approaches/solutions
    researcher = autogen.AssistantAgent(
        name="Researcher",
        system_message="""
You are a brilliant mathematician working with a Verifier agent and the tool `verify_solution`.

Your behavior MUST be incremental and stepwise:

1. Treat every problem as a sequence of small subgoals (lemmas / claims).
2. At each of your turns, you must:
   - Choose ONE atomic subgoal only (e.g., "establish the convergence interval",
     or "show that expression A equals expression B").
   - State it clearly as:

       STEP k: <short statement of the claim to be checked>

   - Provide a concise derivation / reasoning for ONLY this STEP.
   - End your message with a short summary line:

       READY FOR VERIFICATION OF STEP k.

3. Do NOT attempt to solve the entire multi-part problem in one message.
   - Do not address multiple numbered tasks at once.
   - Do not bundle many independent lemmas into a single message.
   - After you finish one STEP, wait for the Verifier to respond before moving on.

4. Make your steps verifiable:
   - Ensure each STEP ends with a concrete mathematical statement the Verifier can test
     symbolically or numerically (e.g., an equality, inequality, explicit formula,
     or precisely described convergence interval).

5. Progression rule:
   - You are allowed to move from STEP k to STEP k+1 ONLY if the Verifier's last
     response clearly indicates VERDICT:PASS for STEP k.
   - If the Verifier reports VERDICT:FAIL for STEP k, you must remain on STEP k
     and revise or correct it.
   - If the Verifier reports VERDICT:INCONCLUSIVE for STEP k, you must rephrase or
     adjust STEP k to be more directly verifiable (e.g., simpler claim, alternative
     characterization) and try again.

Your goal is to progress through the problem via a sequence of such STEPs, each validated
by the Verifier before you proceed to the next.
""",
        llm_config=researcher_llm_config,
    )

    # Verifier Agent: Uses SymPy via tool use to check the Researcher's work
    verifier = autogen.AssistantAgent(
        name="Verifier",
        system_message="""
You are a meticulous verification agent.

Your role:
- Use the 'verify_solution' tool to check the Researcher's STEPs,
  one STEP at a time, using SymPy and numeric methods.
- Provide clear, concise feedback.

Protocol:

1. The Researcher will send messages of the form:

     STEP k: <claim>
     <derivation / explanation>
     READY FOR VERIFICATION OF STEP k.

2. For each such message you must:
   - Identify the current STEP number k.
   - Extract the precise mathematical claim to be checked.
   - Call the tool `verify_solution` with Python code that:
       * imports SymPy / mpmath as needed (ALWAYS keep in mind the limitations of the engine and possible failure modes),
       * reconstructs the relevant expressions / functions,
       * performs a symbolic or numeric test (simplify, differentiation+comparison,
         numeric sampling, asymptotics, etc.),
       * sets:

             verification_passed = True or False
             verification_message = "<short explanation>"

         or raises AssertionError on failure, OR
         sets verification_output for an exploratory result.

3. The tool `verify_solution` will return a string with two lines:

       VERDICT:PASS|FAIL|INCONCLUSIVE
       MESSAGE:<short text>

   You MUST read and obey this verdict:

   - If VERDICT:PASS:
       - Reply in natural language, e.g.
            "STEP k verified successfully: <very short explanation>."
       - Do NOT introduce new claims yourself.
       - This signals that the Researcher may proceed to STEP k+1.

   - If VERDICT:FAIL:
       - Reply, e.g.
            "STEP k failed verification: <short reason>. Please revise STEP k."
       - Make clear that the Researcher must NOT advance to the next step,
         but must correct or refine STEP k.

   - If VERDICT:INCONCLUSIVE:
        - Retry writing the code
            If your first attempt at verification results in VERDICT:INCONCLUSIVE with an exception trace (e.g. NameError, TypeError, NotImplementedError), you must:
            Simplify your approach in the next attempt (e.g., switch from symbolic to numeric).
            Avoid repeating the same pattern that triggered the exception.

            If it turns out there are some fundamental limitations of the symbolic engine (SymPy), then revert to a numerical approximation, 
            but allow the derivation to continue

        - If the problem is not within the code, but in the Researcher's statement, ask to write in a more verifiable form.

4. Verify ONLY the current STEP.
   - If the Researcher accidentally bundles multiple claims, pick the main claim
     explicitly labeled as STEP k and verify that one.

        """,
        llm_config=verifier_llm_config,
    )

    groupchat = autogen.GroupChat(
        agents=[user_proxy, researcher, verifier],
        messages=[],
        max_round=50,
        speaker_selection_method="auto",
    )

    manager_llm_config = {
        "config_list": _build_config_list(model, api_key),
        "temperature": 0.0,
    }

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=manager_llm_config,
        system_message="""
You are the GroupChatManager of a math research duo: Researcher and Verifier.

High-level protocol:

1. The Researcher works in numbered STEPs:

       STEP k: <claim>
       ...
       READY FOR VERIFICATION OF STEP k.

   After such a message, the next speaking agent should be the Verifier.

2. The Verifier uses the `verify_solution` tool and receives output like:

       VERDICT:PASS|FAIL|INCONCLUSIVE
       MESSAGE:<text>

   The Verifier then produces a natural-language summary that clearly indicates
   which verdict occurred.

3. Progression rules:

   - If the Verifier indicates VERDICT:PASS for STEP k:
       * Allow the Researcher to proceed to STEP k+1 in the next turn.

   - If the Verifier indicates VERDICT:FAIL for STEP k:
       * The Researcher must stay on STEP k and revise the step.
       * Do NOT allow the Researcher to advance to a new STEP number.

   - If the Verifier indicates VERDICT:INCONCLUSIVE for STEP k:
       * The Researcher must remain on STEP k and restate or adjust the claim
         to be more verifiable (e.g., simpler or alternative formulation).
       * Again, do NOT allow advancing to a new STEP number.

4. Your job is to enforce this gating logic strictly:
   - Do not let the Researcher advance from STEP k to STEP k+1 unless a PASS
     verdict was clearly reported.
   - If the Verifier's message is unclear or does not contain a verdict, you
     may ask the Verifier to restate their result more clearly before the
     Researcher continues.

5. The Manager does not do mathematics or use tools directly. It only manages
   turn-taking and protocol consistency.

6. Do NOT pass the turn to math_user until all the steps are completed and verified.
        """,
    )

    from autogen.agentchat import register_function as ag_register_function

    # After creating `verifier`:
    ag_register_function(
        verify_solution,
        caller=verifier,   # Verifier can call it
        executor=verifier, # and it is executed "as" Verifier
        name="verify_solution",
        description="Executes Python code using SymPy to verify a mathematical claim.",
    )
    return user_proxy, manager


def start_chat(
    *,
    message: str,
    model: str = "gpt-5.1",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    human_input_mode: str = "ALWAYS",
    max_consecutive_auto_reply: int = 5,
    trace: bool = False,
    trace_file: Optional[str] = None,
) -> None:
    """Spin up the assistant and kick off a chat with the initial message."""
    load_dotenv()
    user_proxy, manager = build_math_assistant(
        model=model,
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        human_input_mode=human_input_mode,
        max_consecutive_auto_reply=max_consecutive_auto_reply,
    )
    user_proxy.initiate_chat(manager, message=message)

    if trace or trace_file:
        trace_entries = collect_trace(manager, user_proxy)
        if trace and trace_entries:
            print("\n=== Agent Trace ===")
            print(render_trace(trace_entries))
        if trace_file and trace_entries:
            write_trace_json(trace_entries, trace_file)
            print(f"\nTrace written to {trace_file}")
        if not trace_entries:
            print("\n(No trace captured.)")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AutoGen Sympy math assistant.")
    parser.add_argument(
        "--model",
        default="gpt-5.1",
        help="Model name in your LLM provider (default: %(default)s).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the model provider (defaults to OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the assistant (default: %(default)s).",
    )
    parser.add_argument(
        "--human-input-mode",
        default="ALWAYS",
        choices=["ALWAYS", "TERMINATE", "NEVER"],
        help="How the user proxy gathers input between tool calls.",
    )
    parser.add_argument(
        "--max-auto-steps",
        type=int,
        default=5,
        help="Max consecutive auto replies before asking for input (default: %(default)s).",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print a detailed trace of the conversation (LLM messages + tool calls).",
    )
    parser.add_argument(
        "--trace-file",
        default=None,
        help="Write the trace to a JSONL file for later inspection.",
    )
    parser.add_argument(
        "message",
        nargs="?",
        default="Simplify (x**2 - 1)/(x - 1) and integrate sin(x) from 0 to pi.",
        help="Initial message to send to the assistant.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    start_chat(
        message=args.message,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        human_input_mode=args.human_input_mode,
        max_consecutive_auto_reply=args.max_auto_steps,
        trace=args.trace,
        trace_file=args.trace_file,
    )


if __name__ == "__main__":
    main()
