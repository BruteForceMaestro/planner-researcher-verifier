"""
Math assistant powered by Microsoft AutoGen.

This module wires the Sympy-powered helpers in ``sympy_tools.py`` into an
AutoGen assistant agent so it can call the math functions directly during a
chat. Run this module as a script to start an interactive session.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Optional
from dotenv import load_dotenv

from autogen import AssistantAgent, UserProxyAgent
from trace_utils import collect_trace, render_trace, write_trace_json

from sympy_tools import (
    numeric_check_equality,
    sympy_diff,
    sympy_integrate,
    sympy_limit,
    sympy_series,
    sympy_simplify,
    sympy_solve_equation,
)


def _build_config_list(model: str, api_key: Optional[str]) -> List[Dict[str, str]]:
    """Create the config_list expected by AutoGen from the provided model/key."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Set OPENAI_API_KEY in your environment or pass api_key to build_math_assistant()."
        )
    return [{"model": model, "api_key": key}]


def register_sympy_tools(assistant: AssistantAgent) -> None:
    """
    Register the Sympy-backed math tools so the assistant can call them via
    function calling.
    """
    assistant.register_function(
        function_map={
            "sympy_simplify": sympy_simplify,
            "sympy_integrate": sympy_integrate,
            "sympy_diff": sympy_diff,
            "sympy_limit": sympy_limit,
            "sympy_series": sympy_series,
            "sympy_solve_equation": sympy_solve_equation,
            "numeric_check_equality": numeric_check_equality,
        }
    )


def build_math_assistant(
    *,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    human_input_mode: str = "ALWAYS",
    max_consecutive_auto_reply: int = 5,
) -> tuple[AssistantAgent, UserProxyAgent]:
    """
    Construct the AutoGen math assistant and its user proxy.

    Parameters mirror the common AutoGen options so you can tune the LLM and
    interaction mode without changing the calling code.
    """
    llm_config = {
        "config_list": _build_config_list(model, api_key),
        "temperature": temperature,
    }
    assistant = AssistantAgent(
        name="math_assistant",
        llm_config=llm_config,
        system_message=(
            "You are a mathematics assistant. Use the registered tools for "
            "symbolic manipulation, calculus, and equation solving. Keep responses concise."
        ),
    )
    user_proxy = UserProxyAgent(
        name="math_user",
        human_input_mode=human_input_mode,
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        code_execution_config=False,
    )
    register_sympy_tools(assistant)
    return assistant, user_proxy


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
    assistant, user_proxy = build_math_assistant(
        model=model,
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        temperature=temperature,
        human_input_mode=human_input_mode,
        max_consecutive_auto_reply=max_consecutive_auto_reply,
    )
    user_proxy.initiate_chat(assistant, message=message)

    if trace or trace_file:
        trace_entries = collect_trace(assistant, user_proxy)
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
        default="gpt-4o-mini",
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
