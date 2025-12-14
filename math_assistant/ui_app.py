


import os
import re
import threading
from datetime import datetime

from openai import BadRequestError
import streamlit as st
from dotenv import load_dotenv

from math_agent import build_math_assistant  # adjust module name if needed
from trace_utils import collect_trace, write_trace_json

load_dotenv()

# ---------- LaTeX normalization ----------

def normalize_latex_for_streamlit(text: str) -> str:
    """
    Heuristically convert ChatGPT-style LaTeX into something Streamlit's
    Markdown+MathJax will render.

    - Convert '\[ ... \]' to '$$ ... $$'
    - Convert standalone lines of the form '[ ... ]' (with lots of backslashes)
      to '$$ ... $$'

    This is intentionally conservative to avoid breaking normal text.
    """

    # 1) Replace \[ ... \] with $$ ... $$
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)

    # 2) Line-by-line: transform [ ... ] lines that *look* like display math
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        # strong heuristic: line starts with '[' and ends with ']' and contains backslash
        if stripped.startswith("[") and stripped.endswith("]") and "\\" in stripped:
            inner = stripped[1:-1].strip()
            new_lines.append(f"$$\n{inner}\n$$")
        else:
            new_lines.append(line)
    text = "\n".join(new_lines)

    return text


# ---------- Utilities for agents & messages ----------

def init_agents(api_key: str, model: str, max_auto_steps: int):
    """
    Initialize or reinitialize the AutoGen agents and store them in session_state.
    """
    st.session_state["user_proxy"], st.session_state["manager"] = build_math_assistant(
        model=model,
        api_key=api_key,
        human_input_mode="NEVER",            # fully automatic inside a run
        max_consecutive_auto_reply=max_auto_steps,
    )


def get_conversation_messages():
    """
    Extract messages from the GroupChat manager in a robust way.
    Returns a list of dicts with keys: name, role, content.
    """
    manager = st.session_state.get("manager")
    if manager is None:
        return []

    raw_messages = getattr(manager.groupchat, "messages", [])
    messages = []

    for m in raw_messages:
        # AutoGen typically uses dict-like messages
        if isinstance(m, dict):
            name = m.get("name") or m.get("role") or "unknown"
            role = m.get("role") or "assistant"
            content = m.get("content", "")
        else:
            # Fallback to string representation
            name = "unknown"
            role = "assistant"
            content = str(m)
        messages.append({"name": name, "role": role, "content": content})

    return messages


def extract_steps(messages):
    """
    Extract lines that look like 'STEP k: ...' from Researcher messages.
    Returns a list of (step_label, index_in_conversation).
    """
    step_pattern = re.compile(r"(STEP\s+\d+\s*:\s*[^\n]*)", re.IGNORECASE)
    steps = []
    for i, msg in enumerate(messages):
        if msg["name"].lower().startswith("researcher"):
            for match in step_pattern.findall(msg["content"]):
                steps.append((match.strip(), i))
    return steps


def run_conversation_in_background(user_proxy, manager, prompt: str):
    """
    Background worker that runs the full AutoGen conversation,
    then writes a JSONL trace to disk using trace_utils.
    """
    try:
        user_proxy.initiate_chat(manager, message=prompt)
    except BadRequestError as e:
        print(f"[Background] LLM request failed: {e}", flush=True)
    except Exception as e:
        import traceback
        print(f"[Background] Unexpected error during conversation: {e}", flush=True)
        traceback.print_exc()

    # After the conversation finishes, collect and save the trace.
    try:
        trace = collect_trace(manager, user_proxy)
        if trace:
            os.makedirs("traces", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join("traces", f"trace-{ts}.jsonl")
            write_trace_json(trace, path)
            print(f"[Background] Trace written to {path}", flush=True)
        else:
            print("[Background] No messages to trace.", flush=True)
    except Exception as e:
        print(f"[Background] Failed to write trace: {e}", flush=True)


# ---------- Streamlit UI ----------

st.set_page_config(
    page_title="Mathematics Research Assistant",
    page_icon="➗",
    layout="wide",
)

st.title("Mathematics Research Assistant")
st.caption(
    "Incremental, stepwise math reasoning with Researcher + Verifier (SymPy-backed)."
)

# Sidebar: configuration
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Will fall back to the OPENAI_API_KEY environment variable if left blank.",
    )

    model = st.text_input(
        "Model name",
        value="gpt-5.1",
        help="Must match the model your AutoGen config is set up to use.",
    )

    max_auto_steps = st.number_input(
        "Max consecutive auto steps",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="Upper bound on automatic agent turns per run.",
    )

    if st.button("Reset session", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

    st.markdown("---")
    st.markdown(
        "Tip: structure your problem so the Researcher works in `STEP k:` units. "
        "The UI will extract and show those steps explicitly."
    )

# Main layout: input + conversation/steps
col_input, col_output = st.columns([1, 2])

with col_input:
    st.subheader("Problem / Instructions")

    prompt = st.text_area(
        "Write your problem here (supports LaTeX, multiline):",
        height=220,
        placeholder=(
            "E.g.\n"
            "Solve the following research problem incrementally. Start with Task 1 only.\n\n"
            "Task 1: Determine the maximal open interval of parameters $\\alpha$ for which\n"
            "$$ I(\\alpha) = \\int_0^{\\infty} \\frac{\\sin x}{x^{\\alpha}(1+x)} \\, dx $$\n"
            "converges. Work in small STEP k units."
        ),
    )

    show_preview = st.checkbox("Show LaTeX preview", value=True)

    if show_preview and prompt.strip():
        st.markdown("**Preview:**")
        preview_text = normalize_latex_for_streamlit(prompt)
        st.markdown(preview_text)

    run_button = st.button("Run assistant", type="primary")

with col_output:
    st.subheader("Conversation")

    # Determine whether a background run is in progress
    run_thread = st.session_state.get("run_thread")
    run_in_progress = run_thread is not None and run_thread.is_alive()

    # Handle Run button: start background conversation if not already running
    if run_button:
        eff_api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not eff_api_key:
            st.error("Please provide an OpenAI API key in the sidebar or via OPENAI_API_KEY.")
        elif not prompt.strip():
            st.error("Please enter a problem or instructions before running.")
        elif run_in_progress:
            st.warning(
                "A conversation is already in progress. "
                "Use the refresh button below to see new messages."
            )
        else:
            try:
                if ("user_proxy" not in st.session_state) or ("manager" not in st.session_state):
                    init_agents(eff_api_key, model, max_auto_steps)
            except Exception as e:
                st.error(f"Failed to initialize agents: {e}")
            else:
                user_proxy = st.session_state["user_proxy"]
                manager = st.session_state["manager"]

                t = threading.Thread(
                    target=run_conversation_in_background,
                    args=(user_proxy, manager, prompt),
                    daemon=True,
                )
                st.session_state["run_thread"] = t
                t.start()
                run_in_progress = True
                st.info(
                    "Assistant run started in the background. "
                    "Use the refresh button to see progress; "
                    "a JSON trace will be written to the `traces/` folder when it finishes."
                )

    # Status line
    if run_in_progress:
        st.info(
            "Conversation in progress… new messages will appear as the agents work. "
            "Click **Refresh conversation** below to update."
        )
    else:
        st.caption(
            "Idle. Click **Run assistant** to start a new conversation. "
            "Finished runs automatically save a JSONL trace under `traces/`."
        )

    # Show conversation
    messages = get_conversation_messages()

    if not messages:
        st.info("No conversation yet. Enter a problem on the left and click **Run assistant**.")
    else:
        for msg in messages:
            role = "user" if msg["name"] == "math_user" else "assistant"
            label = msg["name"]

            with st.chat_message(role):
                st.markdown(f"**{label}**")
                rendered = normalize_latex_for_streamlit(msg["content"])
                st.markdown(rendered)

    if st.button("Refresh conversation"):
        st.rerun()

# Steps timeline below
st.markdown("---")
st.subheader("Steps timeline (parsed from Researcher messages)")

messages = get_conversation_messages()
steps = extract_steps(messages)

if not steps:
    st.info(
        "No `STEP k:` lines detected yet. Ensure your Researcher agent is configured to "
        "label each atomic claim as `STEP k:` in its messages."
    )
else:
    for step_text, msg_index in steps:
        st.markdown(f"- {step_text}  _(from message #{msg_index + 1})_")
