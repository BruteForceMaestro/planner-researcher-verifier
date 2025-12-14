# Math Assistant (AutoGen + Sympy)

An AutoGen-based mathematics assistant that calls the Sympy-backed tools in `sympy_tools.py` for symbolic manipulation, calculus, and equation solving.

## Setup (Poetry)
1. Install Poetry (one-time): https://python-poetry.org/docs/#installation
2. Install dependencies into a local virtualenv:

   ```bash
   poetry install
   ```

   This project is configured to use a repo-local environment at `./.venv`.

3. Provide an LLM API key:

   ```bash
   export OPENAI_API_KEY=...
   ```

   You can also pass `--api-key` when running the CLI.

## Run
- Start an interactive chat with a seed message:  
  `poetry run python -m math_assistant.math_agent "Differentiate sin(x) twice and evaluate the limit as x->0"`
- Adjust model/behavior if needed:  
  `poetry run python -m math_assistant.math_agent --model gpt-4o-mini --temperature 0.2 --max-auto-steps 3 "Series expand e^x around 0 to order 6"`

The assistant will route math requests through the registered Sympy tools via AutoGen function-calling.

## Tracing
- See a structured trace of LLM thinking and tool calls after the run:  
  `poetry run python -m math_assistant.math_agent --trace "Solve x^2 = 4"`
- Persist traces for later analysis (JSONL):  
  `poetry run python -m math_assistant.math_agent --trace --trace-file run.trace.jsonl "Integrate sin(x) from 0 to pi"`

## UI (Streamlit)
- Start the web UI with LaTeX chat + tool traces:  
  `poetry run streamlit run math_assistant/ui_app.py`
- Provide your API key in the sidebar (or via `OPENAI_API_KEY`), choose model/temperature, and chat.  
  Use `$...$`/`$$...$$` for LaTeX; toggle the trace panel to inspect LLM reasoning and SymPy tool calls.

## Lean verifier
`math_assistant.math_agent` includes an optional Lean verification tool that uses the `lean_verifier/` project at the repo root.

- If you have Lean + Lake installed, it will run:
  - `lake env lean temp_verification.lean` with `cwd=lean_verifier/`
- If you don’t need Lean verification, you can ignore this directory.
