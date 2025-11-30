# Math Assistant (AutoGen + Sympy)

An AutoGen-based mathematics assistant that calls the Sympy-backed tools in `sympy_tools.py` for symbolic manipulation, calculus, and equation solving.

## Setup
- Install dependencies: `pip install autogen sympy mpmath`.
- Provide an LLM API key (e.g., `export OPENAI_API_KEY=...`). You can also pass `--api-key` when running.

## Run
- Start an interactive chat with a seed message:  
  `python math_agent.py "Differentiate sin(x) twice and evaluate the limit as x->0"`
- Adjust model/behavior if needed:  
  `python math_agent.py --model gpt-4o-mini --temperature 0.2 --max-auto-steps 3 "Series expand e^x around 0 to order 6"`

The assistant will route math requests through the registered Sympy tools via AutoGen function-calling.

## Tracing
- See a structured trace of LLM thinking and tool calls after the run:  
  `python math_agent.py --trace "Solve x^2 = 4"`
- Persist traces for later analysis (JSONL):  
  `python math_agent.py --trace --trace-file run.trace.jsonl "Integrate sin(x) from 0 to pi"`

## UI (Streamlit)
- Install extras if you haven't: `pip install streamlit autogen sympy mpmath python-dotenv`
- Start the web UI with LaTeX chat + tool traces:  
  `streamlit run ui_app.py`
- Provide your API key in the sidebar (or via `OPENAI_API_KEY`), choose model/temperature, and chat.  
  Use `$...$`/`$$...$$` for LaTeX; toggle the trace panel to inspect LLM reasoning and SymPy tool calls.
