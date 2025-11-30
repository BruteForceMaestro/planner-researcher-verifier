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
