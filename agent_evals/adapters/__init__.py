"""Adapters for agent frameworks. Each module wraps one framework in the `Agent` protocol.

They're imported explicitly (`from agent_evals.adapters.langchain import LangChainAgent`) so the
framework itself stays an optional dependency.
"""
