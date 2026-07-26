"""
Agent
=====

The agent loop itself, built directly on the Anthropic Messages API:

    1. Send conversation history + tool definitions to the model
    2. If the model requests a tool call, execute it (via a Skill, or a
       server-side tool Anthropic runs for you) and feed the result back
    3. Repeat until the model gives a final text answer

Skills (ChromaDB search, etc.) live in skills.py — this file shouldn't
need to change when you add new capabilities there.

Install:
    pip install anthropic chromadb

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python agent.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import anthropic

from .skills import SkillRegistry, build_default_registry


@dataclass
class Agent:
    client: anthropic.Anthropic
    model: str
    system_prompt: str
    skills: SkillRegistry
    server_tools: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: int = 2048
    messages: list[dict[str, Any]] = field(default_factory=list)

    def _tools(self) -> list[dict[str, Any]]:
        # Server tools (like Anthropic's built-in web search) and your own
        # custom skills both go in the same `tools` list in the API call.
        return self.server_tools + self.skills.as_tool_params()

    def _call_model(self) -> anthropic.types.Message:
        return self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=self.messages,
            tools=self._tools(),
        )

    def _execute_tool_use_block(self, block: anthropic.types.ToolUseBlock) -> dict:
        """Run a single tool the model asked for and package the result."""
        skill = self.skills.get(block.name)
        if skill is None:
            # This would happen for a server tool (e.g. web_search) that
            # Anthropic executes on their side — those results arrive
            # already filled in, so you won't normally hit this branch
            # for server tools. This branch is for unknown/misnamed tools.
            result_text = f"Unknown tool: {block.name}"
        else:
            result_text = skill.run(**block.input)

        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result_text,
        }

    def run(self, user_input: str) -> str:
        """Send one user message through the full tool-use loop and
        return the agent's final text reply."""
        self.messages.append({"role": "user", "content": user_input})

        while True:
            response = self._call_model()

            # Always append the assistant turn (text + any tool_use blocks)
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                # No more tools requested -> this is the final answer.
                return "".join(
                    block.text for block in response.content if block.type == "text"
                )

            # The model wants to use one or more tools. Execute each,
            # collect the results, and send them back in a single
            # "user" turn containing tool_result blocks.
            tool_results = [
                self._execute_tool_use_block(block)
                for block in response.content
                if block.type == "tool_use"
            ]
            self.messages.append({"role": "user", "content": tool_results})
            # Loop back around: the model now sees the tool results and
            # either calls another tool or produces a final answer.


def build_agent() -> Agent:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    return Agent(
        client=client,
        model="claude-sonnet-4-6",
        system_prompt=(
            "You are a helpful research assistant. You have access to a "
            "local knowledge base and the public web. Prefer the knowledge "
            "base for questions about the user's own documents; use web "
            "search for anything current or outside that knowledge base."
        ),
        skills=build_default_registry(),
        # Anthropic's server-side web search tool — no handler code needed,
        # Anthropic executes it and returns results directly in the
        # response content.
        server_tools=[{"type": "web_search_20250305", "name": "web_search"}],
    )


def repl() -> None:
    agent = build_agent()
    print("Agent ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        reply = agent.run(user_input)
        print(f"\nAgent: {reply}\n")


if __name__ == "__main__":
    repl()