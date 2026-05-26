"""
Agentic search layer for OpenFlipbook.
Handles web search grounding and prompt engineering.
"""

import os
import json
from typing import Optional
from .templates import (
    SEARCH_QUERY_TEMPLATE,
    GROUNDING_TEMPLATE,
    CLICK_INTERPRET_TEMPLATE,
)


class Agent:
    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai")  # openai, deepseek, minimax
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM client based on provider."""
        if self.llm_provider == "deepseek":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model="deepseek-chat",
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url="https://api.deepseek.com/anthropic",
            )
        elif self.llm_provider == "minimax":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model="MiniMax-M2.7",
                api_key=os.getenv("MINIMAX_API_KEY", ""),
                base_url="https://api.minimax.io/anthropic",
            )
        else:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY", ""),
            )

    async def search(self, query: str) -> list[dict]:
        """Web search using Tavily or fallback to DuckDuckGo."""
        if self.tavily_api_key:
            return await self._tavily_search(query)
        return await self._duckduckgo_search(query)

    async def _tavily_search(self, query: str) -> list[dict]:
        """Search using Tavily API."""
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self.tavily_api_key)
            results = client.search(query=query, max_results=5)
            return [
                {"title": r["title"], "url": r["url"], "content": r["content"]}
                for r in results.get("results", [])
            ]
        except Exception as e:
            print(f"[Agent] Tavily error: {e}, falling back to DuckDuckGo")
            return await self._duckduckgo_search(query)

    async def _duckduckgo_search(self, query: str) -> list[dict]:
        """Fallback search using DuckDuckGo via langchain."""
        try:
            from langchain_community.tools import DuckDuckGoGoSearchResults
            tool = DuckDuckGoGoSearchResults(max_results=5)
            results = await tool.ainvoke({"query": query})
            # Parse results
            parsed = []
            if isinstance(results, str):
                # Format: "title: ..., url: ..., snippet: ..."
                for line in results.split("\n"):
                    if ": " in line:
                        parts = line.split(": ", 1)
                        if len(parts) == 2:
                            parsed.append({"title": parts[0], "url": "", "content": parts[1]})
            return parsed[:5]
        except Exception as e:
            print(f"[Agent] DuckDuckGo error: {e}")
            return []

    async def generate_search_query(self, topic: str, history: str = "") -> str:
        """Generate a focused search query from a topic."""
        prompt = SEARCH_QUERY_TEMPLATE.format(topic=topic, history=history)
        response = await self.llm.ainvoke(prompt)
        return response.content.strip()

    async def ground_prompt(self, topic: str, search_results: list[dict]) -> str:
        """Extract factual grounding from search results."""
        if not search_results:
            return "No web search results available. Use your world knowledge to generate an accurate image."

        results_text = "\n".join([
            f"- {r.get('title', 'Untitled')}: {r.get('content', '')[:300]}"
            for r in search_results
        ])

        prompt = GROUNDING_TEMPLATE.format(topic=topic, search_results=results_text)
        response = await self.llm.ainvoke(prompt)
        return response.content.strip()

    async def interpret_click(
        self,
        image_description: str,
        click_x: float,
        click_y: float,
    ) -> tuple[str, str]:
        """Interpret what user wants to explore based on click coordinates."""
        prompt = CLICK_INTERPRET_TEMPLATE.format(
            image_description=image_description,
            x=click_x,
            y=click_y,
        )
        response = await self.llm.ainvoke(prompt)

        # Parse response
        label = ""
        explore = ""
        for line in response.content.strip().split("\n"):
            if line.startswith("label:"):
                label = line.replace("label:", "").strip()
            elif line.startswith("explore:"):
                explore = line.replace("explore:", "").strip()

        return label, explore

    async def build_image_prompt(
        self,
        user_action: str,
        context: str,
        grounding: str,
        width: int,
        height: int,
    ) -> str:
        """Build the final image generation prompt."""
        from .templates import IMAGE_PROMPT_TEMPLATE

        aspect_ratio = f"{width}:{height}"
        prompt = IMAGE_PROMPT_TEMPLATE.format(
            context=context,
            user_action=user_action,
            grounding=grounding,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
        )
        response = await self.llm.ainvoke(prompt)
        return response.content.strip()

    async def summarize_image(self, image_b64: str) -> str:
        """Use vision model to describe the generated image for context."""
        # This would use GPT-4V or similar vision model
        # For now, return a placeholder
        return "[Image generated — vision description not yet implemented]"
