"""
Improved Google Search Tool

Provides Google-like search with AI Overview when available; otherwise summarizes
organic results using Gemini (if API key available). Returns concise answers.
"""

import sys
sys.path.insert(0, '../..')

from typing import Dict, Any

from mcp_server.improved_base_tool import ImprovedBaseTool

try:
    from core.web_search import Websearch
except Exception as e:
    # Fallback import path when running from MCP server root
    import os
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from core.web_search import Websearch  # type: ignore


class ImprovedGoogleSearchTool(ImprovedBaseTool):
    """Perform a Google-style search and return an AI Overview or a Gemini summary."""

    name = "improved_google_search"
    description = (
        "Search the web for up-to-date information. Returns AI Overview when available; "
        "otherwise summarizes search results concisely (uses Gemini if configured). "
        "CRITICAL: In the final agent response, append a source label exactly as "
        "'(Source: AI Overview)' when the AI Overview is used, or '(Source: Gemini response)' "
        "when the Gemini summary is used. Present only the concise answer followed by the source label."
    )
    version = "1.0.0"

    def __init__(self):
        super().__init__()
        self.searcher = Websearch()

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's search query. Be specific for better results.",
                }
            },
            "required": ["query"]
        }

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query: str = params.get("query", "").strip()

            if not query:
                return {"result": "", "source": ""}

            # Try AI Overview first
            ai = self.searcher.get_ai_overview(query)
            ai_text = self.searcher.flatten_ai_overview(ai)
            if ai_text:
                return {"result": ai_text, "source": "AI Overview"}

            # Fallback: Summarize organic results using Gemini
            results = self.searcher.get_search_results(query, num_results=8)
            summary = self.searcher._summarize_with_gemini(query, results)
            if summary:
                return {"result": summary, "source": "Gemini response"}

            # If Gemini is unavailable, return empty response (caller can handle)
            return {
                "result": "",
                "source": "Gemini response"
            }
        except Exception as e:
            return {"result": f"Search failed: {e}", "source": "Gemini response"}


