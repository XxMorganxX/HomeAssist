"""
Simple Google AI Overview search using SerpAPI, wrapped in a class.

Usage:
    python core/web_search.py
"""

# pip install google-search-results
import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
import requests





class Websearch:
    """Encapsulates SerpAPI Google search with AI Overview support."""

    def __init__(self, api_key: str | None = None, location: str | None = None):
        load_dotenv()
        self.api_key = api_key or os.environ["SERPAPI_API_KEY"]
        # Optional SerpAPI location string, e.g., "Ithaca, New York, United States"
        self.location = location

    def get_ai_overview(self, query: str, *, hl: str = "en", gl: str = "us", location: str | None = None) -> dict | None:
        """Run Google search and return AI Overview payload if available."""
        # Step 1: run a normal Google Search
        params = {
            "engine": "google",
            "q": query,
            "hl": hl,
            "gl": gl,
            "google_domain": "google.com",
            "device": "desktop",
            "safe": "off",
            "ai_overview": "true",  # hint to include AI Overview when available
            "no_cache": "true",
            "api_key": self.api_key,
        }
        loc = location or self.location
        if loc:
            params["location"] = loc
        s1 = GoogleSearch(params).get_dict()

        # If AI Overview is embedded, it's already under 'ai_overview'
        ai = (
            s1.get("ai_overview")
            or s1.get("sg_results")
            or s1.get("sge_summary")
        )
        if not ai:
            # If AI Overview requires a separate request, the first call returns a page_token
            token = (
                (s1.get("ai_overview") or {}).get("page_token")
                or s1.get("search_metadata", {}).get("ai_overview_page_token")
            )
            if token:
                s2 = GoogleSearch({
                    "engine": "google_ai_overview",
                    "page_token": token,
                    "api_key": self.api_key,
                }).get_dict()
                ai = s2.get("ai_overview")
            # Final fallback: attempt direct AI Overview request with the query
            if not ai:
                try:
                    s3_params = {
                        "engine": "google_ai_overview",
                        "q": query,
                        "hl": hl,
                        "gl": gl,
                        "api_key": self.api_key,
                    }
                    if loc:
                        s3_params["location"] = loc
                    s3 = GoogleSearch(s3_params).get_dict()
                    ai = s3.get("ai_overview")
                except Exception:
                    pass
        return ai

    def get_search_results(self, 
                           query: str, 
                           *, 
                           hl: str = "en", 
                           gl: str = "us", 
                           location: str | None = None,
                           num_results: int = 5) -> list[dict]:
        """Fetch standard Google search results (organic) as a fallback.
        Returns a list of dicts: {title, link, snippet}.
        """
        params = {
            "engine": "google",
            "q": query,
            "hl": hl,
            "gl": gl,
            "google_domain": "google.com",
            "device": "desktop",
            "safe": "off",
            "no_cache": "true",
            "num": max(1, min(int(num_results or 5), 20)),
            "api_key": self.api_key,
        }
        loc = location or self.location
        if loc:
            params["location"] = loc
        data = GoogleSearch(params).get_dict()
        organic = data.get("organic_results", []) or []
        results: list[dict] = []
        for r in organic[:num_results]:
            if not isinstance(r, dict):
                continue
            title = (r.get("title") or "").strip()
            link = (r.get("link") or "").strip()
            snippet = (r.get("snippet") or r.get("snippet_highlighted_words") or "")
            if isinstance(snippet, list):
                snippet = " ".join(w for w in snippet if isinstance(w, str))
            snippet = snippet.strip() if isinstance(snippet, str) else ""
            if title or link or snippet:
                results.append({"title": title, "link": link, "snippet": snippet})
        return results

    @staticmethod
    def flatten_ai_overview(ai_overview: dict | None) -> str | None:
        """Flatten AI Overview blocks into markdown-like text."""
        if not ai_overview:
            return None
        parts: list[str] = []
        for block in ai_overview.get("text_blocks", []) or []:
            try:
                btype = block.get("type", "").lower()
                # Prefer 'snippet', fall back to 'text' or 'title'
                content = (
                    block.get("snippet")
                    or block.get("text")
                    or block.get("title")
                    or ""
                )
                content = content.strip() if isinstance(content, str) else ""

                if btype == "heading":
                    if content:
                        parts.append(f"## {content}")
                elif btype == "paragraph":
                    if content:
                        parts.append(content)
                elif btype == "list":
                    items = block.get("list") or block.get("items") or []
                    for item in items:
                        ititle = (item.get("title") or "").strip() if isinstance(item, dict) else ""
                        isnippet = (
                            (item.get("snippet") if isinstance(item, dict) else None)
                            or (item.get("text") if isinstance(item, dict) else None)
                            or ""
                        )
                        isnippet = isnippet.strip() if isinstance(isnippet, str) else ""
                        if ititle or isnippet:
                            bullet = f"- **{ititle}** {isnippet}".strip() if ititle else f"- {isnippet}"
                            parts.append(bullet)
                else:
                    # Unknown types: include any non-empty content
                    if content:
                        parts.append(content)
            except Exception:
                # Skip malformed blocks gracefully
                continue
        return "\n".join(parts) if parts else None

    @staticmethod
    def format_results(results: list[dict]) -> str | None:
        """Format organic results into a readable text block."""
        if not results:
            return None
        lines: list[str] = []
        for r in results:
            title = r.get("title") or ""
            link = r.get("link") or ""
            snippet = r.get("snippet") or ""
            line = f"- {title} ({link})\n  {snippet}".strip()
            lines.append(line)
        return "\n".join(lines)

    def _summarize_with_gemini(self, query: str, results: list[dict]) -> str | None:
        """Summarize search results using Gemini 1.5 Flash, if available.
        Requires GOOGLE_API_KEY and google-generativeai package.
        """
        try:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                return None
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=api_key)
            # Try a list of flash-capable models in order
            model_names = [
                "gemini-2.5-flash",
                "gemini-1.5-flash",
                "gemini-1.5-flash-latest",
            ]
            model = None
            for name in model_names:
                try:
                    model = genai.GenerativeModel(name)
                    break
                except Exception:
                    model = None
            if model is None:
                return None

            # Build a concise, citation-friendly prompt
            items = []
            for r in results:
                title = (r.get("title") or "").strip()
                link = (r.get("link") or "").strip()
                snippet = (r.get("snippet") or "").strip()
                if title or link or snippet:
                    items.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}")
            sources_block = "\n\n".join(items)

            prompt = (
                "You are a concise assistant. Using only the sources below, answer the user's query in 3–5 sentences. "
                "Be factual and neutral. After the answer, add a short 'Sources:' section listing 2–3 of the most relevant URLs.\n\n"
                f"User query: {query}\n\n"
                f"Sources:\n{sources_block}\n\n"
                "Return only plain text."
            )

            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None)
            if not text and hasattr(resp, "candidates") and resp.candidates:
                # Fallback: concatenate parts if text is missing
                try:
                    parts = []
                    for c in resp.candidates:
                        if hasattr(c, "content") and getattr(c.content, "parts", None):
                            for p in c.content.parts:
                                if hasattr(p, "text") and isinstance(p.text, str):
                                    parts.append(p.text)
                    text = "\n".join(parts)
                except Exception:
                    text = None
            return text.strip() if isinstance(text, str) and text.strip() else None
        except Exception:
            return None

    def get_best_answer(self, query: str, *, hl: str = "en", gl: str = "us", num_results: int = 8) -> tuple[str, str]:
        """Return (text, source), where source is 'ai_overview' or 'gemini' or 'results' or 'none'."""
        # Try AI Overview first
        ai = self.get_ai_overview(query, hl=hl, gl=gl, location=self.location)
        text = self.flatten_ai_overview(ai)
        if text:
            return text, "ai_overview"

        # Fallback to organic results
        results = self.get_search_results(query, hl=hl, gl=gl, location=self.location, num_results=num_results)
        if not results:
            return "No results available for this query.", "none"

        # Try Gemini summarization
        summary = self._summarize_with_gemini(query, results)
        if summary:
            return summary, "gemini"

        # Final fallback: formatted list
        formatted = self.format_results(results)
        return (formatted or "No results available for this query."), "results"

    def search_and_print(self, query: str) -> None:
        """Execute search and print only the final response with its source label."""
        text, source = self.get_best_answer(query)
        label = {
            "ai_overview": "[AI Overview]",
            "gemini": "[Gemini Summary]",
            "results": "[Search Results]",
            "none": "[No Result]",
        }.get(source, "")
        if label:
            print(f"{label} {text}")
        else:
            print(text)


if __name__ == "__main__":
    default_query = "What happened in the spanish american war?"
    Websearch().search_and_print(default_query)