"""
Google Search Tool

Provides Google-like search with AI Overview when available; otherwise summarizes
organic results using Gemini (if API key available). Returns concise answers.
Supports different query types for optimized responses.
"""

import os
import re
import sys
import time
from typing import Dict, Any, Optional
from mcp_server.base_tool import BaseTool

# Timing logger for performance analysis - uses stderr to avoid MCP protocol interference
def _log_timing(component: str, elapsed_ms: float, extra: str = ""):
    """Log timing information for search components."""
    extra_str = f" | {extra}" if extra else ""
    print(f"⏱️  [{component}] {elapsed_ms:.0f}ms{extra_str}", file=sys.stderr)

# Import web search with fallback
_import_error = None
try:
    from mcp_server.clients.web_search_client import Websearch
except ImportError as e:
    Websearch = None
    _import_error = str(e)

# Gemini import for custom summarization
try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except ImportError:
    genai = None
    _HAS_GEMINI = False


class GoogleSearchTool(BaseTool):
    """Perform a Google-style search and return an AI Overview or a Gemini summary."""

    name = "google_search"
    description = (
        "Search the web for up-to-date information. IMPORTANT: Only call this tool ONCE per user request. "
        "Combine all aspects of the user's question into a single comprehensive search query. "
        "Supports different query types: 'general' for informational queries, 'link' for when user wants a specific URL/link, "
        "'directions' for navigation/directions queries. "
        "Returns concise, direct answers. For link/directions queries, returns just the relevant link. "
        "Present only the answer - no preamble or fluff."
    )
    version = "1.2.0"

    # Patterns to detect link/directions queries
    LINK_PATTERNS = [
        r'\blink\s+to\b', r'\burl\s+(for|to)\b', r'\bwebsite\s+(for|of)\b',
        r'\bfind\s+(me\s+)?(the\s+)?link\b', r'\bgive\s+(me\s+)?(the\s+)?link\b',
        r'\bget\s+(me\s+)?(the\s+)?link\b', r'\bwhere\s+can\s+i\s+find\b',
    ]
    DIRECTIONS_PATTERNS = [
        r'\bdirections?\s+to\b', r'\bhow\s+(do\s+i\s+)?get\s+to\b',
        r'\broute\s+to\b', r'\bnavigate\s+to\b', r'\bdriving\s+to\b',
        r'\bwalking\s+to\b', r'\btransit\s+to\b',
    ]

    def __init__(self):
        super().__init__()
        self.searcher = None
        self._init_error = None
        self._gemini_model = None
        
        if Websearch is None:
            self._init_error = f"Websearch client not available: {_import_error or 'missing serpapi package'}"
            self.logger.warning(self._init_error)
        else:
            try:
                self.searcher = Websearch()
                if not self.searcher.api_key:
                    self._init_error = "SERPAPI_API_KEY not configured in environment"
                    self.logger.warning(self._init_error)
            except Exception as e:
                self._init_error = f"Failed to initialize Websearch: {e}"
                self.logger.error(self._init_error)
        
        # Initialize Gemini for custom summarization
        if _HAS_GEMINI:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-latest"]:
                        try:
                            self._gemini_model = genai.GenerativeModel(model_name)
                            break
                        except Exception:
                            continue
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Gemini: {e}")

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's search query. Be specific for better results.",
                },
                "query_type": {
                    "type": "string",
                    "enum": ["general", "link", "directions"],
                    "description": (
                        "Type of query to optimize response format. "
                        "'general' for informational questions (default). "
                        "'link' when user wants a specific URL (e.g., 'find me the link to...'). "
                        "'directions' for navigation queries (e.g., 'directions to 123 Main St'). "
                        "For link/directions, returns just the relevant URL."
                    ),
                    "default": "general"
                }
            },
            "required": ["query"]
        }

    def _detect_query_type(self, query: str) -> str:
        """Auto-detect query type from the query text."""
        query_lower = query.lower()
        
        for pattern in self.DIRECTIONS_PATTERNS:
            if re.search(pattern, query_lower):
                return "directions"
        
        for pattern in self.LINK_PATTERNS:
            if re.search(pattern, query_lower):
                return "link"
        
        return "general"

    def _extract_destination(self, query: str) -> str:
        """Extract the destination/target from a directions query."""
        query_lower = query.lower()
        
        # Remove common prefixes
        for pattern in self.DIRECTIONS_PATTERNS + self.LINK_PATTERNS:
            query_lower = re.sub(pattern, '', query_lower)
        
        # Clean up
        destination = query_lower.strip()
        destination = re.sub(r'^(the\s+)?', '', destination)
        return destination.strip() or query

    def _generate_maps_link(self, destination: str) -> str:
        """Generate a Google Maps directions link."""
        from urllib.parse import quote
        encoded = quote(destination)
        return f"https://www.google.com/maps/dir/?api=1&destination={encoded}"

    def _succinct_summarize(self, query: str, results: list, query_type: str) -> Optional[str]:
        """Generate a succinct summary using Gemini with query-type-specific prompts."""
        t_start = time.perf_counter()
        
        if not self._gemini_model or not results:
            _log_timing("Gemini Succinct Summary", 0, "SKIPPED (no model or results)")
            return None
        
        try:
            # Build sources block
            items = []
            for r in results[:5]:  # Limit to top 5
                title = (r.get("title") or "").strip()
                link = (r.get("link") or "").strip()
                snippet = (r.get("snippet") or "").strip()
                if title or link:
                    items.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}")
            
            sources_block = "\n\n".join(items)
            
            if query_type == "directions":
                prompt = (
                    f"User wants directions to a location. From the search results below, "
                    f"provide ONLY a Google Maps link. If a maps link exists in the results, use it. "
                    f"Otherwise, say 'Use Google Maps: https://www.google.com/maps/dir/?api=1&destination=[encoded address]' "
                    f"with the actual destination encoded.\n\n"
                    f"Query: {query}\n\nSources:\n{sources_block}\n\n"
                    f"Return ONLY the link, nothing else."
                )
            elif query_type == "link":
                prompt = (
                    f"User wants a specific link/URL. From the search results below, "
                    f"identify the most relevant URL and return ONLY that URL. "
                    f"No explanation, just the link.\n\n"
                    f"Query: {query}\n\nSources:\n{sources_block}\n\n"
                    f"Return ONLY the most relevant URL."
                )
            else:
                prompt = (
                    f"Answer the user's query in 1-2 sentences max. Be direct and factual. "
                    f"No preamble like 'Based on...' or 'According to...'. Just the answer.\n\n"
                    f"Query: {query}\n\nSources:\n{sources_block}"
                )
            
            gen_start = time.perf_counter()
            resp = self._gemini_model.generate_content(prompt)
            gen_elapsed = (time.perf_counter() - gen_start) * 1000
            
            text = getattr(resp, "text", None)
            total_elapsed = (time.perf_counter() - t_start) * 1000
            
            if text:
                _log_timing("Gemini Succinct Summary", total_elapsed, f"type={query_type} (generation: {gen_elapsed:.0f}ms)")
                return text.strip()
            else:
                _log_timing("Gemini Succinct Summary", total_elapsed, "FAILED (empty response)")
        except Exception as e:
            total_elapsed = (time.perf_counter() - t_start) * 1000
            _log_timing("Gemini Succinct Summary", total_elapsed, f"FAILED: {e}")
            self.logger.warning(f"Gemini summarization failed: {e}")
        
        return None

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        total_start = time.perf_counter()
        print("\n" + "="*60, file=sys.stderr)
        print(f"🔍 GOOGLE SEARCH TIMING ANALYSIS", file=sys.stderr)
        print("="*60, file=sys.stderr)
        
        try:
            query: str = params.get("query", "").strip()
            query_type: str = params.get("query_type", "").strip().lower()

            if not query:
                return {"result": "", "source": "", "success": False, "error": "Empty query"}

            # Auto-detect query type if not specified
            if not query_type or query_type not in ["general", "link", "directions"]:
                query_type = self._detect_query_type(query)
            
            print(f"📝 Query: \"{query}\"", file=sys.stderr)
            print(f"📋 Type: {query_type}", file=sys.stderr)
            print("-"*60, file=sys.stderr)

            # Check if searcher is available
            if self.searcher is None:
                return {
                    "result": "",
                    "source": "",
                    "success": False,
                    "error": self._init_error or "Search client not initialized",
                    "suggestion": "Install serpapi: pip install google-search-results, and set SERPAPI_API_KEY"
                }

            # For directions queries, generate a direct maps link first
            if query_type == "directions":
                destination = self._extract_destination(query)
                maps_link = self._generate_maps_link(destination)
                total_elapsed = (time.perf_counter() - total_start) * 1000
                _log_timing("TOTAL (directions)", total_elapsed, "instant maps link")
                print("="*60 + "\n", file=sys.stderr)
                return {
                    "result": maps_link,
                    "source": "Google Maps",
                    "success": True,
                    "query_type": "directions",
                    "destination": destination
                }

            # For link queries, get search results and return the top link
            if query_type == "link":
                results = self.searcher.get_search_results(query, num_results=3)
                if results:
                    # Try Gemini to pick the best link
                    best_link = self._succinct_summarize(query, results, "link")
                    if best_link and best_link.startswith("http"):
                        total_elapsed = (time.perf_counter() - total_start) * 1000
                        _log_timing("TOTAL (link query)", total_elapsed, "Gemini picked link")
                        print("="*60 + "\n", file=sys.stderr)
                        return {"result": best_link, "source": "Search", "success": True, "query_type": "link"}
                    # Fallback to first result
                    top_result = results[0]
                    total_elapsed = (time.perf_counter() - total_start) * 1000
                    _log_timing("TOTAL (link query)", total_elapsed, "first result fallback")
                    print("="*60 + "\n", file=sys.stderr)
                    return {
                        "result": top_result.get("link", ""),
                        "source": "Search",
                        "success": True,
                        "query_type": "link",
                        "title": top_result.get("title", "")
                    }
                total_elapsed = (time.perf_counter() - total_start) * 1000
                _log_timing("TOTAL (link query)", total_elapsed, "no results")
                print("="*60 + "\n", file=sys.stderr)
                return {"result": "No relevant link found.", "source": "Search", "success": False}

            # General query: Try AI Overview first
            print("\n--- PHASE 1: AI Overview ---", file=sys.stderr)
            ai = self.searcher.get_ai_overview(query)
            ai_text = self.searcher.flatten_ai_overview(ai)
            
            # Check result
            if ai_text:
                total_elapsed = (time.perf_counter() - total_start) * 1000
                print("-"*60, file=sys.stderr)
                _log_timing("✅ TOTAL (general)", total_elapsed, "SUCCESS via AI Overview")
                print("="*60 + "\n", file=sys.stderr)
                return {"result": ai_text, "source": "AI Overview", "success": True, "query_type": "general"}

            # Fallback: Get results and summarize
            print("\n--- PHASE 2: Organic Results ---", file=sys.stderr)
            results = self.searcher.get_search_results(query, num_results=8)
            
            if not results:
                total_elapsed = (time.perf_counter() - total_start) * 1000
                _log_timing("❌ TOTAL (general)", total_elapsed, "no results found")
                print("="*60 + "\n", file=sys.stderr)
                return {
                    "result": "No search results found for this query.",
                    "source": "Search",
                    "success": False
                }
            
            # Try succinct summarization
            print("\n--- PHASE 3: Gemini Succinct ---", file=sys.stderr)
            summary = self._succinct_summarize(query, results, "general")
            if summary:
                total_elapsed = (time.perf_counter() - total_start) * 1000
                print("-"*60, file=sys.stderr)
                _log_timing("✅ TOTAL (general)", total_elapsed, "SUCCESS via Gemini Succinct")
                print("="*60 + "\n", file=sys.stderr)
                return {"result": summary, "source": "Gemini", "success": True, "query_type": "general"}
            
            # Fallback to standard Gemini summarization
            print("\n--- PHASE 4: Gemini Standard ---", file=sys.stderr)
            summary = self.searcher._summarize_with_gemini(query, results)
            if summary:
                total_elapsed = (time.perf_counter() - total_start) * 1000
                print("-"*60, file=sys.stderr)
                _log_timing("✅ TOTAL (general)", total_elapsed, "SUCCESS via Gemini Standard")
                print("="*60 + "\n", file=sys.stderr)
                return {"result": summary, "source": "Gemini", "success": True, "query_type": "general"}

            # Last resort: format results directly
            print("\n--- PHASE 5: Raw Results ---", file=sys.stderr)
            formatted = self.searcher.format_results(results)
            if formatted:
                total_elapsed = (time.perf_counter() - total_start) * 1000
                print("-"*60, file=sys.stderr)
                _log_timing("✅ TOTAL (general)", total_elapsed, "SUCCESS via raw results")
                print("="*60 + "\n", file=sys.stderr)
                return {"result": formatted, "source": "Search results", "success": True}

            total_elapsed = (time.perf_counter() - total_start) * 1000
            _log_timing("❌ TOTAL (general)", total_elapsed, "all methods failed")
            print("="*60 + "\n", file=sys.stderr)
            return {
                "result": "Search completed but no summary available.",
                "source": "Search",
                "success": False
            }
        except Exception as e:
            total_elapsed = (time.perf_counter() - total_start) * 1000
            _log_timing("❌ TOTAL", total_elapsed, f"EXCEPTION: {e}")
            print("="*60 + "\n", file=sys.stderr)
            self.logger.error(f"Google search failed: {e}")
            return {
                "result": f"Search failed: {e}",
                "source": "",
                "success": False,
                "error": str(e)
            }


