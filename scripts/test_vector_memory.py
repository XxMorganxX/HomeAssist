#!/usr/bin/env python3
"""
Vector Memory Test Suite

Interactive CLI for testing the vector memory system.
Allows adding sample conversations, searching, and verifying retrieval.

Usage:
    python scripts/test_vector_memory.py
    
Commands:
    add      - Add a sample conversation summary
    search   - Search for relevant memories
    list     - List all stored memories
    bulk     - Load bulk test data
    clear    - Clear all test memories
    stats    - Show memory statistics
    test     - Run automated test scenarios
    help     - Show help
    quit     - Exit
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_framework.config import VECTOR_MEMORY_CONFIG
from assistant_framework.utils.memory.vector_memory import VectorMemoryManager


# Sample conversation - will be parsed into individual exchanges
RAW_CONVERSATION = """user: My favorite color is orange
assistant: Orange is nice like the sunset.
user: my sister sofia is a zoo keeper.
assistant: You said she is a dog whisperer."""


def parse_conversation_to_exchanges(raw_text: str) -> list:
    """
    Parse raw conversation into individual user/assistant exchanges.
    Each exchange = one user message + one assistant response.
    """
    exchanges = []
    lines = raw_text.strip().split('\n')
    
    current_user = None
    current_assistant = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('user:'):
            # If we have a complete exchange, save it
            if current_user and current_assistant:
                exchanges.append({
                    'user': current_user,
                    'assistant': current_assistant
                })
            # Start new exchange
            current_user = line[5:].strip()  # Remove "user:" prefix
            current_assistant = None
            
        elif line.startswith('assistant:'):
            current_assistant = line[10:].strip()  # Remove "assistant:" prefix
    
    # Don't forget the last exchange
    if current_user and current_assistant:
        exchanges.append({
            'user': current_user,
            'assistant': current_assistant
        })
    
    return exchanges


# Parse into individual exchanges for bulk loading
SAMPLE_CONVERSATIONS = parse_conversation_to_exchanges(RAW_CONVERSATION)


class VectorMemoryTestCLI:
    """Interactive CLI for testing vector memory."""
    
    def __init__(self):
        self.manager: VectorMemoryManager = None
        self.test_prefix = "test_"  # Prefix for test conversation IDs
        
    async def initialize(self):
        """Initialize the vector memory manager."""
        print("\n🔄 Initializing vector memory manager...")
        
        # Override user_id for testing
        config = VECTOR_MEMORY_CONFIG.copy()
        config["user_id"] = "test_user"
        
        self.manager = VectorMemoryManager(config)
        success = await self.manager.initialize()
        
        if success:
            count = await self.manager.get_memory_count()
            print(f"✅ Connected! Found {count} existing memories for test user.")
        else:
            print("❌ Failed to initialize. Check your Supabase credentials.")
            sys.exit(1)
    
    async def run(self):
        """Run the interactive CLI."""
        await self.initialize()
        
        print("\n" + "="*60)
        print("  VECTOR MEMORY TEST SUITE")
        print("="*60)
        print("\nCommands: add, search, list, bulk, clear, stats, test, help, quit\n")
        
        while True:
            try:
                cmd = input(">>> ").strip().lower()
                
                if not cmd:
                    continue
                elif cmd == "quit" or cmd == "exit" or cmd == "q":
                    print("👋 Goodbye!")
                    break
                elif cmd == "help" or cmd == "h":
                    self.show_help()
                elif cmd == "add" or cmd == "a":
                    await self.add_conversation()
                elif cmd == "search" or cmd == "s":
                    await self.search_memories()
                elif cmd == "list" or cmd == "l":
                    await self.list_memories()
                elif cmd == "bulk" or cmd == "b":
                    await self.bulk_load()
                elif cmd == "clear" or cmd == "c":
                    await self.clear_memories()
                elif cmd == "stats":
                    await self.show_stats()
                elif cmd == "test" or cmd == "t":
                    await self.run_tests()
                elif cmd == "debug" or cmd == "d":
                    await self.debug_search()
                elif cmd == "raw":
                    await self.raw_query()
                elif cmd == "selftest" or cmd == "st":
                    await self.selftest()
                else:
                    print(f"❓ Unknown command: {cmd}. Type 'help' for options.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_help(self):
        """Show help information."""
        print("""
╔════════════════════════════════════════════════════════════╗
║                    AVAILABLE COMMANDS                       ║
╠════════════════════════════════════════════════════════════╣
║  add (a)     Add a conversation summary manually            ║
║  search (s)  Search for relevant memories by query          ║
║  list (l)    List all stored memories                       ║
║  bulk (b)    Load pre-defined sample conversations          ║
║  clear (c)   Clear all test memories                        ║
║  stats       Show memory statistics                         ║
║  test (t)    Run automated test scenarios                   ║
║  debug (d)   Debug search - no metadata filter              ║
║  raw         Raw database query - see stored records        ║
║  selftest(st)Search using stored text as query              ║
║  help (h)    Show this help                                 ║
║  quit (q)    Exit the test suite                            ║
╚════════════════════════════════════════════════════════════╝
""")
    
    async def add_conversation(self):
        """Add a conversation summary interactively."""
        print("\n📝 ADD CONVERSATION SUMMARY")
        print("-" * 40)
        
        # Get summary
        print("Enter the conversation summary (multi-line, empty line to finish):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        
        summary = "\n".join(lines)
        
        if not summary.strip():
            print("❌ Empty summary, cancelled.")
            return
        
        # Get topics
        topics_input = input("Topics (comma-separated, or empty): ").strip()
        topics = [t.strip() for t in topics_input.split(",")] if topics_input else []
        
        # Generate ID
        conv_id = f"{self.test_prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store
        success = await self.manager.store_conversation(
            summary=summary,
            conversation_id=conv_id,
            topics=topics
        )
        
        if success:
            print(f"✅ Stored conversation: {conv_id}")
        else:
            print("❌ Failed to store conversation")
    
    async def search_memories(self):
        """Search for relevant memories."""
        print("\n🔍 SEARCH MEMORIES")
        print("-" * 40)
        
        query = input("Enter search query: ").strip()
        if not query:
            print("❌ Empty query, cancelled.")
            return
        
        # Get optional parameters
        top_k_input = input("Max results (default 5): ").strip()
        top_k = int(top_k_input) if top_k_input else 5
        
        print(f"\n🔄 Searching for: '{query}'...")
        
        memories = await self.manager.retrieve_relevant(
            query=query,
            top_k=top_k,
            min_similarity=None  # No gating; return top-K ranked results
        )
        
        if not memories:
            print("📭 No relevant memories found.")
            return
        
        print(f"\n📚 Found {len(memories)} relevant memories:\n")
        
        for i, mem in enumerate(memories, 1):
            print("─" * 60)
            print(f"  #{i} | Similarity: {mem.similarity:.1%}")
            print(f"  Date: {mem.date.strftime('%Y-%m-%d %H:%M') if mem.date else 'Unknown'}")
            print(f"  Topics: {', '.join(mem.topics) if mem.topics else 'None'}")
            print("  Summary:")
            # Wrap summary text
            words = mem.summary.split()
            line = "    "
            for word in words:
                if len(line) + len(word) > 70:
                    print(line)
                    line = "    "
                line += word + " "
            if line.strip():
                print(line)
        print("─" * 60 + "\n")
        
        # Show formatted context
        context = self.manager.format_for_context(memories)
        print("📋 Context injection format:")
        print("-" * 40)
        print(context)
        print()
    
    async def list_memories(self):
        """List all stored memories."""
        print("\n📋 LIST ALL MEMORIES")
        print("-" * 40)
        
        # Use a very generic query to get all
        memories = await self.manager.retrieve_relevant(
            query="conversation discussion topic",
            top_k=100,
            min_similarity=0.0
        )
        
        if not memories:
            print("📭 No memories stored.")
            return
        
        print(f"\n📚 Found {len(memories)} memories:\n")
        
        for i, mem in enumerate(memories, 1):
            date_str = mem.date.strftime('%Y-%m-%d') if mem.date else 'Unknown'
            summary_preview = mem.summary[:80] + "..." if len(mem.summary) > 80 else mem.summary
            print(f"  {i}. [{date_str}] {summary_preview}")
        
        print()
    
    async def bulk_load(self):
        """Load sample conversations as individual exchanges."""
        print("\n📦 BULK LOAD SAMPLE DATA")
        print("-" * 40)
        print(f"This will add {len(SAMPLE_CONVERSATIONS)} individual exchanges.")
        print("Each exchange = 1 user message + 1 assistant response")
        
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Cancelled.")
            return
        
        success_count = 0
        for i, exchange in enumerate(SAMPLE_CONVERSATIONS, 1):
            conv_id = f"{self.test_prefix}exchange_{i:02d}"
            
            # Format as "user: ... | assistant: ..."
            text = f"user: {exchange['user']}\nassistant: {exchange['assistant']}"
            
            success = await self.manager.store_conversation(
                summary=text,
                conversation_id=conv_id,
            )
            
            # Show preview of user message
            user_preview = exchange['user'][:40]
            
            if success:
                success_count += 1
                print(f"  ✅ Exchange {i}: \"{user_preview}...\"")
            else:
                print(f"  ❌ Failed sample {i}")
        
        print(f"\n✅ Loaded {success_count}/{len(SAMPLE_CONVERSATIONS)} samples.")
    
    async def clear_memories(self):
        """Clear all test memories."""
        print("\n🗑️  CLEAR MEMORIES")
        print("-" * 40)
        
        count = await self.manager.get_memory_count()
        print(f"Found {count} memories for test user.")
        
        confirm = input("Delete all? (type 'DELETE' to confirm): ").strip()
        if confirm != 'DELETE':
            print("❌ Cancelled.")
            return
        
        deleted = await self.manager.clear_all_memories()
        print(f"✅ Deleted {deleted} memories.")
    
    async def show_stats(self):
        """Show memory statistics."""
        print("\n📊 MEMORY STATISTICS")
        print("-" * 40)
        
        count = await self.manager.get_memory_count()
        
        print(f"  Total memories: {count}")
        print(f"  User ID: {self.manager.user_id}")
        print(f"  Embedding model: {self.manager._embedding_config.get('model', 'unknown')}")
        print(f"  Table: {self.manager._vector_store_config.get('table_name', 'unknown')}")
        print(f"  Relevance threshold: {self.manager.relevance_threshold}")
        print(f"  Max age (days): {self.manager.max_age_days}")
        print()
    
    async def debug_search(self):
        """Debug search - bypasses metadata filter to check raw similarity."""
        print("\n🔧 DEBUG SEARCH (no metadata filter)")
        print("-" * 40)
        
        query = input("Enter search query: ").strip()
        if not query:
            print("❌ Empty query, cancelled.")
            return
        
        # Generate embedding
        print("🔄 Generating embedding...")
        embedding = await self.manager._embedding_provider.embed(query)
        print(f"   Embedding generated ({len(embedding)} dimensions)")
        
        # Search WITHOUT metadata filter
        print("🔄 Searching vector store (no filter)...")
        results = await self.manager._vector_store.search(
            query_embedding=embedding,
            top_k=5,
            min_similarity=0.0,  # Get everything
            filter_metadata=None  # NO FILTER
        )
        
        if not results:
            print("📭 No results found at all. Table might be empty.")
            return
        
        print(f"\n📚 Found {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.similarity:.1%}] {r.text[:60]}...")
            print(f"     Metadata: {r.metadata}")
        print()
    
    async def raw_query(self):
        """Raw database query to check what's stored."""
        print("\n🗄️  RAW DATABASE QUERY")
        print("-" * 40)
        
        try:
            # Direct Supabase query
            response = self.manager._vector_store._client.table(
                self.manager._vector_store.table_name
            ).select("id, text, metadata, created_at, embedding").limit(10).execute()
            
            if not response.data:
                print("📭 Table is empty!")
                return
            
            print(f"📚 Found {len(response.data)} records:\n")
            for i, row in enumerate(response.data, 1):
                text_preview = row.get("text", "")[:60]
                emb = row.get("embedding")
                has_emb = emb is not None and emb != ""
                emb_type = type(emb).__name__
                print(f"  {i}. ID: {row.get('id', 'unknown')[:20]}...")
                print(f"     Text: {text_preview}...")
                print(f"     Metadata: {row.get('metadata', {})}")
                print(f"     Embedding: present={has_emb} type={emb_type}")
                print()
        except Exception as e:
            print(f"❌ Query failed: {e}")

    async def selftest(self):
        """
        Self-test to diagnose 'empty results' issues.
        Pulls one stored row and searches using its own text as the query.
        Expected: it returns itself with very high similarity (~0.95+).
        """
        print("\n🧪 SELFTEST (stored text → query → should match itself)")
        print("-" * 40)

        try:
            # Fetch one record for this user (same filter used by VectorMemoryManager)
            filter_metadata = {"user_id": self.manager.user_id}
            if getattr(self.manager, "console_token", None):
                filter_metadata["console_token"] = self.manager.console_token

            resp = self.manager._vector_store._client.table(
                self.manager._vector_store.table_name
            ).select("id, text, metadata").contains("metadata", filter_metadata).limit(1).execute()

            if not resp.data:
                print("📭 No records found for current user_id filter.")
                print("   Run: bulk, then raw, then try again.")
                return

            row = resp.data[0]
            record_id = row.get("id")
            text = row.get("text", "")
            print(f"Using record: {str(record_id)[:20]}...")
            print(f"Text preview: {text[:80].replace(chr(10), ' ')}...")

            # Search using the exact stored text
            emb = await self.manager._embedding_provider.embed(text)
            results = await self.manager._vector_store.search(
                query_embedding=emb,
                top_k=5,
                min_similarity=0.0,
                filter_metadata=filter_metadata,
            )

            if not results:
                print("❌ Search returned 0 results even with min_similarity=0.0")
                print("   This usually means embeddings aren't being returned from Supabase,")
                print("   or the table has records but embedding column is NULL.")
                return

            top = results[0]
            print(f"Top match similarity: {top.similarity:.3f}")
            print(f"Top match id: {top.id}")

            if str(top.id) != str(record_id):
                print("⚠️  Top match is not the same record. Still okay, but indicates")
                print("   your dataset is small or the stored text is very generic.")

            if top.similarity < 0.6:
                print("⚠️  Similarity is low. This suggests embeddings might be malformed")
                print("   (wrong dimension / not stored / parsing issue). Run `raw` and")
                print("   confirm Embedding present=true and type looks reasonable.")

        except Exception as e:
            print(f"❌ Selftest failed: {e}")
    
    async def run_tests(self):
        """Run automated test scenarios."""
        print("\n🧪 RUNNING AUTOMATED TESTS")
        print("="*60)
        
        # Ensure we have sample data
        count = await self.manager.get_memory_count()
        if count < 1:
            print("⚠️  Loading sample data first...")
            for i, exchange in enumerate(SAMPLE_CONVERSATIONS, 1):
                text = f"user: {exchange['user']}\nassistant: {exchange['assistant']}"
                await self.manager.store_conversation(
                    summary=text,
                    conversation_id=f"{self.test_prefix}autotest_{i:02d}",
                )
        
        # Test queries based on the real conversation content
        test_queries = [
            ("friend being annoying", "friend"),
            ("closest boys", "closest"),
            ("discord chat", "discord"),
            ("corny behavior", "corny"),
            ("auto send message", "auto"),
            ("advice about communication", "advice"),
        ]
        
        passed = 0
        failed = 0
        
        for query, expected_keyword in test_queries:
            print(f"\n📝 Test: '{query}'")
            
            memories = await self.manager.retrieve_relevant(query, top_k=2, min_similarity=None)
            
            if not memories:
                print("   ❌ FAIL - No memories retrieved")
                failed += 1
                continue
            
            # Check if expected keyword appears in any result
            found = any(expected_keyword.lower() in mem.summary.lower() for mem in memories)
            
            if found:
                preview = memories[0].summary[:60].replace('\n', ' ')
                print(f"   ✅ PASS - Found: {preview}...")
                passed += 1
            else:
                preview = memories[0].summary[:60].replace('\n', ' ')
                print(f"   ⚠️  WEAK - Expected '{expected_keyword}', got: {preview}...")
                passed += 0.5
                failed += 0.5
        
        print("\n" + "="*60)
        print(f"  Results: {passed}/{len(test_queries)} tests passed")
        print("="*60 + "\n")


async def main():
    """Main entry point."""
    cli = VectorMemoryTestCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
