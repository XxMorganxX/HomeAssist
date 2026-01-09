# Memory utilities package
# Persistent memory, vector memory, caching, and conversation summarization
#
# Note: These modules have complex dependencies. Import them directly to avoid
# circular imports:
#   from assistant_framework.utils.memory.vector_memory import VectorMemoryManager
#   from assistant_framework.utils.memory.persistent_memory import PersistentMemoryManager
#   from assistant_framework.utils.memory.conversation_summarizer import ConversationSummarizer
#   from assistant_framework.utils.memory.local_vector_cache import LocalVectorCache

__all__ = [
    "PersistentMemoryManager",
    "VectorMemoryManager",
    "LocalVectorCache",
    "CachedVector",
    "SearchResult",
    "ConversationSummarizer",
]

# Lazy imports - only load when accessed
def __getattr__(name):
    if name == "PersistentMemoryManager":
        from .persistent_memory import PersistentMemoryManager
        return PersistentMemoryManager
    elif name == "VectorMemoryManager":
        from .vector_memory import VectorMemoryManager
        return VectorMemoryManager
    elif name == "LocalVectorCache":
        from .local_vector_cache import LocalVectorCache
        return LocalVectorCache
    elif name == "CachedVector":
        from .local_vector_cache import CachedVector
        return CachedVector
    elif name == "SearchResult":
        from .local_vector_cache import SearchResult
        return SearchResult
    elif name == "ConversationSummarizer":
        from .conversation_summarizer import ConversationSummarizer
        return ConversationSummarizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
