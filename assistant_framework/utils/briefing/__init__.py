# Briefing utilities package
# Scheduled briefing management and processing
#
# Import directly to avoid circular imports:
#   from assistant_framework.utils.briefing.briefing_manager import BriefingManager
#   from assistant_framework.utils.briefing.briefing_processor import BriefingProcessor

__all__ = [
    "BriefingManager",
    "BriefingProcessor",
]

# Lazy imports - only load when accessed
def __getattr__(name):
    if name == "BriefingManager":
        from .briefing_manager import BriefingManager
        return BriefingManager
    elif name == "BriefingProcessor":
        from .briefing_processor import BriefingProcessor
        return BriefingProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
