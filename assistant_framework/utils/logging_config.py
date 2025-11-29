"""
Structured logging configuration for assistant framework.
"""

import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    # Emoji prefixes
    EMOJIS = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️ ',
        'WARNING': '⚠️ ',
        'ERROR': '❌',
        'CRITICAL': '💀'
    }
    
    def __init__(self, use_colors: bool = True, use_emojis: bool = True):
        super().__init__()
        self.use_colors = use_colors
        self.use_emojis = use_emojis
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structure."""
        # Timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # Level
        level = record.levelname
        if self.use_emojis:
            level_str = f"{self.EMOJIS.get(level, '')} {level}"
        else:
            level_str = level
        
        # Color
        if self.use_colors and sys.stdout.isatty():
            color = self.COLORS.get(level, '')
            reset = self.COLORS['RESET']
            level_str = f"{color}{level_str}{reset}"
        
        # Component (from extra context)
        component = getattr(record, 'component', 'general')
        
        # Build message
        parts = [
            f"[{timestamp}]",
            f"[{level_str:15}]",  # Pad for alignment
            f"[{component:15}]",
            record.getMessage()
        ]
        
        # Add exception info if present
        if record.exc_info:
            parts.append('\n' + self.formatException(record.exc_info))
        
        return ' '.join(parts)


class ComponentLogger:
    """
    Logger wrapper for component-specific logging.
    
    Automatically adds component context to all log messages.
    """
    
    def __init__(self, logger: logging.Logger, component: str):
        self.logger = logger
        self.component = component
    
    def _log(self, level: int, msg: str, *args, **kwargs):
        """Log with component context."""
        extra = kwargs.get('extra', {})
        extra['component'] = self.component
        kwargs['extra'] = extra
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        """Log exception with traceback."""
        kwargs['exc_info'] = True
        self._log(logging.ERROR, msg, *args, **kwargs)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    use_colors: bool = True,
    use_emojis: bool = True
) -> logging.Logger:
    """
    Setup structured logging for the framework.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        use_colors: Use ANSI colors in console output
        use_emojis: Use emojis in console output
        
    Returns:
        Configured root logger
    """
    # Get root logger
    logger = logging.getLogger('assistant_framework')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        StructuredFormatter(use_colors=use_colors, use_emojis=use_emojis)
    )
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        # File logs without colors/emojis
        file_handler.setFormatter(
            StructuredFormatter(use_colors=False, use_emojis=False)
        )
        logger.addHandler(file_handler)
    
    return logger


def get_logger(component: str) -> ComponentLogger:
    """
    Get a component-specific logger.
    
    Args:
        component: Component name (e.g., "wakeword", "transcription")
        
    Returns:
        ComponentLogger instance
    """
    base_logger = logging.getLogger('assistant_framework')
    return ComponentLogger(base_logger, component)


# Example usage:
if __name__ == '__main__':
    # Setup logging
    setup_logging(level="DEBUG")
    
    # Get component loggers
    wakeword_logger = get_logger("wakeword")
    transcription_logger = get_logger("transcription")
    
    # Log some messages
    wakeword_logger.info("Wake word detected")
    wakeword_logger.debug("Detection score: 0.95")
    transcription_logger.info("Starting transcription")
    transcription_logger.warning("Audio buffer overflow")
    transcription_logger.error("Connection failed")
    
    # Log exception
    try:
        raise ValueError("Test exception")
    except Exception:
        transcription_logger.exception("An error occurred")

