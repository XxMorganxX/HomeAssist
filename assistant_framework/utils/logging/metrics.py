"""
Metrics collection for assistant framework.
Prometheus-compatible metrics for monitoring and observability.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import asyncio


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metric_type: str = "gauge"  # gauge, counter, histogram


class MetricsCollector(ABC):
    """Abstract base class for metrics collection."""
    
    @abstractmethod
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric (monotonically increasing)."""
        pass
    
    @abstractmethod
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric (can go up or down)."""
        pass
    
    @abstractmethod
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram/timing metric."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> List[Metric]:
        """Get all recorded metrics."""
        pass
    
    @abstractmethod
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of metrics."""
        pass


class InMemoryMetrics(MetricsCollector):
    """
    In-memory metrics collector.
    
    Stores metrics in memory for easy access and debugging.
    Can be exported to Prometheus format.
    """
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self._metrics: List[Metric] = []
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record counter metric."""
        key = self._make_key(name, labels)
        self._counters[key] += value
        
        metric = Metric(
            name=name,
            value=self._counters[key],
            labels=labels or {},
            metric_type="counter"
        )
        self._add_metric(metric)
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record gauge metric."""
        key = self._make_key(name, labels)
        self._gauges[key] = value
        
        metric = Metric(
            name=name,
            value=value,
            labels=labels or {},
            metric_type="gauge"
        )
        self._add_metric(metric)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record histogram metric."""
        key = self._make_key(name, labels)
        self._histograms[key].append(value)
        
        # Keep last 1000 values per histogram
        if len(self._histograms[key]) > 1000:
            self._histograms[key] = self._histograms[key][-1000:]
        
        metric = Metric(
            name=name,
            value=value,
            labels=labels or {},
            metric_type="histogram"
        )
        self._add_metric(metric)
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create unique key from metric name and labels."""
        if not labels:
            return name
        label_str = ','.join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _add_metric(self, metric: Metric):
        """Add metric to history."""
        self._metrics.append(metric)
        
        # Trim if too many
        if len(self._metrics) > self.max_metrics:
            self._metrics = self._metrics[-self.max_metrics:]
    
    def get_metrics(self) -> List[Metric]:
        """Get all metrics."""
        return self._metrics.copy()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {
            'total_metrics': len(self._metrics),
            'counters': dict(self._counters),
            'gauges': dict(self._gauges),
            'histograms': {}
        }
        
        # Calculate histogram stats
        for key, values in self._histograms.items():
            if values:
                summary['histograms'][key] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'sum': sum(values)
                }
        
        return summary
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        
        # Counters
        for key, value in self._counters.items():
            lines.append(f"{key} {value}")
        
        # Gauges
        for key, value in self._gauges.items():
            lines.append(f"{key} {value}")
        
        # Histograms (simplified - sum and count)
        for key, values in self._histograms.items():
            if values:
                lines.append(f"{key}_sum {sum(values)}")
                lines.append(f"{key}_count {len(values)}")
        
        return '\n'.join(lines)


class ComponentMetrics:
    """
    Helper class for component-specific metrics.
    
    Automatically adds component label to all metrics.
    """
    
    def __init__(self, collector: MetricsCollector, component: str):
        self.collector = collector
        self.component = component
    
    def _add_component_label(self, labels: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Add component label to labels dict."""
        result = labels.copy() if labels else {}
        result['component'] = self.component
        return result
    
    def counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record counter with component label."""
        self.collector.record_counter(name, value, self._add_component_label(labels))
    
    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record gauge with component label."""
        self.collector.record_gauge(name, value, self._add_component_label(labels))
    
    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record histogram with component label."""
        self.collector.record_histogram(name, value, self._add_component_label(labels))
    
    def timing(self, name: str, duration_seconds: float, labels: Optional[Dict[str, str]] = None):
        """Record timing metric (convenience method)."""
        self.histogram(name, duration_seconds, labels)


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, metrics: ComponentMetrics, metric_name: str, labels: Optional[Dict[str, str]] = None):
        self.metrics = metrics
        self.metric_name = metric_name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.metrics.timing(self.metric_name, duration, self.labels)


# Standard metric names for framework
class MetricNames:
    """Standard metric names for consistency."""
    
    # Latency metrics (seconds)
    WAKEWORD_DETECTION_LATENCY = "wakeword_detection_latency_seconds"
    TRANSCRIPTION_LATENCY = "transcription_latency_seconds"
    RESPONSE_LATENCY = "response_latency_seconds"
    TTS_LATENCY = "tts_latency_seconds"
    FULL_CONVERSATION_LATENCY = "conversation_latency_seconds"
    
    # Counter metrics
    WAKEWORD_DETECTIONS = "wakeword_detections_total"
    TRANSCRIPTION_REQUESTS = "transcription_requests_total"
    RESPONSE_REQUESTS = "response_requests_total"
    TTS_REQUESTS = "tts_requests_total"
    CONVERSATIONS = "conversations_total"
    
    # Error metrics
    ERRORS = "errors_total"
    RECOVERIES = "recoveries_total"
    
    # State metrics
    STATE_TRANSITIONS = "state_transitions_total"
    CURRENT_STATE = "current_state"
    
    # Provider health
    PROVIDER_ACTIVE = "provider_active"
    PROVIDER_RESTARTS = "provider_restarts_total"


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def setup_metrics(collector: Optional[MetricsCollector] = None) -> MetricsCollector:
    """Setup global metrics collector."""
    global _global_collector
    if collector is None:
        collector = InMemoryMetrics()
    _global_collector = collector
    return collector


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = InMemoryMetrics()
    return _global_collector


def get_component_metrics(component: str) -> ComponentMetrics:
    """Get component-specific metrics helper."""
    return ComponentMetrics(get_metrics_collector(), component)


# Example usage
if __name__ == '__main__':
    # Setup metrics
    setup_metrics()
    
    # Get component metrics
    wakeword_metrics = get_component_metrics("wakeword")
    transcription_metrics = get_component_metrics("transcription")
    
    # Record some metrics
    wakeword_metrics.counter(MetricNames.WAKEWORD_DETECTIONS)
    wakeword_metrics.histogram(MetricNames.WAKEWORD_DETECTION_LATENCY, 0.123)
    
    # Use timer
    with Timer(transcription_metrics, MetricNames.TRANSCRIPTION_LATENCY):
        time.sleep(0.5)  # Simulate work
    
    transcription_metrics.counter(MetricNames.TRANSCRIPTION_REQUESTS)
    
    # Get summary
    collector = get_metrics_collector()
    summary = collector.get_metrics_summary()
    
    print("Metrics Summary:")
    print(f"Total metrics: {summary['total_metrics']}")
    print(f"Counters: {summary['counters']}")
    print(f"Histograms: {summary['histograms']}")
    
    # Export Prometheus format
    print("\nPrometheus Format:")
    print(collector.export_prometheus())

