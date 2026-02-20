"""
vLLM Prometheus metrics collector.

Scrapes and parses Prometheus-format metrics from vLLM's /metrics endpoint.
Useful for observing KV cache hit rates, prefix caching, and other server-level stats.

Usage:
    collector = VLLMMetricsCollector("http://host:port/metrics")
    snap = collector.snapshot()
    # ... run generation ...
    snap2 = collector.snapshot()
    delta = collector.delta(snap, snap2)
"""

import re
import time
from datetime import datetime
from typing import Any, Dict, Optional

import requests

from .logging_config import get_logger

logger = get_logger("vllm_metrics")


class VLLMMetricsCollector:
    """Scrape and parse Prometheus metrics from vLLM /metrics endpoint."""

    def __init__(
        self,
        metrics_url: str,
        target_server: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.metrics_url = metrics_url
        self.target_server = target_server
        self.timeout = timeout

    def scrape_raw(self) -> Optional[str]:
        """Fetch raw Prometheus text from /metrics. Returns None on failure."""
        try:
            resp = requests.get(self.metrics_url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"Failed to scrape metrics from {self.metrics_url}: {e}")
            return None

    def parse_metrics(self, raw: Optional[str] = None) -> Dict[str, float]:
        """Parse Prometheus text format into {metric_name: value} dict.

        Collects ALL vllm:* metrics (no filter). If target_server is set,
        only returns metrics for that server.
        """
        if raw is None:
            raw = self.scrape_raw()
        if not raw:
            return {}

        result = {}
        for line in raw.splitlines():
            if line.startswith("#") or not line.strip():
                continue

            match = re.match(
                r'^([\w:]+)(?:\{([^}]*)\})?\s+([\d.eE+\-]+|NaN|Inf|-Inf)$',
                line,
            )
            if not match:
                continue

            name, labels_str, value_str = match.groups()

            if self.target_server and labels_str:
                if f'server="{self.target_server}"' not in labels_str:
                    continue

            try:
                value = float(value_str)
            except ValueError:
                value = float("nan")

            result[name] = value

        return result

    def snapshot(self) -> Dict[str, Any]:
        """Take a timestamped metrics snapshot. Returns empty metrics on failure."""
        raw = self.scrape_raw()
        metrics = self.parse_metrics(raw) if raw else {}
        return {
            "timestamp": datetime.now().isoformat(),
            "epoch": time.time(),
            "server": self.target_server,
            "metrics": metrics,
        }

    def delta(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute metric deltas between two snapshots."""
        deltas = {}
        before_metrics = before.get("metrics", {})
        after_metrics = after.get("metrics", {})
        for key in after_metrics:
            if key in before_metrics:
                deltas[f"delta_{key}"] = after_metrics[key] - before_metrics[key]
        deltas["elapsed_sec"] = after.get("epoch", 0) - before.get("epoch", 0)
        return deltas
