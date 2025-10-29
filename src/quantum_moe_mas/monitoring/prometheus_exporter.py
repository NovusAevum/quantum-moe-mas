"""
Prometheus Metrics Exporter

Provides HTTP endpoint for Prometheus to scrape metrics and integrates
with the comprehensive metrics collection system.

Requirements: 8.1, 8.2
"""

import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
import threading

from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.exposition import MetricsHandler
from http.server import HTTPServer, BaseHTTPRequestHandler
import structlog

from .metrics_collector import MetricsCollector

logger = structlog.get_logger(__name__)


class PrometheusMetricsHandler(BaseHTTPRequestHandler):
    """Custom HTTP handler for Prometheus metrics endpoint."""
    
    def __init__(self, registry: CollectorRegistry, *args, **kwargs):
        self.registry = registry
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests for metrics endpoint."""
        if self.path == '/metrics':
            try:
                output = generate_latest(self.registry)
                self.send_response(200)
                self.send_header('Content-Type', CONTENT_TYPE_LATEST)
                self.send_header('Content-Length', str(len(output)))
                self.end_headers()
                self.wfile.write(output)
            except Exception as e:
                logger.error("Error generating Prometheus metrics", error=str(e))
                self.send_error(500, "Internal Server Error")
        elif self.path == '/health':
            self._handle_health_check()
        else:
            self.send_error(404, "Not Found")
    
    def _handle_health_check(self):
        """Handle health check endpoint."""
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'service': 'quantum-moe-mas-metrics'
        }
        
        response = str(health_data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)
    
    def log_message(self, format, *args):
        """Override to use structured logging."""
        logger.debug("HTTP request", 
                    method=self.command,
                    path=self.path,
                    client=self.client_address[0])


class PrometheusExporter:
    """
    Prometheus metrics exporter with HTTP server for scraping.
    
    Provides a dedicated HTTP endpoint for Prometheus to scrape metrics
    and integrates with the comprehensive metrics collection system.
    """
    
    def __init__(self, 
                 metrics_collector: MetricsCollector,
                 port: int = 8090,
                 host: str = "0.0.0.0"):
        """Initialize Prometheus exporter."""
        
        self.metrics_collector = metrics_collector
        self.port = port
        self.host = host
        self.registry = metrics_collector.registry
        
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._is_running = False
        
        logger.info("PrometheusExporter initialized", 
                   host=host, port=port)
    
    def start_server(self) -> None:
        """Start the Prometheus metrics HTTP server."""
        
        if self._is_running:
            logger.warning("Prometheus server already running")
            return
        
        try:
            # Create custom handler with registry
            def handler_factory(*args, **kwargs):
                return PrometheusMetricsHandler(self.registry, *args, **kwargs)
            
            self._server = HTTPServer((self.host, self.port), handler_factory)
            
            # Start server in background thread
            self._server_thread = threading.Thread(
                target=self._run_server,
                daemon=True,
                name="PrometheusExporter"
            )
            
            self._is_running = True
            self._server_thread.start()
            
            logger.info("Prometheus metrics server started",
                       endpoint=f"http://{self.host}:{self.port}/metrics")
            
        except Exception as e:
            logger.error("Failed to start Prometheus server", error=str(e))
            self._is_running = False
            raise
    
    def _run_server(self) -> None:
        """Run the HTTP server (called in background thread)."""
        
        try:
            logger.info("Prometheus HTTP server listening",
                       host=self.host, port=self.port)
            self._server.serve_forever()
            
        except Exception as e:
            logger.error("Prometheus server error", error=str(e))
        finally:
            self._is_running = False
    
    def stop_server(self) -> None:
        """Stop the Prometheus metrics HTTP server."""
        
        if not self._is_running or not self._server:
            logger.warning("Prometheus server not running")
            return
        
        try:
            self._is_running = False
            self._server.shutdown()
            self._server.server_close()
            
            if self._server_thread and self._server_thread.is_alive():
                self._server_thread.join(timeout=5.0)
            
            logger.info("Prometheus metrics server stopped")
            
        except Exception as e:
            logger.error("Error stopping Prometheus server", error=str(e))
    
    def get_metrics_text(self) -> str:
        """Get current metrics in Prometheus text format."""
        
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error("Error generating metrics text", error=str(e))
            return f"# Error generating metrics: {str(e)}\n"
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get current server status information."""
        
        return {
            'running': self._is_running,
            'host': self.host,
            'port': self.port,
            'endpoint': f"http://{self.host}:{self.port}/metrics",
            'health_endpoint': f"http://{self.host}:{self.port}/health",
            'registry_collectors': len(self.registry._collector_to_names),
            'thread_alive': self._server_thread.is_alive() if self._server_thread else False
        }
    
    async def export_custom_metrics(self, custom_metrics: Dict[str, float]) -> None:
        """Export custom application metrics to Prometheus."""
        
        # This would be used to export application-specific metrics
        # that aren't covered by the standard metrics collector
        
        for metric_name, value in custom_metrics.items():
            try:
                # Create or update custom gauge metric
                # Implementation would depend on specific metric types needed
                logger.debug("Custom metric exported", 
                           metric=metric_name, value=value)
                
            except Exception as e:
                logger.error("Error exporting custom metric",
                           metric=metric_name, error=str(e))
    
    def __enter__(self):
        """Context manager entry."""
        self.start_server()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()


# Convenience function for easy setup
def create_prometheus_exporter(metrics_collector: MetricsCollector,
                             port: int = 8090,
                             host: str = "0.0.0.0",
                             auto_start: bool = True) -> PrometheusExporter:
    """
    Create and optionally start a Prometheus exporter.
    
    Args:
        metrics_collector: The metrics collector instance
        port: HTTP server port (default: 8090)
        host: HTTP server host (default: 0.0.0.0)
        auto_start: Whether to automatically start the server
    
    Returns:
        PrometheusExporter instance
    """
    
    exporter = PrometheusExporter(
        metrics_collector=metrics_collector,
        port=port,
        host=host
    )
    
    if auto_start:
        exporter.start_server()
    
    return exporter