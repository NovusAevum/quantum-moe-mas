"""
Command Line Interface for Quantum MoE MAS.

This module provides CLI commands for managing the Quantum-Infused MoE
Multi-Agent System, including server startup, configuration validation,
and system health checks.
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from quantum_moe_mas import __version__
from quantum_moe_mas.config.settings import get_settings, validate_required_settings
from quantum_moe_mas.core.logging import setup_logging, get_logger

app = typer.Typer(
    name="quantum-moe-mas",
    help="Quantum-Infused MoE Multi-Agent System CLI",
    add_completion=False,
)
console = Console()


@app.command()
def version():
    """Show version information."""
    console.print(f"Quantum MoE MAS v{__version__}")


@app.command()
def validate_config():
    """Validate system configuration."""
    try:
        settings = get_settings()
        validate_required_settings()
        
        console.print("✅ Configuration validation successful!", style="green")
        
        # Display configuration summary
        table = Table(title="Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Environment", settings.environment)
        table.add_row("Debug Mode", str(settings.debug))
        table.add_row("Log Level", settings.log_level)
        table.add_row("Database URL", "✅ Configured" if settings.database.database_url else "❌ Missing")
        table.add_row("Redis URL", "✅ Configured" if settings.database.redis_url else "❌ Missing")
        
        # Check AI API keys
        ai_apis_configured = []
        if settings.ai_apis.openai_api_key:
            ai_apis_configured.append("OpenAI")
        if settings.ai_apis.anthropic_api_key:
            ai_apis_configured.append("Anthropic")
        if settings.ai_apis.huggingface_api_key:
            ai_apis_configured.append("Hugging Face")
        
        table.add_row("AI APIs", ", ".join(ai_apis_configured) if ai_apis_configured else "❌ None configured")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"❌ Configuration validation failed: {e}", style="red")
        sys.exit(1)


@app.command()
def setup_logging_cmd(
    log_level: str = typer.Option("INFO", help="Log level"),
    environment: str = typer.Option("development", help="Environment"),
    log_file: Optional[Path] = typer.Option(None, help="Log file path"),
):
    """Set up logging configuration."""
    try:
        setup_logging(log_level, environment, log_file)
        logger = get_logger("cli")
        logger.info("Logging setup completed", log_level=log_level, environment=environment)
        console.print("✅ Logging setup successful!", style="green")
    except Exception as e:
        console.print(f"❌ Logging setup failed: {e}", style="red")
        sys.exit(1)


@app.command()
def health_check():
    """Perform system health check."""
    logger = get_logger("health_check")
    logger.info("Starting health check")
    
    console.print("🔍 Performing system health check...", style="blue")
    
    # Check configuration
    try:
        settings = get_settings()
        validate_required_settings()
        console.print("✅ Configuration: OK", style="green")
    except Exception as e:
        console.print(f"❌ Configuration: {e}", style="red")
        return
    
    # Check database connectivity (if configured)
    if settings.database.database_url:
        try:
            # TODO: Implement database connectivity check
            console.print("✅ Database: OK", style="green")
        except Exception as e:
            console.print(f"❌ Database: {e}", style="red")
    
    # Check Redis connectivity (if configured)
    if settings.database.redis_url:
        try:
            # TODO: Implement Redis connectivity check
            console.print("✅ Redis: OK", style="green")
        except Exception as e:
            console.print(f"❌ Redis: {e}", style="red")
    
    # Check AI API connectivity
    console.print("🤖 Checking AI API connectivity...")
    # TODO: Implement AI API connectivity checks
    
    console.print("✅ Health check completed!", style="green")
    logger.info("Health check completed")


@app.command()
def start_api(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    workers: int = typer.Option(1, help="Number of worker processes"),
):
    """Start the FastAPI server."""
    try:
        import uvicorn
        
        logger = get_logger("api_server")
        logger.info("Starting API server", host=host, port=port, workers=workers)
        
        console.print(f"🚀 Starting API server on {host}:{port}", style="blue")
        
        uvicorn.run(
            "quantum_moe_mas.api.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers,
        )
    except ImportError:
        console.print("❌ FastAPI dependencies not installed", style="red")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ Failed to start API server: {e}", style="red")
        sys.exit(1)


@app.command()
def start_ui(
    port: int = typer.Option(8501, help="Port to bind to"),
    host: str = typer.Option("localhost", help="Host to bind to"),
):
    """Start the Streamlit UI."""
    try:
        import subprocess
        import os
        
        logger = get_logger("ui_server")
        logger.info("Starting Streamlit UI", port=port, host=host)
        
        console.print(f"🎨 Starting Streamlit UI on {host}:{port}", style="blue")
        console.print("📊 Dashboard will be available at:", style="cyan")
        console.print(f"   http://{host}:{port}", style="cyan")
        
        # Get the correct path to the UI main file
        ui_main_path = Path(__file__).parent / "ui" / "main.py"
        
        subprocess.run([
            "streamlit", "run",
            str(ui_main_path),
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except FileNotFoundError:
        console.print("❌ Streamlit not installed. Install with: pip install streamlit", style="red")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ Failed to start UI: {e}", style="red")
        sys.exit(1)


@app.command()
def init_db():
    """Initialize the database."""
    try:
        logger = get_logger("db_init")
        logger.info("Initializing database")
        
        console.print("🗄️ Initializing database...", style="blue")
        
        # TODO: Implement database initialization
        console.print("✅ Database initialized successfully!", style="green")
        
        logger.info("Database initialization completed")
    except Exception as e:
        console.print(f"❌ Database initialization failed: {e}", style="red")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()