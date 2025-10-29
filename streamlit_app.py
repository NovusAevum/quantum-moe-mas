"""
Streamlit App Entry Point for Quantum MoE MAS

This file serves as the main entry point for the Streamlit application
when running directly or through deployment platforms.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the main application
from quantum_moe_mas.ui.main import main

if __name__ == "__main__":
    main()