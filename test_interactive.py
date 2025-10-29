#!/usr/bin/env python3
"""
Test script for interactive components
"""
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from quantum_moe_mas.ui.components.interactive import initialize_interactive_components
    from quantum_moe_mas.ui.components.enhanced_query import EnhancedQueryInterface
    print('✅ Interactive components imported successfully')
    
    # Test component initialization
    components = initialize_interactive_components()
    print(f'✅ Components initialized: {list(components.keys())}')
    
    # Test enhanced query interface
    enhanced_query = EnhancedQueryInterface(components)
    print('✅ Enhanced query interface created successfully')
    
    print('\n🎉 All interactive components are working correctly!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()