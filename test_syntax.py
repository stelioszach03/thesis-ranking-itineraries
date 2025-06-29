#!/usr/bin/env python3
"""Test basic Python syntax of all modules without imports"""

import ast
import os

def check_syntax(filepath):
    """Check if a Python file has valid syntax"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"

# Check all Python files
python_files = [
    'src/metrics_definitions.py',
    'src/greedy_algorithms.py',
    'src/astar_itinerary.py',
    'src/lpa_star.py',
    'src/hybrid_planner.py',
    'src/algorithm_selector.py',
    'src/prepare_nyc_data.py',
    'tests/test_metrics.py',
    'tests/test_greedy.py',
    'tests/test_astar.py',
    'tests/test_lpa.py',
    'tests/test_hybrid.py',
    'demo/demo_nyc.py',
    'demo/streamlit_demo.py',
    'benchmarks/benchmark_runner.py'
]

print("Checking Python syntax...")
print("=" * 50)

all_passed = True
for filepath in python_files:
    if os.path.exists(filepath):
        valid, error = check_syntax(filepath)
        if valid:
            print(f"✓ {filepath} - syntax OK")
        else:
            print(f"❌ {filepath} - syntax error: {error}")
            all_passed = False
    else:
        print(f"⚠️  {filepath} - file not found")
        all_passed = False

print("\n" + "=" * 50)
if all_passed:
    print("✅ All files have valid Python syntax!")
else:
    print("❌ Some files have syntax errors or are missing")