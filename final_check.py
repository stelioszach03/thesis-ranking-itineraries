import subprocess
import sys
import os

def run_command(cmd):
    """Run command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

print("=== NYC Itinerary Ranking - Final System Check ===\n")

# Check Python version
success, stdout, _ = run_command("venv/bin/python --version")
print(f"Python Version: {stdout.strip()}")

# Check key packages
packages = ["numpy", "pandas", "networkx", "pytest", "flask", "streamlit"]
for pkg in packages:
    success, stdout, _ = run_command(f"venv/bin/pip show {pkg} | grep Version")
    if success:
        print(f"✓ {pkg}: {stdout.strip()}")
    else:
        print(f"✗ {pkg}: Not installed")

# Check data files
print("\nData Files:")
data_files = ["nyc_pois.json", "distance_matrix.npy", "subway_stations.json", "metadata.json"]
for file in data_files:
    path = f"data/nyc_data/{file}"
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024  # KB
        print(f"✓ {file}: {size:.1f} KB")
    else:
        print(f"✗ {file}: Missing")

# Test imports
print("\nCore Imports:")
try:
    sys.path.insert(0, 'src')
    from metrics_definitions import POI
    from greedy_algorithms import GreedyPOISelection
    from hybrid_planner import HybridPlanner
    print("✓ All core imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")

print("\n✅ System check complete!")