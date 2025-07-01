# save as verify_quality_focus.py
print("=== Verifying Quality-Based Ranking Implementation ===\n")

import ast
import re
import os

# Read metrics file
with open('src/metrics_definitions.py', 'r') as f:
    content = f.read()

# Check for quality vs quantity approach
print("1. Quality vs Quantity Paradigm:")
if "composite_satisfaction_score" in content:
    print("✓ CSS (Composite Satisfaction Score) implemented")
    
    # Extract CSS weights
    css_pattern = r'(\d+\.\d+)\s*\*\s*(tur|sat|fea|div)'
    matches = re.findall(css_pattern, content.lower())
    if matches:
        print("  Found weights:", matches)
else:
    print("✗ CSS not found")

# Check for diversity metrics
if "diversity_score" in content or "shannon_entropy" in content:
    print("✓ Diversity metrics implemented")
else:
    print("✗ Diversity metrics missing")

# Check for user satisfaction beyond POI count
if "user_satisfaction" in content and "preference" in content:
    print("✓ User preference-based satisfaction")
else:
    print("✗ User preference satisfaction missing")

# Verify it's not just maximizing POIs
print("\n2. Checking if system avoids simple POI maximization:")
constraints_found = False
quality_found = False

for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            with open(os.path.join(root, file), 'r') as f:
                file_content = f.read()
                if "min_pois" in file_content and "max_pois" in file_content:
                    constraints_found = True
                if "marginal_utility" in file_content or "quality" in file_content:
                    quality_found = True

print(f"✓ POI count constraints: {constraints_found}")
print(f"✓ Quality-based selection: {quality_found}")