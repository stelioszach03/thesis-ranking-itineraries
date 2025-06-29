# Thesis Figures Description

## Chapter 5: Results - Figures

### Figure 5.1: Memory Usage vs POI Count
- **Type**: Log-log plot with 4 lines
- **Data**: Memory usage (MB) vs number of POIs (10-5000)
- **Lines**: A* (exponential growth, OOM at 5000), SMA* (bounded at 256MB), Hybrid (linear growth), HeapGreedy (most efficient)
- **Key insight**: Hybrid maintains practical memory usage (89MB at 1000 POIs)

### Figure 5.2: User Preference for Daily POI Count
- **Type**: Bar chart with highlighted range
- **Data**: User preference percentage for 1-10 POIs per day
- **Highlight**: 3-7 POI range in green (90% of preferences)
- **Statistics**: Mean=5.2, Median=5, Mode=5
- **Key insight**: Validates 3-7 POI preference from research

### Figure 5.3: Comprehensive Performance Comparison (4-panel)
- **(a) Runtime Scaling**: Log-log plot showing O(n²) for Greedy/Hybrid, exponential for A*
- **(b) Solution Quality**: Bar chart of CSS scores (Random=0.412 to Hybrid=0.823)
- **(c) User Satisfaction**: Bar chart comparing Manual (6.8), ATIPS (7.4), Our System (8.4)
- **(d) Dynamic Update Efficiency**: Bar chart of LPA* computation reuse (45-89%)

### Table 5.1: Algorithm Runtime Performance
- **Format**: LaTeX table with runtime in seconds
- **Columns**: 50, 100, 500, 1000, 5000 POIs + Complexity
- **Rows**: All algorithm variants
- **Key result**: 4.3-17.5x speedup with Numba optimization

### Table 5.2: Solution Quality Comparison
- **Format**: Results table with mean ± std
- **Metrics**: CSS, TUR, SAT, FEA, DIV, Vendi Score
- **Key result**: Hybrid achieves CSS=0.823±0.067

### Table 5.3: Task Success Rate Comparison
- **Format**: Comparison table
- **Data**: TravelPlanner (0.6%) vs Our System (87.5%) on hard constraints
- **Key result**: 145.8x improvement

## Chapter 6: Discussion - Figures

### Figure 6.1: Algorithm Selection Matrix
- **Type**: 2x2 matrix visualization
- **Axes**: Memory availability (x) vs CPU power (y)
- **Quadrants**: 
  - Low Memory/Low CPU: Greedy (Basic)
  - High Memory/Low CPU: HeapGreedy (Optimized)
  - Low Memory/High CPU: Hybrid (Balanced)
  - High Memory/High CPU: A*/SMA* (Optimal)

### Figure 6.2: Quality vs Performance Trade-off Curve
- **Type**: Scatter plot with Pareto frontier
- **X-axis**: Relative runtime (log scale, Random=1.0)
- **Y-axis**: CSS Score
- **Key insight**: Hybrid achieves 96% of optimal quality at 1.5% computational cost
- **Annotation**: "Sweet spot" highlighting Hybrid's position

## Visual Design Guidelines

### Color Scheme
- Random: Red (poor performance)
- Popularity: Orange (baseline)
- Greedy variants: Blue shades (good)
- Hybrid: Dark green (best balance)
- A*: Purple (optimal but expensive)

### Typography
- Font: Times New Roman (publication standard)
- Title size: 14pt bold
- Axis labels: 12pt
- Legend/annotations: 10pt

### Layout
- Figure size: 8x6 inches (single column) or 12x10 inches (multi-panel)
- DPI: 300 for print quality
- Margins: Tight layout with proper padding

## Data Visualization Best Practices Applied

1. **Clear labeling**: All axes, legends, and annotations
2. **Appropriate scales**: Log scales for wide ranges, linear for comparisons
3. **Error bars**: Where applicable (CSS scores, satisfaction ratings)
4. **Highlighting**: Key results emphasized (3-7 POI range, sweet spot)
5. **Accessibility**: Color-blind friendly palette where possible