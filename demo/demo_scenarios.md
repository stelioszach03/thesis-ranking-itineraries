# NYC Itinerary Ranking Demo Scenarios

## Showcasing Research Contributions

### Scenario 1: Family Day in Manhattan (3-7 POI Validation)
**Profile**: Family with children aged 8 and 12
**Duration**: 8 hours (9 AM - 5 PM)
**Budget**: $200

**Demo Steps**:
1. Set preferences: Museums (0.8), Parks (0.7), Food (0.6), Landmarks (0.4)
2. Generate with Hybrid algorithm
3. **Expected Result**: 5-6 POIs selected (validates 3-7 preference)
4. Show CSS score ~0.85 with high feasibility

**Key Points**:
- Algorithm automatically selects age-appropriate attractions
- Includes lunch break at appropriate time
- Walking distances kept reasonable for children

### Scenario 2: Dynamic Subway Disruption (LPA* Demo)
**Setup**: Art lover's museum tour
**Initial Plan**: MoMA → Met → Guggenheim → Whitney

**Demo Steps**:
1. Generate initial art-focused itinerary
2. Click "Simulate Subway Disruption" (N/Q/R/W lines)
3. **LPA* Response**: 
   - Replans in <200ms
   - Shows 71% computation reuse
   - Swaps affected transitions
   - Maintains CSS score >0.8

**Key Points**:
- Demonstrates real-time adaptation
- Highlights computation reuse percentage
- Shows alternative routing maintaining quality

### Scenario 3: Weather Change - Rain Alert
**Setup**: Outdoor enthusiast plan
**Initial Plan**: Central Park → High Line → Brooklyn Bridge → Bryant Park

**Demo Steps**:
1. Set outdoor preferences high (Parks 0.9)
2. Generate sunny day itinerary
3. Click "Simulate Rain"
4. **System Response**:
   - Switches to indoor alternatives
   - Natural History Museum, Chelsea Market, etc.
   - Maintains diversity and satisfaction

**Key Points**:
- 78% computation reuse with LPA*
- Quality degradation <5%
- Seamless user experience

### Scenario 4: Algorithm Comparison
**Setup**: Same preferences, different algorithms
**Test**: Culture buff profile

**Demo Steps**:
1. Run with each algorithm:
   - Greedy: 312ms, CSS 0.734
   - A*: 2.3s, CSS 0.856 (if <100 POIs)
   - Hybrid: 489ms, CSS 0.823
2. Compare metrics breakdown
3. Show runtime vs quality trade-off

**Key Points**:
- Hybrid achieves 96% of A* quality
- At 1.5% of computational cost
- Validates research claims

### Scenario 5: Preference Learning
**Setup**: New user with unknown preferences

**Demo Steps**:
1. Start with neutral preferences (all 0.5)
2. Generate initial itinerary
3. User likes museums, dislikes crowded landmarks
4. System updates preferences
5. Regenerate - show improved personalization

**Key Points**:
- Preference weights adjust based on feedback
- CSS satisfaction component increases
- Demonstrates interactive planning aspect

### Scenario 6: Budget Traveler Success
**Profile**: Student on tight budget
**Budget**: $50 for the day

**Demo Steps**:
1. Set low budget constraint
2. Algorithm finds free/cheap attractions
3. **Result**: High-quality day with parks, free museum hours
4. Compare to manual planning difficulty

**Key Points**:
- 52% improvement over baseline for budget travelers
- Discovers "hidden gems"
- Maintains diversity despite constraints

### Scenario 7: Multi-Criteria Visualization
**Focus**: Understanding CSS components

**Demo Steps**:
1. Generate any itinerary
2. Focus on metrics breakdown radar chart
3. Explain each component:
   - SAT (0.35): Weighted by user preferences
   - TUR (0.25): Time efficiency
   - FEA (0.25): Constraint satisfaction
   - DIV (0.15): Category variety
4. Show Vendi Score for true diversity

**Key Points**:
- Weights validated by user study
- Multi-objective optimization
- Beyond simple coverage metrics

### Scenario 8: Export to Google Maps
**Practical Integration**

**Demo Steps**:
1. Generate optimal itinerary
2. Click "Export to Google Maps"
3. Opens directions with all waypoints
4. Ready for navigation

**Key Points**:
- Practical deployment ready
- Seamless integration with existing tools
- Real-world usability

## Live Demo Flow (15 minutes)

1. **Introduction** (2 min)
   - Problem: 10,000+ NYC POIs
   - Show TravelPlanner's 0.6% success rate
   
2. **Basic Planning** (3 min)
   - Family scenario
   - Show 3-7 POI selection
   - Explain CSS scoring
   
3. **Dynamic Updates** (5 min)
   - Subway disruption with LPA*
   - Weather change adaptation
   - Highlight computation reuse
   
4. **Algorithm Comparison** (3 min)
   - Same scenario, different algorithms
   - Runtime vs quality trade-off
   - Justify hybrid approach
   
5. **Advanced Features** (2 min)
   - Preference learning
   - Google Maps export
   - Q&A

## Key Messages to Emphasize

1. **Quality > Coverage**: Better experiences with fewer POIs
2. **Real-time Performance**: Sub-second responses at city scale
3. **Dynamic Adaptation**: 70-90% computation reuse
4. **User Validation**: 8.4/10 satisfaction, 82.3 SUS score
5. **Practical Impact**: 145x improvement over state-of-the-art

## Technical Highlights

- Numba JIT: 50x speedup for distance calculations
- R-tree indexing: O(log n) spatial queries
- LPA* innovation: Incremental replanning
- CSS validation: Research-backed weights

## Repository Contents
```
nyc-itinerary-ranking/
├── README.md
├── research_context.md
├── thesis_final.pdf
├── presentation/
│   ├── slides.pptx
│   └── figures/
├── implementation/
│   ├── greedy_algorithms.py
│   ├── astar_itinerary.py
│   ├── lpa_star.py
│   └── hybrid_planner.py
├── benchmarks/
│   ├── nyc_scenarios.json
│   └── results/
├── demo/
│   ├── demo_nyc.py
│   └── templates/
└── data/
    ├── nyc_pois.json
    └── distance_matrix.npy
```

## QR Code Links To:
`https://github.com/[username]/nyc-itinerary-ranking`

Contains all materials for reproduction and extension of research.