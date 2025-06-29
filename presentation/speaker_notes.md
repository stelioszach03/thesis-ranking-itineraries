# Speaker Notes for NYC Itinerary Ranking Presentation

## Slide 1: Title (30 seconds)
- Good [morning/afternoon], I'm excited to present our work on ranking itineraries
- This research transforms how we approach tourist trip planning in complex urban environments
- We shift from maximizing coverage to optimizing quality

## Slide 2: NYC Tourism Challenge (1 minute)
- NYC exemplifies the urban tourism challenge perfectly
- With over 10,000 POIs, tourists face decision paralysis
- Current solutions fail - even advanced AI like TravelPlanner achieves only 0.6% success
- The problem isn't finding attractions - it's finding the RIGHT attractions

## Slide 3: Problem Statement (45 seconds)
- Research shows tourists don't want to see everything
- They prefer 3-7 high-quality experiences over rushed coverage
- Quality matters more than quantity for satisfaction
- Dynamic cities require real-time adaptation

## Slide 4: Literature Gap (1 minute)
- Basu Roy et al. 2011 laid the foundation but focused on coverage
- They assume more POIs equals better itineraries - we challenge this
- Missing: quality metrics, dynamic adaptation, true user preferences
- TravelPlanner's 0.6% success rate confirms the gap

## Slide 5: Our Solution (1.5 minutes)
- We developed a hybrid framework combining three approaches
- Greedy selection provides O(n²) scalability for mobile apps
- A* routing ensures optimal paths between selected POIs
- LPA* enables dynamic replanning with 70-90% computation reuse
- Two-phase approach: select quality POIs, then optimize routing

## Slide 6: Complexity Analysis (1 minute)
- Complexity matters for real-world deployment
- Greedy maintains quadratic growth - predictable and fast
- A* becomes exponential - impractical beyond 100 POIs
- Our hybrid achieves 96% of optimal quality at 1.5% cost
- Sub-second response enables interactive planning

## Slide 7: Technology Stack (45 seconds)
- Implementation follows research roadmap exactly
- Python with Numba JIT compilation - 50x speedup
- NetworkX handles graph algorithms elegantly
- OR-Tools for complex routing constraints
- Real NYC data from Foursquare and OpenStreetMap

## Slide 8: Quality Metrics (1 minute)
- CSS formula validated through user study
- Attractiveness gets highest weight (0.35) - matches user priorities
- Time and feasibility equally important (0.25 each)
- Diversity prevents monotonous experiences (0.15)
- Vendi Score captures true experiential variety

## Slide 9: NYC Data Pipeline (45 seconds)
- Started with 15,000+ raw POIs
- Quality filtering reduced to 10,847 verified locations
- R-tree spatial indexing enables fast proximity queries
- Distance matrix precomputed with NYC grid adjustments
- Borough boundaries and subway network integrated

## Slide 10: Performance Results (1.5 minutes)
- This is our key achievement: 87.5% vs 0.6% success rate
- That's a 145x improvement over state-of-the-art
- Sub-second response times for typical queries
- Scales to entire city while fitting in mobile memory
- Not just faster - fundamentally more successful

## Slide 11: User Satisfaction (1 minute)
- 30-participant study validates our approach
- 8.4/10 satisfaction beats all baselines
- 81.7% reduction in planning time - 31 minutes to 5.7
- Confirms 3-7 POI preference from research
- SUS score of 82.3 indicates excellent usability

## Slide 12: Dynamic Adaptation (1.5 minutes)
- LPA* is crucial for real-world deployment
- Subway delays happen daily in NYC - 156ms to replan
- Weather changes require instant adaptation - 234ms
- 70-90% computation reuse makes this possible
- Users experience seamless updates, not full replanning

## Slide 13: Demo Screenshot - Interface (30 seconds)
- Clean, intuitive interface focuses on preferences
- Algorithm selector lets users choose trade-offs
- Real-time metrics show quality scores
- Interactive map provides spatial context

## Slide 14: Demo Screenshot - Dynamic Update (45 seconds)
- Rain notification triggers instant replanning
- Outdoor activities seamlessly replaced with indoor
- Quality score maintained despite constraints
- This scenario happens regularly in NYC

## Slide 15: Case Study (1 minute)
- Real family scenario demonstrates practical impact
- Age-appropriate selections without explicit programming
- Natural meal timing emerges from algorithm
- Reasonable walking distances for children
- Under budget with high satisfaction

## Slide 16: Industry Comparison (45 seconds)
- First system combining all essential features
- 5-7x faster than commercial alternatives
- Only solution with quality guarantees
- Ready for production deployment

## Slide 17: Future Directions (1 minute)
- Graph Neural Networks could capture hidden POI relationships
- Transformers for learning sequential preferences
- Federated learning addresses privacy concerns
- Multi-modal integration with AR for navigation
- Rich research opportunities ahead

## Slide 18: Broader Impact (45 seconds)
- Beyond technology - reshaping urban tourism
- Distributes visitors away from overcrowded attractions
- Supports local businesses through diversity
- Makes cities more accessible to all visitors
- Environmental benefits from efficient routing

## Slide 19: Conclusions (1 minute)
- We've proven quality beats coverage for tourist satisfaction
- 87.5% success rate makes this practically deployable
- Real-time performance enables true interactivity
- User preferences validated through rigorous study
- Dynamic adaptation handles real-world complexity

## Slide 20: Thank You (30 seconds)
- All materials available on GitHub - scan QR code
- Includes full implementation, data, and thesis
- Happy to discuss collaboration opportunities
- Questions welcome!

## Key Papers to Reference in Q&A

1. **Basu Roy et al. 2011**: "They proved NP-completeness and developed the first interactive approach, but focused on coverage"

2. **TravelPlanner 2024**: "Latest benchmark showing even GPT-4 achieves only 0.6% - validates our approach"

3. **Lim et al. 2018**: "Comprehensive study showing 3-7 POI preference - directly supports our design"

4. **Vansteenwegen 2011**: "Orienteering Problem survey - mathematical foundation for our work"

## Common Questions & Answers

**Q: How does this compare to Google Maps?**
A: Google Maps finds routes between points you select. We select the optimal points based on your preferences and constraints, then route between them.

**Q: What about group planning?**
A: Current system optimizes for individuals. Group consensus is an active research area - we're exploring social choice theory approaches.

**Q: How do you handle real-time data?**
A: LPA* enables incremental updates. We can integrate real-time feeds for transit, weather, and events with minimal computation.

**Q: Is this specific to NYC?**
A: The algorithms are general. NYC-specific elements (grid distance, subway) are modular. We've tested on Paris and Tokyo data.

**Q: What about privacy?**
A: Preferences stay on device. No tracking required. Federated learning could enable personalization without central data collection.

## Demo Talking Points

- Emphasize real-time performance - users see instant results
- Show computation reuse percentage for LPA* - unique contribution
- Highlight diversity in results - not just popular attractions
- Demonstrate preference learning - system improves with use
- Export to Google Maps shows practical integration

## Closing Emphasis

"We're not just optimizing routes - we're optimizing experiences. By focusing on quality over quantity, we help tourists truly experience cities rather than just visit them."