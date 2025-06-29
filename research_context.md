---
title: research_context
thesis_title: "Ranking Itineraries: Dynamic algorithms meet user preferences"
university: "NKUA Informatics & Telecommunications"
city: "New York City"
date: "2025-06-29"
---

# Research Context: Ranking Itineraries - Dynamic Algorithms Meet User Preferences

## 1. Thesis Project Description

**Title:** Ranking Itineraries: Dynamic algorithms meet user preferences (Βαθμονόμηση δρομολογίων)

**University:** National and Kapodistrian University of Athens (NKUA), Department of Informatics & Telecommunications

**Supervisor:** To be confirmed (Note: Prof. Dimitrios Gounopoulos is not affiliated with NKUA's Department of Informatics & Telecommunications)

**Motivation:** Recent work by Basu Roy et al. [1] attempts to propose itineraries to users based on specific interests. Their approach allows users to dynamically change Points of Interest (POIs) that they want to visit - if users don't like certain POIs, they can deselect them and new ones can be suggested. This thesis approaches the problem from a different perspective: evaluating the quality of itineraries. While Basu Roy et al. take a first step by assuming that more POIs covered in a route leads to higher user satisfaction, this is not necessarily true as fewer, more enjoyable points might provide better user experience.

**Objectives:**
- Develop algorithms for evaluating and ranking possible itineraries
- Create recommendation systems for users based on quality rankings
- Ensure algorithms are fast and dynamic in nature
- Enable easy incorporation of new points on-the-fly
- Focus implementation and evaluation on New York City data

**Scope:** 
- 1-2 person project
- Focus on algorithmic optimization and user experience design
- Implementation in any programming language
- Emphasis on New York City as the primary use case for data, benchmarks, and demonstrations

**Requirements:**
- Programming skills for software development
- English proficiency for literature review
- 16 ECTS credits (equivalent to two compulsory courses)
- Annual project spanning full academic year

## 2. Research Summary

The research reveals a fascinating evolution in itinerary planning from static optimization to sophisticated interactive systems. The foundational work by Senjuti Basu Roy et al. (2011) established the three-step interactive planning process and proved NP-completeness even for simple scoring functions, justifying heuristic approaches. Recent advances (2019-2025) show dramatic shifts toward AI-powered solutions using Graph Neural Networks for encoding POI relationships and transformer architectures for sequential recommendations.

Modern evaluation frameworks balance multiple objectives through comprehensive metrics. Quantitative measures include travel distance/time, costs, and efficiency ratios, while qualitative assessments capture user satisfaction, diversity (using sophisticated measures like the Vendi Score with Shannon entropy), novelty, and personalization levels. Research indicates users prefer 3-7 POIs per day, with weighted utility functions combining factors where attractiveness typically receives highest weight (0.35), followed by time efficiency and feasibility (0.25 each).

The algorithmic framework layers sophisticated techniques atop proven foundations. Dynamic programming provides optimal solutions but suffers from exponential complexity. Greedy heuristics offer O(n²) complexity suitable for real-time applications. The most promising approach combines graph-based algorithms with machine learning enhancements: R-trees for spatial indexing, A* with admissible heuristics for pathfinding, and Graph Neural Networks for capturing dependencies. For dynamic updates, LPA* (Lifelong Planning A*) enables efficient replanning by reusing previous computations.

The implementation roadmap prioritizes Python with NetworkX for graphs, GeoPandas for spatial data, Google OR-Tools for optimization, and Foursquare/OpenStreetMap for POI data. The TravelPlanner benchmark (2024) provides 1,225 curated queries where current language agents achieve only 0.6% success, highlighting the complexity and opportunity for algorithmic improvements. The six-month timeline allocates equal effort across foundation/research (months 1-2), core implementation (months 3-4), and evaluation/writing (months 5-6).

## 3. Annotated Bibliography

### Interactive Planning

- **S. Basu Roy, G. Das, S. Amer-Yahia, and C. Yu**, "Interactive Itinerary Planning," in *Proceedings of the 27th IEEE International Conference on Data Engineering (ICDE)*, Hannover, Germany, 2011, pp. 15-26. DOI: [10.1109/ICDE.2011.5767920](https://doi.org/10.1109/ICDE.2011.5767920)  
  *Foundational paper introducing interactive itinerary planning as an iterative process with user feedback. Proves NP-completeness and develops efficient heuristics including GreedyPOISelection and HeapPrunGreedyPOI algorithms. Essential baseline for thesis implementation.*  
  **Category:** Interactive_Planning

- **L. Liao, R. Takanobu, Y. Ma, X. Yang, M. Huang, and T.-S. Chua**, "Deep Conversational Recommender in Travel," *arXiv preprint arXiv:1907.00710*, 2019. URL: [https://arxiv.org/abs/1907.00710](https://arxiv.org/abs/1907.00710)  
  *Proposes DCR system using sequence-to-sequence models with neural latent topics and Graph Convolutional Networks. Enables natural language dialogue for interactive travel planning across multiple sub-tasks.*  
  **Category:** Interactive_Planning

- **A. Yahi, A. Chassang, L. Raynaud, H. Duthil, and D. H. Chau**, "Aurigo: An Interactive Tour Planner for Personalized Itineraries," in *Proceedings of the 20th International Conference on Intelligent User Interfaces (IUI)*, Atlanta, GA, USA, 2015, pp. 275-285. DOI: [10.1145/2678025.2701366](https://doi.org/10.1145/2678025.2701366)  
  *Hybrid approach combining recommendation algorithms with interactive visualization. User studies show 70% preference over Google Maps for personalized itinerary creation.*  
  **Category:** Interactive_Planning

- **M. Xie, L. V. S. Lakshmanan, and P. T. Wood**, "CompRec-Trip: A Composite Recommendation System for Travel Planning," in *Proceedings of the 27th IEEE International Conference on Data Engineering (ICDE)*, Hannover, Germany, 2011, pp. 1352-1355.  
  *Automatic generation of composite travel recommendations with rich GUI for customization. Allows flexible package configuration within cost and time budgets.*  
  **Category:** Interactive_Planning

### Ranking Algorithms

- **J. Xie et al.**, "TravelPlanner: A Benchmark for Real-World Planning with Language Agents," *arXiv preprint arXiv:2402.01622*, 2024.  
  *Provides 1,225 meticulously curated planning queries with nearly 4 million data records. Current language agents achieve only 0.6% success rate, highlighting complexity of real-world travel planning.*  
  **Category:** Ranking_Algorithms

- **K. Sun et al.**, "Where to Go Next: Modeling Long- and Short-Term User Preferences for Point-of-Interest Recommendation," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 34, no. 01, pp. 214-221, 2020.  
  *Advanced neural architecture for POI recommendation considering both long-term preferences and short-term intentions. Directly applicable to ranking itinerary segments.*  
  **Category:** Ranking_Algorithms

- **J. Gu, C. Song, W. Jiang, X. Wang, and M. Liu**, "Enhancing Personalized Trip Recommendation with Attractive Routes," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 34, no. 01, pp. 399-406, 2020. DOI: [10.1609/aaai.v34i01.5407](https://doi.org/10.1609/aaai.v34i01.5407)  
  *Introduces "Attractive Routes" concept using gravity model in category space. Provides interactive optimization balancing time cost with user experience.*  
  **Category:** Ranking_Algorithms

- **C. Verbeeck, K. Sörensen, E. H. Aghezzaf, and P. Vansteenwegen**, "A Fast Solution Method for the Time-Dependent Orienteering Problem," *European Journal of Operational Research*, vol. 236, no. 2, pp. 419-432, 2014.  
  *Presents efficient algorithms for time-dependent routing crucial for real-time itinerary ranking. Methods directly applicable to dynamic POI scheduling.*  
  **Category:** Ranking_Algorithms

### User Preferences

- **K. H. Lim, J. Chan, C. Leckie, and S. Karunasekera**, "Personalized Trip Recommendation for Tourists Based on User Interests, Points of Interest Visit Durations and Visit Recency," *Knowledge and Information Systems*, vol. 54, no. 2, pp. 375-406, 2018. DOI: [10.1007/s10115-017-1056-y](https://doi.org/10.1007/s10115-017-1056-y)  
  *Comprehensive framework incorporating user interests, POI visit duration patterns, and temporal preferences. Essential for preference-aware ranking implementation.*  
  **Category:** User_Preferences

- **L. Huang, Y. Ma, S. Wang, and Y. Liu**, "An Attention-Based Spatiotemporal LSTM Network for Next POI Recommendation," *IEEE Transactions on Services Computing*, vol. 14, no. 6, pp. 1585-1597, 2019.  
  *Attention mechanisms for capturing user preferences in sequential contexts. Provides neural architecture for learning complex preference patterns.*  
  **Category:** User_Preferences

- **H. A. Rahmani, Y. Deldjoo, and T. di Noia**, "The Role of Context Fusion on Accuracy, Beyond-Accuracy, and Fairness of Point-of-Interest Recommendation Systems," *Expert Systems with Applications*, vol. 205, p. 117718, 2022.  
  *Addresses fairness and context-awareness in preference modeling. Critical for ensuring equitable recommendations across user groups.*  
  **Category:** User_Preferences

- **H.-T. Chang, Y.-M. Chang, and M.-T. Tsai**, "ATIPS: Automatic Travel Itinerary Planning System for Domestic Areas," *Computational Intelligence and Neuroscience*, vol. 2016, 2016.  
  *Implements automatic preference learning that adapts to user behavior over time. Achieved 82% user satisfaction with gradual profile building approach.*  
  **Category:** User_Preferences

### Dynamic Updates

- **C. Liu et al.**, "Spatio-Temporal Hierarchical Adaptive Dispatching for Ridesharing Systems," in *Proceedings of the 32nd ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems*, Atlanta, GA, USA, 2024.  
  *Hierarchical adaptive dispatching with real-time spatio-temporal adjustments. Techniques transferable to dynamic itinerary updates.*  
  **Category:** Dynamic_Updates

- **M. Khodadadian et al.**, "Time Dependent Orienteering Problem with Time Windows and Service Time Dependent Profits," *Computers & Operations Research*, vol. 143, p. 105795, 2023.  
  *Advanced formulation handling time-dependent constraints and profits. Essential for modeling real-world dynamic conditions.*  
  **Category:** Dynamic_Updates

- **Y. Huang, L. Bian, Z. Li, and M. Zhou**, "Deep Learning-Based Itinerary Recommendation with Queuing Time Awareness," *Information Technology & Tourism*, vol. 26, no. 2, pp. 189-215, 2024.  
  *Incorporates real-time queuing information into recommendations. Crucial for popular tourist destinations with variable wait times.*  
  **Category:** Dynamic_Updates

- **A. Chen, X. Ge, Z. Fu, Y. Xiao, and J. Chen**, "TravelAgent: An AI Assistant for Personalized Travel Planning," *arXiv preprint arXiv:2409.08069*, 2024. URL: [https://arxiv.org/abs/2409.08069](https://arxiv.org/abs/2409.08069)  
  *Features advanced memory module with continual learning through user interactions. Demonstrates superior performance in dynamic personalization.*  
  **Category:** Dynamic_Updates

### Evaluation Methods

- **K. H. Lim, J. Chan, S. Karunasekera, and C. Leckie**, "Tour Recommendation and Trip Planning Using Location-Based Social Media: A Survey," *Knowledge and Information Systems*, vol. 60, no. 3, pp. 1247-1275, 2019.  
  *Comprehensive survey of evaluation methodologies in tour recommendation. Provides taxonomy of metrics and benchmarking approaches.*  
  **Category:** Evaluation_Methods

- **H. Werneck et al.**, "Points of Interest Recommendations: Methods, Evaluation, and Future Directions," *Information Systems*, vol. 101, p. 101789, 2021.  
  *Systematic analysis of POI recommendation evaluation methods. Essential reference for designing comprehensive evaluation framework.*  
  **Category:** Evaluation_Methods

- **J. Ruiz-Meza and J. R. Montoya-Torres**, "A Systematic Literature Review for the Tourist Trip Design Problem: Extensions, Solution Techniques and Future Research Lines," *Operations Research Perspectives*, vol. 9, p. 100228, 2022.  
  *Recent systematic review covering solution techniques and evaluation approaches. Identifies research gaps directly relevant to thesis objectives.*  
  **Category:** Evaluation_Methods

- **D. Gavalas, C. Konstantopoulos, K. Mastakas, and G. Pantziou**, "A Survey on Algorithmic Approaches for Solving Tourist Trip Design Problems," *Journal of Heuristics*, vol. 20, no. 3, pp. 291-328, 2014.  
  *Foundational survey establishing evaluation criteria for tourist trip design algorithms. Provides comprehensive framework for benchmarking.*  
  **Category:** Evaluation_Methods

### Additional Core References

- **P. Vansteenwegen, W. Souffriau, and D. Van Oudheusden**, "The Orienteering Problem: A Survey," *European Journal of Operational Research*, vol. 209, no. 1, pp. 1-10, 2011.  
  *Essential mathematical foundations for the Orienteering Problem underlying most itinerary optimization formulations.*  
  **Category:** Ranking_Algorithms

- **A. Gunawan, H. C. Lau, and P. Vansteenwegen**, "Orienteering Problem: A Survey of Recent Variants, Solution Approaches and Applications," *European Journal of Operational Research*, vol. 255, no. 2, pp. 315-332, 2016.  
  *Updated survey covering variants directly applicable to dynamic itinerary planning including time-dependent and stochastic versions.*  
  **Category:** Ranking_Algorithms

- **J. L. Sarkar, A. Majumder, C. R. Panigrahi, and S. Roy**, "MULTITOUR: A Multiple Itinerary Tourists Recommendation Engine," *Electronic Commerce Research and Applications*, vol. 40, 2020. DOI: [10.1016/j.elerap.2020.100943](https://doi.org/10.1016/j.elerap.2020.100943)  
  *Algorithm for multiple itinerary recommendations using geo-tagged photos. Demonstrates dynamic sequence-based recommendations.*  
  **Category:** Dynamic_Updates

- **Y. Kurata and T. Hara**, "CT-Planner4: Toward a More User-Friendly Interactive Day-Tour Planner," in *Information and Communication Technologies in Tourism 2014*, Dublin, Ireland, 2014, pp. 73-86. DOI: [10.1007/978-3-319-03973-2_6](https://doi.org/10.1007/978-3-319-03973-2_6)  
  *Implements collaborative planning through iterative cycles with hot-start mechanisms. Validated through international student user tests.*  
  **Category:** Interactive_Planning

- **L. Liao, L. Kennedy, L. Wilcox, and T.-S. Chua**, "Crowd Knowledge Enhanced Multimodal Conversational Assistant in Travel Domain," in *MultiMedia Modeling (MMM)*, 2020, pp. 405-417. DOI: [10.1007/978-3-030-37731-1_33](https://doi.org/10.1007/978-3-030-37731-1_33)  
  *First multimodal conversational system for travel combining text and image inputs. Advances toward human-like interactive planning assistants.*  
  **Category:** Interactive_Planning