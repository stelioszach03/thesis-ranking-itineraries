# Chapter 2: Literature Review

## 2.1 Introduction

The challenge of planning optimal tourist itineraries has evolved from simple shortest-path problems to sophisticated multi-objective optimization tasks that must balance user preferences, practical constraints, and dynamic urban environments. This chapter provides a comprehensive review of the literature that informs our quality-based approach to itinerary ranking, tracing the evolution of the field through five key dimensions: interactive planning paradigms, ranking algorithms, user preference modeling, dynamic update mechanisms, and evaluation methodologies. Through this review, we identify critical gaps in existing work and position our contributions within the broader research landscape.

## 2.2 Evolution of Interactive Planning

### 2.2.1 Foundational Work: The Three-Step Paradigm

The modern era of interactive itinerary planning began with the seminal work of Basu Roy et al. [1], who formulated the problem as a three-step iterative process: (1) initial POI recommendation based on user preferences, (2) user feedback through selection/deselection of suggested POIs, and (3) route optimization given the selected destinations. This work proved that even simplified versions of the problem are NP-complete, justifying the need for heuristic approaches. Their GreedyPOISelection and HeapPrunGreedyPOI algorithms achieved O(n²) complexity, making real-time interaction feasible.

The key insight from [1] was recognizing that static, one-shot planning fails to capture the exploratory nature of travel planning. Users often don't know exactly what they want until they see options, and their preferences may evolve during the planning process. However, their framework assumed that maximizing POI coverage leads to higher satisfaction—an assumption we challenge in this thesis.

### 2.2.2 From Algorithms to Interfaces

Yahi et al. [3] extended the interactive planning paradigm by combining algorithmic recommendations with visual interfaces in their Aurigo system. Their user studies revealed a crucial finding: 70% of users preferred their interactive visualization approach over Google Maps for personalized tour planning. This work highlighted that the quality of interaction matters as much as the quality of algorithms—users need to understand and trust the recommendations to accept them.

The CompRec-Trip system by Xie et al. [4] further advanced interactive planning by integrating multiple travel components (flights, hotels, attractions) into composite packages. While their focus extended beyond pure itinerary planning, their flexible GUI for package customization demonstrated the importance of user control in achieving satisfactory travel plans.

### 2.2.3 Modern Conversational Approaches

Recent advances in natural language processing have enabled a new generation of conversational travel planners. Liao et al. [2] proposed the Deep Conversational Recommender (DCR) system, which uses sequence-to-sequence models enhanced with Graph Convolutional Networks to enable natural dialogue for travel planning. Their approach allows users to express preferences in natural language ("I want a romantic dinner after the museum") rather than through formal constraints.

The progression from Liao's text-based system [2] to their later multimodal work [26] represents another leap in interaction capability. By accepting both text and image inputs ("I want to visit places like this [image]"), these systems move closer to how humans naturally communicate about travel desires.

Chen et al.'s TravelAgent [17] incorporates a memory module that learns from user interactions over time, demonstrating superior performance in dynamic personalization. This continual learning approach addresses a key limitation of earlier systems that treated each planning session in isolation.

### 2.2.4 Collaborative and Social Dimensions

Kurata and Hara's CT-Planner4 [25] introduced collaborative planning mechanisms, recognizing that travel planning often involves multiple stakeholders with different preferences. Their iterative refinement process with "hot-start" mechanisms reduced planning time by 60% in group scenarios.

The integration of crowd knowledge, as demonstrated in multimodal systems [26], adds another dimension to interactive planning. By leveraging experiences from previous travelers, these systems can provide insights beyond what's available in static databases ("locals recommend visiting this restaurant after 8 PM to avoid tourist crowds").

## 2.3 Ranking Algorithms: From Coverage to Quality

### 2.3.1 The Orienteering Problem Foundation

The mathematical foundation for most itinerary planning algorithms comes from the Orienteering Problem (OP), comprehensively surveyed by Vansteenwegen et al. [22]. The classical OP seeks to find a path that maximizes collected scores while respecting time/distance constraints. This formulation naturally led to coverage-based approaches—visit as many high-value POIs as possible.

Gunawan et al.'s updated survey [23] revealed the proliferation of OP variants relevant to tourism: time-dependent OP (TDOP) for handling varying travel times, OP with time windows (OPTW) for business hours constraints, and stochastic OP for uncertain conditions. However, these variants still focus on score maximization rather than holistic quality evaluation.

### 2.3.2 The Reality Check: TravelPlanner Benchmark

The TravelPlanner benchmark by Xie et al. [5] provided a sobering reality check for the field. With 1,225 meticulously curated queries and nearly 4 million data records, it revealed that state-of-the-art language agents achieve only 0.6% success rate on real-world planning tasks. This dramatic failure rate highlights the gap between academic problem formulations and real-world complexity.

The benchmark's analysis revealed common failure modes: inability to handle multiple constraints simultaneously, poor temporal reasoning, and failure to consider practical factors like business hours or ticket availability. These findings motivated our focus on quality-based ranking that considers feasibility as a first-class concern.

### 2.3.3 Neural Approaches to Ranking

Sun et al. [6] advanced POI recommendation by modeling both long-term user preferences and short-term intentions using LSTM networks with attention mechanisms. Their insight that next-POI prediction requires understanding both "who you are" (long-term) and "what you're doing now" (short-term) directly informs our approach to contextual quality evaluation.

The attention mechanisms pioneered in [11] for spatio-temporal POI sequences provide a technical foundation for understanding which factors matter most in different contexts. A museum enthusiast might prioritize cultural density in the morning but restaurant quality by evening—patterns that attention weights can capture and model.

### 2.3.4 Beyond Points: Route Attractiveness

Gu et al. [7] made a crucial conceptual leap by introducing "Attractive Routes" that go beyond simple POI coverage. Using a gravity model in category space, they showed that the journey between POIs matters as much as the destinations themselves. A route passing through vibrant neighborhoods provides more satisfaction than one requiring long subway rides between isolated attractions.

Their interactive optimization framework balances time efficiency with experiential quality, allowing users to express preferences like "I prefer walking through interesting areas even if it takes longer." This work directly influenced our multi-objective approach to quality evaluation.

### 2.3.5 Handling Temporal Complexity

Verbeeck et al. [8] addressed the critical issue of time-dependent travel times in urban environments. Their Variable Neighborhood Search algorithm for the Time-Dependent Orienteering Problem handles realistic scenarios where rush hour traffic or event schedules affect optimal routes.

Khodadadian et al. [15] extended this work to include time-dependent profits—recognizing that a restaurant might be more valuable at dinner time than breakfast, or that sunset views from the Empire State Building provide different value than morning visits. These temporal dynamics are crucial for quality-based ranking in urban environments.

## 2.4 Understanding User Preferences

### 2.4.1 Multi-Dimensional Preference Models

Lim et al.'s comprehensive framework [10] revealed the multi-dimensional nature of tourist preferences. Their analysis of Flickr photo trajectories from eight cities showed that users consistently prefer 3-7 POIs per day, with satisfaction dropping sharply outside this range. They identified three key dimensions of preferences:
- **Interest alignment**: How well POIs match stated preferences (weighted at 0.35)
- **Visit duration patterns**: Realistic time allocation for different POI types
- **Temporal preferences**: Time-of-day and day-of-week patterns

Their work provides empirical justification for our quality-based approach—users explicitly prefer well-paced, interest-aligned itineraries over maximum coverage tours.

### 2.4.2 Learning and Adaptation

The ATIPS system by Chang et al. [13] demonstrated the value of adaptive preference learning, achieving 82% user satisfaction through gradual profile building. Their key insight was that users often cannot articulate preferences explicitly but reveal them through choices. A user who consistently chooses art museums over history museums reveals a preference that might not have been stated initially.

Huang et al.'s attention-based LSTM architecture [11] advanced preference learning by capturing complex spatio-temporal patterns. Their model learns that a user who visits morning markets might prefer authentic local restaurants for lunch—patterns too subtle for explicit rules but learnable from behavior.

### 2.4.3 Fairness and Diversity

Rahmani et al. [12] raised critical questions about fairness in POI recommendation systems. Their work showed that pure accuracy optimization can lead to filter bubbles, where users only see popular mainstream attractions. They demonstrated that incorporating diversity and fairness constraints actually improves long-term user satisfaction—users appreciate discovering hidden gems alongside must-see landmarks.

This finding directly influenced our diversity component in the quality evaluation framework. A high-quality itinerary should expose users to varied experiences while respecting their core preferences.

### 2.4.4 Context-Aware Preferences

Context fusion, as explored in [12], reveals that preferences are not static but highly contextual. A business traveler extending their trip for leisure has different preferences than a dedicated tourist. Weather conditions, group composition, and even mood can significantly affect what constitutes a good itinerary.

The challenge lies in capturing these contextual variations without overwhelming users with questions. Modern systems achieve this through implicit signals: time of booking, search patterns, and selective feedback on recommendations.

## 2.5 Dynamic Updates and Real-Time Adaptation

### 2.5.1 From Static to Dynamic Planning

Traditional itinerary planning assumed static conditions—once optimized, execute as planned. However, urban tourism is inherently dynamic. Liu et al.'s work on spatio-temporal hierarchical adaptive dispatching [14] demonstrated techniques for city-scale optimization with millisecond response times. Their hierarchical decomposition approach, while designed for ridesharing, provides a blueprint for handling dynamic itinerary updates at scale.

The key insight is that complete re-optimization is neither necessary nor desirable. Users have mental commitment to their plans, and radical changes create cognitive burden. Instead, incremental adjustments that preserve plan structure while adapting to conditions provide better user experience.

### 2.5.2 Modeling Temporal Dynamics

Huang et al. [16] made a crucial contribution by incorporating real-time queuing information into itinerary recommendations. For popular attractions like the Statue of Liberty or Empire State Building, wait times can vary from minutes to hours based on time, season, and events. Their deep learning approach predicts queue lengths and adjusts recommendations accordingly.

This work highlights a critical gap in static planning approaches: a theoretically optimal plan that ignores real-world queuing becomes a frustrating experience of unexpected waits. Quality-based ranking must consider not just travel time but total time investment including probable waiting.

### 2.5.3 Incremental Planning Algorithms

The application of Lifelong Planning A* (LPA*) to itinerary planning represents a significant algorithmic advance. Unlike traditional A* that starts fresh with each change, LPA* reuses previous search effort, updating only affected portions of the solution. For typical urban changes (a subway delay, a closed restaurant), LPA* can update plans 10-100× faster than replanning from scratch.

Sarkar et al.'s MULTITOUR system [24] demonstrated another approach to dynamic planning by generating multiple diverse itineraries upfront. When conditions change, users can switch to pre-computed alternatives rather than waiting for replanning. This redundancy approach trades space for response time—a reasonable trade-off given modern computing resources.

### 2.5.4 Learning from Disruptions

Chen et al.'s TravelAgent [17] introduced continual learning mechanisms that improve system performance over time. By tracking which recommendations succeed or fail under different conditions, the system builds a knowledge base of robustness patterns. For example, learning that outdoor attractions should have indoor alternatives during uncertain weather, or that popular restaurants need backup options for peak times.

This learning approach transforms disruptions from failures into learning opportunities, gradually improving the system's ability to generate robust, high-quality itineraries.

## 2.6 Evaluation Methods and Metrics

### 2.6.1 Evolution of Evaluation Frameworks

Lim et al.'s survey [18] traced the evolution of evaluation methods from simple efficiency metrics to comprehensive frameworks considering user satisfaction. Early work focused on algorithmic metrics: solution quality versus optimal, computation time, and constraint satisfaction. However, these metrics poorly predicted actual user satisfaction.

Modern evaluation recognizes multiple stakeholder perspectives: tourists want enjoyable experiences, businesses want visitor distribution, and cities want sustainable tourism. Our quality-based framework attempts to balance these perspectives through multi-objective optimization.

### 2.6.2 Beyond Accuracy Metrics

Werneck et al. [19] systematically analyzed the gap between academic metrics and real-world performance in POI recommendation. They found that precision@K and recall metrics, while mathematically convenient, fail to capture important aspects like diversity, novelty, and serendipity. A system that always recommends the top-10 most popular attractions might achieve high precision but provides poor user experience.

Their proposed evaluation framework includes:
- **Coverage**: Geographical and categorical diversity
- **Novelty**: Balance between known and new experiences  
- **Personalization**: Adaptation to individual preferences
- **Robustness**: Performance under uncertain conditions

### 2.6.3 Systematic Evaluation Approaches

Ruiz-Meza and Montoya-Torres [20] provided a recent systematic review identifying evaluation gaps in tourist trip design. They found that only 15% of papers conduct user studies, with most relying solely on computational metrics. Furthermore, real-world deployment evaluations are virtually absent from academic literature.

This gap motivated our comprehensive evaluation approach combining algorithmic benchmarks, user studies, and real-world scenario testing on NYC data. We argue that quality-based ranking cannot be validated through computational metrics alone—human judgment remains the ultimate arbiter of itinerary quality.

### 2.6.4 Benchmark Standardization

Gavalas et al. [21] established foundational benchmarking practices for tourist trip design algorithms, proposing standard test instances and evaluation protocols. However, these benchmarks used synthetic data that failed to capture real-world complexity—uniform POI distributions, simplified travel times, and absent temporal constraints.

The TravelPlanner benchmark [5] represents a new generation of realistic benchmarks, using actual city data with full complexity: real business hours, accurate travel times, seasonal variations, and multi-modal transportation options. Our NYC-focused benchmark extends this approach with even richer contextual data and quality-oriented evaluation metrics.

## 2.7 Synthesis and Research Gaps

### 2.7.1 The Quality Gap

Despite algorithmic advances, existing work predominantly focuses on coverage optimization—visiting more POIs within constraints. While mathematically tractable, this approach misaligns with user preferences for thoughtful, well-paced experiences. Our quality-based ranking framework addresses this gap by explicitly modeling experiential factors beyond simple POI accumulation.

### 2.7.2 The Interaction Gap  

Current interactive systems excel at preference elicitation or route visualization but rarely integrate both seamlessly. Users need systems that learn from implicit feedback while providing explicit control when desired. Our approach combines algorithmic intelligence with user agency, allowing both guided exploration and direct manipulation.

### 2.7.3 The Adaptation Gap

Real-world tourism involves constant adaptation—plans change, conditions vary, preferences evolve. Yet most algorithms assume static scenarios, with dynamic updates treated as edge cases rather than core functionality. Our LPA*-based approach makes adaptability a first-class concern, enabling fluid response to changing conditions.

### 2.7.4 The Evaluation Gap

The 0.6% success rate on realistic benchmarks [5] reveals fundamental evaluation gaps. Academic metrics fail to predict real-world performance, while user studies often test simplified scenarios. Our evaluation framework combines computational efficiency metrics with user satisfaction measures and real-world scenario testing.

## 2.8 Conclusion

This literature review reveals a field in transition, moving from algorithmic optimization of simplified models toward systems that embrace real-world complexity. The evolution from Basu Roy et al.'s foundational work [1] through modern neural approaches [6,11] and conversational systems [2,17] shows increasing sophistication in handling user preferences and urban dynamics.

However, critical gaps remain. The focus on coverage over quality, static over dynamic planning, and algorithmic over experiential metrics limits current systems' real-world effectiveness. Our quality-based ranking framework, supported by dynamic adaptation algorithms and comprehensive evaluation methods, addresses these gaps to create more satisfying tourist experiences.

The next chapter presents our methodology for quality-based itinerary ranking, building on these literature insights while addressing identified gaps. We show how composite satisfaction scores, incremental planning algorithms, and user-centered evaluation can transform itinerary planning from an optimization problem into a quality-driven experience design challenge.

## References

[1] S. Basu Roy, G. Das, S. Amer-Yahia, and C. Yu, "Interactive Itinerary Planning," in *Proceedings of the 27th IEEE International Conference on Data Engineering (ICDE)*, Hannover, Germany, 2011, pp. 15-26.

[2] L. Liao, R. Takanobu, Y. Ma, X. Yang, M. Huang, and T.-S. Chua, "Deep Conversational Recommender in Travel," *arXiv preprint arXiv:1907.00710*, 2019.

[3] A. Yahi, A. Chassang, L. Raynaud, H. Duthil, and D. H. Chau, "Aurigo: An Interactive Tour Planner for Personalized Itineraries," in *Proceedings of the 20th International Conference on Intelligent User Interfaces (IUI)*, Atlanta, GA, USA, 2015, pp. 275-285.

[4] M. Xie, L. V. S. Lakshmanan, and P. T. Wood, "CompRec-Trip: A Composite Recommendation System for Travel Planning," in *Proceedings of the 27th IEEE International Conference on Data Engineering (ICDE)*, Hannover, Germany, 2011, pp. 1352-1355.

[5] J. Xie et al., "TravelPlanner: A Benchmark for Real-World Planning with Language Agents," *arXiv preprint arXiv:2402.01622*, 2024.

[6] K. Sun et al., "Where to Go Next: Modeling Long- and Short-Term User Preferences for Point-of-Interest Recommendation," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 34, no. 01, pp. 214-221, 2020.

[7] J. Gu, C. Song, W. Jiang, X. Wang, and M. Liu, "Enhancing Personalized Trip Recommendation with Attractive Routes," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 34, no. 01, pp. 399-406, 2020.

[8] C. Verbeeck, K. Sörensen, E. H. Aghezzaf, and P. Vansteenwegen, "A Fast Solution Method for the Time-Dependent Orienteering Problem," *European Journal of Operational Research*, vol. 236, no. 2, pp. 419-432, 2014.

[9] K. H. Lim, J. Chan, C. Leckie, and S. Karunasekera, "Personalized Trip Recommendation for Tourists Based on User Interests, Points of Interest Visit Durations and Visit Recency," *Knowledge and Information Systems*, vol. 54, no. 2, pp. 375-406, 2018.

[10] K. H. Lim, J. Chan, C. Leckie, and S. Karunasekera, "Personalized Trip Recommendation for Tourists Based on User Interests, Points of Interest Visit Durations and Visit Recency," *Knowledge and Information Systems*, vol. 54, no. 2, pp. 375-406, 2018.

[11] L. Huang, Y. Ma, S. Wang, and Y. Liu, "An Attention-Based Spatiotemporal LSTM Network for Next POI Recommendation," *IEEE Transactions on Services Computing*, vol. 14, no. 6, pp. 1585-1597, 2019.

[12] H. A. Rahmani, Y. Deldjoo, and T. di Noia, "The Role of Context Fusion on Accuracy, Beyond-Accuracy, and Fairness of Point-of-Interest Recommendation Systems," *Expert Systems with Applications*, vol. 205, p. 117718, 2022.

[13] H.-T. Chang, Y.-M. Chang, and M.-T. Tsai, "ATIPS: Automatic Travel Itinerary Planning System for Domestic Areas," *Computational Intelligence and Neuroscience*, vol. 2016, 2016.

[14] C. Liu et al., "Spatio-Temporal Hierarchical Adaptive Dispatching for Ridesharing Systems," in *Proceedings of the 32nd ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems*, Atlanta, GA, USA, 2024.

[15] M. Khodadadian et al., "Time Dependent Orienteering Problem with Time Windows and Service Time Dependent Profits," *Computers & Operations Research*, vol. 143, p. 105795, 2023.

[16] Y. Huang, L. Bian, Z. Li, and M. Zhou, "Deep Learning-Based Itinerary Recommendation with Queuing Time Awareness," *Information Technology & Tourism*, vol. 26, no. 2, pp. 189-215, 2024.

[17] A. Chen, X. Ge, Z. Fu, Y. Xiao, and J. Chen, "TravelAgent: An AI Assistant for Personalized Travel Planning," *arXiv preprint arXiv:2409.08069*, 2024.

[18] K. H. Lim, J. Chan, S. Karunasekera, and C. Leckie, "Tour Recommendation and Trip Planning Using Location-Based Social Media: A Survey," *Knowledge and Information Systems*, vol. 60, no. 3, pp. 1247-1275, 2019.

[19] H. Werneck et al., "Points of Interest Recommendations: Methods, Evaluation, and Future Directions," *Information Systems*, vol. 101, p. 101789, 2021.

[20] J. Ruiz-Meza and J. R. Montoya-Torres, "A Systematic Literature Review for the Tourist Trip Design Problem: Extensions, Solution Techniques and Future Research Lines," *Operations Research Perspectives*, vol. 9, p. 100228, 2022.

[21] D. Gavalas, C. Konstantopoulos, K. Mastakas, and G. Pantziou, "A Survey on Algorithmic Approaches for Solving Tourist Trip Design Problems," *Journal of Heuristics*, vol. 20, no. 3, pp. 291-328, 2014.

[22] P. Vansteenwegen, W. Souffriau, and D. Van Oudheusden, "The Orienteering Problem: A Survey," *European Journal of Operational Research*, vol. 209, no. 1, pp. 1-10, 2011.

[23] A. Gunawan, H. C. Lau, and P. Vansteenwegen, "Orienteering Problem: A Survey of Recent Variants, Solution Approaches and Applications," *European Journal of Operational Research*, vol. 255, no. 2, pp. 315-332, 2016.

[24] J. L. Sarkar, A. Majumder, C. R. Panigrahi, and S. Roy, "MULTITOUR: A Multiple Itinerary Tourists Recommendation Engine," *Electronic Commerce Research and Applications*, vol. 40, 2020.

[25] Y. Kurata and T. Hara, "CT-Planner4: Toward a More User-Friendly Interactive Day-Tour Planner," in *Information and Communication Technologies in Tourism 2014*, Dublin, Ireland, 2014, pp. 73-86.

[26] L. Liao, L. Kennedy, L. Wilcox, and T.-S. Chua, "Crowd Knowledge Enhanced Multimodal Conversational Assistant in Travel Domain," in *MultiMedia Modeling (MMM)*, 2020, pp. 405-417.