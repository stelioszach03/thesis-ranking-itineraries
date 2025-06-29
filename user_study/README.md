# User Study Protocol

This directory contains materials for conducting user evaluation of the NYC itinerary planning system, following research findings that users prefer 3-7 POIs per day with attractiveness weighted at 0.35.

## Study Overview

**Research Question**: How effectively do our dynamic algorithms meet user preferences compared to baseline approaches?

**Target**: n=30 participants (following research standards)
**Baseline**: 82% user satisfaction from literature
**Duration**: 45-60 minutes per session

## Directory Structure

```
user_study/
├── ethics/                 # Ethics compliance documents
│   ├── GDPR_compliance.md
│   ├── consent_form.md
│   └── ethics_approval_template.md
├── scenarios/             # NYC-specific evaluation tasks
│   ├── museum_tour.json
│   ├── broadway_dinner.json
│   ├── family_day.json
│   ├── budget_tour.json
│   ├── rainy_day.json
│   └── accessible_tour.json
├── evaluation/            # Evaluation protocols
│   ├── pre_study_questionnaire.md
│   ├── task_protocol.md
│   ├── post_task_questionnaire.md
│   └── final_interview_guide.md
├── analysis/              # Analysis scripts
│   ├── calculate_metrics.py
│   ├── statistical_analysis.py
│   └── generate_report.py
└── results/              # Study results (not in repo)
    └── .gitkeep
```

## Study Protocol

### 1. Pre-Study (10 min)
- Consent form signing
- Demographics questionnaire
- Travel preference assessment

### 2. Training (5 min)
- System introduction
- Practice scenario

### 3. Main Tasks (30 min)
- 6 NYC scenarios (5 min each)
- Think-aloud protocol
- Screen recording

### 4. Post-Study (10 min)
- System Usability Scale (SUS)
- CSS component ratings
- Semi-structured interview

## Evaluation Metrics

Based on research_context.md findings:

1. **Objective Metrics**
   - Task completion rate
   - Time to complete
   - Number of modifications
   - Algorithm selection choices

2. **Subjective Metrics**
   - CSS components (TUR, SAT, FEA, DIV)
   - Overall satisfaction (1-10 scale)
   - Preference vs manual planning
   - Intent to use

3. **Qualitative Feedback**
   - Feature requests
   - Pain points
   - Unexpected uses

## Running the Study

### Setup
```bash
# Install study dependencies
pip install -r requirements_study.txt

# Generate participant IDs
python analysis/generate_participants.py --n 30

# Prepare scenario randomization
python analysis/randomize_scenarios.py
```

### During Study
```bash
# Launch study interface
python run_study.py --participant P001

# Interface will:
# - Present scenarios in randomized order
# - Log all interactions
# - Record screen with consent
# - Save responses automatically
```

### Analysis
```bash
# Calculate metrics
python analysis/calculate_metrics.py

# Run statistical tests
python analysis/statistical_analysis.py

# Generate report
python analysis/generate_report.py
```

## Recruitment

### Inclusion Criteria
- Age 18-65
- Visited NYC or planning to visit
- Basic computer skills
- English proficiency

### Recruitment Channels
- University mailing lists
- Social media (travel groups)
- Local community boards
- Compensated recruitment services

### Compensation
- $20 Amazon gift card
- Summary of personalized NYC itinerary

## Ethics Compliance

All procedures approved by [Institution] IRB (#pending).
See ethics/ directory for:
- GDPR compliance checklist
- Data handling procedures
- Consent forms
- Withdrawal process

## Contact

Principal Investigator: [Name]
Email: [email]
Ethics Officer: [Name]