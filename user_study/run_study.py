"""
User Study Runner

Main interface for conducting user studies with:
- Participant management
- Task presentation
- Data collection
- Real-time logging
"""

import streamlit as st
import json
import time
from datetime import datetime
from pathlib import Path
import random
from typing import Dict, List
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UserStudyInterface:
    """Streamlit-based user study interface"""
    
    def __init__(self):
        self.scenarios_dir = Path("user_study/scenarios")
        self.results_dir = Path("user_study/results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize session state
        if 'participant_id' not in st.session_state:
            st.session_state.participant_id = None
        if 'current_scenario' not in st.session_state:
            st.session_state.current_scenario = 0
        if 'start_time' not in st.session_state:
            st.session_state.start_time = None
        if 'interactions' not in st.session_state:
            st.session_state.interactions = []
        if 'scenario_order' not in st.session_state:
            st.session_state.scenario_order = None
            
    def load_scenarios(self) -> List[Dict]:
        """Load all scenario files"""
        scenarios = []
        for scenario_file in sorted(self.scenarios_dir.glob("*.json")):
            with open(scenario_file, 'r') as f:
                scenarios.append(json.load(f))
        return scenarios
    
    def save_participant_data(self, data_type: str, data: Dict):
        """Save participant data to appropriate file"""
        participant_dir = self.results_dir / st.session_state.participant_id
        participant_dir.mkdir(exist_ok=True)
        
        if data_type == 'task':
            # Save individual task results
            task_dir = participant_dir / 'tasks'
            task_dir.mkdir(exist_ok=True)
            filename = f"{data['scenario_id']}.json"
            filepath = task_dir / filename
        else:
            # Save other data types
            filename = f"{data_type}.json"
            filepath = participant_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {data_type} data for {st.session_state.participant_id}")
    
    def log_interaction(self, action: str, details: Dict = None):
        """Log user interactions"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'scenario': st.session_state.current_scenario,
            'details': details or {}
        }
        st.session_state.interactions.append(interaction)
    
    def run_pre_study(self):
        """Pre-study questionnaire"""
        st.header("Pre-Study Questionnaire")
        
        with st.form("pre_study_form"):
            # Demographics
            st.subheader("Demographics")
            age_range = st.selectbox(
                "Age Range",
                ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
            )
            
            gender = st.selectbox(
                "Gender (optional)",
                ["Prefer not to say", "Male", "Female", "Non-binary"]
            )
            
            education = st.selectbox(
                "Education Level",
                ["High school", "Bachelor's degree", "Master's degree", 
                 "Doctoral degree", "Other"]
            )
            
            # Travel Experience
            st.subheader("Travel Experience")
            travel_frequency = st.selectbox(
                "How often do you travel for leisure?",
                ["Never", "Once a year", "2-3 times a year", 
                 "4-6 times a year", "More than 6 times a year"]
            )
            
            nyc_visits = st.selectbox(
                "Have you visited New York City?",
                ["Never", "Once", "2-3 times", "4-5 times", "More than 5 times"]
            )
            
            poi_preference = st.select_slider(
                "How many POIs do you typically visit per day?",
                options=["1-2", "3-4", "5-6", "7-8", "More than 8"]
            )
            
            # Planning preferences
            st.subheader("Planning Preferences")
            planning_time = st.selectbox(
                "How much time do you typically spend planning a day's itinerary?",
                ["Less than 15 minutes", "15-30 minutes", "30-60 minutes",
                 "1-2 hours", "More than 2 hours"]
            )
            
            planning_tools = st.multiselect(
                "How do you typically plan your travel itineraries?",
                ["Travel guidebooks", "Online travel websites", "Google Maps",
                 "Travel planning apps", "Social media", "Friends/family",
                 "Travel agents", "No planning - spontaneous"]
            )
            
            submitted = st.form_submit_button("Continue to Training")
            
            if submitted:
                pre_study_data = {
                    'participant_id': st.session_state.participant_id,
                    'timestamp': datetime.now().isoformat(),
                    'age_range': age_range,
                    'gender': gender,
                    'education': education,
                    'travel_frequency': travel_frequency,
                    'nyc_visits': nyc_visits,
                    'poi_preference': poi_preference,
                    'planning_time': planning_time,
                    'planning_tools': planning_tools
                }
                
                self.save_participant_data('pre_study', pre_study_data)
                st.session_state.study_phase = 'training'
                st.rerun()
    
    def run_training(self):
        """System training phase"""
        st.header("System Training")
        
        st.markdown("""
        ### Welcome to the NYC Itinerary Planning System!
        
        In this study, you'll use our system to plan tourist itineraries for different 
        scenarios in New York City.
        
        #### How the System Works:
        1. **Read the Scenario**: Each task presents a specific planning scenario
        2. **Use the Planning Tools**: Search for POIs, adjust routes, set constraints
        3. **Review Your Itinerary**: Check the suggested route and timing
        4. **Make Adjustments**: Modify as needed to meet the scenario requirements
        5. **Submit When Ready**: Save your final itinerary
        
        #### Important:
        - **Think Aloud**: Please verbalize your thoughts as you work
        - **Be Honest**: We want your genuine reactions and feedback
        - **Take Your Time**: There's no rush - quality matters more than speed
        - **Ask Questions**: If something is unclear, please ask
        
        Let's start with a practice scenario!
        """)
        
        if st.button("Start Practice Scenario"):
            st.session_state.study_phase = 'practice'
            st.rerun()
    
    def run_scenario(self, scenario: Dict, is_practice: bool = False):
        """Run a single scenario task"""
        if not is_practice and st.session_state.start_time is None:
            st.session_state.start_time = time.time()
        
        # Scenario header
        st.header(f"{'Practice: ' if is_practice else ''}{scenario['name']}")
        st.markdown(f"**Scenario**: {scenario['description']}")
        
        # Display narrative
        with st.expander("Scenario Details", expanded=True):
            st.write(scenario['narrative'])
            
            # Constraints
            st.subheader("Constraints")
            constraints = scenario['constraints']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Time Available", f"{constraints['duration_hours']}h")
                st.metric("Start Time", constraints['start_time'])
            with col2:
                st.metric("Budget", f"${constraints['budget']}")
                st.metric("End Time", constraints['end_time'])
            with col3:
                st.metric("Walking Limit", f"{constraints['walking_limit_km']}km")
                st.metric("Start/End", constraints['start_location'])
        
        # Task instructions
        st.info(scenario['task_instructions'])
        
        # Placeholder for actual planning interface
        st.markdown("---")
        st.subheader("Planning Interface")
        st.warning("Planning interface would be integrated here")
        
        # Simulated planning results
        with st.form("scenario_completion"):
            st.subheader("Your Itinerary")
            
            # Satisfaction rating
            satisfaction = st.slider(
                "How satisfied are you with this itinerary?",
                min_value=1, max_value=10, value=7
            )
            
            # Success checkbox
            requirements_met = st.checkbox(
                "I believe this itinerary meets all the scenario requirements"
            )
            
            # Algorithm selection (simulated)
            algorithm = st.selectbox(
                "Which algorithm did you use?",
                ["Greedy", "HeapPrunGreedy", "A*", "Two-Phase", "Auto"]
            )
            
            # Additional notes
            notes = st.text_area("Any additional comments about this scenario?")
            
            submitted = st.form_submit_button("Submit Itinerary")
            
            if submitted:
                completion_time = time.time() - st.session_state.start_time if st.session_state.start_time else 0
                
                task_data = {
                    'participant_id': st.session_state.participant_id,
                    'scenario_id': scenario['scenario_id'],
                    'timestamp': datetime.now().isoformat(),
                    'completion_time': completion_time,
                    'satisfaction_rating': satisfaction,
                    'requirements_met': requirements_met,
                    'algorithm_used': algorithm,
                    'notes': notes,
                    'success': requirements_met and satisfaction >= 7,
                    'final_itinerary': {
                        'stops': scenario.get('baseline_itinerary', {}).get('stops', []),
                        'total_cost': scenario.get('baseline_itinerary', {}).get('total_cost', 0),
                        'total_distance_km': scenario.get('baseline_itinerary', {}).get('total_distance_km', 0)
                    }
                }
                
                if not is_practice:
                    self.save_participant_data('task', task_data)
                    st.session_state.current_scenario += 1
                    st.session_state.start_time = None
                else:
                    st.session_state.study_phase = 'main_tasks'
                
                st.rerun()
    
    def run_post_study(self):
        """Post-study questionnaire"""
        st.header("Post-Study Questionnaire")
        
        with st.form("post_study_form"):
            # SUS questions
            st.subheader("System Usability Scale")
            st.write("Please rate your agreement with each statement (1=Strongly Disagree, 5=Strongly Agree)")
            
            sus_questions = [
                "I think that I would like to use this system frequently",
                "I found the system unnecessarily complex",
                "I thought the system was easy to use",
                "I think that I would need the support of a technical person to be able to use this system",
                "I found the various functions in this system were well integrated",
                "I thought there was too much inconsistency in this system",
                "I would imagine that most people would learn to use this system very quickly",
                "I found the system very cumbersome to use",
                "I felt very confident using the system",
                "I needed to learn a lot of things before I could get going with this system"
            ]
            
            sus_responses = []
            for i, question in enumerate(sus_questions):
                response = st.slider(question, 1, 5, 3, key=f"sus_{i}")
                sus_responses.append(response)
            
            # Overall satisfaction
            st.subheader("Overall Experience")
            overall_satisfaction = st.selectbox(
                "Overall, how satisfied were you with the itineraries created?",
                ["Very dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very satisfied"]
            )
            
            comparison = st.selectbox(
                "How do the generated itineraries compare to what you would plan manually?",
                ["Much worse", "Somewhat worse", "About the same", "Somewhat better", "Much better"]
            )
            
            would_use = st.selectbox(
                "Would you use this system for planning real trips?",
                ["Definitely not", "Probably not", "Not sure", "Probably yes", "Definitely yes"]
            )
            
            would_recommend = st.selectbox(
                "Would you recommend this system to others?",
                ["Definitely not", "Probably not", "Not sure", "Probably yes", "Definitely yes"]
            )
            
            # Open feedback
            liked_most = st.text_area("What did you like most about the system?")
            liked_least = st.text_area("What did you like least about the system?")
            feature_requests = st.text_area("What features would you add or change?")
            
            submitted = st.form_submit_button("Complete Study")
            
            if submitted:
                post_study_data = {
                    'participant_id': st.session_state.participant_id,
                    'timestamp': datetime.now().isoformat(),
                    'sus_responses': sus_responses,
                    'overall_satisfaction': overall_satisfaction,
                    'comparison': comparison,
                    'would_use': would_use,
                    'would_recommend': would_recommend,
                    'liked_most': liked_most,
                    'liked_least': liked_least,
                    'feature_requests': feature_requests
                }
                
                self.save_participant_data('post_study', post_study_data)
                
                # Save interaction log
                self.save_participant_data('interactions', {
                    'participant_id': st.session_state.participant_id,
                    'interactions': st.session_state.interactions
                })
                
                st.session_state.study_phase = 'complete'
                st.rerun()
    
    def run(self):
        """Main study runner"""
        st.set_page_config(
            page_title="NYC Itinerary Planning User Study",
            page_icon="🗽",
            layout="wide"
        )
        
        st.title("NYC Itinerary Planning User Study")
        
        # Participant ID entry
        if st.session_state.participant_id is None:
            st.header("Welcome!")
            participant_id = st.text_input(
                "Please enter your participant ID (e.g., P001):",
                max_chars=4
            )
            
            if st.button("Begin Study") and participant_id:
                st.session_state.participant_id = participant_id
                st.session_state.study_phase = 'pre_study'
                
                # Load randomized scenario order
                scenarios = self.load_scenarios()
                st.session_state.scenario_order = list(range(len(scenarios)))
                random.shuffle(st.session_state.scenario_order)
                
                logger.info(f"Study started for participant {participant_id}")
                st.rerun()
        
        else:
            # Display participant ID in sidebar
            st.sidebar.write(f"**Participant**: {st.session_state.participant_id}")
            
            # Study phase management
            if 'study_phase' not in st.session_state:
                st.session_state.study_phase = 'pre_study'
            
            phase = st.session_state.study_phase
            
            if phase == 'pre_study':
                self.run_pre_study()
                
            elif phase == 'training':
                self.run_training()
                
            elif phase == 'practice':
                # Load practice scenario (using museum tour as practice)
                scenarios = self.load_scenarios()
                practice_scenario = scenarios[0]
                self.run_scenario(practice_scenario, is_practice=True)
                
            elif phase == 'main_tasks':
                scenarios = self.load_scenarios()
                
                if st.session_state.current_scenario < len(scenarios):
                    # Show progress
                    progress = st.session_state.current_scenario / len(scenarios)
                    st.progress(progress)
                    st.write(f"Scenario {st.session_state.current_scenario + 1} of {len(scenarios)}")
                    
                    # Get current scenario
                    scenario_idx = st.session_state.scenario_order[st.session_state.current_scenario]
                    current_scenario = scenarios[scenario_idx]
                    
                    self.run_scenario(current_scenario)
                else:
                    # All scenarios complete
                    st.session_state.study_phase = 'post_study'
                    st.rerun()
                    
            elif phase == 'post_study':
                self.run_post_study()
                
            elif phase == 'complete':
                st.success("Study Complete!")
                st.balloons()
                
                st.markdown("""
                ### Thank you for participating!
                
                Your responses have been saved successfully.
                
                **Next Steps:**
                1. You will receive your compensation within 24 hours
                2. A personalized NYC itinerary will be sent to your email
                3. If you opted in, you'll receive the study results when complete
                
                Please let the researcher know you have finished.
                """)
                
                # Clear session state for next participant
                if st.button("Reset for Next Participant"):
                    for key in st.session_state.keys():
                        del st.session_state[key]
                    st.rerun()


def main():
    """Run the user study interface"""
    interface = UserStudyInterface()
    interface.run()


if __name__ == "__main__":
    main()