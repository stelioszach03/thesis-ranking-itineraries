"""
Streamlit Demo for LPA* Dynamic Itinerary Planning

Run with: streamlit run streamlit_demo.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
from datetime import datetime, timedelta
import time
import os
import sys

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lpa_star import (
    LPAStarPlanner, DynamicUpdate, UpdateType, Constraints
)

# Page configuration
st.set_page_config(
    page_title="LPA* Dynamic Itinerary Planning - NYC",
    page_icon="🗽",
    layout="wide"
)

st.title("🗽 LPA* Dynamic Itinerary Planning for NYC")
st.markdown("""
**Demonstrating efficient replanning with Lifelong Planning A***

Based on research from "Ranking Itineraries: Dynamic algorithms meet user preferences"
- Handles real-time POI closures and subway disruptions
- Reuses 70-90% of previous computations
- Optimized for NYC tourism scenarios
""")

# Initialize session state
if 'planner' not in st.session_state:
    # Check if data exists
    if not os.path.exists('data/nyc_pois.json'):
        st.error("Please run prepare_nyc_data.py first to generate NYC data!")
        st.stop()
    
    # Load NYC data
    with open('data/nyc_pois.json', 'r') as f:
        pois_data = json.load(f)[:50]  # Use subset for demo
    
    if os.path.exists('data/distance_matrix.npy'):
        distance_matrix = np.load('data/distance_matrix.npy')[:50, :50]
    else:
        distance_matrix = None
    
    st.session_state.planner = LPAStarPlanner(pois_data, distance_matrix)
    st.session_state.current_path = None
    st.session_state.updates_applied = []
    st.session_state.performance_history = []
    st.session_state.comparison_mode = False

# Sidebar controls
with st.sidebar:
    st.header("🎛️ Planning Controls")
    
    # User preferences
    st.subheader("User Preferences")
    prefs = {}
    prefs['museum'] = st.slider("Museums 🏛️", 0.0, 1.0, 0.8)
    prefs['park'] = st.slider("Parks 🌳", 0.0, 1.0, 0.7)
    prefs['landmark'] = st.slider("Landmarks 🗽", 0.0, 1.0, 0.6)
    prefs['restaurant'] = st.slider("Restaurants 🍽️", 0.0, 1.0, 0.5)
    prefs['entertainment'] = st.slider("Entertainment 🎭", 0.0, 1.0, 0.4)
    prefs['shopping'] = st.slider("Shopping 🛍️", 0.0, 1.0, 0.3)
    prefs['nature'] = st.slider("Nature 🦆", 0.0, 1.0, 0.6)
    prefs['cultural'] = st.slider("Cultural 🎨", 0.0, 1.0, 0.7)
    
    # Constraints
    st.subheader("Constraints")
    budget = st.number_input("Budget ($)", 50, 500, 150)
    max_pois = st.slider("Max POIs per day", 3, 7, 5)
    transport = st.selectbox("Transportation", 
                           ["public_transit", "walking", "taxi"],
                           help="Walking is limited to 2km between POIs")
    start_time = st.slider("Start time", 6, 12, 9, 
                          help="Hour of day to start (24h format)")
    
    # Plan button
    if st.button("🗺️ Generate Initial Plan", type="primary"):
        with st.spinner("Planning optimal itinerary..."):
            constraints = Constraints(
                budget=budget,
                max_pois=max_pois,
                min_pois=3,
                transportation_mode=transport,
                start_time=float(start_time)
            )
            
            start = time.time()
            path = st.session_state.planner.plan_with_updates(
                prefs, constraints
            )
            elapsed = time.time() - start
            
            st.session_state.current_path = path
            stats = st.session_state.planner.get_statistics()
            st.session_state.performance_history.append({
                'type': 'Initial Plan',
                'time': elapsed,
                'nodes': stats['nodes_updated'],
                'reuse': 0.0
            })
            
            if path:
                st.success(f"✅ Found itinerary with {len(path)} POIs in {elapsed:.3f}s")
            else:
                st.error("❌ No feasible itinerary found with current constraints")
    
    # Performance comparison toggle
    st.session_state.comparison_mode = st.checkbox(
        "Enable Performance Comparison",
        help="Compare LPA* with complete replanning"
    )

# Dynamic updates section
st.header("🚨 Dynamic Updates")

col_update1, col_update2, col_update3 = st.columns(3)

with col_update1:
    st.subheader("📍 POI Updates")
    
    if st.session_state.current_path:
        # POI closure
        poi_to_close = st.selectbox(
            "Select POI to close",
            ["None"] + [f"{poi.name} ({poi.category})" 
                       for poi in st.session_state.current_path]
        )
        
        if st.button("❌ Close POI", disabled=(poi_to_close == "None")):
            # Extract POI name
            poi_name = poi_to_close.split(" (")[0]
            poi = next(p for p in st.session_state.current_path 
                      if p.name == poi_name)
            
            update = DynamicUpdate(
                update_type=UpdateType.POI_CLOSED,
                poi_ids=[poi.id],
                timestamp=datetime.now()
            )
            
            col1, col2 = st.columns(2)
            
            # LPA* replanning
            with col1:
                with st.spinner("LPA* replanning..."):
                    start = time.time()
                    new_path = st.session_state.planner.replan_after_update(update)
                    lpa_time = time.time() - start
                    
                    st.session_state.current_path = new_path
                    st.session_state.updates_applied.append(update)
                    
                    stats = st.session_state.planner.get_statistics()
                    st.session_state.performance_history.append({
                        'type': f'LPA*: Close {poi_name}',
                        'time': lpa_time,
                        'nodes': stats['nodes_updated'],
                        'reuse': stats['computation_reuse']
                    })
                    
                    st.success(f"LPA* completed in {lpa_time:.3f}s")
                    st.metric("Computation Reuse", f"{stats['computation_reuse']:.1%}")
            
            # Comparison with full replanning
            if st.session_state.comparison_mode:
                with col2:
                    with st.spinner("Full replanning..."):
                        # Create new planner for comparison
                        comparison_planner = LPAStarPlanner(
                            st.session_state.planner.pois,
                            st.session_state.planner.distance_matrix
                        )
                        comparison_planner.closed_pois.add(poi.id)
                        
                        start = time.time()
                        comparison_path = comparison_planner.plan_with_updates(
                            prefs, constraints
                        )
                        full_time = time.time() - start
                        
                        st.info(f"Full replan: {full_time:.3f}s")
                        speedup = full_time / lpa_time if lpa_time > 0 else 1
                        st.metric("LPA* Speedup", f"{speedup:.1f}x faster")

with col_update2:
    st.subheader("🚇 Transit Updates")
    
    subway_lines = {
        "4/5/6 (East Side)": ["museum", "park"],
        "N/Q/R/W (Broadway)": ["entertainment", "shopping"],
        "L (14th St)": ["restaurant", "cultural"],
        "A/C/E (8th Ave)": ["landmark", "park"]
    }
    
    line = st.selectbox("Subway line", list(subway_lines.keys()))
    
    if st.button("🚧 Disrupt Line"):
        update = DynamicUpdate(
            update_type=UpdateType.SUBWAY_DISRUPTION,
            poi_ids=[],
            timestamp=datetime.now(),
            details={'lines': [line], 'affected_categories': subway_lines[line]}
        )
        
        with st.spinner(f"Replanning for {line} disruption..."):
            start = time.time()
            new_path = st.session_state.planner.replan_after_update(update)
            elapsed = time.time() - start
            
            st.session_state.current_path = new_path
            st.session_state.updates_applied.append(update)
            
            stats = st.session_state.planner.get_statistics()
            st.session_state.performance_history.append({
                'type': f'Subway: {line}',
                'time': elapsed,
                'nodes': stats['nodes_updated'],
                'reuse': stats['computation_reuse']
            })
            
            st.warning(f"⚠️ {line} disrupted! Replanned in {elapsed:.3f}s")

with col_update3:
    st.subheader("🌦️ Weather/Events")
    
    weather_event = st.selectbox(
        "Scenario",
        ["Hurricane Warning", "Times Square NYE", "Broadway Matinee", "Heat Wave"]
    )
    
    if st.button("🌪️ Apply Scenario"):
        if weather_event == "Hurricane Warning":
            # Close all outdoor attractions
            outdoor_categories = ['park', 'nature']
            outdoor_pois = [p.id for p in st.session_state.planner.pois 
                          if p.category in outdoor_categories]
            
            update = DynamicUpdate(
                update_type=UpdateType.WEATHER_CLOSURE,
                poi_ids=outdoor_pois[:10],  # Limit for demo
                timestamp=datetime.now(),
                duration_hours=48.0,
                details={'weather': 'Hurricane Warning', 'categories': outdoor_categories}
            )
            
        elif weather_event == "Times Square NYE":
            # Close Times Square area
            times_square_pois = [p.id for p in st.session_state.planner.pois 
                               if 40.755 < p.lat < 40.760 and -73.988 < p.lon < -73.983]
            
            update = DynamicUpdate(
                update_type=UpdateType.EVENT_CLOSURE,
                poi_ids=times_square_pois,
                timestamp=datetime.now(),
                duration_hours=12.0,
                details={'event': 'New Years Eve Celebration'}
            )
        
        elif weather_event == "Broadway Matinee":
            # Adjust timing for theater district
            theater_pois = [p.id for p in st.session_state.planner.pois 
                          if p.category == 'entertainment'][:3]
            
            update = DynamicUpdate(
                update_type=UpdateType.COST_CHANGE,
                poi_ids=theater_pois,
                timestamp=datetime.now(),
                details={'cost_factors': {pid: 1.5 for pid in theater_pois}}
            )
        
        else:  # Heat Wave
            # Reduce outdoor visit durations
            outdoor_pois = [p.id for p in st.session_state.planner.pois 
                          if p.category in ['park', 'nature']][:5]
            
            update = DynamicUpdate(
                update_type=UpdateType.COST_CHANGE,
                poi_ids=outdoor_pois,
                timestamp=datetime.now(),
                details={'cost_factors': {pid: 0.5 for pid in outdoor_pois}}
            )
        
        with st.spinner(f"Applying {weather_event}..."):
            start = time.time()
            new_path = st.session_state.planner.replan_after_update(update)
            elapsed = time.time() - start
            
            st.session_state.current_path = new_path
            st.session_state.updates_applied.append(update)
            
            st.info(f"🎯 Adapted to {weather_event} in {elapsed:.3f}s")

# Main content area
st.header("📍 Current Itinerary")

if st.session_state.current_path:
    # Map and details in columns
    col_map, col_details = st.columns([3, 2])
    
    with col_map:
        # Create interactive map
        fig = go.Figure()
        
        # Extract coordinates
        lats = [poi.lat for poi in st.session_state.current_path]
        lons = [poi.lon for poi in st.session_state.current_path]
        names = [f"{i+1}. {poi.name}" for i, poi in enumerate(st.session_state.current_path)]
        categories = [poi.category for poi in st.session_state.current_path]
        
        # Color by category
        color_map = {
            'museum': 'red',
            'park': 'green',
            'restaurant': 'orange',
            'landmark': 'blue',
            'entertainment': 'purple',
            'shopping': 'pink',
            'nature': 'lightgreen',
            'cultural': 'brown'
        }
        colors = [color_map.get(cat, 'gray') for cat in categories]
        
        # Add POI markers
        fig.add_trace(go.Scattermapbox(
            mode="markers+text",
            lon=lons,
            lat=lats,
            marker=dict(size=15, color=colors),
            text=names,
            textposition="top right",
            hovertemplate="<b>%{text}</b><br>Category: %{customdata}<extra></extra>",
            customdata=categories
        ))
        
        # Add route lines
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=lons,
            lat=lats,
            line=dict(width=3, color='blue'),
            hoverinfo='skip'
        ))
        
        # Map layout
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lon=np.mean(lons), lat=np.mean(lats)),
                zoom=12
            ),
            showlegend=False,
            height=500,
            margin=dict(r=0, t=0, l=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_details:
        # Itinerary timeline
        st.subheader("📅 Timeline")
        
        current_time = st.session_state.planner.constraints.start_time
        timeline_data = []
        total_cost = 0.0
        
        for i, poi in enumerate(st.session_state.current_path):
            # Travel time (skip for first POI)
            if i > 0:
                prev_poi = st.session_state.current_path[i-1]
                travel_time = st.session_state.planner._get_travel_time_dynamic(
                    prev_poi, poi, current_time
                )
                current_time += travel_time
            
            # Visit details
            timeline_data.append({
                '⏰ Time': f"{int(current_time):02d}:{int((current_time % 1) * 60):02d}",
                '📍 POI': poi.name,
                '🏷️ Type': poi.category.title(),
                '⏱️ Duration': f"{poi.avg_visit_duration:.1f}h",
                '💵 Cost': f"${poi.entrance_fee:.0f}"
            })
            
            current_time += poi.avg_visit_duration
            total_cost += poi.entrance_fee
        
        st.dataframe(pd.DataFrame(timeline_data), use_container_width=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total POIs", len(st.session_state.current_path))
        with col2:
            st.metric("Total Cost", f"${total_cost:.0f}")
        with col3:
            duration = current_time - st.session_state.planner.constraints.start_time
            st.metric("Duration", f"{duration:.1f}h")

# Performance metrics
if st.session_state.performance_history:
    st.header("📊 Performance Analysis")
    
    perf_df = pd.DataFrame(st.session_state.performance_history)
    
    # Performance chart
    fig_perf = go.Figure()
    
    # Solution time bars
    fig_perf.add_trace(go.Bar(
        x=perf_df.index,
        y=perf_df['time'],
        name='Solution Time (s)',
        marker_color='lightblue',
        yaxis='y'
    ))
    
    # Computation reuse line
    fig_perf.add_trace(go.Scatter(
        x=perf_df.index,
        y=perf_df['reuse'] * 100,
        name='Computation Reuse (%)',
        line=dict(color='green', width=3),
        yaxis='y2',
        mode='lines+markers'
    ))
    
    # Layout
    fig_perf.update_layout(
        title="LPA* Performance Over Updates",
        xaxis=dict(
            title="Update Sequence",
            tickmode='array',
            tickvals=perf_df.index,
            ticktext=[f"{i+1}: {row['type']}" for i, row in perf_df.iterrows()],
            tickangle=-45
        ),
        yaxis=dict(
            title="Time (seconds)",
            side='left'
        ),
        yaxis2=dict(
            title="Reuse %",
            side='right',
            overlaying='y',
            range=[0, 100]
        ),
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # Statistics summary
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    stats = st.session_state.planner.get_statistics()
    
    with col_stat1:
        st.metric("Total Updates", stats['total_updates'])
    with col_stat2:
        st.metric("Avg Reuse", f"{perf_df['reuse'].mean():.1%}")
    with col_stat3:
        st.metric("Avg Time", f"{perf_df['time'].mean():.3f}s")
    with col_stat4:
        if len(perf_df) > 1:
            speedup = perf_df['time'].iloc[0] / perf_df['time'].iloc[1:].mean()
            st.metric("Avg Speedup", f"{speedup:.1f}x")
        else:
            st.metric("Avg Speedup", "N/A")

# Update history
if st.session_state.updates_applied:
    with st.expander("📜 Update History", expanded=False):
        updates_data = []
        for update in st.session_state.updates_applied:
            updates_data.append({
                'Time': update.timestamp.strftime("%H:%M:%S"),
                'Type': update.update_type.value.replace('_', ' ').title(),
                'POIs': len(update.poi_ids),
                'Details': str(update.details.get('event', update.details.get('weather', 'N/A')))
            })
        
        st.dataframe(pd.DataFrame(updates_data), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎓 Based on "Ranking Itineraries: Dynamic algorithms meet user preferences"</p>
    <p>LPA* enables efficient replanning with O(k log k) complexity where k = affected nodes</p>
</div>
""", unsafe_allow_html=True)