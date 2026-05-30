import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import re

# 1. Page Configuration
st.set_page_config(page_title="AI Synthetic Edge", layout="wide", initial_sidebar_state="expanded")

# --- LOCAL SYNTHESIS ENGINE (ZERO-KEY ALTERNATIVE) ---
def local_architect_and_profiler(user_custom_prompt: str) -> tuple:
    """
    Simulates multi-agent schema extraction and mathematical profiling locally 
    by parsing the structural intent of the natural language prompt.
    """
    prompt_lower = user_custom_prompt.lower()
    columns = []
    math_config = {}

    # Scenario 1: Industrial Hardware / Robotic Lock System Logs
    if any(k in prompt_lower for k in ['lock', 'solenoid', 'robot', 'industrial', 'rfid', 'voltage']):
        columns = ['cycle_index', 'solenoid_voltage_mv', 'current_ma', 'lock_status', 'coil_temperature_c']
        math_config = {
            'solenoid_voltage_mv': {'min': 4950.0, 'max': 5050.0, 'profile': 'noise'},  # Steady 5V rail
            'current_ma': {'min': 200.0, 'max': 210.0, 'profile': 'noise'},
            'lock_status': {'min': 0.0, 'max': 1.0, 'profile': 'noise'},
            'coil_temperature_c': {'min': 25.0, 'max': 45.0, 'profile': 'linear'}       # Heats up over cycles
        }
    
    # Scenario 2: Smart Energy Management / HEMS / PV Baseline
    elif any(k in prompt_lower for k in ['pv', 'energy', 'solar', 'power', 'generation', 'kwh']):
        columns = ['cycle_index', 'pv_generation_kwh', 'load_demand_kwh', 'grid_export_kwh', 'battery_soc']
        math_config = {
            'pv_generation_kwh': {'min': 0.0, 'max': 12.0, 'profile': 'noise'},
            'load_demand_kwh': {'min': 1.5, 'max': 8.0, 'profile': 'noise'},
            'grid_export_kwh': {'min': 0.0, 'max': 5.0, 'profile': 'noise'},
            'battery_soc': {'min': 20.0, 'max': 95.0, 'profile': 'linear'}              # Charges up
        }
        
    # Default Fallback: Generic Matrix Parameters
    else:
        # Extract words to look like custom column names
        words = re.findall(r'\b[a-zA-Z_]{3,15}\b', user_custom_prompt)
        custom_cols = [w.strip() for w in words if w.lower() not in ['and', 'with', 'for', 'the', 'columns']]
        
        if len(custom_cols) >= 2:
            columns = ['cycle_index'] + custom_cols[:4]
        else:
            columns = ['cycle_index', 'parameter_alpha', 'parameter_beta', 'operational_efficiency']
            
        for col in columns:
            if col != 'cycle_index':
                math_config[col] = {'min': 10.0, 'max': 100.0, 'profile': 'noise'}

    return columns, math_config

def generate_tabular_from_prompt_local(user_custom_prompt: str, num_rows: int) -> pd.DataFrame:
    """
    Compiles data frames deterministically based on inferred mathematical profiles.
    """
    columns, math_config = local_architect_and_profiler(user_custom_prompt)
    data = {}
    
    # Establish sequential execution timeline
    data['cycle_index'] = np.arange(1, num_rows + 1)
        
    for col in columns:
        if col == 'cycle_index':
            continue
            
        col_cfg = math_config.get(col, {"min": 10.0, "max": 100.0, "profile": "noise"})
        c_min = float(col_cfg.get("min", 10.0))
        c_max = float(col_cfg.get("max", 100.0))
        c_profile = col_cfg.get("profile", "noise")
        
        if c_profile == "linear":
            base = np.linspace(c_min, c_max, num_rows)
            deviation = (c_max - c_min) * 0.03
            vector_array = base + np.random.normal(0, deviation if deviation > 0 else 1, num_rows)
        else:
            midpoint = (c_min + c_max) / 2
            spread = (c_max - c_min) / 6
            vector_array = np.random.normal(midpoint, spread if spread > 0 else 1, num_rows)
            
        data[col] = np.round(vector_array, 2)
        
    return pd.DataFrame(data)

# --- FRONTEND INTERFACE CUSTOM STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #F5F5F5; }
    header[data-testid="stHeader"] { background-color: rgba(0, 0, 0, 0) !important; }
    div.block-container { padding-top: 2rem !important; }
    [data-testid="stSidebar"] { background-color: #0A0A0A; border-right: 1px solid #333; }
    .stButton>button {
        background: linear-gradient(45deg, #00E5FF, #DEFF9A) !important;
        color: #000 !important; font-weight: 700 !important; border-radius: 8px !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.4) !important; text-transform: uppercase;
        letter-spacing: 1px; padding: 10px 24px !important; border: none !important;
    }
    div.stButton { text-align: center; }
    .stTextArea textarea { background-color: #111 !important; color: #00E5FF !important; border: 1px solid #333 !important; border-radius: 12px !important; }
    h1, h2, h3 { font-family: 'Urbanist', sans-serif; color: #F5F5F5 !important; }
    </style>
""", unsafe_allow_html=True)

if "synthetic_df" not in st.session_state:
    st.session_state.synthetic_df = None

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.markdown("#  MATRIX PARAMETERS")
st.sidebar.markdown("---")
num_rows = st.sidebar.slider(
    "TARGET ROW COUNT", 
    min_value=10, 
    max_value=1000, 
    value=100, 
    step=10, 
    on_change=lambda: st.session_state.update({"synthetic_df": None})
)
st.sidebar.markdown("---")
st.sidebar.info("Engine Status: **LOCAL/OFFLINE**")
st.sidebar.info("Engine: **Deterministic Profiler Engine**")

# --- MAIN DASHBOARD LAYOUT ---
st.markdown("<h1 style='font-size: 64px; text-align: center;'>SYNTHETE <span style='color: #00E5FF;'>- PRO</span></h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #A0A0A0; font-weight: 400;'>Advanced Embedded System Signal Synthesis</h3>", unsafe_allow_html=True)
st.markdown("---")

user_prompt = st.text_area(
    "DEFINE DATASET ARCHITECTURE", 
    placeholder="Describe your requirements (e.g., 'Robotic lock voltage logs' or 'Smart home power tracking')...", 
    height=150, 
    on_change=lambda: st.session_state.update({"synthetic_df": None})
)

btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    submit_trigger = st.button("EXECUTE GENERATION", use_container_width=True)

if submit_trigger:
    if not user_prompt.strip():
        st.warning("Please define your dataset requirements first.")
    else:
        st.session_state.synthetic_df = None
        with st.spinner("PROCESSING LOCAL VECTORS..."):
            try:
                fresh_df = generate_tabular_from_prompt_local(user_prompt, num_rows)
                if fresh_df is not None and not fresh_df.empty:
                    st.session_state.synthetic_df = fresh_df
                    st.success("SYNTHESIS COMPLETE")
                    st.rerun()
                else:
                    st.error("ENGINE ERROR: Failed to process local matrix geometry.")
            except Exception as e:
                st.error(f"SYSTEM ANOMALY: {e}")

# --- RENDER OUTPUT GEOMETRY ---
if st.session_state.synthetic_df is not None:
    df = st.session_state.synthetic_df
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### GENERATED DATA")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            label="EXPORT CSV", 
            data=df.to_csv(index=False).encode('utf-8'), 
            file_name="synthetic_edge_data.csv"
        )
        
    with col2:
        st.markdown("####  TREND VISUALIZATION")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if 'cycle_index' in numeric_cols:
            numeric_cols.remove('cycle_index')
            
        if numeric_cols:
            selected_col = st.selectbox("VARIABLE", numeric_cols)
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 5))
            
            x_axis = df['cycle_index']
            
            ax.plot(x_axis, df[selected_col], color='#00E5FF', linewidth=2.5, marker='o', markerfacecolor='white', markersize=3)
            ax.fill_between(x_axis, df[selected_col], color='#00E5FF', alpha=0.08)
            ax.set_title(f"Signal Tracking: {selected_col}", color='#F5F5F5')
            ax.grid(color='#222222', linestyle='--')
            
            st.pyplot(fig)
            plt.close(fig)