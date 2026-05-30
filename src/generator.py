import os
import re
import json
import requests
import numpy as np
import pandas as pd

# Google AI Studio Gemini API Endpoint
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# Secure API configuration using your key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", #"API key here")

def extract_columns_from_response(schema_raw: str) -> list:
    """
    Cleans and extracts column lists from text.
    """
    text = re.sub(r'```.*?```', '', schema_raw, flags=re.DOTALL)
    text = re.sub(r'`', '', text)
    text = re.sub(r'^(Here are the columns?|Columns?needed|The columns? are):?\s*', '', text, flags=re.IGNORECASE)
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return []
    
    first_line = lines[0]
    columns = []
    for col in first_line.split(','):
        col = col.strip().strip('"').strip("'").strip()
        if col and len(col) < 50 and not col.startswith('#'):
            columns.append(col)
    
    return columns

def generate_tabular_from_prompt(user_custom_prompt: str, num_rows: int) -> pd.DataFrame:
    """
    Multi-agent data synthesizer routed seamlessly through Google's Free Gemini Infrastructure.
    """
    if num_rows < 1 or num_rows > 100000:
        raise ValueError("num_rows must be between 1 and 100,000")
        
    request_url = f"{API_URL}?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    # AGENT 1: SCHEMA DESIGNER
    schema_system_instruction = (
        "You are an elite data engineer. Analyze the user request and return ONLY a "
        "comma-separated list of relevant column names for this dataset. Do not include data, "
        "indexes, markdown formatting, explanations, or numbering. Just output: col1,col2,col3"
    )
    
    schema_payload = {
        "contents": [{
            "parts": [{"text": f"System Instruction: {schema_system_instruction}\n\nUser Request: What columns are needed for: {user_custom_prompt}?"}]
        }],
        "generationConfig": {"temperature": 0.1}
    }
    
    try:
        response = requests.post(request_url, json=schema_payload, headers=headers, timeout=15)
        response.raise_for_status()
        schema_raw = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        columns = extract_columns_from_response(schema_raw)
    except Exception as e:
        raise RuntimeError(f"Google Gemini Schema Agent Gateway Connection Failure: {e}")

    if not columns:
        columns = ['metric_alpha', 'metric_beta', 'metric_gamma']

    # AGENT 2: MATHEMATICAL PROFILER
    config_system_instruction = (
        "You are a mathematical profiling agent. Your task is to analyze a list of columns and "
        "return a JSON dictionary defining their numeric boundaries based on context.\n"
        "Return ONLY a clean JSON object matching this structure, no explanations, no markdown ticks:\n"
        "{\n"
        "  \"column_name\": {\"min\": 0.0, \"max\": 100.0, \"profile\": \"linear\"}\n"
        "}\n"
        "Profiles: 'linear' (slopes over time) or 'noise' (deviations around a steady baseline operational rail)."
    )
    
    config_payload = {
        "contents": [{
            "parts": [{"text": f"System Instruction: {config_system_instruction}\n\nContext: {user_custom_prompt}\nColumns: {columns}"}]
        }],
        "generationConfig": {"temperature": 0.1}
    }
    
    try:
        cfg_response = requests.post(request_url, json=config_payload, headers=headers, timeout=15)
        cfg_response.raise_for_status()
        cfg_raw = cfg_response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        
        json_string = cfg_raw.replace("```json", "").replace("```", "").strip()
        math_config = json.loads(json_string)
    except Exception:
        math_config = {col: {"min": 10.0, "max": 100.0, "profile": "noise"} for col in columns}

    # STEP 3: DATA COMPILATION LOGIC
    data = {}
    
    has_clock_keywords = any(k in [c.lower() for c in columns] for k in ['cycle', 'clock', 'step', 'index'])
    has_date_keywords = any(k in [c.lower() for c in columns] for k in ['date', 'timestamp', 'time'])
    
    if has_clock_keywords or not has_date_keywords:
        data['cycle_index'] = np.arange(1, num_rows + 1)
    else:
        data['date'] = pd.date_range(start="2015-01-01", periods=num_rows, freq='D').strftime('%Y-%m-%d')
        
    for col in columns:
        c_low = col.lower()
        if c_low in ['date', 'index', 'cycle_index', 'timestamp', 'time']:
            continue
            
        col_cfg = math_config.get(col, {"min": 10.0, "max": 100.0, "profile": "noise"})
        c_min = float(col_cfg.get("min", 10.0))
        c_max = float(col_cfg.get("max", 100.0))
        c_profile = col_cfg.get("profile", "noise")
        
        if c_profile == "linear":
            base = np.linspace(c_min, c_max, num_rows)
            deviation = (c_max - c_min) * 0.04
            vector_array = base + np.random.normal(0, deviation if deviation > 0 else 1, num_rows)
        else:
            midpoint = (c_min + c_max) / 2
            spread = (c_max - c_min) / 4
            vector_array = np.random.normal(midpoint, spread if spread > 0 else 1, num_rows)
            
        data[col] = np.round(vector_array, 2)
        
    return pd.DataFrame(data)