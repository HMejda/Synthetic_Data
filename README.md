# SYNTHETE-PRO 
> **Advanced Foundation Model Tabular Data & Signal Synthesis Engine**

## 👤 Author 
* **Mejda Harbaoui 2EAN**

---

##  Key Features

* **Zero-Dependency Local Engine:** Operates completely offline with zero external API key requirements, bypassing institutional firewalls and network restrictions.
* **Context-Aware Semantic Mapping:** Dynamic token-matching parser that distinguishes between hardware signals (solenoid voltage rails, current telemetry) and environmental data tracks.
* **Vectorized Mathematical Profiler:** Utilizes optimized NumPy structures to synthesize high-fidelity signals following deterministic profiles:
    * `noise`: Normal Gaussian distributions around stable baseline operational rails.
    * `linear`: Continuous slope sequences with micro-deviations to simulate thermal or load climbing trends over time.
* **Cyberpunk Neon UI:** Custom-built Streamlit glassmorphic interface tailored for maximum legibility, high scannability, and seamless interactive variable tracking.

---

##  System Architecture

The application implements a discrete three-step data compilation pipeline:

1. **Schema Architect Agent:** Parses the user's natural language requirements to extract semantic intent and determine relevant, non-overlapping columns.
2. **Mathematical Profiler:** Automatically configures numeric operational limits ($V_{min}$, $V_{max}$) and behavioral signal models based on the domain context.
3. **Vector Synthesis Compiler:** Executes vector calculations to build a complete `pandas.DataFrame` indexed by sequential execution cycles (`cycle_index`).

---

##  Project Structure


Synthetic_Data/
│
├── assets/
│   └── .png       # interface demos
├── src/
│   └── generator.py       # Core architectural simulation logic
│
├── app.py                 # Streamlit UI dashboard and charting pipeline
│
├── requirements.txt       # Project python dependencies
└── README.md              # Documentation