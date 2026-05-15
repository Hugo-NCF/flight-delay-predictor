"""
Flight Delay Predictor — Gradio Application
Served on Hugging Face Spaces (free tier).

Calls the Databricks Model Serving REST endpoint and displays
a rich prediction with confidence gauge and feature breakdown.
"""

import os
import json
import requests
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
# Set these as HuggingFace Space Secrets (never hard-code tokens)
DATABRICKS_HOST  = os.environ.get("DATABRICKS_HOST", "")   # e.g. https://adb-xxx.azuredatabricks.net
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
ENDPOINT_NAME    = os.environ.get("ENDPOINT_NAME", "flight-delay-clf")

SERVING_URL = f"{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations"

# ── Airline options (top US carriers) ────────────────────────────────────────
AIRLINES = {
    "American Airlines (AA)": "AA",
    "Delta Air Lines (DL)":   "DL",
    "United Airlines (UA)":   "UA",
    "Southwest Airlines (WN)": "WN",
    "JetBlue Airways (B6)":   "B6",
    "Alaska Airlines (AS)":   "AS",
    "Spirit Airlines (NK)":   "NK",
    "Frontier Airlines (F9)": "F9",
}

TOP_AIRPORTS = [
    "ATL","LAX","ORD","DFW","DEN","JFK","SFO","SEA","LAS","MCO",
    "EWR","CLT","PHX","MIA","IAH","BOS","MSP","FLL","DTW","LGA",
]

MONTHS = {
    "January":1,"February":2,"March":3,"April":4,
    "May":5,"June":6,"July":7,"August":8,
    "September":9,"October":10,"November":11,"December":12,
}

DAYS = {
    "Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,
    "Friday":5,"Saturday":6,"Sunday":7,
}

# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(airline_name, origin, dest, month_name, day_name,
            dep_hour, distance, sched_duration):
    """Send features to Databricks endpoint and return prediction + confidence."""

    airline_code = AIRLINES[airline_name]
    month        = MONTHS[month_name]
    day_of_week  = DAYS[day_name]
    is_weekend   = 1 if day_of_week >= 6 else 0

    # Rough route/airline delay rates (fallback when no live DB connection)
    route_delay_rate   = 0.20   # neutral prior; replace with live lookup if desired
    airline_delay_rate = 0.18

    payload = {
        "dataframe_records": [{
            "month":              month,
            "day_of_week":        day_of_week,
            "dep_hour":           int(dep_hour),
            "is_weekend":         is_weekend,
            "distance":           float(distance),
            "sched_duration":     float(sched_duration),
            "route_delay_rate":   route_delay_rate,
            "airline_delay_rate": airline_delay_rate,
            "airline_enc":        list(AIRLINES.values()).index(airline_code),
            "origin_enc":         TOP_AIRPORTS.index(origin) if origin in TOP_AIRPORTS else 0,
            "dest_enc":           TOP_AIRPORTS.index(dest) if dest in TOP_AIRPORTS else 0,
        }]
    }

    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type":  "application/json",
    }

    try:
        resp = requests.post(SERVING_URL, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Databricks returns {"predictions": [...]}
        predictions = data.get("predictions", [])
        if not predictions:
            return None, None, "⚠️ Empty response from model endpoint."

        pred = predictions[0]
        # pred is either a scalar (0/1) or dict with label + probability
        if isinstance(pred, dict):
            label = int(pred.get("label", pred.get("0", 0)))
            proba = float(pred.get("probability", pred.get("1", 0.5)))
        else:
            label = int(pred)
            proba = 0.75 if label == 1 else 0.25  # fallback when no proba returned

        return label, proba, None

    except Exception as e:
        return None, None, f"❌ Endpoint error: {e}"


# ── Chart builders ─────────────────────────────────────────────────────────────
def make_gauge(confidence, is_delayed):
    color = "#ef4444" if is_delayed else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = round(confidence * 100, 1),
        title = {"text": "Delay Probability (%)", "font": {"size": 18, "color": "#e2e8f0"}},
        number= {"suffix": "%", "font": {"size": 36, "color": color}},
        gauge = {
            "axis":  {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar":   {"color": color},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0,  40], "color": "#166534"},
                {"range": [40, 60], "color": "#713f12"},
                {"range": [60,100], "color": "#7f1d1d"},
            ],
            "threshold": {
                "line": {"color": "#f8fafc", "width": 3},
                "thickness": 0.8,
                "value": confidence * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        height=280, margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def make_feature_bar(month, day_of_week, dep_hour, distance, sched_duration, is_weekend):
    """Simple normalised feature-importance bar chart (illustrative)."""
    # Static importances from a trained model — replace with real ones from MLflow if desired
    static_importances = {
        "route_delay_rate":   0.28,
        "airline_delay_rate": 0.22,
        "dep_hour":           0.16,
        "distance":           0.12,
        "month":              0.09,
        "sched_duration":     0.07,
        "day_of_week":        0.04,
        "is_weekend":         0.02,
    }
    user_values = {
        "dep_hour":       dep_hour / 23,
        "distance":       min(distance / 5000, 1),
        "month":          month / 12,
        "sched_duration": min(sched_duration / 600, 1),
        "day_of_week":    day_of_week / 7,
        "is_weekend":     is_weekend,
        "route_delay_rate":   0.20,
        "airline_delay_rate": 0.18,
    }
    labels = list(static_importances.keys())
    values = [static_importances[k] * user_values.get(k, 0.5) for k in labels]

    df_chart = pd.DataFrame({"Feature": labels, "Contribution": values})
    df_chart = df_chart.sort_values("Contribution", ascending=True)

    fig = px.bar(
        df_chart, x="Contribution", y="Feature", orientation="h",
        color="Contribution",
        color_continuous_scale=["#1e40af", "#3b82f6", "#ef4444"],
        title="Feature Contributions to Prediction",
    )
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font={"color": "#e2e8f0"},
        title_font={"size": 15, "color": "#94a3b8"},
        coloraxis_showscale=False,
        height=300, margin=dict(t=50, b=20, l=10, r=20),
        xaxis={"gridcolor": "#334155"},
        yaxis={"gridcolor": "#334155"},
    )
    return fig


# ── Main prediction handler ───────────────────────────────────────────────────
def run_prediction(airline_name, origin, dest, month_name, day_name,
                   dep_hour, distance, sched_duration):

    if origin == dest:
        return (
            "⚠️ Origin and destination cannot be the same.",
            None, None, None
        )

    label, proba, err = predict(
        airline_name, origin, dest, month_name, day_name,
        dep_hour, distance, sched_duration
    )

    if err:
        return err, None, None, None

    is_delayed  = label == 1
    confidence  = proba if is_delayed else (1 - proba)

    # Verdict text
    if is_delayed:
        if proba > 0.75:
            verdict = f"✈️  HIGH RISK OF DELAY  ({proba*100:.0f}% probability)"
        else:
            verdict = f"⚠️  MODERATE DELAY RISK  ({proba*100:.0f}% probability)"
    else:
        verdict = f"🟢  ON-TIME LIKELY  ({(1-proba)*100:.0f}% confidence)"

    month      = MONTHS[month_name]
    day_of_week= DAYS[day_name]
    is_weekend = 1 if day_of_week >= 6 else 0

    gauge_fig   = make_gauge(proba, is_delayed)
    feature_fig = make_feature_bar(month, day_of_week, dep_hour, distance, sched_duration, is_weekend)

    tips = []
    if int(dep_hour) < 9:
        tips.append("🌅 Early morning departures typically see fewer delays.")
    if day_of_week == 5:
        tips.append("🗓️  Fridays have higher delay rates — consider Thursday travel.")
    if month in [6, 7, 8, 12]:
        tips.append("📅 Peak travel months (summer & December) inflate delay risk.")
    tip_text = "\n".join(tips) if tips else "No specific tips for this route / time."

    return verdict, gauge_fig, feature_fig, tip_text


# ── Custom CSS ────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

body, .gradio-container {
    background: #0f172a !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.02em;
}

.gr-button-primary {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    border: none !important;
    color: white !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    padding: 12px 32px !important;
    transition: all 0.2s !important;
}

.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}

.gr-form, .gr-box {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}

label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

.gr-input, .gr-dropdown {
    background: #0f172a !important;
    border: 1px solid #475569 !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}

.verdict-box {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    padding: 16px;
    border-radius: 10px;
    text-align: center;
    background: #1e293b;
    border: 1px solid #334155;
}
"""

# ── Layout ────────────────────────────────────────────────────────────────────
HEADER = """
<div style="text-align:center; padding: 32px 0 16px; font-family:'Space Mono',monospace;">
  <div style="font-size:2.8rem; font-weight:700; background:linear-gradient(135deg,#38bdf8,#818cf8,#f472b6);
       -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-0.03em;">
    ✈ FLIGHT DELAY PREDICTOR
  </div>
  <div style="color:#64748b; font-size:0.9rem; margin-top:8px; letter-spacing:0.1em;">
    DISTRIBUTED DATA PIPELINE · MLFLOW · DATABRICKS
  </div>
  <div style="color:#475569; font-size:0.75rem; margin-top:4px;">
    Trained on 500,000+ BTS 2023 flight records
  </div>
</div>
"""

with gr.Blocks(css=CSS, title="Flight Delay Predictor") as demo:
    gr.HTML(HEADER)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ✈ Flight Details")

            airline_input = gr.Dropdown(
                choices=list(AIRLINES.keys()),
                value="Delta Air Lines (DL)",
                label="Airline",
            )
            with gr.Row():
                origin_input = gr.Dropdown(
                    choices=TOP_AIRPORTS, value="ATL", label="Origin Airport"
                )
                dest_input = gr.Dropdown(
                    choices=TOP_AIRPORTS, value="LAX", label="Destination"
                )
            with gr.Row():
                month_input = gr.Dropdown(
                    choices=list(MONTHS.keys()), value="June", label="Month"
                )
                day_input = gr.Dropdown(
                    choices=list(DAYS.keys()), value="Friday", label="Day of Week"
                )
            dep_hour_input = gr.Slider(
                minimum=0, maximum=23, step=1, value=8,
                label="Scheduled Departure Hour (24h)"
            )
            with gr.Row():
                distance_input = gr.Number(
                    value=2475, label="Distance (miles)", precision=0
                )
                duration_input = gr.Number(
                    value=330, label="Scheduled Duration (min)", precision=0
                )

            predict_btn = gr.Button("🔮 Predict Delay", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Prediction Results")
            verdict_output = gr.Textbox(
                label="Verdict", interactive=False, lines=2
            )
            gauge_output   = gr.Plot(label="Delay Probability Gauge")
            feature_output = gr.Plot(label="Feature Contributions")
            tips_output    = gr.Textbox(
                label="💡 Travel Tips", interactive=False, lines=3
            )

    predict_btn.click(
        fn=run_prediction,
        inputs=[
            airline_input, origin_input, dest_input,
            month_input, day_input,
            dep_hour_input, distance_input, duration_input,
        ],
        outputs=[verdict_output, gauge_output, feature_output, tips_output],
    )

    gr.Markdown("""
    ---
    <div style="text-align:center; color:#475569; font-size:0.75rem; font-family:'Space Mono',monospace;">
    Pipeline: S3 → Databricks PySpark (bronze/silver/gold) → MLflow RandomForest → Databricks Model Serving → Gradio<br>
    Data: Bureau of Transportation Statistics 2023 On-Time Performance Dataset
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
