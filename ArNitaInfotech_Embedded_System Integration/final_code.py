import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import time
import requests

st.set_page_config(
    page_title="Smart Light Monitoring",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Intelligent Light Monitoring System")
st.write("Week 10 - Intelligent Detection with Telegram Alerts")

# -------------------------
# TELEGRAM SETTINGS
# -------------------------
# NOTE: replace these with YOUR OWN regenerated token and real numeric chat ID.
# Get your chat ID from @userinfobot on Telegram.
# Do not commit or share a file that has a real token pasted into it.
HARDCODED_BOT_TOKEN = "8566936493:AAET7OTUAq99sLPuDf94HQmM7-v6K7Vi6uM"
HARDCODED_CHAT_ID = "1964147286"

st.sidebar.header("Telegram Settings")

def get_secret(key):
    """Safely read from st.secrets without crashing if no secrets.toml exists."""
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""

default_token = get_secret("BOT_TOKEN") or HARDCODED_BOT_TOKEN
default_chat_id = get_secret("CHAT_ID") or HARDCODED_CHAT_ID

BOT_TOKEN = st.sidebar.text_input(
    "Bot Token",
    value=default_token,
    type="password",
    help="Get this from @BotFather on Telegram"
)

CHAT_ID = st.sidebar.text_input(
    "Chat ID",
    value=default_chat_id,
    help="Your numeric Telegram user ID (get it from @userinfobot). "
         "You must have sent /start to your bot at least once."
)

# -------------------------
# SEND TELEGRAM ALERT
# -------------------------
def send_telegram_alert(lux, prediction, reading):
    """
    Sends a Telegram alert and returns (success: bool, message: str)
    so the caller can surface real errors in the UI instead of only
    printing to the console.
    """
    if not BOT_TOKEN or not CHAT_ID or "PASTE_YOUR" in BOT_TOKEN or "PASTE_YOUR" in CHAT_ID:
        return False, "Bot Token or Chat ID is missing/placeholder. Fill them in the sidebar."

    message = (
        "🚨 HIGH LIGHT INTENSITY ALERT\n\n"
        f"Reading : {reading}\n"
        f"Lux : {lux:.2f}\n"
        f"Prediction : {prediction}\n"
        "Status : HIGH INTENSITY DETECTED"
    )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}

    try:
        response = requests.post(url, data=payload, timeout=10)
        data = response.json()

        if response.status_code == 200 and data.get("ok"):
            return True, "Alert sent successfully."
        else:
            return False, f"Telegram API error: {data.get('description', response.text)}"

    except Exception as e:
        return False, f"Request failed: {e}"


# -------------------------
# TEST BUTTON (debug independently of the monitoring loop)
# -------------------------
if st.sidebar.button("📨 Test Telegram Connection"):
    ok, msg = send_telegram_alert(lux=9999.0, prediction="Test", reading="TEST")
    if ok:
        st.sidebar.success(msg)
    else:
        st.sidebar.error(msg)

st.sidebar.divider()
st.sidebar.header("Monitoring Settings")

speed = st.sidebar.slider("Refresh Time (seconds)", 0.1, 2.0, 0.5, 0.1)
threshold = st.sidebar.slider("High Intensity Threshold", 100, 10000, 2000)

# -------------------------
# LOAD MODEL + DATA
# -------------------------
try:
    model = joblib.load("decision_tree_model.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

try:
    df = pd.read_csv("Combined_Dataset.csv")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

required = ["Lux", "Smoothed Lux"]
for c in required:
    if c not in df.columns:
        st.error(f"Missing Column : {c}")
        st.stop()

label_map = {0: "Bright", 1: "Dark", 2: "Dim"}

metric_placeholder = st.empty()
graph_placeholder = st.empty()
event_placeholder = st.empty()
alert_status_placeholder = st.empty()

graph_data = []
event_history = []
event_count = 0

if "alert_sent" not in st.session_state:
    st.session_state.alert_sent = False

if st.button("▶ Start Live Monitoring"):

    for index, row in df.iterrows():

        lux = float(row["Lux"])
        smooth = float(row["Smoothed Lux"])

        start = time.perf_counter()

        features = pd.DataFrame([[lux, smooth]], columns=["Lux", "Smoothed Lux"])
        pred = model.predict(features)[0]
        prediction = label_map.get(pred, str(pred))

        prediction_time = (time.perf_counter() - start) * 1000

        # -------------------------
        # ALERT
        # -------------------------
        if lux >= threshold:
            status = "🔴 HIGH INTENSITY DETECTED"
            event_count += 1

            event_history.append({
                "Reading": index + 1,
                "Lux": round(lux, 2),
                "Prediction": prediction
            })

            if not st.session_state.alert_sent:
                ok, msg = send_telegram_alert(lux, prediction, index + 1)
                if ok:
                    alert_status_placeholder.success(f"Telegram: {msg}")
                else:
                    alert_status_placeholder.error(f"Telegram: {msg}")
                st.session_state.alert_sent = True
        else:
            status = "🟢 Normal"
            st.session_state.alert_sent = False

        # -------------------------
        # GRAPH
        # -------------------------
        graph_data.append(lux)
        if len(graph_data) > 100:
            graph_data.pop(0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=graph_data, mode="lines", name="Lux"))
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig.update_layout(title="Live Light Intensity", height=400,
                           xaxis_title="Samples", yaxis_title="Lux")

        graph_placeholder.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # METRICS
        # -------------------------
        with metric_placeholder.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Reading", index + 1)
            c2.metric("Lux", round(lux, 2))
            c3.metric("Smoothed Lux", round(smooth, 2))
            c4.metric("Prediction", prediction)

            a1, a2 = st.columns(2)
            a1.metric("Prediction Time", f"{prediction_time:.3f} ms")
            a2.metric("Events", event_count)

            if lux >= threshold:
                st.error(status)
            else:
                st.success(status)

        # -------------------------
        # EVENT TABLE
        # -------------------------
        with event_placeholder.container():
            st.subheader("Event History")
            if len(event_history):
                st.dataframe(pd.DataFrame(event_history), use_container_width=True)
            else:
                st.info("No Events Detected")

        time.sleep(speed)
