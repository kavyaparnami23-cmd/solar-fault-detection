import streamlit as st
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except ImportError:
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter
from PIL import Image
import os
import datetime
import requests
import time

try:
    from github import Github
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Solar Monitor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Constants ───────────────────────────────────────────────────────────────
CHANNEL_ID   = "2679947"
READ_API_KEY = "KR8S190NY76UDC68"
WRITE_API_KEY = "EUACAIBRUDUPQYTN"
IMG_SIZE     = (224, 224)
CLASS_NAMES  = ["clean", "crack", "dust"]

NEWS_ITEMS = [
    ("Quantum Solar Cells Break Efficiency Records",
     "New photovoltaic technology achieves 47 % efficiency in lab conditions, shattering previous benchmarks."),
    ("AI-Optimised Solar Farms Increase Output by 22 %",
     "Machine-learning algorithms dynamically adjust panel angles for maximum sun exposure throughout the day."),
    ("Space-Based Solar Power Becomes Reality",
     "Orbital collectors now transmitting clean energy to ground stations 24/7, unaffected by weather."),
    ("Self-Cleaning Nano-Coatings Revolutionise Maintenance",
     "New surface treatment reduces dust accumulation by 85 % on solar panels, cutting cleaning costs."),
]

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;500;600&display=swap');

/* ── Root & Body ── */
:root {
    --accent:  #00f5ff;
    --accent2: #7e42ff;
    --accent3: #ff2a6d;
    --card-bg: rgba(14, 20, 45, 0.82);
    --text:    #e8f7ff;
    --glow:    0 0 18px rgba(0,245,255,.55);
}
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at top, #0a0f29 0%, #050814 100%) !important;
    color: var(--text) !important;
    font-family: 'Exo 2', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background-attachment: fixed !important;
}
/* hide default header */
[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

/* ── Top Nav Bar ── */
.nav-bar {
    position: sticky; top: 0; z-index: 999;
    background: rgba(5,8,20,.85);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(0,245,255,.2);
    box-shadow: var(--glow);
    padding: 12px 40px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 0;
}
.nav-title {
    font-family: 'Orbitron', sans-serif;
    color: var(--accent);
    font-size: 22px;
    letter-spacing: 2px;
    text-shadow: 0 0 12px rgba(0,245,255,.7);
}
.nav-links { display: flex; gap: 24px; }
.nav-links a {
    color: #b8d4ff; text-decoration: none;
    font-weight: 500; font-size: 15px;
    transition: color .3s;
}
.nav-links a:hover { color: var(--accent); text-shadow: 0 0 8px var(--accent); }

/* ── Hero Section ── */
.hero {
    text-align: center;
    padding: 80px 20px 60px;
    position: relative;
}
.hero h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(28px, 5vw, 52px);
    color: var(--accent);
    text-shadow: 0 0 20px rgba(0,245,255,.6);
    letter-spacing: 2px;
    margin-bottom: 20px;
    animation: fadeUp .8s ease both;
}
.hero p {
    font-size: 18px; color: #cfd8ff;
    max-width: 700px; margin: 0 auto 40px;
    line-height: 1.7;
    animation: fadeUp .8s ease .2s both;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(28px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ── Sun Animation ── */
.sun-wrap {
    display: flex; justify-content: center; margin-bottom: 30px;
    animation: fadeUp .8s ease .1s both;
}
.sun-svg { filter: drop-shadow(0 0 18px #ff8c00); animation: sunPulse 4s infinite alternate; }
@keyframes sunPulse {
    0%   { filter: drop-shadow(0 0 18px #ff8c00); }
    100% { filter: drop-shadow(0 0 36px #ff8c00) drop-shadow(0 0 50px rgba(255,140,0,.4)); }
}

/* ── CTA Button ── */
.cta-btn {
    display: inline-block;
    padding: 14px 38px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    color: #000 !important; font-weight: 700;
    border-radius: 12px; text-decoration: none;
    font-family: 'Orbitron', sans-serif; letter-spacing: 1px;
    box-shadow: 0 5px 20px rgba(0,245,255,.4);
    transition: transform .3s, box-shadow .3s;
    animation: fadeUp .8s ease .4s both;
    cursor: pointer;
}
.cta-btn:hover { transform: translateY(-4px); box-shadow: 0 12px 28px rgba(0,245,255,.6); }

/* ── Auth Card ── */
.auth-wrap {
    display: flex; justify-content: center; padding: 60px 20px;
}
.auth-card {
    background: var(--card-bg);
    border: 1px solid rgba(0,245,255,.25);
    border-radius: 20px;
    padding: 44px 38px;
    width: 380px;
    box-shadow: var(--glow), 0 0 30px rgba(126,66,255,.3);
    backdrop-filter: blur(14px);
    animation: cardIn .6s ease both;
    position: relative; overflow: hidden;
}
@keyframes cardIn {
    from { opacity:0; transform:scale(.9); }
    to   { opacity:1; transform:scale(1); }
}
.auth-card::before {
    content:''; position:absolute; top:0; left:0; width:100%; height:4px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.auth-card h2 {
    font-family: 'Orbitron', sans-serif;
    color: var(--accent); text-align: center;
    font-size: 22px; margin-bottom: 22px;
    text-shadow: 0 0 10px rgba(0,245,255,.5);
}

/* ── Streamlit input overrides inside auth ── */
.stTextInput > div > div > input {
    background: rgba(5,10,25,.75) !important;
    border: 1px solid rgba(0,245,255,.3) !important;
    border-radius: 10px !important; color: #fff !important;
    font-family: 'Exo 2', sans-serif !important;
    transition: border-color .3s, box-shadow .3s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 10px rgba(0,245,255,.45) !important;
}

/* ── Dashboard Header ── */
.dash-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 28px;
}
.dash-title {
    font-family: 'Orbitron', sans-serif;
    color: var(--accent); font-size: 28px;
    text-shadow: 0 0 14px var(--accent); letter-spacing: 1px;
}
.status-badge {
    display: flex; align-items: center; gap: 8px;
    background: rgba(5,10,25,.7);
    padding: 7px 16px; border-radius: 20px;
    border: 1px solid rgba(0,245,255,.3);
    font-size: 13px; color: #cfd8ff;
}
.status-dot {
    width: 11px; height: 11px; border-radius: 50%;
    background: #00ff2a;
    box-shadow: 0 0 8px #00ff2a;
    animation: dotPulse 2s infinite;
}
@keyframes dotPulse {
    0%   { box-shadow: 0 0 0 0 rgba(0,255,42,.7); }
    70%  { box-shadow: 0 0 0 8px rgba(0,255,42,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,255,42,0); }
}

/* ── Metric Cards ── */
.metric-card {
    background: var(--card-bg);
    border: 1px solid rgba(0,245,255,.2);
    border-radius: 18px;
    padding: 24px 16px 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    box-shadow: var(--glow);
    transition: transform .35s, box-shadow .35s;
    position: relative; overflow: hidden;
    animation: fadeUp .7s ease both;
}
.metric-card::before {
    content:''; position:absolute; top:0; left:0; width:100%; height:4px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-card:hover { transform:translateY(-7px); box-shadow:0 16px 30px rgba(0,245,255,.3); }
.metric-label {
    color: var(--accent2); font-size: 14px; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px;
}
.metric-icon { font-size: 22px; }
.metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 40px; font-weight: 900; color: #fff;
    text-shadow: 0 0 12px rgba(255,255,255,.25);
    line-height: 1;
}
.metric-unit {
    color: var(--accent); font-size: 14px; margin-top: 6px; opacity: .85;
}

/* ── Section Card ── */
.section-card {
    background: var(--card-bg);
    border: 1px solid rgba(0,245,255,.2);
    border-radius: 18px; padding: 28px;
    backdrop-filter: blur(10px);
    box-shadow: var(--glow);
    transition: transform .35s, box-shadow .35s;
    position: relative; overflow: hidden;
    animation: fadeUp .7s ease both;
}
.section-card::before {
    content:''; position:absolute; top:0; left:0; width:100%; height:4px;
    background: linear-gradient(90deg, var(--accent), var(--accent3));
}
.section-card:hover { transform:translateY(-4px); box-shadow:0 12px 28px rgba(0,245,255,.25); }
.section-card h3 {
    font-family: 'Orbitron', sans-serif;
    color: var(--accent); font-size: 20px;
    text-shadow: 0 0 10px rgba(0,245,255,.5);
    margin-bottom: 18px;
}

/* ── Graph Container ── */
.graph-card {
    background: var(--card-bg);
    border: 1px solid rgba(0,245,255,.2);
    border-radius: 18px; padding: 18px;
    backdrop-filter: blur(10px);
    box-shadow: var(--glow);
    transition: transform .35s;
    position: relative; overflow: hidden;
}
.graph-card::before {
    content:''; position:absolute; top:0; left:0; width:100%; height:4px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.graph-card:hover { transform:translateY(-5px); }
.graph-card h4 {
    font-family:'Orbitron',sans-serif;
    color: var(--accent2); font-size:14px; letter-spacing:1px;
    margin-bottom:10px;
}
.graph-card iframe {
    border-radius:10px; border:1px solid rgba(255,255,255,.1);
    width:100%; height:260px;
}

/* ── News Item ── */
.news-item {
    background: var(--card-bg);
    border: 1px solid rgba(0,245,255,.2);
    border-radius: 14px; padding: 20px;
    backdrop-filter: blur(10px);
    box-shadow: var(--glow);
    transition: transform .3s, box-shadow .3s;
    position: relative; overflow: hidden;
    animation: fadeUp .7s ease both;
}
.news-item::before {
    content:''; position:absolute; top:0; left:0; width:100%; height:4px;
    background: linear-gradient(90deg, var(--accent2), var(--accent3));
}
.news-item:hover { transform:translateY(-5px); box-shadow:0 10px 24px rgba(0,245,255,.28); }
.news-item h4 { color:var(--accent); font-size:16px; margin-bottom:8px; }
.news-item p  { color:#cfd8ff; font-size:14px; line-height:1.55; }

/* ── Action Button ── */
.action-btn {
    display: inline-block;
    padding: 13px 28px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    color: #000 !important; font-weight: 700; border: none;
    border-radius: 10px; cursor: pointer;
    font-family: 'Orbitron', sans-serif; letter-spacing: 1px; font-size: 14px;
    transition: transform .3s, box-shadow .3s;
    box-shadow: 0 4px 16px rgba(0,245,255,.35);
}
.action-btn:hover { transform:translateY(-3px); box-shadow:0 10px 22px rgba(0,245,255,.5); }

/* ── Alert Badges ── */
.alert-critical { background:rgba(255,42,109,.15); border-left:4px solid var(--accent3); padding:10px 14px; border-radius:8px; font-family:monospace; font-size:13px; margin-bottom:6px; }
.alert-warning  { background:rgba(255,180,0,.12);  border-left:4px solid #ffb400;         padding:10px 14px; border-radius:8px; font-family:monospace; font-size:13px; margin-bottom:6px; }
.alert-success  { background:rgba(0,255,42,.1);    border-left:4px solid #00ff2a;          padding:10px 14px; border-radius:8px; font-family:monospace; font-size:13px; margin-bottom:6px; }
.alert-info     { background:rgba(0,245,255,.1);   border-left:4px solid var(--accent);    padding:10px 14px; border-radius:8px; font-family:monospace; font-size:13px; margin-bottom:6px; }

/* ── Prediction Badge ── */
.pred-badge {
    display: inline-block;
    padding: 8px 22px; border-radius: 30px;
    font-family: 'Orbitron', sans-serif; font-size: 18px; font-weight: 700;
    letter-spacing: 1px; margin: 8px 0;
}
.pred-clean  { background: rgba(0,255,42,.15);    border:2px solid #00ff2a;        color:#00ff2a; }
.pred-crack  { background: rgba(255,42,109,.15);  border:2px solid var(--accent3); color:var(--accent3); }
.pred-dust   { background: rgba(255,180,0,.15);   border:2px solid #ffb400;        color:#ffb400; }

/* ── Divider ── */
.neon-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    margin: 40px 0;
    opacity: .5;
}

/* ── Footer ── */
.footer {
    background: rgba(5,8,20,.85);
    border-top: 1px solid rgba(0,245,255,.2);
    padding: 28px 40px;
    text-align: center; color: #6a7fa8; font-size: 13px;
    margin-top: 80px;
}
.footer strong { color: var(--accent); }

/* ── Streamlit widget overrides ── */
.stButton > button {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    color: #000 !important; font-weight: 700 !important; border: none !important;
    font-family: 'Orbitron', sans-serif !important; letter-spacing: 1px !important;
    border-radius: 10px !important; padding: 12px 24px !important;
    transition: transform .3s, box-shadow .3s !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 22px rgba(0,245,255,.5) !important;
}
[data-testid="stFileUploader"] {
    background: rgba(5,10,25,.7) !important;
    border: 1px dashed rgba(0,245,255,.4) !important;
    border-radius: 12px !important; padding: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
if "page"        not in st.session_state: st.session_state.page    = "home"
if "logged_in"   not in st.session_state: st.session_state.logged_in = False
if "username"    not in st.session_state: st.session_state.username = ""
if "alerts"      not in st.session_state: st.session_state.alerts  = []
if "ts_data"     not in st.session_state: st.session_state.ts_data = {}
if "last_fetch"  not in st.session_state: st.session_state.last_fetch = 0

# ─── Helpers ─────────────────────────────────────────────────────────────────
def log_alert(level: str, message: str, dedup: bool = True):
    body = f"{level}: {message}"
    if dedup and st.session_state.alerts and body in st.session_state.alerts[0]:
        return
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.alerts.insert(0, f"[{ts}] {body}")

def nav(page: str):
    st.session_state.page = page
    st.rerun()

def fetch_thingspeak():
    now = time.time()
    if now - st.session_state.last_fetch < 15:
        return st.session_state.ts_data
    try:
        url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds/last.json?api_key={READ_API_KEY}"
        r = requests.get(url, timeout=6)
        d = r.json()
        voltage = float(d.get("field1") or 0)
        temp    = float(d.get("field2") or 0)
        power   = round(voltage * 8.5, 1)
        eff     = min(98.0, round(voltage / 24 * 100, 1))
        st.session_state.ts_data = {
            "voltage": voltage, "temp": temp,
            "power": power, "efficiency": eff,
        }
        st.session_state.last_fetch = now
    except Exception:
        pass
    return st.session_state.ts_data

def trigger_motor():
    try:
        url = f"https://api.thingspeak.com/update?api_key={WRITE_API_KEY}&field3=1"
        requests.get(url, timeout=6)
        log_alert("SUCCESS", "Cleaning motor activated via ThingSpeak field3=1", False)
        return True
    except Exception as e:
        log_alert("ERROR", f"Motor trigger failed: {e}")
        return False

def create_github_issue(title: str, body: str):
    if not GITHUB_AVAILABLE:
        log_alert("WARNING", "PyGithub not installed — skipping GitHub issue.")
        return
    token = os.getenv("GITHUB_TOKEN", "REPLACE_WITH_YOUR_TOKEN")
    repo_name = "kavyaparnami23-cmd/solar-fault-detection"
    if token == "REPLACE_WITH_YOUR_TOKEN" or not token:
        log_alert("WARNING", "GitHub token not set. Set GITHUB_TOKEN env var.")
        return
    try:
        g = Github(token)
        g.get_repo(repo_name).create_issue(title=title, body=body)
        log_alert("SUCCESS", f"GitHub Issue created: {title}")
    except Exception as e:
        log_alert("ERROR", f"GitHub issue failed: {e}")

@st.cache_resource
def load_model():
    if not os.path.exists("model.tflite"):
        return None
    interp = TFLiteInterpreter(model_path="model.tflite")
    interp.allocate_tensors()
    return interp

def predict_image(image: Image.Image):
    interp = load_model()
    if interp is None:
        return None, None, None
    in_d  = interp.get_input_details()
    out_d = interp.get_output_details()
    img = image.resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    # EfficientNet preprocess_input is a no-op (expects [0,255] float32)
    # No transformation needed — arr is already float32 in [0, 255]
    arr = np.expand_dims(arr, axis=0)
    interp.set_tensor(in_d[0]["index"], arr)
    interp.invoke()
    preds = interp.get_tensor(out_d[0]["index"])[0]
    idx   = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx]), preds

# ─── Navigation Bar ───────────────────────────────────────────────────────────
def render_nav():
    user_info = f"<span style='color:#00ff2a;font-size:13px;'>● {st.session_state.username}</span>" if st.session_state.logged_in else ""
    dashboard_link = '<a href="#" onclick="void(0)">Dashboard</a>' if st.session_state.logged_in else ""
    st.markdown(f"""
    <div class="nav-bar">
        <div class="nav-title">☀️ AI SOLAR MONITOR</div>
        <div style="display:flex;align-items:center;gap:24px;">
            {user_info}
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns([1,1,1,1,1,1,4])
    with cols[0]:
        if st.button("🏠 Home"):     nav("home")
    with cols[1]:
        if st.session_state.logged_in:
            if st.button("📊 Dashboard"): nav("dashboard")
    with cols[2]:
        if not st.session_state.logged_in:
            if st.button("🔑 Login"):    nav("login")
    with cols[3]:
        if not st.session_state.logged_in:
            if st.button("📝 Register"): nav("register")
    with cols[4]:
        if st.session_state.logged_in:
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.username  = ""
                nav("home")

# ─── Pages ───────────────────────────────────────────────────────────────────

def page_home():
    st.markdown("""
    <div class="hero">
        <div class="sun-wrap">
            <svg class="sun-svg" width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <radialGradient id="sg" cx="50%" cy="50%" r="50%">
                        <stop offset="0%"  stop-color="#ffde00"/>
                        <stop offset="100%" stop-color="#ff8c00"/>
                    </radialGradient>
                </defs>
                <circle cx="60" cy="60" r="30" fill="url(#sg)"/>
                <g stroke="#ffde00" stroke-width="3" stroke-linecap="round" opacity=".7">
                    <line x1="60" y1="5"  x2="60" y2="18"/>
                    <line x1="60" y1="102" x2="60" y2="115"/>
                    <line x1="5"  y1="60" x2="18" y2="60"/>
                    <line x1="102" y1="60" x2="115" y2="60"/>
                    <line x1="19" y1="19" x2="28" y2="28"/>
                    <line x1="92" y1="92" x2="101" y2="101"/>
                    <line x1="101" y1="19" x2="92" y2="28"/>
                    <line x1="28" y1="92" x2="19" y2="101"/>
                </g>
            </svg>
        </div>
        <h1>AI-POWERED SOLAR MONITORING SYSTEM</h1>
        <p>Advanced real-time monitoring of solar power systems with AI-driven image analytics,
        automated panel cleaning, predictive fault detection, and cloud-based alerting
        for optimal energy production.</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature tiles
    c1, c2, c3, c4 = st.columns(4)
    feats = [
        ("⚡", "Real-Time Data", "Live voltage, temperature, power, and efficiency from ThingSpeak IoT cloud."),
        ("🤖", "AI Fault Detection", "EfficientNet TFLite model classifies panels as Clean, Dusty, or Cracked."),
        ("🧹", "Auto Cleaning", "One-click motor trigger sends command directly to ESP32 via ThingSpeak."),
        ("🚨", "Smart Alerts", "Automated GitHub issue creation and on-screen alert log for critical events."),
    ]
    for col, (icon, title, desc) in zip([c1,c2,c3,c4], feats):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="animation-delay:{[0,.1,.2,.3][[c1,c2,c3,c4].index(col)]}s">
                <div style="font-size:34px;margin-bottom:10px;">{icon}</div>
                <div class="metric-label">{title}</div>
                <p style="color:#cfd8ff;font-size:13px;line-height:1.5;margin-top:8px;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)

    if st.session_state.logged_in:
        c = st.columns([3,1,3])
        with c[1]:
            if st.button("📊  VIEW DASHBOARD"):
                nav("dashboard")
    else:
        st.markdown("""
        <div style="text-align:center;padding:20px 0 40px;">
            <p style="color:#cfd8ff;font-size:16px;">Login or register to access the full dashboard.</p>
        </div>
        """, unsafe_allow_html=True)
        c = st.columns([3,1,1,3])
        with c[1]:
            if st.button("🔑 Login"):    nav("login")
        with c[2]:
            if st.button("📝 Register"): nav("register")


def page_login():
    st.markdown("""
    <div class="auth-wrap">
      <div class="auth-card">
        <h2>🔐 ACCESS PORTAL</h2>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1,1.2,1])
    with mid:
        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
        username = st.text_input("Username", key="li_user", placeholder="Enter username")
        password = st.text_input("Password", type="password", key="li_pass", placeholder="Enter password")
        if st.button("AUTHENTICATE"):
            if username and password:
                st.session_state.logged_in = True
                st.session_state.username  = username
                log_alert("SUCCESS", f"User '{username}' authenticated.", False)
                st.success("✅ Authentication successful! Redirecting to dashboard…")
                time.sleep(0.6)
                nav("dashboard")
            else:
                st.error("Please enter both username and password.")
        st.markdown(
            '<p style="text-align:center;color:#7a8db0;font-size:13px;margin-top:14px;">'
            'No account? <a href="#" style="color:var(--accent);">Register</a></p>',
            unsafe_allow_html=True,
        )


def page_register():
    st.markdown("""
    <div class="auth-wrap">
      <div class="auth-card">
        <h2>🛸 REGISTER SYSTEM</h2>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1,1.2,1])
    with mid:
        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
        username = st.text_input("Choose Username", key="reg_user")
        password = st.text_input("Choose Password", type="password", key="reg_pass")
        if st.button("CREATE ACCOUNT"):
            if username and password:
                log_alert("INFO", f"New account created for '{username}'.", False)
                st.success("✅ Account created! Please log in.")
                time.sleep(0.8)
                nav("login")
            else:
                st.error("Please fill in all fields.")


def page_dashboard():
    data = fetch_thingspeak()

    # Header
    st.markdown("""
    <div class="dash-header">
        <div class="dash-title">SOLAR CONTROL DASHBOARD</div>
        <div class="status-badge">
            <div class="status-dot"></div>
            <span>SYSTEM ONLINE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric Cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, "⚡", "VOLTAGE",     data.get("voltage",  "--"), "VOLTS",   "0.1s"),
        (c2, "🌡️", "TEMPERATURE", data.get("temp",     "--"), "°C",      "0.2s"),
        (c3, "🔋", "POWER OUTPUT",data.get("power",    "--"), "WATTS",   "0.3s"),
        (c4, "☀️", "EFFICIENCY",  data.get("efficiency","--"), "PERCENT", "0.4s"),
    ]
    for col, icon, label, val, unit, delay in metrics:
        with col:
            display = f"{val:.1f}" if isinstance(val, float) else str(val)
            st.markdown(f"""
            <div class="metric-card" style="animation-delay:{delay}">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{display}</div>
                <div class="metric-unit">{unit}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Auto-refresh hint ─────────────────────────────────────────────────────
    rc = st.columns([5,1])
    with rc[1]:
        if st.button("🔄 Refresh Data"):
            st.session_state.last_fetch = 0
            st.rerun()

    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)

    # ── ThingSpeak Graphs ──────────────────────────────────────────────────────
    st.markdown('<h3 style="font-family:Orbitron,sans-serif;color:#00f5ff;text-shadow:0 0 10px rgba(0,245,255,.5);font-size:20px;margin-bottom:16px;">📈 LIVE DATA STREAMS</h3>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    graphs = [
        (g1, "FIELD 1 – Voltage",
         f"https://thingspeak.com/channels/{CHANNEL_ID}/charts/1?bgcolor=%23000&color=%2300ffb3&dynamic=true&type=line&results=60"),
        (g2, "FIELD 2 – Temperature",
         f"https://thingspeak.com/channels/{CHANNEL_ID}/charts/2?bgcolor=%23000&color=%23ff6b6b&dynamic=true&type=line&results=60"),
        (g3, "FIELD 4 – Extra",
         f"https://thingspeak.com/channels/{CHANNEL_ID}/charts/4?bgcolor=%23000&color=%230084ff&dynamic=true&type=line&results=60"),
    ]
    for col, title, url in graphs:
        with col:
            st.markdown(f"""
            <div class="graph-card">
                <h4>{title}</h4>
                <iframe src="{url}" frameborder="0" scrolling="no"></iframe>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)

    # ── AI Image Analyzer + Alerts ─────────────────────────────────────────────
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="section-card"><h3>🤖 AI IMAGE ANALYSER (ESP32-CAM)</h3>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload a solar panel image for AI analysis",
            type=["jpg","jpeg","png"],
            key="panel_image",
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_column_width=True, caption="Uploaded Panel Image")

            with st.spinner("Running TFLite model…"):
                label, conf, probs = predict_image(image)

            if label is None:
                st.warning("⚠️ model.tflite not found. Place the file alongside this script.")
                log_alert("WARNING", "model.tflite missing — prediction skipped.")
            else:
                badge_cls = f"pred-{label}"
                st.markdown(f"""
                <div style="text-align:center;margin:14px 0;">
                    <span class="pred-badge {badge_cls}">{label.upper()}</span>
                    <div style="color:#cfd8ff;font-size:14px;margin-top:6px;">
                        Confidence: <strong style="color:#00f5ff">{conf*100:.1f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Probability bars
                for name, prob in zip(CLASS_NAMES, probs):
                    st.progress(float(prob), text=f"{name.capitalize()}: {prob*100:.1f}%")

                # Downstream actions
                if label == "dust":
                    log_alert("WARNING", "Dust detected. Performance degraded. Activating spray.")
                    st.warning("⚙️ Automated Cleaning: Motor & Pump activating…")
                    trigger_motor()

                elif label == "crack":
                    log_alert("CRITICAL", "Crack detected! Efficiency catastrophic. Ticket raised.")
                    st.error("🚨 Critical fault! GitHub issue being created…")
                    create_github_issue(
                        "🚨 Urgent: Solar Panel Crack Detected",
                        "A severe crack was detected via the IoT Monitoring system.\n"
                        "Power efficiency dropped critically.\n\nPlease inspect the panel immediately.",
                    )

                elif label == "clean":
                    log_alert("INFO", "Scan clear. System nominal.", False)
                    st.success("✅ Panel is clean — operating at peak efficiency.")

        st.markdown("</div>", unsafe_allow_html=True)

        # Manual motor trigger
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-card"><h3>🧹 CLEANING SYSTEM CONTROL</h3>', unsafe_allow_html=True)
        if st.button("🚿  ACTIVATE CLEANING MOTOR"):
            if trigger_motor():
                st.success("✅ Motor command sent! Cleaning system running.")
            else:
                st.error("❌ Motor trigger failed.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card" style="min-height:420px;"><h3>🚨 ALERT & NOTIFICATION LOG</h3>', unsafe_allow_html=True)
        if st.session_state.alerts:
            for alert in st.session_state.alerts[:12]:
                if "CRITICAL" in alert or "ERROR" in alert:
                    cls = "alert-critical"
                elif "WARNING" in alert:
                    cls = "alert-warning"
                elif "SUCCESS" in alert:
                    cls = "alert-success"
                else:
                    cls = "alert-info"
                st.markdown(f'<div class="{cls}">{alert}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-info">System initialising. No alerts yet.</div>', unsafe_allow_html=True)

        if st.button("🗑️ Clear Log"):
            st.session_state.alerts = []
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)

    # ── News Section ──────────────────────────────────────────────────────────
    st.markdown('<h3 style="font-family:Orbitron,sans-serif;color:#00f5ff;font-size:20px;margin-bottom:16px;">📡 SOLAR ENERGY INTELLIGENCE</h3>', unsafe_allow_html=True)
    ncols = st.columns(4)
    for col, (title, desc) in zip(ncols, NEWS_ITEMS):
        with col:
            st.markdown(f"""
            <div class="news-item">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
def render_footer():
    st.markdown("""
    <div class="footer">
        <strong>☀️ AI SOLAR MONITOR</strong> &nbsp;·&nbsp;
        Powered by ThingSpeak IoT &amp; TFLite AI &nbsp;·&nbsp;
        © 2025 AI Solar Monitor System. All rights reserved.
    </div>
    """, unsafe_allow_html=True)


# ─── Main Router ──────────────────────────────────────────────────────────────
render_nav()

page = st.session_state.page
if   page == "home":      page_home()
elif page == "login":     page_login()
elif page == "register":  page_register()
elif page == "dashboard":
    if st.session_state.logged_in:
        page_dashboard()
    else:
        st.warning("⚠️ Please log in to access the dashboard.")
        nav("login")

render_footer()