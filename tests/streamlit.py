import streamlit as st
import requests
import json
import time
from datetime import datetime
from io import BytesIO


st.set_page_config(page_title="ESG Report Analyzer",
                   page_icon = "🌿",
                   initial_sidebar_state="expanded",
                   layout="wide")


BASE_URL = "http://localhost:8000"

ESG_BADGE_COLORS = {
    "Environmental": {"bg": "#d4edda", "text": "#155724", "border": "#28a745", "emoji": "🌍"},
    "Social":        {"bg": "#cce5ff", "text": "#004085", "border": "#0d6efd", "emoji": "🤝"},
    "Governance":    {"bg": "#fff3cd", "text": "#856404", "border": "#ffc107", "emoji": "⚖️"},
    "None":          {"bg": "#f8d7da", "text": "#721c24", "border": "#dc3545", "emoji": "❓"},
    "UNKNOWN":       {"bg": "#e2e3e5", "text": "#383d41", "border": "#6c757d", "emoji": "❓"},
}

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* --- Fonts --- */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
 
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
 
    /* --- Global background --- */
    .stApp {
        background-color: #f5f7f2;
    }
 
    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background-color: #1a2e1a;
        border-right: none;
    }
    [data-testid="stSidebar"] * {
        color: #e8f0e8 !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background-color: #2d5a27;
        color: #e8f0e8 !important;
        border: 1px solid #3d7a35;
        border-radius: 8px;
        width: 100%;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #3d7a35;
        border-color: #52a347;
    }
 
    /* --- Page title --- */
    .app-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: #1a2e1a;
        margin-bottom: 0.1rem;
        line-height: 1.1;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #5a7a5a;
        margin-bottom: 1.5rem;
        font-weight: 300;
    }
 
    /* --- Chat messages --- */
    .chat-message {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    .chat-message.user {
        background-color: #1a2e1a;
        color: #e8f0e8;
        margin-left: 15%;
        border-bottom-right-radius: 4px;
    }
    .chat-message.assistant {
        background-color: #ffffff;
        color: #1a2e1a;
        margin-right: 5%;
        border: 1px solid #dde8dd;
        border-bottom-left-radius: 4px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .chat-message .sender {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
        opacity: 0.7;
    }
 
    /* --- ESG Badge --- */
    .esg-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid;
        margin: 0.15rem;
    }
 
    /* --- Status cards --- */
    .status-card {
        background: white;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        border: 1px solid #dde8dd;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
    }
    .status-card.success { border-left: 4px solid #28a745; }
    .status-card.info    { border-left: 4px solid #0d6efd; }
    .status-card.warning { border-left: 4px solid #ffc107; }
 
    /* --- Section divider --- */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #5a7a5a;
        margin: 1.2rem 0 0.5rem 0;
    }
 
    /* --- Download button styling --- */
    .stDownloadButton > button {
        background-color: #2d5a27 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        background-color: #3d7a35 !important;
    }
 
    /* --- Chat input --- */
    .stChatInput > div {
        border-color: #dde8dd !important;
        border-radius: 10px !important;
    }
 
    /* --- Hide default Streamlit branding --- */
    #MainMenu, footer, header { visibility: hidden; }
 
    /* --- Analysis complete banner --- */
    .analysis-banner {
        background: linear-gradient(135deg, #1a2e1a 0%, #2d5a27 100%);
        color: #e8f0e8;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    .analysis-banner strong {
        font-family: 'DM Serif Display', serif;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)
 
 
# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
def init_session():
    defaults = {
        "session_id": None,
        "uploaded_files": [],
        "analysis_done": False,
        "messages": [],          # {"role": "user"|"assistant", "content": str, "timestamp": str}
        "analysis_report": None, # raw markdown string of the initial analysis
        "esg_labels": [],        # list of detected ESG label strings
        "sidebar_open": True,    # tracks sidebar visibility so we can offer a reopen button
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
 
init_session()
 
 
# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def upload_to_api(files) -> dict | None:
    """Uploads files to the FastAPI /upload endpoint."""
    try:
        file_tuples = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        response = requests.post(f"{BASE_URL}/upload", files=file_tuples, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the API. Make sure the FastAPI server is running.")
        return None
    except Exception as e:
        st.error(f"Upload error: {e}")
        return None
 
 
def query_api(query: str, session_id: str) -> str | None:
    """Sends a query to the FastAPI /query endpoint."""
    try:
        payload = {"query": query, "session_id": session_id}
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            timeout=120  # LLM calls can take a while
        )
        if response.status_code == 200:
            return response.json().get("response")
        else:
            st.error(f"Query failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the API. Make sure the FastAPI server is running.")
        return None
    except Exception as e:
        st.error(f"Query error: {e}")
        return None
 
 
def check_api_health() -> bool:
    """Pings /health to confirm the API is reachable."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False
 
 
# ---------------------------------------------------------------------------
# ESG badge renderer
# ---------------------------------------------------------------------------
def render_esg_badge(label: str) -> str:
    style = ESG_BADGE_COLORS.get(label, ESG_BADGE_COLORS["UNKNOWN"])
    return (
        f'<span class="esg-badge" '
        f'style="background:{style["bg"]};color:{style["text"]};border-color:{style["border"]}">'
        f'{style["emoji"]} {label}</span>'
    )
 
 
def extract_esg_labels_from_response(response_text: str) -> list[str]:
    """
    Naively scans the response text for E/S/G label mentions.
    Used to surface the badge summary after analysis.
    """
    found = []
    for label in ["Environmental", "Social", "Governance"]:
        if label.lower() in response_text.lower():
            found.append(label)
    return list(set(found))
 
 
# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="padding: 0.5rem 0 1.5rem 0;">
            <div style="font-family:'DM Serif Display',serif; font-size:1.4rem; color:#e8f0e8;">
                🌿 ESG Analyzer
            </div>
            <div style="font-size:0.75rem; color:#8aaa8a; margin-top:0.2rem;">
                GRI · SASB · IFRS
            </div>
        </div>
    """, unsafe_allow_html=True)
 
    # --- API status ---
    api_ok = check_api_health()
    status_color = "#52a347" if api_ok else "#dc3545"
    status_text = "API Connected" if api_ok else "API Offline"
    st.markdown(
        f'<div style="font-size:0.78rem; color:{status_color}; margin-bottom:1rem;">'
        f'● {status_text}</div>',
        unsafe_allow_html=True
    )
 
    st.markdown('<div class="section-label">Upload ESG Report</div>', unsafe_allow_html=True)
 
    uploaded_files = st.file_uploader(
        label="Upload ESG Report",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Supported: PDF, DOCX, TXT"
    )
 
    if uploaded_files:
        for f in uploaded_files:
            size_kb = round(len(f.getvalue()) / 1024, 1)
            st.markdown(
                f'<div class="status-card info">📄 {f.name}<br>'
                f'<span style="opacity:0.6">{size_kb} KB</span></div>',
                unsafe_allow_html=True
            )
 
    # --- Analyze button ---
    analyze_clicked = st.button(
        "🔍 Run ESG Analysis",
        disabled=not (uploaded_files and api_ok),
        use_container_width=True
    )
 
    # --- Session info ---
    if st.session_state.session_id:
        st.markdown('<div class="section-label">Session</div>', unsafe_allow_html=True)
        short_id = st.session_state.session_id[:8]
        st.markdown(
            f'<div style="font-size:0.78rem; color:#8aaa8a; font-family:monospace;">'
            f'ID: {short_id}...</div>',
            unsafe_allow_html=True
        )
 
    # --- ESG coverage badges (shown after analysis) ---
    if st.session_state.esg_labels:
        st.markdown('<div class="section-label">Detected Coverage</div>', unsafe_allow_html=True)
        badges_html = " ".join(render_esg_badge(l) for l in st.session_state.esg_labels)
        st.markdown(badges_html, unsafe_allow_html=True)
 
    # --- Downloadable report ---
    if st.session_state.analysis_report:
        st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        report_bytes = st.session_state.analysis_report.encode("utf-8")
        st.download_button(
            label="⬇️ Download Report (.md)",
            data=BytesIO(report_bytes),
            file_name=f"esg_analysis_{timestamp}.md",
            mime="text/markdown",
            use_container_width=True
        )
 
    # --- Reset session ---
    if st.session_state.session_id:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺ New Session", use_container_width=True):
            # Attempt to clean up on the server side
            try:
                requests.delete(
                    f"{BASE_URL}/session/{st.session_state.session_id}",
                    timeout=5
                )
            except Exception:
                pass
            # Reset all local state
            for key in ["session_id", "uploaded_files", "analysis_done",
                        "messages", "analysis_report", "esg_labels"]:
                st.session_state[key] = [] if key in ["messages", "uploaded_files", "esg_labels"] else None
            st.session_state.analysis_done = False
            st.rerun()
    
    # --- Collapse sidebar button ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("✕ Close sidebar", use_container_width=True, key="close_sidebar"):
        st.session_state.sidebar_open = False
        st.rerun()
 
 
# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if not st.session_state.sidebar_open:
    st.markdown('<div class="sidebar-toggle-btn">', unsafe_allow_html=True)
    if st.button("🌿 Open sidebar", key="open_sidebar"):
        st.session_state.sidebar_open = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
 
st.markdown(
    '<div class="app-title">ESG Report Analyzer</div>'
    '<div class="app-subtitle">Upload a sustainability report · Analyze against GRI, SASB & IFRS · Ask follow-up questions</div>',
    unsafe_allow_html=True
)
 
# --- Handle Analyze button click ---
if analyze_clicked and uploaded_files:
    with st.spinner("Uploading documents and building vector index..."):
        result = upload_to_api(uploaded_files)
 
    if result:
        st.session_state.session_id = result["session_id"]
        st.session_state.uploaded_files = [f.name for f in uploaded_files]
        st.session_state.analysis_done = False
        st.session_state.messages = []
        st.session_state.analysis_report = None
        st.session_state.esg_labels = []
 
        # Run the initial structured analysis
        initial_query = (
            "Provide a full ESG analysis of this report. "
            "Cover environmental, social, and governance topics. "
            "Identify compliance gaps, vague disclosures, and missing sections "
            "against GRI, SASB, and IFRS standards. "
            "Include the full structured table of ESG percepts."
        )
 
        with st.spinner("Running ESG analysis — this may take a moment..."):
            analysis = query_api(initial_query, st.session_state.session_id)
 
        if analysis:
            st.session_state.analysis_done = True
            st.session_state.analysis_report = analysis
            st.session_state.esg_labels = extract_esg_labels_from_response(analysis)
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.messages.append({
                "role": "assistant",
                "content": analysis,
                "timestamp": timestamp,
                "is_analysis": True
            })
            st.rerun()
 
# --- Empty state (no session yet) ---
if not st.session_state.session_id:
    st.markdown("""
        <div style="
            text-align:center;
            padding: 4rem 2rem;
            color: #5a7a5a;
        ">
            <div style="font-size:3rem; margin-bottom:1rem;">🌿</div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.3rem; 
                        color:#1a2e1a; margin-bottom:0.5rem;">
                Ready to analyze your ESG report
            </div>
            <div style="font-size:0.9rem; max-width:400px; margin:0 auto; line-height:1.7;">
                Upload a sustainability report in the sidebar and click 
                <strong>Run ESG Analysis</strong> to get started.
                <br><br>
                The tool will assess coverage, detect vague disclosures, 
                and map your report against <strong>GRI</strong>, 
                <strong>SASB</strong>, and <strong>IFRS</strong> standards.
            </div>
        </div>
    """, unsafe_allow_html=True)
 
# --- Analysis complete banner ---
elif st.session_state.analysis_done:
    file_list = ", ".join(st.session_state.uploaded_files)
    st.markdown(
        f'<div class="analysis-banner">'
        f'<strong>Analysis complete</strong><br>'
        f'<span style="opacity:0.8;font-size:0.85rem;">📄 {file_list} &nbsp;·&nbsp; '
        f'Ask follow-up questions below</span>'
        f'</div>',
        unsafe_allow_html=True
    )
 
# --- Render chat history ---
if st.session_state.messages:
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        timestamp = msg.get("timestamp", "")
        is_analysis = msg.get("is_analysis", False)
 
        if role == "user":
            st.markdown(
                f'<div class="chat-message user">'
                f'<div class="sender">You · {timestamp}</div>'
                f'{content}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            # For the initial analysis, render as markdown (tables, bold, etc.)
            # For follow-up responses, also render as markdown
            st.markdown(
                f'<div class="chat-message assistant">'
                f'<div class="sender">{"📊 ESG Analysis" if is_analysis else "🤖 Assistant"} · {timestamp}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            # Render the actual content as markdown (supports tables from the LLM)
            st.markdown(content)
 
            # Show ESG badges under the initial analysis
            if is_analysis and st.session_state.esg_labels:
                badges_html = (
                    '<div style="margin-top:0.5rem;">'
                    + " ".join(render_esg_badge(l) for l in st.session_state.esg_labels)
                    + "</div>"
                )
                st.markdown(badges_html, unsafe_allow_html=True)
 
            st.markdown("---")
 
# --- Follow-up chat input (only shown after initial analysis) ---
if st.session_state.analysis_done:
    user_input = st.chat_input(
        "Ask a follow-up question about the report...",
    )
 
    if user_input:
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })
 
        with st.spinner("Thinking..."):
            response = query_api(user_input, st.session_state.session_id)
 
        if response:
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M"),
                "is_analysis": False
            })
 
        st.rerun()