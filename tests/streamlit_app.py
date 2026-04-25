import streamlit as st
import requests
from datetime import datetime
from io import BytesIO

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ESG Report Analyzer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"  # fully collapse native sidebar — we don't use it
)

BASE_URL = "http://localhost:8000"

ESG_BADGE_COLORS = {
    "Environmental": {"bg": "#d4edda", "text": "#155724", "border": "#28a745", "emoji": "🌍"},
    "Social":        {"bg": "#cce5ff", "text": "#004085", "border": "#0d6efd", "emoji": "🤝"},
    "Governance":    {"bg": "#fff3cd", "text": "#856404", "border": "#ffc107", "emoji": "⚖️"},
    "None":          {"bg": "#f8d7da", "text": "#721c24", "border": "#dc3545", "emoji": "❓"},
    "UNKNOWN":       {"bg": "#e2e3e5", "text": "#383d41", "border": "#6c757d", "emoji": "❓"},
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Remove all default padding so our layout fills edge to edge */
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    .stApp {
        background-color: #f5f7f2;
    }

    /* Left panel (our custom "sidebar") */
    .left-panel {
        background-color: #1a2e1a;
        min-height: 100vh;
        padding: 2rem 1.4rem;
        position: sticky;
        top: 0;
    }

    /* Right panel (main content) */
    .right-panel {
        padding: 2rem 2.5rem;
        min-height: 100vh;
    }

    .panel-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
        color: #e8f0e8;
        margin-bottom: 0.1rem;
    }
    .panel-subtitle {
        font-size: 0.72rem;
        color: #8aaa8a;
        margin-bottom: 1.5rem;
        letter-spacing: 0.06em;
    }

    .section-label {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #8aaa8a;
        margin: 1.2rem 0 0.5rem 0;
    }

    .status-card {
        background: rgba(255,255,255,0.07);
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 0.5rem;
        font-size: 0.83rem;
        color: #c8dcc8;
    }

    .esg-badge {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 600;
        border: 1px solid;
        margin: 0.12rem;
    }

    .app-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.1rem;
        color: #1a2e1a;
        margin-bottom: 0.15rem;
        line-height: 1.1;
    }
    .app-subtitle {
        font-size: 0.9rem;
        color: #5a7a5a;
        margin-bottom: 1.8rem;
        font-weight: 300;
    }

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
        margin-left: 12%;
        border-bottom-right-radius: 4px;
    }
    .chat-message.assistant {
        background-color: #ffffff;
        color: #1a2e1a;
        margin-right: 3%;
        border: 1px solid #dde8dd;
        border-bottom-left-radius: 4px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .chat-message .sender {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
        opacity: 0.65;
    }

    .analysis-banner {
        background: linear-gradient(135deg, #1a2e1a 0%, #2d5a27 100%);
        color: #e8f0e8;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.2rem;
        font-size: 0.9rem;
    }
    .analysis-banner strong {
        font-family: 'DM Serif Display', serif;
        font-size: 1.05rem;
    }

    /* Style all buttons inside our left column */
    [data-testid="stVerticalBlock"] .stButton > button {
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stDownloadButton > button {
        background-color: #2d5a27 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        background-color: #3d7a35 !important;
    }

    .stChatInput > div {
        border-color: #dde8dd !important;
        border-radius: 10px !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_session():
    defaults = {
        "session_id": None,
        "uploaded_files": [],
        "analysis_done": False,
        "messages": [],
        "analysis_report": None,
        "esg_labels": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def upload_to_api(files) -> dict | None:
    try:
        file_tuples = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        r = requests.post(f"{BASE_URL}/upload", files=file_tuples, timeout=60)
        if r.status_code == 200:
            return r.json()
        st.error(f"Upload failed: {r.json().get('detail', 'Unknown error')}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the API. Is the FastAPI server running?")
        return None
    except Exception as e:
        st.error(f"Upload error: {e}")
        return None


def query_api(query: str, session_id: str) -> str | None:
    try:
        r = requests.post(
            f"{BASE_URL}/query",
            json={"query": query, "session_id": session_id},
            timeout=300
        )
        if r.status_code == 200:
            return r.json().get("response")
        st.error(f"Query failed: {r.json().get('detail', 'Unknown error')}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the API. Is the FastAPI server running?")
        return None
    except Exception as e:
        st.error(f"Query error: {e}")
        return None


def check_api_health() -> bool:
    try:
        return requests.get(f"{BASE_URL}/health", timeout=5).status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# ESG helpers
# ---------------------------------------------------------------------------
def render_esg_badge(label: str) -> str:
    s = ESG_BADGE_COLORS.get(label, ESG_BADGE_COLORS["UNKNOWN"])
    return (
        f'<span class="esg-badge" style="background:{s["bg"]};'
        f'color:{s["text"]};border-color:{s["border"]}">'
        f'{s["emoji"]} {label}</span>'
    )


def extract_esg_labels(text: str) -> list[str]:
    return list({l for l in ["Environmental", "Social", "Governance"] if l.lower() in text.lower()})


# ---------------------------------------------------------------------------
# Layout — two permanent columns instead of a collapsible sidebar
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1, 3], gap="small")

# ===========================================================================
# LEFT PANEL
# ===========================================================================
with left_col:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)

    st.markdown("""
        <div class="panel-title">🌿 ESG Analyzer</div>
        <div class="panel-subtitle">GRI &nbsp;·&nbsp; SASB &nbsp;·&nbsp; IFRS</div>
    """, unsafe_allow_html=True)

    # API status
    api_ok = check_api_health()
    dot_color = "#52a347" if api_ok else "#dc3545"
    dot_text  = "API Connected" if api_ok else "API Offline"
    st.markdown(
        f'<div style="font-size:0.76rem;color:{dot_color};margin-bottom:1.2rem;">'
        f'● {dot_text}</div>',
        unsafe_allow_html=True
    )

    # Upload
    st.markdown('<div class="section-label">Upload ESG Report</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        label="Upload",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="PDF, DOCX, or TXT"
    )

    if uploaded_files:
        for f in uploaded_files:
            size_kb = round(len(f.getvalue()) / 1024, 1)
            st.markdown(
                f'<div class="status-card">📄 {f.name}<br>'
                f'<span style="opacity:0.55;font-size:0.78rem;">{size_kb} KB</span></div>',
                unsafe_allow_html=True
            )

    analyze_clicked = st.button(
        "🔍 Run ESG Analysis",
        disabled=not (uploaded_files and api_ok),
        use_container_width=True
    )

    # Session info
    if st.session_state.session_id:
        st.markdown('<div class="section-label">Session</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.75rem;color:#8aaa8a;font-family:monospace;">'
            f'ID: {st.session_state.session_id[:8]}...</div>',
            unsafe_allow_html=True
        )

    # ESG coverage badges
    if st.session_state.esg_labels:
        st.markdown('<div class="section-label">Detected Coverage</div>', unsafe_allow_html=True)
        st.markdown(
            " ".join(render_esg_badge(l) for l in st.session_state.esg_labels),
            unsafe_allow_html=True
        )

    # Download
    if st.session_state.analysis_report:
        st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Report (.md)",
            data=BytesIO(st.session_state.analysis_report.encode("utf-8")),
            file_name=f"esg_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True
        )

    # New session
    if st.session_state.session_id:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺ New Session", use_container_width=True):
            try:
                requests.delete(f"{BASE_URL}/session/{st.session_state.session_id}", timeout=5)
            except Exception:
                pass
            for key in ["session_id", "uploaded_files", "messages", "analysis_report", "esg_labels"]:
                st.session_state[key] = [] if isinstance(st.session_state[key], list) else None
            st.session_state.analysis_done = False
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# RIGHT PANEL
# ===========================================================================
with right_col:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)

    st.markdown(
        '<div class="app-title">ESG Report Analyzer</div>'
        '<div class="app-subtitle">Upload a sustainability report · Analyze against GRI, SASB & IFRS · Ask follow-up questions</div>',
        unsafe_allow_html=True
    )

    # --- Run analysis ---
    if analyze_clicked and uploaded_files:
        with st.status("📤 Uploading and indexing documents...", expanded=True):
            result = upload_to_api(uploaded_files)

        if result:
            st.session_state.session_id = result["session_id"]
            st.session_state.uploaded_files = [f.name for f in uploaded_files]
            st.session_state.analysis_done = False
            st.session_state.messages = []
            st.session_state.analysis_report = None
            st.session_state.esg_labels = []

            initial_query = (
                "Provide a full ESG analysis of this report. "
                "Cover environmental, social, and governance topics. "
                "Identify compliance gaps, vague disclosures, and missing sections "
                "against GRI, SASB, and IFRS standards. "
                "Include the full structured table of ESG percepts."
            )

            with st.status("🔄 Running ESG analysis — this may take a moment...", expanded=True):
                analysis = query_api(initial_query, st.session_state.session_id)

            if analysis:
                st.session_state.analysis_done = True
                st.session_state.analysis_report = analysis
                st.session_state.esg_labels = extract_esg_labels(analysis)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": analysis,
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "is_analysis": True
                })
                st.rerun()

    # --- Empty state ---
    if not st.session_state.session_id:
        st.markdown("""
            <div style="text-align:center; padding:5rem 2rem; color:#5a7a5a;">
                <div style="font-size:3rem; margin-bottom:1rem;">🌿</div>
                <div style="font-family:'DM Serif Display',serif; font-size:1.3rem;
                            color:#1a2e1a; margin-bottom:0.6rem;">
                    Ready to analyze your ESG report
                </div>
                <div style="font-size:0.9rem; max-width:380px; margin:0 auto; line-height:1.8;">
                    Upload a sustainability report on the left and click
                    <strong>Run ESG Analysis</strong> to get started.<br><br>
                    The tool will assess coverage, detect vague disclosures,
                    and map your report against <strong>GRI</strong>,
                    <strong>SASB</strong>, and <strong>IFRS</strong> standards.
                </div>
            </div>
        """, unsafe_allow_html=True)

    # --- Analysis banner ---
    elif st.session_state.analysis_done:
        file_list = ", ".join(st.session_state.uploaded_files)
        st.markdown(
            f'<div class="analysis-banner"><strong>Analysis complete</strong><br>'
            f'<span style="opacity:0.8;font-size:0.83rem;">📄 {file_list} &nbsp;·&nbsp; '
            f'Ask follow-up questions below</span></div>',
            unsafe_allow_html=True
        )

    # --- Chat history ---
    for msg in st.session_state.messages:
        role        = msg["role"]
        content     = msg["content"]
        timestamp   = msg.get("timestamp", "")
        is_analysis = msg.get("is_analysis", False)

        if role == "user":
            st.markdown(
                f'<div class="chat-message user">'
                f'<div class="sender">You · {timestamp}</div>'
                f'{content}</div>',
                unsafe_allow_html=True
            )
        else:
            label = "📊 ESG Analysis" if is_analysis else "🤖 Assistant"
            st.markdown(
                f'<div class="chat-message assistant">'
                f'<div class="sender">{label} · {timestamp}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown(content)

            if is_analysis and st.session_state.esg_labels:
                st.markdown(
                    '<div style="margin-top:0.5rem;">'
                    + " ".join(render_esg_badge(l) for l in st.session_state.esg_labels)
                    + '</div>',
                    unsafe_allow_html=True
                )
            st.markdown("---")

    # --- Follow-up chat input ---
    if st.session_state.analysis_done:
        user_input = st.chat_input("Ask a follow-up question about the report...")

        if user_input:
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            with st.status("🤔 Thinking...", expanded=True):
                response = query_api(user_input, st.session_state.session_id)
            if response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "is_analysis": False
                })
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)