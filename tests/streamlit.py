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