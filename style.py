style_sheet = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
 
/* Global font */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
 
/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
 
/* App background */
.stApp {
    background-color: #0f0f0f;
    color: #e8e8e8;
}
 
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #141414;
    border-right: 1px solid #2a2a2a;
}
[data-testid="stSidebar"] * {
    color: #c8c8c8 !important;
}
 
/* Sidebar header */
.sidebar-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 18px;
    font-weight: 500;
    color: #f0f0f0 !important;
    letter-spacing: -0.5px;
    padding-bottom: 4px;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 16px;
}
 
/* Context meter */
.ctx-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #666 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.ctx-numbers {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #c8c8c8 !important;
    margin-bottom: 6px;
}
 
/* File list */
.file-pill {
    display: inline-block;
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 3px;
    padding: 3px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #888 !important;
    margin: 2px 2px 2px 0;
}
 
/* Chat area header */
.chat-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 24px;
    font-weight: 500;
    color: #f0f0f0;
    letter-spacing: -1px;
    margin-bottom: 2px;
}
.chat-sub {
    font-size: 13px;
    color: #555;
    margin-bottom: 24px;
    font-weight: 300;
}
 
/* Message bubbles */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 8px 0 !important;
}
 
/* User messages */
[data-testid="stChatMessage"][data-testid*="user"] {
    background: #1a1a1a !important;
}
 
/* Status/info boxes */
.stAlert {
    background-color: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    color: #c8c8c8 !important;
    border-radius: 4px !important;
}
 
/* Progress bar */
.stProgress > div > div {
    background-color: #2a2a2a !important;
}
.stProgress > div > div > div {
    background-color: #4a9eff !important;
    transition: width 0.4s ease;
}
 
/* Buttons */
.stButton > button {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    color: #c8c8c8;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    border-radius: 3px;
    padding: 6px 14px;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #222;
    border-color: #444;
    color: #f0f0f0;
}
 
/* File uploader */
[data-testid="stFileUploader"] {
    background: #141414;
    border: 1px dashed #2a2a2a;
    border-radius: 4px;
    padding: 8px;
}
 
/* Chat input */
[data-testid="stChatInput"] textarea {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    color: #e8e8e8 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    border-radius: 4px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #4a9eff !important;
    box-shadow: 0 0 0 1px #4a9eff22 !important;
}
 
/* Section dividers */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #444 !important;
    margin: 16px 0 8px;
}
 
/* Empty state */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #333;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    line-height: 2;
}
.empty-icon {
    font-size: 32px;
    margin-bottom: 12px;
    opacity: 0.4;
}
</style>
"""