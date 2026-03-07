"""
Interactive UI for the Metadata Extraction Agent.

In containerised mode the UI talks to the FastAPI backend (agent-api service)
via HTTP.  Set AGENT_API_URL in your environment or .env file.

Run locally (no Docker):
    export AGENT_API_URL=http://localhost:8000
    streamlit run app.py
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Metadata Agent",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API client ────────────────────────────────────────────────────────────────
AGENT_API_URL = os.environ.get("AGENT_API_URL", "http://localhost:8000").rstrip("/")


class APIClient:
    def __init__(self, base: str):
        self._base = base

    def _get(self, path: str, **kwargs) -> Any:
        r = requests.get(f"{self._base}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: Dict) -> Any:
        r = requests.post(f"{self._base}{path}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> Any:
        r = requests.delete(f"{self._base}{path}", timeout=15)
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        try:
            r = requests.get(f"{self._base}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def start_extraction(self, payload: Dict) -> str:
        return self._post("/extract", payload)["job_id"]

    def get_job(self, job_id: str) -> Dict:
        return self._get(f"/jobs/{job_id}")

    def get_job_report(self, job_id: str) -> Dict:
        return self._get(f"/jobs/{job_id}/report")

    def ask(self, run_id: str, question: str) -> str:
        return self._post(f"/history/{run_id}/ask", {"question": question})["answer"]

    def get_history(self) -> List[Dict]:
        try:
            return self._get("/history")
        except Exception:
            return []

    def delete_history(self, run_id: str) -> None:
        self._delete(f"/history/{run_id}")

    def get_history_report(self, run_id: str) -> Dict:
        return self._get(f"/history/{run_id}/report")

    def search(self, q: str, scope: str = "all", db_type: str = "all") -> List[Dict]:
        return self._get("/search", params={"q": q, "scope": scope, "db_type": db_type})


api = APIClient(AGENT_API_URL)

# ── Constants ─────────────────────────────────────────────────────────────────
DB_META: Dict[str, Dict] = {
    "postgres":   {"icon": "🐘", "label": "PostgreSQL", "color": "#60a5fa"},
    "oracle":     {"icon": "🏛️",  "label": "Oracle",     "color": "#f87171"},
    "sqlserver":  {"icon": "🪟",  "label": "SQL Server", "color": "#34d399"},
    "teradata":   {"icon": "📊",  "label": "Teradata",   "color": "#fbbf24"},
    "redshift":   {"icon": "🔺",  "label": "Redshift",   "color": "#a78bfa"},
    "bigquery":   {"icon": "📈",  "label": "BigQuery",   "color": "#f472b6"},
    "delta_lake": {"icon": "⬡",   "label": "Delta Lake", "color": "#2dd4bf"},
}

PIPELINE_NODES = [
    ("connection",  "🔌", "Connecting to database"),
    ("discovery",   "🔍", "Discovering tables"),
    ("extraction",  "📦", "Extracting schema & statistics"),
    ("analysis",    "🧬", "Analyzing dependencies"),
    ("report",      "📋", "Generating report"),
]


# ── CSS ────────────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1a 0%, #0f1623 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a1f35 50%, #0f2744 100%);
    border: 1px solid rgba(99,179,237,0.18); border-radius: 18px;
    padding: 2rem 2.5rem 1.8rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
}
.hero-title {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(90deg, #63b3ed 0%, #b794f4 50%, #68d391 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; letter-spacing: -0.8px;
}
.hero-sub { color: #718096; font-size: 0.95rem; margin-top: 0.4rem; }

.stat-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
.stat-card {
    background: linear-gradient(135deg, #131929 0%, #1e2740 100%);
    border: 1px solid rgba(255,255,255,0.07); border-radius: 14px;
    padding: 1.3rem 1.1rem; text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(0,0,0,0.4); }
.stat-val { font-size: 2rem; font-weight: 700; line-height: 1; }
.stat-lbl { font-size: 0.7rem; color: #4a5568; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.35rem; }

.sec-head {
    font-size: 1.2rem; font-weight: 700; color: #e2e8f0;
    display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 1.2rem; padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

.hcard {
    background: linear-gradient(135deg, #131929 0%, #1a2035 100%);
    border: 1px solid rgba(255,255,255,0.07); border-radius: 14px;
    padding: 1.3rem 1.5rem; margin-bottom: 0.8rem;
    position: relative; overflow: hidden;
    transition: border-color 0.2s, transform 0.15s;
}
.hcard:hover { border-color: rgba(99,179,237,0.3); transform: translateX(2px); }
.hcard-title { font-size: 1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.2rem; }
.hcard-meta  { font-size: 0.78rem; color: #64748b; }

.dbbadge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.22rem 0.65rem; border-radius: 20px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
}

.node-item {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.55rem 1rem; border-radius: 9px;
    margin-bottom: 0.35rem; font-size: 0.85rem; font-weight: 500;
    transition: all 0.3s;
}
.node-pending { background: rgba(255,255,255,0.02); color: #4a5568; }
.node-running {
    background: rgba(99,179,237,0.1); color: #63b3ed;
    border: 1px solid rgba(99,179,237,0.2);
    animation: pulse-blue 1.5s ease-in-out infinite;
}
.node-done  { background: rgba(104,211,145,0.1); color: #68d391; border: 1px solid rgba(104,211,145,0.2); }
.node-error { background: rgba(252,129,74,0.1);  color: #fc814a; border: 1px solid rgba(252,129,74,0.2); }
@keyframes pulse-blue {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99,179,237,0.2); }
    50%       { box-shadow: 0 0 0 6px rgba(99,179,237,0); }
}

.s-result {
    background: #111827; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 11px; padding: 1rem 1.2rem; margin-bottom: 0.55rem;
    transition: border-color 0.2s;
}
.s-result:hover { border-color: rgba(99,179,237,0.3); }
.s-result-title { font-size: 0.95rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.2rem; }
.s-result-sub   { font-size: 0.78rem; color: #64748b; }
mark { background: rgba(251,191,36,0.25); color: #fbbf24; padding: 0 2px; border-radius: 3px; }

.empty-state { text-align: center; padding: 4rem 2rem; color: #374151; }
.empty-state-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-state-text { font-size: 1rem; font-weight: 500; }

.pill {
    display: inline-block; background: rgba(99,179,237,0.12);
    color: #63b3ed; padding: 0.15rem 0.6rem; border-radius: 12px;
    font-size: 0.72rem; margin: 2px; font-weight: 500;
}
.banner-ok  {
    background: rgba(104,211,145,0.1); border: 1px solid rgba(104,211,145,0.3);
    border-radius: 10px; padding: 0.9rem 1.2rem; color: #68d391; font-weight: 500;
}
.banner-err {
    background: rgba(252,129,74,0.1); border: 1px solid rgba(252,129,74,0.3);
    border-radius: 10px; padding: 0.9rem 1.2rem; color: #fc814a; font-weight: 500;
}
.api-status-ok  { color: #68d391; font-size: 0.75rem; }
.api-status-err { color: #fc814a; font-size: 0.75rem; }
.sidebar-logo {
    font-size: 1.4rem; font-weight: 800; margin-bottom: 0.2rem;
    background: linear-gradient(90deg, #63b3ed, #b794f4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def db_badge(db_type: str) -> str:
    m = DB_META.get(db_type, {"icon": "🗄️", "label": db_type, "color": "#94a3b8"})
    c = m["color"]
    r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    return (
        f'<span class="dbbadge" style="background:rgba({r},{g},{b},0.15);'
        f'color:{c};border:1px solid rgba({r},{g},{b},0.3)">'
        f'{m["icon"]} {m["label"]}</span>'
    )


def _highlight(text: str, q: str) -> str:
    if not q:
        return text
    return re.sub(f"({re.escape(q)})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)


def _stat_card(value: Any, label: str, color: str = "#63b3ed") -> str:
    return (
        f'<div class="stat-card">'
        f'<div class="stat-val" style="color:{color}">{value}</div>'
        f'<div class="stat-lbl">{label}</div></div>'
    )


def _node_html(nodes_state: Dict[str, str]) -> str:
    icons  = {n[0]: n[1] for n in PIPELINE_NODES}
    labels = {n[0]: n[2] for n in PIPELINE_NODES}
    items  = []
    for name, _, _ in PIPELINE_NODES:
        state  = nodes_state.get(name, "pending")
        prefix = {"running": "⟳ ", "done": "✓ ", "error": "✗ "}.get(state, "○ ")
        items.append(
            f'<div class="node-item node-{state}">'
            f'{icons[name]} {prefix}{labels[name]}</div>'
        )
    return "<div>" + "".join(items) + "</div>"


def _summary_stats(summary: Dict) -> str:
    cards = [
        _stat_card(summary.get("total_tables", 0),                    "Tables",             "#63b3ed"),
        _stat_card(summary.get("total_functional_dependencies", 0),   "Func. Dependencies", "#b794f4"),
        _stat_card(summary.get("total_inclusion_dependencies", 0),    "Incl. Dependencies", "#68d391"),
        _stat_card(summary.get("total_cardinality_relationships", 0), "Cardinality Links",  "#f6ad55"),
    ]
    return '<div class="stat-row">' + "".join(cards) + "</div>"


def _nodes_from_status(status: Dict) -> Dict[str, str]:
    completed = set(status.get("completed_nodes") or [])
    current   = status.get("current_node") or ""
    job_status= status.get("status", "queued")
    out = {}
    for name, _, _ in PIPELINE_NODES:
        if name in completed:
            out[name] = "done"
        elif name == current and job_status == "running":
            out[name] = "running"
        elif job_status == "error" and name == current:
            out[name] = "error"
        else:
            out[name] = "pending"
    return out


# ── Session state init ─────────────────────────────────────────────────────────
for _k, _v in [("page", "extract"), ("last_report", None),
                ("last_run_id", None), ("last_run_meta", None),
                ("running_job_id", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sidebar ────────────────────────────────────────────────────────────────────
def _sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">⬡ Metadata Agent</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.72rem;color:#475569;margin-bottom:0.4rem">Database schema intelligence</div>',
            unsafe_allow_html=True,
        )
        # API status indicator
        ok = api.health()
        dot = "🟢" if ok else "🔴"
        st.markdown(
            f'<div class="{"api-status-ok" if ok else "api-status-err"}">'
            f'{dot} Agent API: {"connected" if ok else "unreachable"}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        for key, icon, label in [("extract", "⚡", "New Extraction"),
                                   ("history", "🗂️",  "History"),
                                   ("search",  "🔍", "Search Metadata")]:
            active = st.session_state.page == key
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.page = key
                st.rerun()

        history = api.get_history()
        if history:
            st.markdown("---")
            st.markdown(
                '<div style="font-size:0.72rem;color:#475569;text-transform:uppercase;'
                'letter-spacing:0.08em;margin-bottom:0.5rem">Recent runs</div>',
                unsafe_allow_html=True,
            )
            for h in history[:5]:
                m  = DB_META.get(h["db_type"], {"icon": "🗄️"})
                ts = h.get("timestamp", "")[:16].replace("T", " ")
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:0.4rem;'
                    f'padding:0.4rem 0.6rem;background:rgba(255,255,255,0.03);border-radius:7px">'
                    f'{m["icon"]} <b>{h.get("database","?")}</b><br>'
                    f'<span style="color:#475569">{ts}</span></div>',
                    unsafe_allow_html=True,
                )


# ── Inline result panel (shared between fresh run & history load) ──────────────
def _show_result_panel() -> None:
    report   = st.session_state.last_report
    run_id   = st.session_state.last_run_id
    run_meta = st.session_state.last_run_meta or {}
    if not report:
        return

    summary = report.get("summary", {})
    st.markdown(
        '<div class="banner-ok" style="margin-bottom:1rem">✓ Extraction complete!</div>',
        unsafe_allow_html=True,
    )
    st.markdown(_summary_stats(summary), unsafe_allow_html=True)

    with st.expander("📊 Report details", expanded=False):
        tables: Dict = report.get("tables") or {}
        if tables:
            st.markdown(f"**Tables ({len(tables)})**")
            for tname, tmeta in list(tables.items())[:20]:
                cols = tmeta.get("columns", []) if isinstance(tmeta, dict) else []
                col_names = [c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in cols]
                pills = " ".join(f'<span class="pill">{c}</span>' for c in col_names[:12])
                extra = f'<span class="pill">+{len(col_names)-12} more</span>' if len(col_names) > 12 else ""
                rows  = tmeta.get("row_count", "?") if isinstance(tmeta, dict) else "?"
                st.markdown(
                    f'<div style="margin-bottom:0.6rem">'
                    f'<span style="color:#e2e8f0;font-weight:600">{tname}</span>'
                    f'<span style="color:#475569;font-size:0.8rem;margin-left:0.5rem">{rows} rows</span>'
                    f'<br>{pills}{extra}</div>',
                    unsafe_allow_html=True,
                )

        fds = report.get("functional_dependencies") or []
        if fds:
            st.markdown(f"**Functional Dependencies ({len(fds)})**")
            for fd in fds[:10]:
                det  = ", ".join(fd.get("determinant", []))
                dep  = ", ".join(fd.get("dependent", []))
                conf = fd.get("confidence", 0)
                st.markdown(
                    f'<div style="font-size:0.82rem;margin-bottom:0.3rem;color:#94a3b8">'
                    f'<span style="color:#63b3ed">[{det}]</span> → '
                    f'<span style="color:#b794f4">[{dep}]</span> '
                    f'<span style="color:#475569">conf={conf:.2f}</span></div>',
                    unsafe_allow_html=True,
                )

        st.download_button(
            "⬇  Download full JSON report",
            data=__import__("json").dumps(report, indent=2, default=str),
            file_name=f'metadata_{run_meta.get("db_type","?")}.json',
            mime="application/json",
            use_container_width=True,
        )

    # LLM Q&A
    st.markdown('<div class="sec-head" style="margin-top:1.5rem">💬 Ask the Agent</div>', unsafe_allow_html=True)
    question = st.text_input("Ask a question about the extracted metadata",
                              placeholder="Which tables have the most null values?",
                              key="llm_question")
    if st.button("Ask", key="ask_btn") and question and run_id:
        try:
            with st.spinner("Thinking…"):
                answer = api.ask(run_id, question)
            st.markdown(
                f'<div style="background:rgba(183,148,244,0.08);border:1px solid rgba(183,148,244,0.2);'
                f'border-radius:10px;padding:1rem 1.2rem;color:#e2e8f0;font-size:0.9rem">{answer}</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"LLM error: {e}")


# ── View: Extract ──────────────────────────────────────────────────────────────
def _extract_view() -> None:
    st.markdown(
        '<div class="hero"><div class="hero-title">New Metadata Extraction</div>'
        '<div class="hero-sub">Connect to any database and extract complete schema intelligence.</div></div>',
        unsafe_allow_html=True,
    )

    col_form, col_right = st.columns([3, 2], gap="large")

    with col_form:
        st.markdown('<div class="sec-head">🔌 Database Connection</div>', unsafe_allow_html=True)
        db_type = st.selectbox(
            "Database type",
            options=list(DB_META.keys()),
            format_func=lambda k: f'{DB_META[k]["icon"]}  {DB_META[k]["label"]}',
        )

        needs_bq    = db_type == "bigquery"
        needs_spark = db_type == "delta_lake"

        if needs_bq:
            c1, c2 = st.columns(2)
            bq_project = c1.text_input("GCP Project", placeholder="my-gcp-project")
            bq_dataset = c2.text_input("Dataset (schema)", placeholder="my_dataset")
            bq_creds   = st.text_input("Service account JSON path", placeholder="/path/to/sa.json")
            host = port = database = schema = username = password = ""
            spark_master = http_path = odbc_driver = ""
        elif needs_spark:
            c1, c2 = st.columns(2)
            host         = c1.text_input("Databricks host", placeholder="adb-xxx.azuredatabricks.net")
            database     = c1.text_input("Catalog / Database", placeholder="my_catalog")
            schema       = c2.text_input("Schema", placeholder="my_schema")
            spark_master = c2.text_input("Spark master (local only)", placeholder="local[*]")
            http_path    = st.text_input("HTTP path", placeholder="/sql/1.0/warehouses/abc123")
            password     = st.text_input("Databricks token", type="password")
            port         = 443
            username     = ""
            bq_project = bq_dataset = bq_creds = odbc_driver = ""
        else:
            c1, c2 = st.columns([3, 1])
            host = c1.text_input("Host", placeholder="localhost")
            defaults = {"postgres": 5432, "oracle": 1521, "sqlserver": 1433,
                        "teradata": 1025, "redshift": 5439}
            port = c2.number_input("Port", value=defaults.get(db_type, 5432), step=1)
            c3, c4 = st.columns(2)
            database = c3.text_input("Database / Service name", placeholder="mydb")
            schema   = c4.text_input("Schema", placeholder="public")
            c5, c6  = st.columns(2)
            username = c5.text_input("Username")
            password = c6.text_input("Password", type="password")
            odbc_driver = ""
            if db_type == "sqlserver":
                odbc_driver = st.text_input("ODBC Driver", value="ODBC Driver 18 for SQL Server")
            bq_project = bq_dataset = bq_creds = spark_master = http_path = ""

        st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1rem 0">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head">⚙️ Agent Settings</div>', unsafe_allow_html=True)

        c7, c8 = st.columns(2)
        sample_size    = c7.number_input("Sample size (rows)", value=10_000, min_value=100, step=1000)
        fd_threshold   = c7.slider("FD threshold", 0.80, 1.0, 1.0, 0.01,
                                    help="1.0 = exact FDs only; lower = approximate")
        id_threshold   = c8.slider("IND threshold", 0.70, 1.0, 0.95, 0.01)
        target_raw     = c8.text_input("Target tables (blank = all)", placeholder="users, orders")

        run_btn = st.button("⚡  Run Extraction", type="primary", use_container_width=True,
                            disabled=bool(st.session_state.running_job_id))

    # ── Right column: pipeline progress only ──────────────────────────────────
    with col_right:
        st.markdown('<div class="sec-head">📡 Pipeline Progress</div>', unsafe_allow_html=True)
        progress_area = st.empty()
        error_area    = st.empty()

        job_id = st.session_state.running_job_id
        if job_id:
            try:
                status = api.get_job(job_id)
            except Exception as e:
                error_area.markdown(
                    f'<div class="banner-err">⚠ API error while polling: {e}</div>',
                    unsafe_allow_html=True,
                )
                st.session_state.running_job_id = None
                status = None

            if status:
                nodes_state = _nodes_from_status(status)
                progress_area.markdown(_node_html(nodes_state), unsafe_allow_html=True)

                if status["status"] == "done":
                    try:
                        report = api.get_job_report(job_id)
                        st.session_state.last_report    = report
                        st.session_state.last_run_id    = job_id
                        st.session_state.last_run_meta  = {"db_type": db_type}
                        st.session_state.running_job_id = None
                        st.balloons()
                        st.rerun()          # re-render so the result panel appears
                    except Exception as e:
                        error_area.markdown(
                            f'<div class="banner-err">⚠ Could not fetch report: {e}</div>',
                            unsafe_allow_html=True,
                        )
                        st.session_state.running_job_id = None

                elif status["status"] == "error":
                    for k, v in nodes_state.items():
                        if v == "running":
                            nodes_state[k] = "error"
                    progress_area.markdown(_node_html(nodes_state), unsafe_allow_html=True)
                    error_area.markdown(
                        f'<div class="banner-err">⚠ Extraction failed: {status.get("error","unknown error")}</div>',
                        unsafe_allow_html=True,
                    )
                    st.session_state.running_job_id = None
                else:
                    # Still running — poll again
                    time.sleep(1.5)
                    st.rerun()
        else:
            idle = {n[0]: "pending" for n in PIPELINE_NODES}
            progress_area.markdown(_node_html(idle), unsafe_allow_html=True)

    # ── Result panel — rendered in main body so widgets work correctly ─────────
    if st.session_state.last_report and not st.session_state.running_job_id:
        st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.5rem 0'>",
                    unsafe_allow_html=True)
        _show_result_panel()

    # ── Launch extraction ──────────────────────────────────────────────────────
    if run_btn:
        if not api.health():
            st.error(f"Cannot reach the Agent API at {AGENT_API_URL}. Is the agent-api container running?")
            return

        if needs_bq:
            db_cfg_payload: Dict[str, Any] = {
                "db_type": db_type, "project": bq_project or None,
                "schema_name": bq_dataset or None, "credentials_path": bq_creds or None,
            }
        elif needs_spark:
            extra: Dict = {}
            if http_path:
                extra["http_path"] = http_path
            db_cfg_payload = {
                "db_type": db_type, "host": host or None, "port": int(port) if host else None,
                "database": database or None, "schema_name": schema or None,
                "password": password or None, "spark_master": spark_master or None, "extra": extra,
            }
        else:
            extra = {}
            if db_type == "sqlserver" and odbc_driver:
                extra["driver"] = odbc_driver
            db_cfg_payload = {
                "db_type": db_type, "host": host or None, "port": int(port) if port else None,
                "database": database or None, "schema_name": schema or None,
                "username": username or None, "password": password or None, "extra": extra,
            }

        target_tables = (
            [t.strip() for t in target_raw.split(",") if t.strip()]
            if target_raw.strip() else None
        )
        payload = {
            "db_config":     db_cfg_payload,
            "target_tables": target_tables,
            "sample_size":   int(sample_size),
            "fd_threshold":  float(fd_threshold),
            "id_threshold":  float(id_threshold),
        }

        try:
            job_id = api.start_extraction(payload)
            st.session_state.running_job_id = job_id
            st.session_state.last_report    = None
            st.session_state.last_run_id    = None
            st.rerun()
        except Exception as e:
            st.error(f"Failed to start extraction: {e}")


# ── View: History ──────────────────────────────────────────────────────────────
def _history_view() -> None:
    st.markdown(
        '<div class="hero"><div class="hero-title">Extraction History</div>'
        '<div class="hero-sub">Browse, inspect, and manage all previous metadata extractions.</div></div>',
        unsafe_allow_html=True,
    )

    history = api.get_history()

    if not history:
        st.markdown(
            '<div class="empty-state"><div class="empty-state-icon">🗂️</div>'
            '<div class="empty-state-text">No extractions yet. Run your first extraction!</div></div>',
            unsafe_allow_html=True,
        )
        return

    total_tables = sum(h.get("summary", {}).get("total_tables", 0) for h in history)
    total_fds    = sum(h.get("summary", {}).get("total_functional_dependencies", 0) for h in history)
    db_types     = len({h["db_type"] for h in history})
    st.markdown(
        '<div class="stat-row">'
        + _stat_card(len(history), "Total Runs",     "#63b3ed")
        + _stat_card(total_tables, "Tables Scanned", "#b794f4")
        + _stat_card(total_fds,    "FDs Found",       "#68d391")
        + _stat_card(db_types,     "DB Types Used",   "#f6ad55")
        + "</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3, 1])
    filter_text = c1.text_input("Filter by database / host", placeholder="Search history…",
                                 label_visibility="collapsed")
    filter_type = c2.selectbox("DB type", ["All"] + list(DB_META.keys()), label_visibility="collapsed")

    filtered = [
        h for h in history
        if (filter_type == "All" or h.get("db_type") == filter_type)
        and (not filter_text or any(
            filter_text.lower() in str(h.get(k, "")).lower()
            for k in ("host", "database", "schema", "db_type")
        ))
    ]
    st.markdown(
        f'<div style="color:#475569;font-size:0.8rem;margin-bottom:1rem">'
        f'Showing {len(filtered)} of {len(history)} runs</div>',
        unsafe_allow_html=True,
    )

    to_delete = None
    for h in filtered:
        m       = DB_META.get(h["db_type"], {"icon": "🗄️", "color": "#94a3b8"})
        summary = h.get("summary", {})
        ts      = h.get("timestamp", "")[:19].replace("T", " ")
        color   = m["color"]

        cols = st.columns([6, 1, 1, 1])
        with cols[0]:
            st.markdown(
                f'<div class="hcard" style="border-left:3px solid {color}">'
                f'<div class="hcard-title">{db_badge(h["db_type"])} &nbsp; '
                f'{h.get("database","—")} / {h.get("schema","—")}</div>'
                f'<div class="hcard-meta">🕐 {ts} &nbsp;|&nbsp; host: {h.get("host","—")}'
                f' &nbsp;|&nbsp; id: {h["id"][:8]}…</div>'
                f'<div style="margin-top:0.7rem;display:flex;gap:1.2rem;font-size:0.82rem">'
                f'<span style="color:#63b3ed">📋 {summary.get("total_tables",0)} tables</span>'
                f'<span style="color:#b794f4">⟶ {summary.get("total_functional_dependencies",0)} FDs</span>'
                f'<span style="color:#68d391">⊆ {summary.get("total_inclusion_dependencies",0)} INDs</span>'
                f'<span style="color:#f6ad55">⇌ {summary.get("total_cardinality_relationships",0)} cardinalities</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⬇ JSON", key=f"dl_{h['id']}"):
                try:
                    report_data = api.get_history_report(h["id"])
                    st.download_button(
                        "Save JSON",
                        data=__import__("json").dumps(report_data, indent=2, default=str),
                        file_name=f"metadata_{h['db_type']}_{h['timestamp'][:10]}.json",
                        mime="application/json",
                        key=f"save_{h['id']}",
                    )
                except Exception as e:
                    st.error(str(e))
        with cols[2]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔎 Load", key=f"load_{h['id']}"):
                try:
                    report = api.get_history_report(h["id"])
                    st.session_state.last_report   = report
                    st.session_state.last_run_id   = h["id"]
                    st.session_state.last_run_meta = h
                    st.session_state.page          = "extract"
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load report: {e}")
        with cols[3]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Del", key=f"del_{h['id']}"):
                to_delete = h["id"]

    if to_delete:
        try:
            api.delete_history(to_delete)
        except Exception as e:
            st.error(f"Delete failed: {e}")
        st.rerun()


# ── View: Search ───────────────────────────────────────────────────────────────
def _search_view() -> None:
    st.markdown(
        '<div class="hero"><div class="hero-title">Search Metadata</div>'
        '<div class="hero-sub">Full-text search across all saved metadata — tables, columns, dependencies.</div></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([4, 1, 1])
    query     = c1.text_input("Search", placeholder="e.g. customer_id, VARCHAR, orders …",
                               label_visibility="collapsed")
    scope_map = {"Tables & Columns": "tables", "Functional Deps": "fds", "All": "all"}
    scope_lbl = c2.selectbox("Scope", list(scope_map.keys()), label_visibility="collapsed")
    filter_db = c3.selectbox("DB type", ["All"] + list(DB_META.keys()), label_visibility="collapsed")

    if not query.strip():
        st.markdown(
            '<div class="empty-state"><div class="empty-state-icon">🔍</div>'
            '<div class="empty-state-text">Type a keyword to search all saved metadata.</div></div>',
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Searching…"):
        try:
            results = api.search(
                q=query.strip(),
                scope=scope_map[scope_lbl],
                db_type=filter_db.lower() if filter_db != "All" else "all",
            )
        except Exception as e:
            st.error(f"Search failed: {e}")
            return

    if not results:
        st.markdown(
            '<div class="empty-state"><div class="empty-state-icon">😶</div>'
            f'<div class="empty-state-text">No results for "<b>{query}</b>".</div></div>',
            unsafe_allow_html=True,
        )
        return

    kind_icons  = {"table": "📋", "column": "🔲", "fd": "⟶"}
    kind_colors = {"table": "#63b3ed", "column": "#b794f4", "fd": "#68d391"}

    st.markdown(
        f'<div style="color:#475569;font-size:0.82rem;margin-bottom:1rem">'
        f'{len(results)} result{"s" if len(results)!=1 else ""} for '
        f'<b style="color:#e2e8f0">{query}</b></div>',
        unsafe_allow_html=True,
    )

    for r in results:
        icon    = kind_icons.get(r["kind"], "•")
        color   = kind_colors.get(r["kind"], "#94a3b8")
        hi_m    = _highlight(r["match"],   query)
        hi_d    = _highlight(r["detail"],  query)
        hi_c    = _highlight(r["context"], query)
        db_bdg  = db_badge(r.get("db_type", ""))

        col_res, col_load = st.columns([8, 1])
        with col_res:
            st.markdown(
                f'<div class="s-result">'
                f'<div class="s-result-title">'
                f'<span style="color:{color};margin-right:0.4rem">{icon}</span>{hi_m}'
                f'<span style="font-size:0.72rem;color:#475569;margin-left:0.6rem">{r["kind"].upper()}</span>'
                f'&nbsp; {db_bdg}</div>'
                f'<div class="s-result-sub">{hi_d} &nbsp;·&nbsp; {hi_c}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_load:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Load", key=f"sr_load_{r['run_id']}_{r['match'][:20]}"):
                try:
                    report = api.get_history_report(r["run_id"])
                    st.session_state.last_report   = report
                    st.session_state.last_run_id   = r["run_id"]
                    st.session_state.last_run_meta = {"db_type": r.get("db_type", "")}
                    st.session_state.page          = "extract"
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    _inject_css()
    _sidebar()
    page = st.session_state.page
    if page == "extract":
        _extract_view()
    elif page == "history":
        _history_view()
    elif page == "search":
        _search_view()


if __name__ == "__main__":
    main()
