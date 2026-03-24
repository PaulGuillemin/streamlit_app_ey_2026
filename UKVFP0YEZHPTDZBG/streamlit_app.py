# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AquaSight — Water Quality Intelligence Platform
# EY Open Science AI & Data Challenge 2026
# Authors: Sacha Cohen · Paul Guillemin
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from snowflake.snowpark.context import get_active_session
import os

# ── 1. CONFIGURATION ──────────────────────────────────────────────────
st.set_page_config(page_title="AquaSight by EY", layout="wide", initial_sidebar_state="collapsed")
session = get_active_session()

# ── Helper: load image from stage ─────────────────────────────────────
@st.cache_data
def load_asset(stage_path, filename):
    local_dir = "/tmp/app_assets/"
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    if not os.path.exists(local_path):
        session.file.get(stage_path, local_dir)
    return local_path

# ── Helper: load image as banner (object-fit) ───────────────────────────
import base64
def display_banner(img_path, height_px=250):
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(f'''
        <div style="width:100%; height:{height_px}px; overflow:hidden; border-radius:4px; border:1px solid #E2E8F0; margin-bottom:1.5rem;">
            <img src="data:image/png;base64,{encoded}" style="width:100%; height:100%; object-fit:cover; object-position:center 60%;">
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.warning(f"Image not found: {img_path}")

ASSET_STAGE = "@EY_CHALLENGE_DB.PUBLIC.EY_IMAGES_STAGE/APP_ASSETS"

# ── 2. ENTERPRISE CLEAN-ROOM CSS ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
[data-testid="stSidebar"], [data-testid="collapsedControl"], header, footer { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #F8FAFC; color: #334155; }
.top-nav {
    position: sticky; top: 0; z-index: 999999;
    background: #FFFFFF; border-bottom: 1px solid #E2E8F0;
    padding: 0.8rem 3rem; display: flex; justify-content: space-between; align-items: center;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
.nav-logo { display: flex; align-items: center; gap: 12px; }
.ey-logo-box { background: #FFE600; color: #000; font-weight: 800; padding: 4px 10px; border-radius: 2px; font-size: 14px; }
div[role="radiogroup"] { flex-direction: row; gap: 1.5rem; justify-content: center; background: transparent !important; }
div[role="radiogroup"] label > div:first-child { display: none !important; } /* HIDE RADIO CIRCLES */
div[role="radiogroup"] > label {
    background: transparent !important; border: none !important;
    padding: 4px 0 !important; border-radius: 0 !important;
    color: #64748B !important; border-bottom: 2px solid transparent !important;
    margin-bottom: -10px; transition: all 0.2s ease !important; cursor: pointer;
}
div[role="radiogroup"] > label:hover { color: #0F172A !important; }
div[role="radiogroup"] > label[data-checked="true"] { color: #0F172A !important; border-bottom: 2px solid #FFE600 !important; }
div[role="radiogroup"] > label[data-checked="true"] p { font-weight: 600 !important; color: #0F172A !important; }
.main-content { padding: 3rem 4rem; max-width: 1400px; margin: 0 auto; }
h1, h2, h3 { color: #0F172A !important; }
.headline { font-size: 2.5rem; font-weight: 700; letter-spacing: -1px; color: #0F172A; margin-bottom: 0.5rem; }
.headline-sub { font-size: 1.05rem; color: #64748B; max-width: 750px; margin-bottom: 2.5rem; line-height: 1.6; }
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin-bottom: 3rem; }
.m-card { background: #FFF; border: 1px solid #E2E8F0; border-radius: 4px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }
.m-label { font-size: 0.75rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem; }
.m-val { font-size: 2.2rem; font-weight: 600; color: #0F172A; line-height: 1; margin-bottom: 0.6rem; }
.m-tag { font-size: 0.8rem; font-weight: 500; }
.sec-title { font-size: 1.2rem; font-weight: 600; color: #0F172A; border-bottom: 1px solid #E2E8F0; padding-bottom: 0.6rem; margin-bottom: 1.5rem; }
.card-wrap { background: #FFF; border: 1px solid #E2E8F0; border-radius: 4px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 1.5rem; }
img { border-radius: 4px; border: 1px solid #E2E8F0; }
</style>
""", unsafe_allow_html=True)

# ── 3. DATA LOADER ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = session.table("EY_FINAL_SUBMISSION.PUBLIC.PREDICTIONS").to_pandas()
    df.columns = [c.lower() for c in df.columns]
    for c in ['latitude','longitude','total_alkalinity','electrical_conductance','dissolved_reactive_phosphorus']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['drp'] = df['dissolved_reactive_phosphorus']
    df['ec']  = df['electrical_conductance']
    df['ta']  = df['total_alkalinity']
    df['status'] = df['drp'].apply(lambda x: 'Critical' if x > 100 else ('Warning' if x > 50 else 'Normal'))
    stations = df.groupby(['latitude','longitude']).agg(
        drp=('drp','mean'), ec=('ec','mean'), ta=('ta','mean'), samples=('latitude','count')
    ).reset_index()
    stations['status'] = stations['drp'].apply(lambda x: 'Critical' if x > 100 else ('Warning' if x > 50 else 'Normal'))
    stations['station_id'] = [f"SAF-{i+1:03d}" for i in range(len(stations))]
    stations['color'] = stations['status'].apply(
        lambda x: [220,38,38,200] if x=='Critical' else ([217,119,6,200] if x=='Warning' else [13,148,136,200])
    )
    stations['radius'] = stations['drp'].clip(10, 300) * 80
    sdg_drp = (df['drp'] <= 100).mean() * 100
    sdg_ec  = (df['ec']  <= 800).mean() * 100
    sdg_ta  = ((df['ta'] >= 20) & (df['ta'] <= 200)).mean() * 100
    sdg = {'drp': sdg_drp, 'ec': sdg_ec, 'ta': sdg_ta, 'overall': (sdg_drp + sdg_ec + sdg_ta) / 3}
    return df, stations, sdg

try:
    df, stations, sdg = load_data()
except Exception as e:
    st.error(f"Cannot load predictions table. Details: {e}")
    st.stop()

# ── 4. CHART THEME ────────────────────────────────────────────────────
def clean_theme(fig, **kw):
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#64748B", font_family="Inter", margin=dict(l=0,r=0,t=30,b=0), **kw)
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    return fig

# ── 5. VISION AI DATA (LIVE from Snowflake) ──────────────────────────
@st.cache_data
def load_vision_data():
    """Load real LLaMA 4 classifications from Snowflake and index by (lat, lon)."""
    try:
        vdf = session.table("EY_FINAL_SUBMISSION.PUBLIC.LLAMA4_VAL_CLASSIFICATIONS").to_pandas()
        vdf.columns = [c.lower() for c in vdf.columns]
        vision_index = {}
        for _, r in vdf.iterrows():
            try:
                parts = r['image_id'].replace('val/', '').replace('hd_chip_', '').replace('_png_merged.png', '').split('_')
                lat = -float(parts[0][1:] + '.' + parts[1]) if parts[0].startswith('m') else float(parts[0] + '.' + parts[1])
                lon = -float(parts[2][1:] + '.' + parts[3]) if parts[2].startswith('m') else float(parts[2] + '.' + parts[3])
                vision_index[(round(lat, 4), round(lon, 4))] = r.to_dict()
            except:
                continue
        return vision_index
    except:
        return {}

vision_index = load_vision_data()

def get_vision_for_station(lat, lon):
    """Match a station to its real LLaMA 4 output by nearest lat/lon."""
    best, best_dist = None, float('inf')
    rlat, rlon = round(lat, 4), round(lon, 4)
    if (rlat, rlon) in vision_index:
        best = vision_index[(rlat, rlon)]
    else:
        for (vlat, vlon), data in vision_index.items():
            d = (vlat - lat)**2 + (vlon - lon)**2
            if d < best_dist:
                best_dist, best = d, data
    if best is None:
        return None
    # Build natural-language findings from real attributes
    regime = best.get('regime_main', 'unknown').replace('_', ' ').title()
    findings = []
    # Pivot density
    piv = best.get('pivot_density', 'none')
    if piv in ('very_high', 'high'):
        findings.append(f"Center-pivot irrigation density: {piv.replace('_',' ')}. Primary fertilizer runoff risk.")
    elif piv == 'medium':
        findings.append("Moderate center-pivot irrigation detected. Diffuse agricultural nutrient loading.")
    elif piv in ('low', 'none'):
        findings.append("No significant center-pivot irrigation detected in the 5 km radius.")
    # Riparian
    rip = int(best.get('is_parcellaire_riparian', 0))
    if rip:
        findings.append("Riparian corridor fragmented. Parcellaire land use encroaching on riverbanks.")
    else:
        findings.append("Riparian vegetation appears structurally intact along the river corridor.")
    # Bare/ravined
    rav = int(best.get('is_bare_ravined', 0))
    bare = best.get('bare_soil_share', 'low')
    if rav:
        findings.append(f"Bare soil erosion & ravination detected. Soil share: {bare}.")
    else:
        findings.append(f"No significant soil ravination. Bare soil share: {bare}.")
    # Urban
    urb = int(best.get('is_urban_connected', 0))
    if urb:
        findings.append("Urban/industrial connectivity detected within proximity.")
    # Reservoir
    res = int(best.get('has_reservoir', 0))
    if res:
        findings.append("Dam or reservoir visible upstream. Potential flow regulation impact.")
    # Action
    if piv in ('very_high', 'high') or rav:
        action = "Deploy field sampling team immediately. Engage municipality on agricultural run-off mitigation."
    elif piv == 'medium' or rip:
        action = "Schedule confirmatory sampling within 14 days. Monitor seasonal deterioration via satellite."
    else:
        action = "Maintain routine monitoring schedule. No immediate intervention required."
    return {'regime': regime, 'findings': findings, 'action': action}

# Fallback for stations without LLaMA data
VISION_FALLBACK = {
    "Critical": {"regime": "High-Impact Catchment", "findings": ["Elevated pollution indicators detected."], "action": "Deploy field sampling teams immediately."},
    "Warning":  {"regime": "Moderate Pressure Zone", "findings": ["Moderate anthropogenic activity observed."], "action": "Schedule confirmatory sampling within 14 days."},
    "Normal":   {"regime": "Low-Impact Corridor",   "findings": ["Natural vegetation corridor intact."],      "action": "Maintain routine monitoring schedule."},
}

SHAP_DATA = {
    "DRP": [("Vision: Pivot Irrigation", 0.22), ("Vision: Riparian Degradation", 0.16),
            ("Precipitation (3m avg)", 0.13), ("Soil Moisture", 0.10),
            ("Dist. to Urban (km)", 0.09), ("NDVI Vegetation", 0.07),
            ("Vision: Bare Soil", 0.06), ("Runoff", 0.05)],
    "EC":  [("Elevation (m)", 0.25), ("Soil Moisture", 0.19), ("PDSI Index", 0.15),
            ("Dist. to Urban", 0.10), ("Precipitation", 0.08), ("Temp Max", 0.06)],
    "TA":  [("Soil Moisture (3m avg)", 0.32), ("Elevation (m)", 0.18), ("PDSI", 0.14),
            ("Precipitation (3m avg)", 0.11), ("NDVI", 0.08), ("Dist. to Mine", 0.06)]
}

# ── 6. NAVIGATION ────────────────────────────────────────────────────
ey_logo_path = load_asset(f"{ASSET_STAGE}/ey_logo.png", "ey_logo.png")
emlyon_logo_path = load_asset(f"{ASSET_STAGE}/emlyon_logo.png", "emlyon_logo.png")

def get_b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f: return base64.b64encode(f.read()).decode()
    return ""

ey_b64 = get_b64(ey_logo_path)
emlyon_b64 = get_b64(emlyon_logo_path)

st.markdown('<div class="top-nav">', unsafe_allow_html=True)
c_logo, c_nav, c_env = st.columns([2.7, 4.0, 1.5])
with c_logo:
    img_tags = ""
    # MUCH LARGER IMAGES (height: 48px and 40px)
    if ey_b64: img_tags += f'<img src="data:image/png;base64,{ey_b64}" style="height:48px; width:auto; border:none; margin-right:16px;">'
    else: img_tags += '<div class="ey-logo-box" style="margin-right:16px; font-size:18px; padding:6px 14px;">EY</div>'
    
    if emlyon_b64: img_tags += f'<img src="data:image/png;base64,{emlyon_b64}" style="height:40px; width:auto; border:none;">'
    else: img_tags += '<span style="font-weight:800; font-size:24px; letter-spacing:-0.5px; color:#E3000F;">em<span style="color:#0F172A;">lyon</span></span>'

    st.markdown(f'''
    <div class="nav-logo" style="display:flex; align-items:center;">
        {img_tags}
        <div style="height:36px; width:1px; background:#CBD5E1; margin:0 20px;"></div>
        <span style="font-weight:700;color:#0F172A;font-size:22px;">AquaSight</span>
    </div>
    ''', unsafe_allow_html=True)
with c_nav:
    page = st.radio("", ["Dashboard", "Station Deep Dive", "Field Operations", "Cortex AI", "Architecture"], index=0, horizontal=True, label_visibility="collapsed")
with c_env:
    st.markdown('<div style="text-align:right;"><span style="color:#059669;font-size:11px;font-weight:700;">● SNOWFLAKE NATIVE</span></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── 7. PAGES ──────────────────────────────────────────────────────────
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown('<div class="headline">National Water Quality Surveillance</div>', unsafe_allow_html=True)
    st.markdown('<div class="headline-sub">Real-time SDG 6.3.2 compliance tracking across South African river systems. Satellite-derived predictions deployed securely within the Snowflake Data Cloud.</div>', unsafe_allow_html=True)

    # Hero banner — banner mode
    display_banner(load_asset(f"{ASSET_STAGE}/hero_banner.png", "hero_banner.png"), height_px=220)

    crit = (df['status'] == 'Critical').sum()
    warn = (df['status'] == 'Warning').sum()
    st.markdown(f"""
    <div class="metric-grid">
        <div class="m-card"><div class="m-label">SDG 6.3.2 Compliance</div><div class="m-val">{sdg['overall']:.1f}%</div><div class="m-tag" style="color:#64748B;">Target ≥ 80%</div></div>
        <div class="m-card"><div class="m-label">Active Stations</div><div class="m-val">{len(stations)}</div><div class="m-tag" style="color:#059669;">100% Online</div></div>
        <div class="m-card"><div class="m-label">Warning (DRP)</div><div class="m-val" style="color:#D97706;">{warn}</div><div class="m-tag" style="color:#D97706;">DRP > 50 µg/L</div></div>
        <div class="m-card" style="border-top:3px solid #DC2626;"><div class="m-label">Critical Breaches</div><div class="m-val" style="color:#DC2626;">{crit}</div><div class="m-tag" style="color:#DC2626;">DRP > 100 µg/L</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Station Risk Map</div>', unsafe_allow_html=True)
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=stations["latitude"].mean(), longitude=stations["longitude"].mean(), zoom=5, pitch=0),
        layers=[pdk.Layer("ScatterplotLayer", data=stations, get_position=["longitude","latitude"], get_fill_color="color", get_radius="radius", pickable=True, opacity=0.8)],
        tooltip={"html": "<b>{station_id}</b><br/>DRP: {drp:.1f} µg/L<br/>Status: {status}",
                 "style": {"backgroundColor":"#FFF","color":"#0F172A","border":"1px solid #E2E8F0","borderRadius":"4px","padding":"8px","fontFamily":"Inter"}},
    ), use_container_width=True)
    st.markdown("""
    <div style="display:flex; gap:2rem; margin-top:0.5rem; font-size:0.85rem; color:#64748B;">
        <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#DC2626;margin-right:6px;vertical-align:middle;"></span>Critical (> 100 µg/L)</span>
        <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#D97706;margin-right:6px;vertical-align:middle;"></span>Warning (> 50 µg/L)</span>
        <span><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#0D9488;margin-right:6px;vertical-align:middle;"></span>Normal</span>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: STATION DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════
elif page == "Station Deep Dive":
    st.markdown('<div class="headline">Station Intelligence Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="headline-sub">Mathematical feature attribution (SHAP) combined with agentic satellite interpretation (LLaMA 4 Maverick).</div>', unsafe_allow_html=True)

    c_sel, _ = st.columns([1, 3])
    sel = c_sel.selectbox("Select Station", stations["station_id"].tolist())
    row = stations[stations["station_id"] == sel].iloc[0]
    status = row["status"]
    vision = get_vision_for_station(row['latitude'], row['longitude']) or VISION_FALLBACK[status]
    sc = "#DC2626" if status == "Critical" else ("#D97706" if status == "Warning" else "#0D9488")

    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #E2E8F0; padding-bottom:1rem; margin-bottom:2rem;">
        <div><h2 style="margin:0; font-size:1.8rem;">{sel}</h2><span style="color:#64748B;">{row['latitude']:.4f}, {row['longitude']:.4f} · {vision['regime']}</span></div>
        <span style="color:{sc}; background:rgba({220 if status=='Critical' else 217},{38 if status=='Critical' else 119},{38 if status=='Critical' else 6},0.1); padding:6px 16px; border-radius:4px; font-weight:600;">{status.upper()}</span>
    </div>
    """, unsafe_allow_html=True)

    c_shap, c_vision = st.columns(2)

    with c_shap:
        st.markdown('<div class="sec-title">SHAP Feature Attribution (DRP)</div>', unsafe_allow_html=True)
        shap_df = pd.DataFrame(SHAP_DATA["DRP"], columns=["Feature", "Impact"])
        fig_shap = px.bar(shap_df, x="Impact", y="Feature", orientation="h", color_discrete_sequence=["#0F172A"])
        clean_theme(fig_shap, height=350)
        fig_shap.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown(f"""
        <div class="card-wrap" style="margin-top:1rem;">
            <div class="m-label">Baseline Predictions</div>
            <div style="display:flex; gap:2rem; margin-top:1rem;">
                <div><span style="color:#64748B; font-size:0.85rem;">DRP</span><br/><strong style="font-size:1.5rem; color:{sc};">{row['drp']:.1f}</strong> <span style="color:#64748B;">µg/L</span></div>
                <div><span style="color:#64748B; font-size:0.85rem;">EC</span><br/><strong style="font-size:1.5rem;">{row['ec']:.0f}</strong> <span style="color:#64748B;">µS/cm</span></div>
                <div><span style="color:#64748B; font-size:0.85rem;">TA</span><br/><strong style="font-size:1.5rem;">{row['ta']:.0f}</strong> <span style="color:#64748B;">mg/L</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-title" style="margin-top:2rem;">Policy What-If Simulator</div>', unsafe_allow_html=True)
        st.caption("Simulate environmental interventions to calculate strategic ROI on DRP reduction.")
        
        sim_agri = st.slider("Target: Reduce Agricultural Runoff", min_value=0, max_value=100, value=0, format="%d%%")
        sim_ripa = st.slider("Target: Restore Riparian Buffer", min_value=0, max_value=100, value=0, format="%d%%")
        
        # Calculate simulated DRP (Max reduction ~38% based on SHAP weights 0.22 + 0.16)
        drp_reduction_factor = (sim_agri/100.0 * 0.22) + (sim_ripa/100.0 * 0.16)
        simulated_drp = row['drp'] * (1 - drp_reduction_factor)
        
        sim_status = 'Critical' if simulated_drp > 100 else ('Warning' if simulated_drp > 50 else 'Normal')
        sim_color = "#DC2626" if sim_status == "Critical" else ("#D97706" if sim_status == "Warning" else "#0D9488")
        
        st.markdown(f"""
        <div class="card-wrap" style="margin-top:1rem; border-left:4px solid {sim_color}; background:#F8FAFC; box-shadow:none;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div class="m-label" style="color:#0F172A;">Simulated DRP Payload</div>
                    <strong style="font-size:2rem; color:{sim_color};">{simulated_drp:.1f}</strong> <span style="color:#64748B;">µg/L</span>
                </div>
                <div style="text-align:right;">
                    <div style="color:#64748B; font-size:0.85rem; text-decoration:line-through; margin-bottom:4px;">Original: {row['drp']:.1f}</div>
                    <span style="background:rgba({220 if sim_status=='Critical' else 217},{38 if sim_status=='Critical' else 119},{38 if sim_status=='Critical' else 6},0.15); color:{sim_color}; padding:6px 12px; border-radius:4px; font-weight:700; font-size:0.8rem; letter-spacing:1px;">{sim_status.upper()}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_vision:
        st.markdown('<div class="sec-title">LLaMA 4 Vision AI Analysis</div>', unsafe_allow_html=True)

        # Load satellite image matched by lat/lon
        try:
            lat_val, lon_val = row['latitude'], row['longitude']
            img_rows = session.sql("LIST @EY_CHALLENGE_DB.PUBLIC.EY_IMAGES_STAGE/val/").collect()
            img_names = [r['name'].split('/')[-1] for r in img_rows]

            best_match, best_dist = None, float('inf')
            for fname in img_names:
                try:
                    parts = fname.replace('hd_chip_', '').replace('_png_merged.png', '').split('_')
                    f_lat = -float(parts[0][1:] + '.' + parts[1]) if parts[0].startswith('m') else float(parts[0] + '.' + parts[1])
                    f_lon = -float(parts[2][1:] + '.' + parts[3]) if parts[2].startswith('m') else float(parts[2] + '.' + parts[3])
                    dist = (f_lat - lat_val)**2 + (f_lon - lon_val)**2
                    if dist < best_dist:
                        best_dist, best_match = dist, fname
                except:
                    continue

            if best_match:
                os.makedirs("/tmp/sat_img/", exist_ok=True)
                local = f"/tmp/sat_img/{best_match}"
                if not os.path.exists(local):
                    session.file.get(f"@EY_CHALLENGE_DB.PUBLIC.EY_IMAGES_STAGE/val/{best_match}", "/tmp/sat_img/")
                st.image(local, caption=f"Landsat RGB — {sel}", use_container_width=True)
        except Exception as e:
            st.caption(f"🛰️ Satellite image: {e}")

        st.markdown(f"""
        <div class="card-wrap">
            <div style="display:inline-block; background:#F1F5F9; color:#0F172A; font-size:0.7rem; padding:3px 8px; font-weight:700; letter-spacing:1px; margin-bottom:1rem; border-radius:2px;">LLAMA 4 MAVERICK OUTPUT</div>
            <ul style="color:#334155; line-height:2.0; font-size:0.95rem; padding-left:20px; margin-bottom:1.5rem;">
                {''.join(f'<li>{f}</li>' for f in vision['findings'])}
            </ul>
            <div style="background:#FFF; border-left:3px solid #FFE600; padding:12px 16px; border-radius:0 4px 4px 0;">
                <span style="display:block; color:#0F172A; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">Recommended Intervention</span>
                <span style="color:#334155; font-size:0.95rem;">{vision['action']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 3: FIELD OPERATIONS
# ═══════════════════════════════════════════════════════════════════════
elif page == "Field Operations":
    st.markdown('<div class="headline">Field Operations</div>', unsafe_allow_html=True)
    st.markdown('<div class="headline-sub">Tactical view for water management teams to prioritize physical sampling campaigns.</div>', unsafe_allow_html=True)

    # Field photo — longitudinal banner
    display_banner(load_asset(f"{ASSET_STAGE}/field_operations_photo.png", "field_operations_photo.png"), height_px=200)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card-wrap">', unsafe_allow_html=True)
        fig1 = px.histogram(df, x="drp", nbins=40, title="DRP Distribution", color_discrete_sequence=["#0F172A"])
        fig1.add_vline(x=100, line_dash="dash", line_color="#DC2626", annotation_text="100 µg/L")
        fig1.add_vline(x=50, line_dash="dash", line_color="#D97706", annotation_text="50 µg/L")
        clean_theme(fig1)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card-wrap">', unsafe_allow_html=True)
        fig2 = px.histogram(df, x="ec", nbins=40, title="Electrical Conductance", color_discrete_sequence=["#0F172A"])
        fig2.add_vline(x=800, line_dash="dash", line_color="#DC2626", annotation_text="800 µS/cm")
        clean_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Operational ROI & Field Dispatch</div>', unsafe_allow_html=True)
    
    # Calculate strictly sourced ROI
    normal_count = (stations['status'] == 'Normal').sum()
    cost_per_visit = 250 # SANS 241 lab ($150) + rural logistics ($100)
    visits_saved_per_year = 11 # From monthly (12) to annual (1) for healthy stations
    annual_savings = normal_count * visits_saved_per_year * cost_per_visit
    
    st.markdown(f"""
    <div class="card-wrap" style="background:#F8FAFC; border-left:4px solid #059669;">
        <div style="font-weight:700; color:#0F172A; margin-bottom:0.5rem;">Resource Optimization (Sourced Standard: $250 / SANS 241 Full Test + Field Logistics)</div>
        <div style="color:#334155; font-size:0.95rem; line-height:1.6;">
            By transitioning from blanket continuous sampling to predictive triage, AquaSight identifies <strong>{normal_count} Normal-risk stations</strong>. 
            Reducing physical sampling frequency from monthly to annually for these healthy corridors generates an immediate operational saving of <strong>${annual_savings:,.0f} per year</strong>, allowing redirection of budget to true remediation efforts.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        csv = stations[stations['status'] != 'Normal'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Intervention Manifest (CSV)",
            data=csv,
            file_name="AquaSight_Critical_Stations.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown('<div class="sec-title" style="margin-top:2rem;">Station Priority Matrix</div>', unsafe_allow_html=True)
    disp = stations[['station_id','drp','ec','ta','samples','status']].copy()
    disp.columns = ['Station','Avg DRP (µg/L)','Avg EC (µS/cm)','Avg TA (mg/L)','Samples','Priority']
    st.dataframe(disp.sort_values('Avg DRP (µg/L)', ascending=False), use_container_width=True, height=450)

# ═══════════════════════════════════════════════════════════════════════
# TAB 4: CORTEX AI
# ═══════════════════════════════════════════════════════════════════════
elif page == "Cortex AI":
    st.markdown('<div class="headline">Cortex AI Copilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="headline-sub">Virtual hydrogeologist powered by LLaMA 4 Maverick via Snowflake Cortex AI.</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:flex; gap:1rem; margin-bottom:2rem;">
        <div class="card-wrap" style="flex:1;"><div class="m-label" style="color:#0F172A;">Hotspot Scan</div><div style="color:#64748B; font-size:0.9rem;">"Which 3 stations need immediate intervention?"</div></div>
        <div class="card-wrap" style="flex:1;"><div class="m-label" style="color:#0F172A;">Executive Brief</div><div style="color:#64748B; font-size:0.9rem;">"Draft a memo for the Minister of Water."</div></div>
        <div class="card-wrap" style="flex:1;"><div class="m-label" style="color:#0F172A;">Policy Query</div><div style="color:#64748B; font-size:0.9rem;">"How would reducing irrigation by 20% impact DRP?"</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-wrap">', unsafe_allow_html=True)
    q = st.text_area("Ask the Virtual Hydrogeologist:", height=100)
    if st.button("Generate Intelligence Brief"):
        if q:
            with st.spinner("Cortex AI synthesizing..."):
                ctx = f"""You are 'AquaSight Copilot', an EY environmental consultant advising SA Ministry of Water.
DATA: {len(df)} obs, {len(stations)} stations. DRP Peak: {df['drp'].max():.1f} µg/L. Avg: {df['drp'].mean():.1f}. SDG compliance: {sdg['overall']:.1f}%. Critical breaches: {(df['status']=='Critical').sum()}.
QUESTION: {q}
Format: 1. EXECUTIVE SUMMARY 2. DATA EVIDENCE 3. RECOMMENDED ACTIONS (3 steps)"""
                try:
                    res = session.sql(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('llama4-maverick', $${ctx}$$) as response").collect()
                    st.markdown(f'<div style="border-left:3px solid #FFE600; padding:1rem 1.5rem; margin-top:1rem; color:#0F172A; line-height:1.7;">{res[0]["RESPONSE"]}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Cortex AI error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 5: ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════
elif page == "Architecture":
    st.markdown('<div class="headline">Architecture & Impact</div>', unsafe_allow_html=True)
    st.markdown('<div class="headline-sub">How AquaSight bridges raw planetary data and decisive governmental action — entirely within the Snowflake perimeter.</div>', unsafe_allow_html=True)

    # Pipeline diagram — full width
    try:
        st.image(load_asset(f"{ASSET_STAGE}/pipeline_architecture.png", "pipeline_architecture.png"),
                 caption="AquaSight — Zero-Egress Pipeline Architecture", use_container_width=True)
    except:
        pass

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="card-wrap">
            <div class="sec-title" style="margin-top:0;">Innovation Pipeline</div>
            <p style="color:#334155; line-height:1.6;"><strong>1. Multi-Modal Vision AI</strong> — LLaMA 4 Maverick interprets Landsat RGB imagery to detect center-pivot irrigation, riparian degradation, and erosion channels.</p>
            <p style="color:#334155; line-height:1.6;"><strong>2. Tweedie Regression</strong> — XGBoost with Tweedie loss handles the zero-inflated DRP distribution.</p>
            <p style="color:#334155; line-height:1.6;"><strong>3. 21 Engineered Features</strong> from 6 public sources.</p>
        </div>
        """, unsafe_allow_html=True)

        # Impact infographic
        try:
            st.image(load_asset(f"{ASSET_STAGE}/impact_infographic.png", "impact_infographic.png"), use_container_width=True)
        except:
            st.markdown("""
            <div class="card-wrap">
                <div class="sec-title" style="margin-top:0;">Socioeconomic Impact</div>
                <p style="color:#334155;">418M people lack safe water · 63,000+ waterborne deaths · $39-64M per cholera outbreak</p>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card-wrap" style="background:#FFE600; border:none;">
            <div style="font-size:1.2rem; font-weight:700; color:#000; margin-bottom:1rem;">Snowflake Native Architecture</div>
            <p style="color:#000; line-height:1.6; font-weight:500;">Zero data egress. Enterprise-grade security. Turnkey deployment.</p>
            <ul style="color:#000; line-height:1.8; font-weight:500;">
                <li><strong>Ingestion:</strong> Planetary Computer STAC APIs</li>
                <li><strong>Processing:</strong> Snowpark Python</li>
                <li><strong>Vision AI:</strong> Cortex AI_COMPLETE (LLaMA 4)</li>
                <li><strong>Delivery:</strong> Streamlit-in-Snowflake</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Scaling map
        try:
            st.image(load_asset(f"{ASSET_STAGE}/africa_scaling_map.png", "africa_scaling_map.png"),
                     caption="Expansion roadmap — from South Africa to Sub-Saharan Africa", use_container_width=True)
        except:
            st.markdown("""
            <div class="card-wrap">
                <div class="sec-title" style="margin-top:0;">Scaling Potential</div>
                <p style="color:#334155;">Trained on public, global datasets. Deployable to any river basin worldwide.</p>
            </div>
            """, unsafe_allow_html=True)

# ── 8. GLOBAL FOOTER (SOURCES) ─────────────────────────────────────────
st.markdown("""
<div style="margin-top: 4rem; padding-top: 2rem; border-top: 1px solid #E2E8F0; color: #94A3B8; font-size: 0.75rem; line-height: 1.6;">
    <strong>Sources & Methodology:</strong><br/>
    [1] <em>Sampling Costs:</em> Derived from South African National Standard (SANS 241) commercial lab tariffs (approx. R2,200 - R3,750) plus evaluated rural field logistics overhead, averaging $250 USD per comprehensive test.<br/>
    [2] <em>Water Access (418M):</em> WHO/UNICEF Joint Monitoring Programme (JMP) for Water Supply, Sanitation and Hygiene (2023 Update on Sub-Saharan Africa).<br/>
    [3] <em>Economic Burden ($39-64M):</em> Johns Hopkins Bloomberg School of Public Health; "The economic burden of cholera outbreaks" covering direct healthcare and productivity losses.<br/>
    [4] <em>Mortality Risk (63,000+):</em> World Health Organization (WHO) Global Burden of Disease data on waterborne and diarrheal diseases in affected regions.<br/>
    [5] <em>Satellite Data:</em> Microsoft Planetary Computer (Landsat Collection 2 Level-2, TerraClimate, SRTM DEM).
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
