# -*- coding: utf-8 -*-
"""
Sistem Prediksi dan Deteksi Dini Bencana Berbasis Integrasi Citra Drone,
Data Meteorologi BMKG, dan Sistem Informasi Geografis (GIS)
==========================================================================
Jalankan dengan:  streamlit run app.py
"""
import os
import io
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
from PIL import Image
from scipy import ndimage

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors as rl_colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")

st.set_page_config(page_title="Deteksi Dini Multi-Bencana Drone-BMKG-GIS", page_icon="🛰️",
                    layout="wide", initial_sidebar_state="expanded")

PRIMARY, GOLD = "#2E5339", "#B08B4F"
st.markdown(f"""
<style>
    .main-header {{ background: linear-gradient(90deg, {PRIMARY} 0%, #3d6b4a 100%);
        padding: 1.3rem 1.8rem; border-radius: 10px; color: white; margin-bottom: 1.2rem; }}
    .main-header h1 {{ margin: 0; font-size: 1.6rem; }}
    .main-header p {{ margin: 0.2rem 0 0 0; opacity: 0.9; font-size: 0.93rem; }}
    .metric-card {{ background: #f7f5ef; border-left: 5px solid {GOLD}; border-radius: 6px;
        padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; }}
    .status-siaga {{ color: #C0392B; font-weight: 700; }}
    .status-waspada {{ color: #D68910; font-weight: 700; }}
    .status-aman {{ color: {PRIMARY}; font-weight: 700; }}
    div[data-testid="stMetricValue"] {{ color: {PRIMARY}; }}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    gis_df = pd.read_csv(os.path.join(DATA, "gis_grid_multihazard.csv"))
    kec_df = pd.read_csv(os.path.join(DATA, "kecamatan_meta.csv"))
    stasiun_df = pd.read_csv(os.path.join(DATA, "stasiun_bmkg.csv"))
    met_df = pd.read_csv(os.path.join(DATA, "data_meteorologi_bmkg.csv"), parse_dates=["tanggal"])
    return gis_df, kec_df, stasiun_df, met_df


@st.cache_resource
def load_models():
    mh_dnn = joblib.load(os.path.join(MODELS, "multihazard_dnn_model.joblib"))
    mh_scaler = joblib.load(os.path.join(MODELS, "multihazard_scaler.joblib"))
    mh_le = joblib.load(os.path.join(MODELS, "multihazard_label_encoder.joblib"))
    mh_features = joblib.load(os.path.join(MODELS, "multihazard_feature_list.joblib"))
    mh_metrics = joblib.load(os.path.join(MODELS, "multihazard_metrics.joblib"))
    img_dnn = joblib.load(os.path.join(MODELS, "image_classifier_dnn.joblib"))
    img_scaler = joblib.load(os.path.join(MODELS, "image_feature_scaler.joblib"))
    img_le = joblib.load(os.path.join(MODELS, "image_label_encoder.joblib"))
    img_metrics = joblib.load(os.path.join(MODELS, "image_metrics.joblib"))
    return mh_dnn, mh_scaler, mh_le, mh_features, mh_metrics, img_dnn, img_scaler, img_le, img_metrics


gis_df, kec_df, stasiun_df, met_df = load_data()
(mh_dnn, mh_scaler, mh_le, MH_FEATURES, mh_metrics,
 img_dnn, img_scaler, img_le, img_metrics) = load_models()

gis_df["nearest_station"] = gis_df.apply(
    lambda r: stasiun_df.loc[((stasiun_df["lat"] - r["lat"]) ** 2 + (stasiun_df["lon"] - r["lon"]) ** 2).idxmin(), "stasiun_id"],
    axis=1,
) if "nearest_station" not in gis_df.columns else gis_df["nearest_station"]

met_by_station = {sid: met_df[met_df["stasiun_id"] == sid].reset_index(drop=True) for sid in stasiun_df["stasiun_id"]}
ALL_DATES = met_by_station[stasiun_df["stasiun_id"].iloc[0]]["tanggal"]
N_DAYS = len(ALL_DATES)


def rolling_sum(arr, w):
    c = np.cumsum(np.insert(arr, 0, 0))
    out = np.zeros_like(arr)
    out[w - 1:] = c[w:] - c[:-w]
    out[: w - 1] = arr[: w - 1].cumsum()
    return out


def extract_image_features(pil_img):
    arr = np.array(pil_img.convert("RGB")).astype(np.float32)
    feats = []
    for c in range(3):
        ch = arr[:, :, c]
        feats += [ch.mean(), ch.std(), np.percentile(ch, 10), np.percentile(ch, 90)]
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=8, range=(0, 255))
        feats += (hist / hist.sum()).tolist()
    gray = arr.mean(axis=2)
    lap = ndimage.laplace(gray)
    feats.append(lap.var())
    sx, sy = ndimage.sobel(gray, axis=0), ndimage.sobel(gray, axis=1)
    grad_mag = np.hypot(sx, sy)
    feats += [grad_mag.mean(), grad_mag.std()]
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    feats += [(g - r).mean(), (b - g).mean(), (r - b).mean()]
    return np.array(feats).reshape(1, -1)


def predict_multihazard_for_day(day_idx, rain_multiplier=1.0):
    r1s, r3s, r7s = {}, {}, {}
    for sid, mdf in met_by_station.items():
        rain = mdf["curah_hujan_mm"].values * rain_multiplier
        r1s[sid] = rain[day_idx]
        r3s[sid] = rolling_sum(rain, 3)[day_idx]
        r7s[sid] = rolling_sum(rain, 7)[day_idx]

    feat = pd.DataFrame({
        "curah_hujan_1h": gis_df["nearest_station"].map(r1s),
        "curah_hujan_3h": gis_df["nearest_station"].map(r3s),
        "curah_hujan_7h": gis_df["nearest_station"].map(r7s),
        "elevation_m": gis_df["elevation_m"], "slope_deg": gis_df["slope_deg"],
        "dist_volcano_km": gis_df["dist_volcano_km"], "dist_river_km": gis_df["dist_river_km"],
        "land_cover": gis_df["land_cover"], "soil_infiltration_mmhr": gis_df["soil_infiltration_mmhr"],
        "drainage_density": gis_df["drainage_density"],
        "indeks_drone_genangan": gis_df["indeks_drone_genangan"],
        "indeks_drone_retakan": gis_df["indeks_drone_retakan"],
        "indeks_drone_lahan_terbuka": gis_df["indeks_drone_lahan_terbuka"],
    })[MH_FEATURES]

    X_scaled = mh_scaler.transform(feat.values)
    proba = mh_dnn.predict_proba(X_scaled)
    pred_class = mh_le.inverse_transform(np.argmax(proba, axis=1))
    return pred_class, proba, mh_le.classes_


def make_pdf_report(title, summary_rows, fig_paths=None, table_data=None, table_title=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleC", parent=styles["Title"], textColor=rl_colors.HexColor(PRIMARY), fontSize=16)
    h2_style = ParagraphStyle("H2C", parent=styles["Heading2"], textColor=rl_colors.HexColor(PRIMARY))
    normal = styles["Normal"]
    story = [Paragraph(title, title_style), Spacer(1, 10)]
    story.append(Paragraph("Sistem Prediksi dan Deteksi Dini Bencana — Drone, BMKG & GIS", normal))
    story.append(Spacer(1, 14))
    story.append(Paragraph("Ringkasan Hasil Analisis", h2_style))
    for row in summary_rows:
        story.append(Paragraph(f"• {row}", normal))
    story.append(Spacer(1, 12))
    if table_data:
        story.append(Paragraph(table_title or "Detail Data", h2_style))
        t = Table(table_data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(PRIMARY)),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor("#F7F5EF"), rl_colors.white]),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))
    if fig_paths:
        for fp, cap in fig_paths:
            if os.path.exists(fp):
                story.append(RLImage(fp, width=15 * cm, height=15 * cm * 0.62, kind="proportional"))
                story.append(Paragraph(f"<i>{cap}</i>", normal))
                story.append(Spacer(1, 10))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Catatan: seluruh data pada sistem ini bersifat sintetis untuk keperluan riset dan "
                            "demonstrasi. Lihat laporan lengkap untuk metodologi dan keterbatasan.", normal))
    doc.build(story)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🛰️ Sistem Prediksi dan Deteksi Dini Bencana Multi-Hazard</h1>
    <p>Integrasi Citra Drone, Data Meteorologi BMKG, dan Sistem Informasi Geografis (GIS) —
    Studi Kasus DAS Marapi–Singgalang, Sumatera Barat</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Panel Kontrol")
    page = st.radio("Navigasi", [
        "🏠 Beranda", "📷 Analisis Citra Drone", "🌦️ Dashboard Meteorologi BMKG",
        "🗺️ Peta GIS Multi-Hazard", "📈 Prediksi Spatio-Temporal", "🔔 Peringatan Dini",
        "📊 Evaluasi Model", "🖨️ Cetak Laporan PDF", "📖 Tentang & Panduan",
    ], label_visibility="collapsed")
    st.markdown("---")
    default_day = N_DAYS - 30
    day_idx = st.slider("Indeks hari (dataset historis)", 7, N_DAYS - 1, default_day)
    selected_date = ALL_DATES.iloc[day_idx]
    st.caption(f"Tanggal: **{selected_date.strftime('%d %B %Y')}**")
    rain_multiplier = st.slider("Faktor pengali curah hujan (skenario)", 0.5, 3.0, 1.0, 0.1)
    st.markdown("---")
    st.caption("👤 Ade Kurniawan, M.Kom. — Teknik Informatika")
    st.caption("Studi kasus: DAS Marapi–Singgalang, Sumatera Barat")

pred_class, proba, classes = predict_multihazard_for_day(day_idx, rain_multiplier)
gis_disp = gis_df.copy()
gis_disp["prediksi"] = pred_class
gis_disp["prob_max"] = proba.max(axis=1)
color_map_hz = {"AMAN": "#2E5339", "BANJIR": "#2E86C1", "LONGSOR": "#D68910", "GALODO": "#C0392B"}

# ===========================================================================
if page == "🏠 Beranda":
    counts = gis_disp["prediksi"].value_counts()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sel Grid", f"{len(gis_disp):,}")
    c2.metric("🟢 AMAN", f"{counts.get('AMAN', 0)}")
    c3.metric("🔵 BANJIR", f"{counts.get('BANJIR', 0)}")
    c4.metric("🟠 LONGSOR", f"{counts.get('LONGSOR', 0)}")
    c5.metric("🔴 GALODO", f"{counts.get('GALODO', 0)}")
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📍 Peta Prediksi Multi-Hazard Saat Ini")
        m = folium.Map(location=[-0.40, 100.42], zoom_start=10, tiles="CartoDB positron")
        for _, row in gis_disp.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]], radius=3, color=color_map_hz[row["prediksi"]],
                fill=True, fill_opacity=0.75, weight=0,
                popup=f"Sel {int(row['cell_id'])} | {row['prediksi']} ({row['prob_max']:.0%})",
            ).add_to(m)
        for _, s in stasiun_df.iterrows():
            folium.Marker([s["lat"], s["lon"]], icon=folium.Icon(color="red", icon="cloud", prefix="fa"),
                          tooltip=s["nama"]).add_to(m)
        for _, k in kec_df.iterrows():
            folium.Marker([k["lat"], k["lon"]], icon=folium.Icon(color="blue", icon="home", prefix="fa"),
                          tooltip=f"{k['kecamatan']}, {k['kabupaten']}").add_to(m)
        st_folium(m, height=470, use_container_width=True, key="beranda_map")
    with col2:
        st.subheader("🚦 Ringkasan Status")
        n_bahaya = (gis_disp["prediksi"] != "AMAN").sum()
        if n_bahaya > 0:
            top_hazard = gis_disp[gis_disp["prediksi"] != "AMAN"]["prediksi"].value_counts().idxmax()
            st.markdown(f"""<div class="metric-card"><div style="font-size:0.85rem;color:#666;">
                Bencana Dominan Terdeteksi</div><div class="status-siaga" style="font-size:1.3rem;">
                {top_hazard}</div><div style="font-size:0.85rem;color:#666;">{n_bahaya} sel berisiko dari {len(gis_disp)}</div></div>""",
                        unsafe_allow_html=True)
        else:
            st.markdown("""<div class="metric-card"><div class="status-aman" style="font-size:1.3rem;">
                SELURUH WILAYAH AMAN</div></div>""", unsafe_allow_html=True)
        top5 = gis_disp[gis_disp["prediksi"] != "AMAN"].nlargest(5, "prob_max")[["cell_id", "prediksi", "prob_max"]]
        if len(top5) > 0:
            st.markdown("**5 Sel Berisiko Tertinggi**")
            st.dataframe(top5.rename(columns={"cell_id": "ID Sel", "prediksi": "Jenis", "prob_max": "Probabilitas"}),
                        hide_index=True, use_container_width=True)
        st.info("ℹ️ Data pada dashboard ini bersifat **sintetis** untuk riset & demonstrasi model.", icon="ℹ️")

# ===========================================================================
elif page == "📷 Analisis Citra Drone":
    st.subheader("📷 Klasifikasi Citra Drone — Deteksi Indikator Visual Bencana")
    st.markdown("Unggah citra ubin (tile) drone, atau pilih salah satu contoh untuk melihat hasil klasifikasi "
                "model Deep Learning (5 kelas: vegetasi normal, genangan air, retakan tanah, lahan terbuka "
                "vulkanik, permukiman).")

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Unggah citra (JPG/PNG)", type=["jpg", "jpeg", "png"])
        st.markdown("**Atau pilih contoh citra:**")
        img_dir = os.path.join(DATA, "drone_images")
        classes_dir = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
        example_class = st.selectbox("Kelas contoh", classes_dir)
        example_files = sorted(os.listdir(os.path.join(img_dir, example_class)))[:5]
        example_choice = st.selectbox("File contoh", example_files)

    target_img = None
    if uploaded is not None:
        target_img = Image.open(uploaded)
    elif example_choice:
        target_img = Image.open(os.path.join(img_dir, example_class, example_choice))

    if target_img is not None:
        with col2:
            st.image(target_img, caption="Citra yang dianalisis", width=280)
        feats = extract_image_features(target_img)
        feats_scaled = img_scaler.transform(feats)
        proba_img = img_dnn.predict_proba(feats_scaled)[0]
        pred_idx = np.argmax(proba_img)
        pred_label = img_le.classes_[pred_idx]
        confidence = proba_img[pred_idx]

        st.markdown("---")
        st.markdown(f"### Hasil Klasifikasi: **{pred_label.replace('_', ' ').title()}** ({confidence:.1%} keyakinan)")
        proba_df = pd.DataFrame({"Kelas": [c.replace("_", " ").title() for c in img_le.classes_], "Probabilitas": proba_img})
        fig = px.bar(proba_df.sort_values("Probabilitas"), x="Probabilitas", y="Kelas", orientation="h",
                     color="Probabilitas", color_continuous_scale=["#EAF2EC", PRIMARY])
        fig.update_layout(height=300, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        interpretasi = {
            "genangan_air": "🔵 Indikasi genangan/permukaan air terdeteksi — dapat menjadi indikator awal risiko **banjir** bila terjadi di area permukiman/pertanian.",
            "retakan_tanah": "🟠 Indikasi retakan/pergerakan tanah terdeteksi — dapat menjadi indikator awal risiko **longsor**, khususnya pada lereng curam.",
            "lahan_terbuka_vulkanik": "🔴 Indikasi lahan terbuka dengan material vulkanik lepas — dapat menjadi indikator awal risiko **galodo/lahar dingin** bila berada di alur sungai dekat gunung berapi.",
            "vegetasi_normal": "🟢 Tutupan vegetasi normal terdeteksi — indikasi kondisi lahan stabil, tidak menunjukkan tanda bahaya visual.",
            "permukiman": "🏘️ Area permukiman terdeteksi — relevan sebagai referensi elemen berisiko (exposure) dalam analisis kerentanan.",
        }
        st.info(interpretasi.get(pred_label, ""), icon="🔎")

# ===========================================================================
elif page == "🌦️ Dashboard Meteorologi BMKG":
    st.subheader("🌦️ Dashboard Data Meteorologi BMKG")
    station_names = stasiun_df["nama"] + " (" + stasiun_df["stasiun_id"] + ")"
    selected_station = st.selectbox("Pilih Stasiun", station_names)
    sid = stasiun_df.iloc[station_names.tolist().index(selected_station)]["stasiun_id"]
    mdf = met_by_station[sid]

    window = 90
    start = max(0, day_idx - window)
    sub = mdf.iloc[start:day_idx + 1]

    c1, c2, c3, c4 = st.columns(4)
    latest = mdf.iloc[day_idx]
    c1.metric("Curah Hujan Hari Ini", f"{latest['curah_hujan_mm']:.1f} mm")
    c2.metric("Suhu", f"{latest['suhu_C']:.1f} °C")
    c3.metric("Kelembaban", f"{latest['kelembaban_persen']:.0f}%")
    c4.metric("Kecepatan Angin", f"{latest['kecepatan_angin_ms']:.1f} m/s")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=sub["tanggal"], y=sub["curah_hujan_mm"], name="Curah Hujan (mm)", marker_color="#7FB3D5"))
    fig.update_layout(title=f"Curah Hujan Harian — {selected_station}", height=320)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sub["tanggal"], y=sub["suhu_C"], name="Suhu (°C)", line=dict(color="#C0392B")))
        fig2.update_layout(title="Suhu Udara", height=300)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=sub["tanggal"], y=sub["kelembaban_persen"], name="Kelembaban (%)", line=dict(color=PRIMARY)))
        fig3.update_layout(title="Kelembaban Udara", height=300)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### 📋 Tabel Data Meteorologi (30 hari terakhir dari tanggal terpilih)")
    st.dataframe(mdf.iloc[max(0, day_idx - 30):day_idx + 1][
        ["tanggal", "curah_hujan_mm", "suhu_C", "kelembaban_persen", "kecepatan_angin_ms", "tekanan_udara_hPa"]
    ].rename(columns={"tanggal": "Tanggal", "curah_hujan_mm": "Hujan (mm)", "suhu_C": "Suhu (°C)",
                       "kelembaban_persen": "Kelembaban (%)", "kecepatan_angin_ms": "Angin (m/s)",
                       "tekanan_udara_hPa": "Tekanan (hPa)"}),
                hide_index=True, use_container_width=True, height=280)

# ===========================================================================
elif page == "🗺️ Peta GIS Multi-Hazard":
    st.subheader("🗺️ Peta Interaktif GIS Multi-Hazard")
    tab1, tab2, tab3 = st.tabs(["Peta Prediksi Gabungan", "Peta Kerentanan per Jenis", "Peta Layer GIS Dasar"])

    with tab1:
        m1 = folium.Map(location=[-0.40, 100.42], zoom_start=10, tiles="CartoDB positron")
        for _, row in gis_disp.iterrows():
            folium.CircleMarker(location=[row["lat"], row["lon"]], radius=3.3, color=color_map_hz[row["prediksi"]],
                                fill=True, fill_opacity=0.8, weight=0,
                                popup=f"Sel {int(row['cell_id'])}<br>{row['prediksi']} ({row['prob_max']:.0%})").add_to(m1)
        legend_html = """<div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; background:white;
            padding:10px 14px; border-radius:6px; box-shadow:0 1px 6px rgba(0,0,0,0.3); font-size:13px;">
            <b>Legenda</b><br><span style="color:#2E5339;">●</span> AMAN<br>
            <span style="color:#2E86C1;">●</span> BANJIR<br><span style="color:#D68910;">●</span> LONGSOR<br>
            <span style="color:#C0392B;">●</span> GALODO</div>"""
        m1.get_root().html.add_child(folium.Element(legend_html))
        st_folium(m1, height=520, use_container_width=True, key="pred_map")

    with tab2:
        hz_choice = st.selectbox("Pilih jenis kerentanan", ["Banjir", "Longsor", "Galodo"])
        col_map = {"Banjir": "flood_susceptibility", "Longsor": "landslide_susceptibility", "Galodo": "galodo_susceptibility"}
        col = col_map[hz_choice]
        m2 = folium.Map(location=[-0.40, 100.42], zoom_start=10, tiles="CartoDB positron")
        heat_data = gis_df[["lat", "lon", col]].values.tolist()
        HeatMap(heat_data, radius=16, blur=20).add_to(m2)
        st_folium(m2, height=500, use_container_width=True, key="hz_heatmap")

    with tab3:
        layer_choice = st.selectbox("Pilih layer GIS", ["Kemiringan Lereng", "Elevasi", "Jarak Sungai",
                                                          "Jarak Gunung Berapi", "Tutupan Lahan", "Densitas Drainase"])
        col_map2 = {"Kemiringan Lereng": "slope_deg", "Elevasi": "elevation_m", "Jarak Sungai": "dist_river_km",
                    "Jarak Gunung Berapi": "dist_volcano_km", "Tutupan Lahan": "land_cover", "Densitas Drainase": "drainage_density"}
        col2_ = col_map2[layer_choice]
        m3 = folium.Map(location=[-0.40, 100.42], zoom_start=10, tiles="CartoDB positron")
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        vals = gis_df[col2_].values
        norm = mcolors.Normalize(vmin=vals.min(), vmax=vals.max())
        cmap = cm.get_cmap("YlOrRd")
        for _, row in gis_df.iterrows():
            hexcolor = mcolors.to_hex(cmap(norm(row[col2_])))
            folium.CircleMarker(location=[row["lat"], row["lon"]], radius=3, color=hexcolor,
                                fill=True, fill_opacity=0.8, weight=0).add_to(m3)
        st_folium(m3, height=500, use_container_width=True, key="gis_layer_map")

# ===========================================================================
elif page == "📈 Prediksi Spatio-Temporal":
    st.subheader("📈 Prediksi Spatio-Temporal per Kecamatan")
    kec_names = kec_df["kecamatan"] + " (" + kec_df["kabupaten"] + ")"
    selected_kec = st.selectbox("Pilih Kecamatan", kec_names)
    kec_row = kec_df.iloc[kec_names.tolist().index(selected_kec)]
    gis_df["dist_to_kec"] = np.sqrt((gis_df["lat"] - kec_row["lat"]) ** 2 + (gis_df["lon"] - kec_row["lon"]) ** 2)
    nearest_cell = int(gis_df.loc[gis_df["dist_to_kec"].idxmin(), "cell_id"])
    nearest_row = gis_df[gis_df["cell_id"] == nearest_cell].iloc[0]
    sid = nearest_row["nearest_station"]
    mdf = met_by_station[sid]

    st.markdown(f"**Sel grid terdekat**: #{nearest_cell} | **Stasiun BMKG rujukan**: {sid} | "
                f"**Kerentanan**: Banjir {nearest_row['flood_susceptibility']:.2f}, "
                f"Longsor {nearest_row['landslide_susceptibility']:.2f}, Galodo {nearest_row['galodo_susceptibility']:.2f}")

    st.markdown("#### 🔮 Proyeksi Status Bencana 7 Hari ke Depan")
    proj_rows = []
    for fd in range(0, 8):
        target_day = min(day_idx + fd, N_DAYS - 1)
        pc, pb, cls = predict_multihazard_for_day(target_day, rain_multiplier)
        idx = gis_df.index[gis_df["cell_id"] == nearest_cell][0]
        proj_rows.append({"Hari ke-": fd, "Tanggal": (selected_date + timedelta(days=fd)).strftime("%d %b %Y"),
                           "Prediksi": pc[idx], "Probabilitas": pb[idx].max()})
    proj_df = pd.DataFrame(proj_rows)
    fig = px.bar(proj_df, x="Tanggal", y="Probabilitas", color="Prediksi",
                 color_discrete_map=color_map_hz)
    fig.update_layout(height=380, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(proj_df, hide_index=True, use_container_width=True)

    st.markdown("#### 🌧️ Riwayat Curah Hujan Stasiun Rujukan")
    window = 60
    start = max(0, day_idx - window)
    sub = mdf.iloc[start:day_idx + 1]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=sub["tanggal"], y=sub["curah_hujan_mm"], marker_color="#7FB3D5"))
    fig2.update_layout(title=f"Curah Hujan — {sid}", height=300)
    st.plotly_chart(fig2, use_container_width=True)

# ===========================================================================
elif page == "🔔 Peringatan Dini":
    st.subheader("🔔 Sistem Peringatan Dini Multi-Hazard")
    alert_rows = []
    for _, k in kec_df.iterrows():
        d = np.sqrt((gis_disp["lat"] - k["lat"]) ** 2 + (gis_disp["lon"] - k["lon"]) ** 2)
        nearest = gis_disp.loc[d.idxmin()]
        alert_rows.append({"Kecamatan": k["kecamatan"], "Kabupaten/Kota": k["kabupaten"],
                            "Status": nearest["prediksi"], "Probabilitas": round(nearest["prob_max"], 3)})
    alert_df = pd.DataFrame(alert_rows).sort_values("Probabilitas", ascending=False)
    n_bahaya = (alert_df["Status"] != "AMAN").sum()

    if n_bahaya > 0:
        hz_list = alert_df[alert_df["Status"] != "AMAN"]["Status"].value_counts()
        st.error(f"🚨 **PERINGATAN**: {n_bahaya} kecamatan berstatus bahaya pada {selected_date.strftime('%d %B %Y')} "
                 f"— {', '.join([f'{k}: {v}' for k, v in hz_list.items()])}")
    else:
        st.success("✅ Seluruh kecamatan berstatus AMAN.")

    def style_status(val):
        return f"color: {color_map_hz.get(val, 'black')}; font-weight: 700;"
    st.dataframe(alert_df.style.applymap(style_status, subset=["Status"]), hide_index=True, use_container_width=True)

    st.markdown("#### 📋 Rekomendasi Tindakan")
    recs = {
        "BANJIR": "Siagakan pompa air & jalur drainase; siapkan evakuasi warga di dataran rendah/bantaran sungai; pantau tinggi muka air.",
        "LONGSOR": "Larang aktivitas di lereng curam; siapkan jalur evakuasi alternatif; pantau retakan tanah baru di lapangan.",
        "GALODO": "Evakuasi warga di alur sungai dekat gunung berapi; tutup akses jalan rawan; koordinasi BPBD & PVMBG.",
        "AMAN": "Pemantauan rutin; edukasi mitigasi berkelanjutan kepada masyarakat.",
    }
    for status in ["GALODO", "LONGSOR", "BANJIR", "AMAN"]:
        subset = alert_df[alert_df["Status"] == status]
        if len(subset) > 0:
            st.markdown(f"**{status}** ({len(subset)} kecamatan): {recs[status]}")

# ===========================================================================
elif page == "📊 Evaluasi Model":
    st.subheader("📊 Evaluasi Kinerja Model")
    tab1, tab2 = st.tabs(["Model Fusi Multi-Hazard", "Model Klasifikasi Citra Drone"])
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Akurasi DNN", f"{mh_metrics['acc_dnn']:.2%}")
        c2.metric("F1-Macro DNN", f"{mh_metrics['f1_dnn']:.3f}")
        c3.metric("Akurasi RF", f"{mh_metrics['acc_rf']:.2%}")
        c4.metric("Data Uji", f"{mh_metrics['n_test']:,}")
        art = os.path.join(MODELS, "artifacts")
        col1, col2 = st.columns(2)
        with col1:
            st.image(os.path.join(art, "confusion_matrix_multihazard.png"), use_container_width=True)
        with col2:
            st.image(os.path.join(art, "training_loss_multihazard.png"), use_container_width=True)
        st.image(os.path.join(art, "feature_importance_multihazard.png"), use_container_width=True)
    with tab2:
        c1, c2, c3 = st.columns(3)
        c1.metric("Akurasi DNN", f"{img_metrics['acc_dnn']:.2%}")
        c2.metric("Akurasi RF", f"{img_metrics['acc_rf']:.2%}")
        c3.metric("Data Uji", f"{img_metrics['n_test']}")
        art2 = os.path.join(MODELS, "artifacts_citra")
        col1, col2 = st.columns(2)
        with col1:
            st.image(os.path.join(art2, "confusion_matrix_citra.png"), use_container_width=True)
        with col2:
            st.image(os.path.join(art2, "training_loss_citra.png"), use_container_width=True)
        st.image(os.path.join(art2, "feature_importance_citra.png"), use_container_width=True)

# ===========================================================================
elif page == "🖨️ Cetak Laporan PDF":
    st.subheader("🖨️ Cetak Laporan Hasil Analisis (PDF)")
    st.markdown("Hasilkan laporan PDF ringkas berisi ringkasan status wilayah, peta kerentanan, dan tabel "
                "peringatan dini berdasarkan tanggal dan skenario yang dipilih pada panel kontrol.")

    if st.button("📄 Buat Laporan PDF", type="primary"):
        counts = gis_disp["prediksi"].value_counts()
        summary_rows = [
            f"Tanggal analisis: {selected_date.strftime('%d %B %Y')} (faktor hujan ×{rain_multiplier})",
            f"Total sel grid dipantau: {len(gis_disp)}",
            f"Sel AMAN: {counts.get('AMAN', 0)} | BANJIR: {counts.get('BANJIR', 0)} | "
            f"LONGSOR: {counts.get('LONGSOR', 0)} | GALODO: {counts.get('GALODO', 0)}",
            f"Akurasi model fusi multi-hazard (data uji): {mh_metrics['acc_dnn']:.2%}",
        ]
        alert_rows = []
        for _, k in kec_df.iterrows():
            d = np.sqrt((gis_disp["lat"] - k["lat"]) ** 2 + (gis_disp["lon"] - k["lon"]) ** 2)
            nearest = gis_disp.loc[d.idxmin()]
            alert_rows.append([k["kecamatan"], k["kabupaten"], nearest["prediksi"], f"{nearest['prob_max']:.0%}"])
        table_data = [["Kecamatan", "Kabupaten/Kota", "Status", "Probabilitas"]] + alert_rows

        fig_paths = [
            (os.path.join(BASE, "report_assets", "peta_kerentanan_multihazard.png"), "Peta Kerentanan Multi-Hazard Wilayah Studi"),
        ]
        pdf_buf = make_pdf_report(
            "Laporan Hasil Analisis Deteksi Dini Multi-Hazard",
            summary_rows, fig_paths=fig_paths, table_data=table_data,
            table_title="Status Peringatan Dini per Kecamatan",
        )
        st.download_button("⬇️ Unduh Laporan PDF", data=pdf_buf, file_name=f"Laporan_Bencana_{selected_date.strftime('%Y%m%d')}.pdf",
                            mime="application/pdf")
        st.success("Laporan PDF berhasil dibuat. Klik tombol di atas untuk mengunduh.")

# ===========================================================================
elif page == "📖 Tentang & Panduan":
    st.subheader("📖 Tentang Aplikasi & Panduan Penggunaan")
    st.markdown("""
    ### Tentang
    Aplikasi ini adalah purwarupa **Sistem Prediksi dan Deteksi Dini Bencana Multi-Hazard**
    (Banjir, Longsor, Galodo) yang mengintegrasikan tiga sumber data: **citra drone** (indikator
    visual permukaan lahan), **data meteorologi BMKG** (curah hujan, suhu, kelembaban, angin), dan
    **Sistem Informasi Geografis/GIS** (topografi, jaringan sungai, tutupan lahan).

    ### Alur Kerja Sistem
    1. **Citra drone** diklasifikasikan menjadi 5 indikator visual menggunakan model Deep Learning.
    2. **Data BMKG** diproses menjadi fitur temporal (curah hujan kumulatif 1/3/7 hari).
    3. **Data GIS** dianalisis menjadi skor kerentanan tiga jenis bencana.
    4. Ketiga sumber fitur **difusikan** oleh model Deep Neural Network untuk memprediksi status
       bencana (AMAN/BANJIR/LONGSOR/GALODO) per sel grid per hari.
    5. Hasil ditampilkan pada **WebGIS interaktif** dan dapat **dicetak sebagai laporan PDF**.

    ### Panduan Navigasi
    - **📷 Analisis Citra Drone**: unggah/pilih contoh citra untuk klasifikasi indikator visual.
    - **🌦️ Dashboard Meteorologi BMKG**: eksplorasi data cuaca per stasiun.
    - **🗺️ Peta GIS Multi-Hazard**: peta prediksi gabungan, kerentanan per jenis, dan layer GIS dasar.
    - **📈 Prediksi Spatio-Temporal**: analisis & proyeksi 7 hari per kecamatan.
    - **🔔 Peringatan Dini**: status dan rekomendasi tindakan per kecamatan.
    - **📊 Evaluasi Model**: metrik kinerja kedua model (fusi multi-hazard & klasifikasi citra).
    - **🖨️ Cetak Laporan PDF**: unduh ringkasan hasil analisis dalam format PDF.

    ---
    *Seluruh data (citra drone, data BMKG, dan basis data GIS) pada purwarupa ini bersifat
    **sintetis**, dibangun berdasarkan karakteristik fisik nyata wilayah studi untuk keperluan
    riset dan demonstrasi. Detail metodologi lengkap tersedia pada laporan Bab I–V yang menyertai
    aplikasi ini.*
    """)

st.markdown("---")
st.caption("Sistem Prediksi dan Deteksi Dini Bencana Multi-Hazard © 2026 — Drone · BMKG · GIS · Deep Learning")
