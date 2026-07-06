import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 1. KONFIGURASI DAN TAMPILAN HALAMAN
# ==========================================
st.set_page_config(
    page_title="OptiSolve Pro | Linear Programming Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kustomisasi CSS untuk tampilan kartu metrik yang lebih modern
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .main-header {
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1rem;
        margin-top: 5px;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>⚡ OptiSolve Pro: Studio Optimasi Linear 2D</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Sistem Pendukung Keputusan Berbasis Analisis Grafik dan Aljabar Matriks</p>", unsafe_allow_html=True)

# ==========================================
# 2. PANEL INPUT DATA (SIDEBAR INTERAKTIF)
# ==========================================
with st.sidebar:
    st.header("🎛️ Pengaturan Simulasi")
    st.caption("Tentukan parameter studi kasus bisnis Anda:")
    
    with st.expander("🏷️ Penamaan Variabel Keputusan", expanded=True):
        nama_var_x = st.text_input("Nama Produk/Variabel 1 (X):", value="Sepatu")
        nama_var_y = st.text_input("Nama Produk/Variabel 2 (Y):", value="Tas")
        
    st.markdown("---")
    
    st.subheader("🎯 Target Optimasi (Fungsi Z)")
    tujuan_opt = st.selectbox("Pilih Target:", ["Maksimalisasi (Profit/Keuntungan)", "Minimalisasi (Biaya/Cost)"])
    is_max = "Maksimalisasi" in tujuan_opt
    
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        c1 = st.number_input(f"Koef. {nama_var_x} (C1)", value=120.0, step=10.0)
    with col_z2:
        c2 = st.number_input(f"Koef. {nama_var_y} (C2)", value=150.0, step=10.0)
        
    st.markdown("---")
    st.subheader("🚧 Batasan Kendala (Constraints)")
    
    # Kendala 1
    st.markdown("**Kendala 1:**")
    c_k1_1, c_k1_2, c_k1_3 = st.columns([2, 1.5, 2])
    with c_k1_1: a1 = st.number_input("Koef X1", value=1.0, step=1.0, key="a1")
    with c_k1_2: op1 = st.selectbox("Tanda", ["≤", "≥"], key="op1")
    with c_k1_3: b1 = st.number_input("Koef Y1", value=1.0, step=1.0, key="b1")
    k1 = st.number_input("Kapasitas/Batas Kendala 1 (K1):", value=100.0, step=10.0, key="k1")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Kendala 2
    st.markdown("**Kendala 2:**")
    c_k2_1, c_k2_2, c_k2_3 = st.columns([2, 1.5, 2])
    with c_k2_1: a2 = st.number_input("Koef X2", value=2.0, step=1.0, key="a2")
    with c_k2_2: op2 = st.selectbox("Tanda", ["≤", "≥"], key="op2")
    with c_k2_3: b2 = st.number_input("Koef Y2", value=4.0, step=1.0, key="b2")
    k2 = st.number_input("Kapasitas/Batas Kendala 2 (K2):", value=300.0, step=10.0, key="k2")

# ==========================================
# 3. MESIN PERHITUNGAN MATEMATIKA UNIVERSAL
# ==========================================

# Fungsi untuk mengecek apakah suatu titik memenuhi semua kendala
def cek_layak(x, y):
    if x < -1e-5 or y < -1e-5:
        return False
    # Cek Kendala 1
    syarat1 = (a1*x + b1*y <= k1 + 1e-5) if op1 == "≤" else (a1*x + b1*y >= k1 - 1e-5)
    # Cek Kendala 2
    syarat2 = (a2*x + b2*y <= k2 + 1e-5) if op2 == "≤" else (a2*x + b2*y >= k2 - 1e-5)
    return syarat1 and syarat2

# Mengumpulkan 6 kandidat titik potong dari seluruh garis pembatas (Garis 1, Garis 2, Sumbu X, Sumbu Y)
kandidat_titik = []

# 1. Titik (0,0)
kandidat_titik.append((0.0, 0.0))
# 2. Titik potong Kendala 1 dengan sumbu
if a1 != 0: kandidat_titik.append((k1/a1, 0.0))
if b1 != 0: kandidat_titik.append((0.0, k1/b1))
# 3. Titik potong Kendala 2 dengan sumbu
if a2 != 0: kandidat_titik.append((k2/a2, 0.0))
if b2 != 0: kandidat_titik.append((0.0, k2/b2))
# 4. Titik potong antara Kendala 1 dan Kendala 2 (Aljabar Linear Matriks NumPy)
try:
    A_mat = np.array([[a1, b1], [a2, b2]])
    B_mat = np.array([k1, k2])
    pt_potong = np.linalg.solve(A_mat, B_mat)
    kandidat_titik.append((pt_potong[0], pt_potong[1]))
except np.linalg.LinAlgError:
    pass # Garis sejajar

# Filter hanya titik yang memenuhi semua kendala (Daerah Layak / Feasible Region)
titik_layak = []
for pt in kandidat_titik:
    tx, ty = round(pt[0], 4), round(pt[1], 4)
    if cek_layak(tx, ty):
        if (tx, ty) not in titik_layak:
            titik_layak.append((tx, ty))

# Mengurutkan titik secara berputar (polar) agar membentuk poligon tertutup yang rapi
if len(titik_layak) > 2:
    pts = np.array(titik_layak)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    titik_layak = [titik_layak[i] for i in np.argsort(angles)]

# Hitung Nilai Tujuan (Z) untuk semua titik layak
data_evaluasi = []
for idx, (tx, ty) in enumerate(titik_layak):
    nz = (c1 * tx) + (c2 * ty)
    data_evaluasi.append({
        "Titik Ekstrem": f"Titik {chr(65+idx)} ({tx:g}, {ty:g})",
        f"Produksi {nama_var_x} (X)": tx,
        f"Produksi {nama_var_y} (Y)": ty,
        "Nilai Tujuan Z": nz
    })

df_eval = pd.DataFrame(data_evaluasi)

# Tentukan Solusi Optimum
if not df_eval.empty:
    idx_opt = df_eval['Nilai Tujuan Z'].idxmax() if is_max else df_eval['Nilai Tujuan Z'].idxmin()
    solusi_opt = df_eval.loc[idx_opt]
else:
    solusi_opt = None

# ==========================================
# 4. TAMPILAN DASHBOARD UTAMA
# ==========================================

# Banner Model Aktif
tanda_opt_teks = "Maksimum" if is_max else "Minimum"
st.info(f"📌 **Model Matematika Aktif:** Mencari nilai **{tanda_opt_teks}** untuk fungsi tujuan $Z = {c1}X + {c2}Y$, dengan batasan: $(1)\; {a1}X + {b1}Y {op1} {k1}$ dan $(2)\; {a2}X + {b2}Y {op2} {k2}$ ($X, Y \ge 0$).")

if solusi_opt is None:
    st.error("❌ **Tidak Ditemukan Daerah Layak (Infeasible Region):** Kombinasi batasan kendala yang Anda masukkan saling bertentangan sehingga tidak ada titik yang memenuhi semua syarat.")
else:
    # --- BAGIAN KPI EKSEKUTIF ---
    st.markdown("### 🏆 Rekomendasi Keputusan Optimum")
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric(label=f"💰 Nilai Z ({tanda_opt_teks})", value=f"{solusi_opt['Nilai Tujuan Z']:,.2f}")
    with kpi2:
        st.metric(label=f"📦 Target Produksi [{nama_var_x}]", value=f"{solusi_opt[f'Produksi {nama_var_x} (X)']:g} unit")
    with kpi3:
        st.metric(label=f"📦 Target Produksi [{nama_var_y}]", value=f"{solusi_opt[f'Produksi {nama_var_y} (Y)']:g} unit")
        
    st.markdown("---")
    
    # --- BAGIAN GRAFIK & TABEL (2 KOLOM) ---
    col_left, col_right = st.columns([1.6, 1.1])
    
    with col_left:
        st.subheader("📊 Grafik Interaktif Daerah Penyelesaian")
        st.caption("💡 *Tips: Arahkan kursor ke titik/garis untuk melihat koordinat detail, atau gunakan fitur zoom.*")
        
        # Penentuan batas maksimal sumbu grafik
        max_val_x = max([pt[0] for pt in kandidat_titik] + [10]) * 1.3
        max_val_y = max([pt[1] for pt in kandidat_titik] + [10]) * 1.3
        
        fig = go.Figure()
        
        # 1. Menggambar Garis Kendala 1
        if b1 != 0 and a1 != 0:
            x_line = np.array([0, max_val_x])
            y_line = (k1 - a1 * x_line) / b1
            fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name=f'Kendala 1 ({a1}X + {b1}Y {op1} {k1})', line=dict(color='#3b82f6', width=3)))
        elif a1 == 0 and b1 != 0: # Garis horizontal
            fig.add_hline(y=k1/b1, line_dash="dash", line_color="#3b82f6", annotation_text="Kendala 1")
        elif b1 == 0 and a1 != 0: # Garis vertikal
            fig.add_vline(x=k1/a1, line_dash="dash", line_color="#3b82f6", annotation_text="Kendala 1")
            
        # 2. Menggambar Garis Kendala 2
        if b2 != 0 and a2 != 0:
            x_line2 = np.array([0, max_val_x])
            y_line2 = (k2 - a2 * x_line2) / b2
            fig.add_trace(go.Scatter(x=x_line2, y=y_line2, mode='lines', name=f'Kendala 2 ({a2}X + {b2}Y {op2} {k2})', line=dict(color='#f97316', width=3, dash='dash')))
        elif a2 == 0 and b2 != 0:
            fig.add_hline(y=k2/b2, line_dash="dash", line_color="#f97316", annotation_text="Kendala 2")
        elif b2 == 0 and a2 != 0:
            fig.add_vline(x=k2/a2, line_dash="dash", line_color="#f97316", annotation_text="Kendala 2")

        # 3. Menggambar Poligon Daerah Layak (Feasible Region)
        if len(titik_layak) >= 3:
            poly_x = [pt[0] for pt in titik_layak] + [titik_layak[0][0]]
            poly_y = [pt[1] for pt in titik_layak] + [titik_layak[0][1]]
            fig.add_trace(go.Scatter(
                x=poly_x, y=poly_y, fill='toself', fillcolor='rgba(16, 185, 129, 0.25)',
                line=dict(color='#10b981', width=2), name='Daerah Layak (Feasible Region)',
                hoverinfo='skip'
            ))
            
        # 4. Menandai Titik Sudut Ekstrem & Titik Optimum
        for idx, (tx, ty) in enumerate(titik_layak):
            is_optimal = (tx == solusi_opt[f'Produksi {nama_var_x} (X)'] and ty == solusi_opt[f'Produksi {nama_var_y} (Y)'])
            warna = '#ef4444' if is_optimal else '#4b5563'
            ukuran = 14 if is_optimal else 9
            simbol = 'star' if is_optimal else 'circle'
            label_teks = f"<b>★ OPTIMUM ({tx:g}, {ty:g})</b>" if is_optimal else f"({tx:g}, {ty:g})"
            
            fig.add_trace(go.Scatter(
                x=[tx], y=[ty], mode='markers+text',
                marker=dict(color=warna, size=ukuran, symbol=simbol, line=dict(width=2, color='white')),
                text=[label_teks], textposition="top right",
                name=f"Titik Sudut ({tx:g}, {ty:g})",
                showlegend=False,
                hovertemplate=f"<b>Titik {chr(65+idx)}</b><br>{nama_var_x} (X): {tx:g}<br>{nama_var_y} (Y): {ty:g}<br>Nilai Z: {(c1*tx + c2*ty):,.2f}<extra></extra>"
            ))

        # Konfigurasi Tampilan Layout Plotly
        fig.update_layout(
            xaxis=dict(title=f"<b>Produksi {nama_var_x} (Sumbu X)</b>", range=[0, max_val_x], showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black'),
            yaxis=dict(title=f"<b>Produksi {nama_var_y} (Sumbu Y)</b>", range=[0, max_val_y], showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black'),
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='white',
            height=480
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📋 Evaluasi Titik Ekstrem")
        st.caption("Perbandingan nilai Z pada setiap sudut daerah layak:")
        
        # Format angka dataframe agar rapi
        df_display = df_eval.copy()
        df_display['Nilai Tujuan Z'] = df_display['Nilai Tujuan Z'].apply(lambda v: f"{v:,.2f}")
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        st.success(f"""
        **Kesimpulan Analisis:** Untuk mencapai hasil **{tanda_opt_teks.lower()}**, kombinasi produksi terbaik adalah menghasilkan **{solusi_opt[f'Produksi {nama_var_x} (X)']:g} unit {nama_var_x}** dan **{solusi_opt[f'Produksi {nama_var_y} (Y)']:g} unit {nama_var_y}**.
        """)
        
        with st.expander("🔍 Lihat Bedah Aljabar Matriks"):
            st.markdown("**Perhitungan Titik Potong Garis:**")
            st.latex(r"\begin{bmatrix}" f"{a1} & {b1} \\\\ {a2} & {b2}" r"\end{bmatrix} \begin{bmatrix} X \\\\ Y \end{bmatrix} = \begin{bmatrix}" f"{k1} \\\\ {k2}" r"\end{bmatrix}")
            st.markdown("Dengan metode eliminasi/substitusi sistem persamaan linear, diperoleh koordinat perpotongan pada batas kendala tersebut.")
