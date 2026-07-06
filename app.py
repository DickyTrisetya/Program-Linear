import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Konfigurasi Halaman Mode Lebar (Wide Dashboard)
st.set_page_config(
    page_title="LP Solver Dashboard | Metode Grafik",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. MENU SAMPING (SIDEBAR) - KUMPULAN INPUT DATA
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Model")
    st.caption("Sesuaikan parameter program linear Anda di sini:")
    st.markdown("---")
    
    # Jenis Optimasi dengan desain tombol horizontal
    jenis_optimasi = st.radio(
        "Tujuan Optimasi:", 
        ("Maksimum", "Minimum"), 
        horizontal=True
    )
    
    st.markdown("#### 🎯 Fungsi Tujuan (Z)")
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        c1 = st.number_input("Koef. X (C1)", value=120.0, step=10.0)
    with col_z2:
        c2 = st.number_input("Koef. Y (C2)", value=150.0, step=10.0)
        
    st.markdown("#### 🛡️ Batasan Kendala 1")
    col_k1a, col_k1b = st.columns(2)
    with col_k1a:
        a1 = st.number_input("Koef. X (A1)", value=1.0, step=1.0)
    with col_k1b:
        b1 = st.number_input("Koef. Y (B1)", value=1.0, step=1.0)
    k1 = st.number_input("Maksimal Kapasitas (K1)", value=100.0, step=10.0)
    
    st.markdown("#### 🛡️ Batasan Kendala 2")
    col_k2a, col_k2b = st.columns(2)
    with col_k2a:
        a2 = st.number_input("Koef. X (A2)", value=2.0, step=1.0)
    with col_k2b:
        b2 = st.number_input("Koef. Y (B2)", value=4.0, step=1.0)
    k2 = st.number_input("Maksimal Kapasitas (K2)", value=300.0, step=10.0)
    
    st.markdown("---")
    hitung_btn = st.button("🚀 KALKULASI SOLUSI", type="primary", use_container_width=True)

# 3. HALAMAN UTAMA (DASHBOARD)
st.title("📈 Dashboard Analisis Program Linear")
st.markdown("*Penyelesaian masalah optimasi dua variabel menggunakan pemodelan metode grafik.*")

# Tampilkan ringkasan model saat ini dalam kotak informasi bergaya
st.info(f"**Model Aktif:** Mencari nilai **{jenis_optimasi}** untuk $Z = {c1}x + {c2}y$, dengan kendala: $(1)\; {a1}x + {b1}y \le {k1}$ dan $(2)\; {a2}x + {b2}y \le {k2}$ di mana $x, y \ge 0$.")

# Perhitungan Titik Potong Sumbu
x_k1 = k1 / a1 if a1 != 0 else 0
y_k1 = k1 / b1 if b1 != 0 else 0
x_k2 = k2 / a2 if a2 != 0 else 0
y_k2 = k2 / b2 if b2 != 0 else 0

# Perhitungan Titik Potong Antar Garis menggunakan Matriks NumPy
try:
    A = np.array([[a1, b1], [a2, b2]])
    B = np.array([k1, k2])
    xp, yp = np.linalg.solve(A, B)
except np.linalg.LinAlgError:
    xp, yp = None, None

# Menentukan Kandidat Titik Sudut Daerah Layak (Feasible Region)
titik_sudut = [[0, 0]]
if x_k1 <= x_k2: titik_sudut.append([x_k1, 0])
else: titik_sudut.append([x_k2, 0])
    
if y_k1 <= y_k2: titik_sudut.append([0, y_k1])
else: titik_sudut.append([0, y_k2])
    
if xp is not None and xp >= 0 and yp >= 0:
    if (a1*xp + b1*yp <= k1 + 1e-5) and (a2*xp + b2*yp <= k2 + 1e-5):
        titik_sudut.append([xp, yp])
        
titik_sudut = np.unique(titik_sudut, axis=0)

# Perhitungan Data Tabel Pandas
data_hasil = []
for t in titik_sudut:
    tx, ty = t[0], t[1]
    data_hasil.append({
        "Koordinat Titik": f"({tx:.1f}, {ty:.1f})",
        "Nilai X": tx,
        "Nilai Y": ty,
        "Nilai Tujuan (Z)": (c1 * tx) + (c2 * ty)
    })
df = pd.DataFrame(data_hasil)

# Menentukan Nilai Optimum
idx_optimum = df['Nilai Tujuan (Z)'].idxmax() if jenis_optimasi == "Maksimum" else df['Nilai Tujuan (Z)'].idxmin()
solusi = df.loc[idx_optimum]

# 4. TAMPILAN TAB MODERN
tab1, tab2 = st.tabs(["📊 Visualisasi & Solusi Optimum", "📝 Langkah Pembuktian & Matriks"])

with tab1:
    # Bagian Atas: Kartu Metrik Solusi
    st.markdown("#### 🏆 Kesimpulan Solusi Optimum")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label=f"Nilai Z ({jenis_optimasi})", value=f"{solusi['Nilai Tujuan (Z)']:.2f}")
    with m2:
        st.metric(label="Rekomendasi Nilai X", value=f"{solusi['Nilai X']:.2f}")
    with m3:
        st.metric(label="Rekomendasi Nilai Y", value=f"{solusi['Nilai Y']:.2f}")
    
    st.markdown("---")
    
    # Bagian Bawah: Grafik & Tabel Utama
    col_grafik, col_tabel = st.columns([1.5, 1])
    
    with col_grafik:
        st.markdown("#### 📈 Pemetaan Daerah Layak (Feasible Region)")
        
        # Desain Grafik Custom yang Lebih Modern
        fig, ax = plt.subplots(figsize=(8, 6))
        max_x = max(x_k1, x_k2, xp if xp else 0) * 1.25
        max_y = max(y_k1, y_k2, yp if yp else 0) * 1.25
        
        # Gambar Garis Kendala dengan gaya modern
        ax.plot([x_k1, 0], [0, y_k1], label=f'Kendala 1 ({a1}x + {b1}y ≤ {k1})', color='#2980B9', linewidth=2.5)
        ax.plot([x_k2, 0], [0, y_k2], label=f'Kendala 2 ({a2}x + {b2}y ≤ {k2})', color='#E67E22', linewidth=2.5, linestyle='--')
        
        # Arsiran Daerah Layak dengan warna lembut
        center = np.mean(titik_sudut, axis=0)
        angles = np.arctan2(titik_sudut[:,1] - center[1], titik_sudut[:,0] - center[0])
        titik_sorted = titik_sudut[np.argsort(angles)]
        
        polygon = plt.Polygon(titik_sorted, closed=True, color='#27AE60', alpha=0.25, label='Daerah Layak')
        ax.add_patch(polygon)
        
        # Menambahkan Anotasi / Label otomatis pada Titik Sudut
        for t in titik_sudut:
            is_opt = (t[0] == solusi['Nilai X'] and t[1] == solusi['Nilai Y'])
            marker_color = 'red' if is_opt else '#2C3E50'
            marker_size = 8 if is_opt else 5
            
            ax.plot(t[0], t[1], marker='o', color=marker_color, markersize=marker_size)
            ax.annotate(f"({t[0]:.0f}, {t[1]:.0f})", (t[0], t[1]), 
                        textcoords="offset points", xytext=(8,8), 
                        ha='left', fontweight='bold' if is_opt else 'normal',
                        color=marker_color)

        # Styling Desain Grafik (Menghilangkan garis bingkai atas & kanan)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)
        ax.set_xlabel('Variabel Keputusan X', fontweight='bold')
        ax.set_ylabel('Variabel Keputusan Y', fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        
        st.pyplot(fig)
        
    with col_tabel:
        st.markdown("#### 📋 Evaluasi Titik Sudut")
        st.caption("Perbandingan nilai fungsi tujuan di setiap titik ekstrem:")
        
        # Menyoroti baris solusi optimum pada tabel
        def highlight_optimum(row):
            if row['Koordinat Titik'] == solusi['Koordinat Titik']:
                return ['background-color: #d4edda; font-weight: bold; color: #155724;'] * len(row)
            return [''] * len(row)
            
        st.dataframe(df.style.apply(highlight_optimum, axis=1), use_container_width=True, hide_index=True)
        
        st.success(f"**Kesimpulan Klinis:** Solusi terbaik berada pada titik **{solusi['Koordinat Titik']}**, karena menghasilkan nilai {jenis_optimasi.lower()} paling ekstrem dibandingkan titik sudut lainnya di dalam area daerah layak.")

with tab2:
    st.markdown("#### 🔍 Rincian Tahapan Matematika")
    
    with st.expander("1️⃣ Perhitungan Titik Potong Sumbu (Kendala 1 & 2)", expanded=True):
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown(f"**Kendala 1:** `{a1}x + {b1}y ≤ {k1}`")
            st.write(f"- Saat $x = 0$, maka $y = {k1}/{b1} = \mathbf{{{y_k1:.1f}}}$ → Titik $(0, {y_k1:.1f})$")
            st.write(f"- Saat $y = 0$, maka $x = {k1}/{a1} = \mathbf{{{x_k1:.1f}}}$ → Titik $(x={x_k1:.1f}, 0)$")
        with col_e2:
            st.markdown(f"**Kendala 2:** `{a2}x + {b2}y ≤ {k2}`")
            st.write(f"- Saat $x = 0$, maka $y = {k2}/{b2} = \mathbf{{{y_k2:.1f}}}$ → Titik $(0, {y_k2:.1f})$")
            st.write(f"- Saat $y = 0$, maka $x = {k2}/{a2} = \mathbf{{{x_k2:.1f}}}$ → Titik $(x={x_k2:.1f}, 0)$")
            
    with st.expander("2️⃣ Eliminasi & Substitusi Titik Potong Antar Garis"):
        if xp is not None and xp >= 0 and yp >= 0:
            st.write("Dengan menyelesaikan sistem persamaan linear dua variabel (SPLDV) menggunakan aljabar matriks:")
            st.latex(r"\begin{bmatrix}" f"{a1} & {b1} \\\\ {a2} & {b2}" r"\end{bmatrix} \begin{bmatrix} x \\\\ y \end{bmatrix} = \begin{bmatrix}" f"{k1} \\\\ {k2}" r"\end{bmatrix}")
            st.write(f"Diperoleh titik potong kedua garis tepat pada koordinat **$(x={xp:.2f}, y={yp:.2f})$**.")
        else:
            st.write("Kedua garis tidak berpotongan pada kuadran positif (x≥0, y≥0) atau garis bersifat sejajar.")
