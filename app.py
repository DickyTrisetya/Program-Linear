import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. KONFIGURASI HALAMAN KHUSUS MOBILE
# layout="centered" sangat penting agar pas dan rapi di layar HP
st.set_page_config(
    page_title="LP Mobile Solver",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Kustomisasi judul bergaya aplikasi mobile
st.markdown("""
    <div style="text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 15px;">
        <h2 style="margin: 0; color: #1e3d59;">📱 Kalkulator Program Linear</h2>
        <p style="margin: 0; font-size: 13px; color: #6883ba;">Metode Grafik • Cepat & Responsive di HP</p>
    </div>
""", unsafe_allow_html=True)

# 2. INPUT DATA BERGAYA ACCORDION (KOTAK LIPAT)
# Sangat ramah untuk HP karena hemat tempat dan mudah disentuh jari
with st.expander("✏️ 1. TEKAN DI SINI UNTUK UBAH SOAL / ANGKA", expanded=True):
    jenis_optimasi = st.selectbox("🎯 Tujuan Optimasi:", ["Maksimum (Keuntungan/Profit)", "Minimum (Biaya/Cost)"])
    is_max = "Maksimum" in jenis_optimasi
    
    st.markdown("---")
    st.markdown("**💰 Fungsi Tujuan Z = C1(X) + C2(Y)**")
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        c1 = st.number_input("Koefisien X (C1)", value=120.0, step=10.0)
    with col_z2:
        c2 = st.number_input("Koefisien Y (C2)", value=150.0, step=10.0)
        
    st.markdown("---")
    st.markdown("**🛡️ Batasan Kendala 1: A1(X) + B1(Y) ≤ K1**")
    col_k1a, col_k1b = st.columns(2)
    with col_k1a:
        a1 = st.number_input("Koefisien X (A1)", value=1.0, step=1.0)
    with col_k1b:
        b1 = st.number_input("Koefisien Y (B1)", value=1.0, step=1.0)
    k1 = st.number_input("Batas Kendala 1 (K1)", value=100.0, step=10.0)
    
    st.markdown("---")
    st.markdown("**🛡️ Batasan Kendala 2: A2(X) + B2(Y) ≤ K2**")
    col_k2a, col_k2b = st.columns(2)
    with col_k2a:
        a2 = st.number_input("Koefisien X (A2)", value=2.0, step=1.0)
    with col_k2b:
        b2 = st.number_input("Koefisien Y (B2)", value=4.0, step=1.0)
    k2 = st.number_input("Batas Kendala 2 (K2)", value=300.0, step=10.0)
    
    st.markdown("---")
    # Tombol besar di HP (use_container_width=True)
    btn_hitung = st.button("🔍 HITUNG SEKARANG", type="primary", use_container_width=True)

# 3. PERHITUNGAN MATEMATIKA
# Hitung titik potong sumbu
x_k1 = k1 / a1 if a1 != 0 else 0
y_k1 = k1 / b1 if b1 != 0 else 0
x_k2 = k2 / a2 if a2 != 0 else 0
y_k2 = k2 / b2 if b2 != 0 else 0

# Hitung titik potong antar dua garis
try:
    A_mat = np.array([[a1, b1], [a2, b2]])
    B_mat = np.array([k1, k2])
    xp, yp = np.linalg.solve(A_mat, B_mat)
except np.linalg.LinAlgError:
    xp, yp = None, None

# Tentukan titik sudut daerah penyelesaian (Feasible Region)
titik_sudut = [[0, 0]]
if x_k1 <= x_k2: titik_sudut.append([x_k1, 0])
else: titik_sudut.append([x_k2, 0])
    
if y_k1 <= y_k2: titik_sudut.append([0, y_k1])
else: titik_sudut.append([0, y_k2])
    
if xp is not None and xp >= 0 and yp >= 0:
    # Cek apakah titik potong memenuhi semua kendala
    if (a1*xp + b1*yp <= k1 + 1e-5) and (a2*xp + b2*yp <= k2 + 1e-5):
        titik_sudut.append([xp, yp])
        
titik_sudut = np.unique(titik_sudut, axis=0)

# Buat DataFrame hasil perhitungan di tiap titik sudut
data_hasil = []
for t in titik_sudut:
    tx, ty = t[0], t[1]
    data_hasil.append({
        "Titik": f"({tx:.1f}, {ty:.1f})",
        "X": tx,
        "Y": ty,
        "Nilai Z": (c1 * tx) + (c2 * ty)
    })
df = pd.DataFrame(data_hasil)

# Tentukan pemenang solusi optimum
tanda = "Maksimum" if is_max else "Minimum"
idx_opt = df['Nilai Z'].idxmax() if is_max else df['Nilai Z'].idxmin()
solusi = df.loc[idx_opt]

# 4. TAMPILAN HASIL (VERTIKAL ALIGNMENT KHUSUS MOBILE)
st.markdown("### 🏆 2. HASIL KEPUTUSAN OPTIMUM")

# Banner Utama Hasil (Langsung terlihat tanpa perlu scroll jauh ke bawah)
st.success(f"""
💡 **Kesimpulan:** Nilai **{tanda}** terbaik adalah **{solusi['Nilai Z']:.2f}** yang dicapai pada koordinat **{solusi['Titik']}** *(Nilai X = {solusi['X']:.1f} dan Nilai Y = {solusi['Y']:.1f})*.
""")

# Kotak Angka Besar (Metrik) disusun 2 kolom di HP
col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric("📦 Target Produk X", f"{solusi['X']:.1f} unit")
with col_m2:
    st.metric("📦 Target Produk Y", f"{solusi['Y']:.1f} unit")
st.metric(f"💰 Total Nilai Z ({tanda})", f"{solusi['Nilai Z']:.2f}")

st.markdown("---")

# 5. GRAFIK DAERAH PENYELESAIAN (SQUARE & RESPONSIVE)
st.markdown("### 📈 3. GRAFIK METODE GRAFIK")
st.caption("Area berwarna hijau adalah Daerah Layak (Feasible Region):")

# Rasio 6x6 adalah rasio emas untuk tampilan layar HP (bujursangkar tidak kepanjangan)
fig, ax = plt.subplots(figsize=(6, 6))
max_x = max(x_k1, x_k2, xp if xp else 0) * 1.25
max_y = max(y_k1, y_k2, yp if yp else 0) * 1.25

# Gambar garis kendala
ax.plot([x_k1, 0], [0, y_k1], label=f'{a1}x+{b1}y≤{k1}', color='#2563eb', linewidth=2.5)
ax.plot([x_k2, 0], [0, y_k2], label=f'{a2}x+{b2}y≤{k2}', color='#ea580c', linewidth=2.5, linestyle='--')

# Arsiran daerah layak
center = np.mean(titik_sudut, axis=0)
angles = np.arctan2(titik_sudut[:,1] - center[1], titik_sudut[:,0] - center[0])
titik_sorted = titik_sudut[np.argsort(angles)]

polygon = plt.Polygon(titik_sorted, closed=True, color='#16a34a', alpha=0.3, label='Daerah Layak')
ax.add_patch(polygon)

# Gambar titik sudut & titik optimum
for t in titik_sudut:
    is_opt = (t[0] == solusi['X'] and t[1] == solusi['Y'])
    warna = 'red' if is_opt else '#1e293b'
    ukuran = 8 if is_opt else 5
