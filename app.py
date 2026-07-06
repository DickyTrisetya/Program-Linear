import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Pengaturan halaman utama web
st.set_page_config(page_title="Program Linear Metode Grafik Kelompok 4", layout="centered")

# Bagian Header / Judul Aplikasi
st.markdown("""
    <div style="background-color:#1e1e38; padding:20px; border-radius:10px; text-align:center; margin-bottom:25px;">
        <h1 style="color:white; margin:0;">Program Linear Metode Grafik</h1>
        <p style="color:#b0b0d0; margin:5px 0 0 0;">Masukkan fungsi tujuan dan dua kendala, kemudian tekan tombol Hitung</p>
    </div>
""", unsafe_allow_html=True)

st.header("Input Data")

# 1. Pilihan Jenis Optimasi
jenis_optimasi = st.radio("Jenis Optimasi", ("Maksimum", "Minimum"))

# 2. Input Fungsi Tujuan Z = C1*x + C2*y
st.subheader("Fungsi Tujuan")
col1, col2 = st.columns(2)
with col1:
    c1 = st.number_input("Koefisien x", value=120.0, step=1.0)
with col2:
    c2 = st.number_input("Koefisien y", value=150.0, step=1.0)

# 3. Input Kendala 1: A1*x + B1*y <= / >= K1
st.subheader("Kendala 1")
col3, col4, col5 = st.columns(3)
with col3:
    a1 = st.number_input("Koefisien x (A1)", value=1.0, step=1.0)
with col4:
    b1 = st.number_input("Koefisien y (B1)", value=1.0, step=1.0)
with col5:
    k1 = st.number_input("Nilai Kendala 1", value=100.0, step=1.0)

# 4. Input Kendala 2: A2*x + B2*y <= / >= K2
st.subheader("Kendala 2")
col6, col7, col8 = st.columns(3)
with col6:
    a2 = st.number_input("Koefisien x (A2)", value=2.0, step=1.0)
with col7:
    b2 = st.number_input("Koefisien y (B2)", value=4.0, step=1.0)
with col8:
    k2 = st.number_input("Nilai Kendala 2", value=300.0, step=1.0)

# Anggap kendala bertipe kurang dari atau sama dengan (<=) sesuai contoh umum metode grafik
tanda_kendala = "<=" 

# Tombol Hitung
if st.button("HITUNG", use_container_width=True):
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; background-color: #1e1e38; color: white; padding: 10px; border-radius: 5px;'>HASIL PENYELESAIAN</h2>", unsafe_allow_html=True)
    
    # Menampilkan Fungsi Tujuan dan Kendala yang terbentuk
    st.write("### Fungsi Tujuan")
    st.latex(f"Z = {c1}x + {c2}y")
    
    st.write("### Fungsi Kendala")
    st.latex(f"{a1}x + {b1}y \\le {k1}")
    st.latex(f"{a2}x + {b2}y \\le {k2}")
    st.latex("x \\ge 0, y \\ge 0")
    
    # ----------------------------------------------------
    # Perhitungan Perubahan Titik Kendala terhadap Sumbu
    # ----------------------------------------------------
    st.write("### 3. Menentukan Titik pada Setiap Kendala")
    
    # Titik Potong Kendala 1
    x_k1 = k1 / a1 if a1 != 0 else 0
    y_k1 = k1 / b1 if b1 != 0 else 0
    
    col_k1_a, col_k1_b = st.columns(2)
    with col_k1_a:
        st.info(f"**Kendala 1:** Jika x = 0, maka y = {y_k1} → Titik (0, {y_k1})")
    with col_k1_b:
        st.info(f"**Kendala 1:** Jika y = 0, maka x = {x_k1} → Titik ({x_k1}, 0)")
        
    # Titik Potong Kendala 2
    x_k2 = k2 / a2 if a2 != 0 else 0
    y_k2 = k2 / b2 if b2 != 0 else 0
    
    col_k2_a, col_k2_b = st.columns(2)
    with col_k2_a:
        st.info(f"**Kendala 2:** Jika x = 0, maka y = {y_k2} → Titik (0, {y_k2})")
    with col_k2_b:
        st.info(f"**Kendala 2:** Jika y = 0, maka x = {x_k2} → Titik ({x_k2}, 0)")

    # ----------------------------------------------------
    # Perhitungan Titik Potong menggunakan NumPy (Aljabar Linear)
    # ----------------------------------------------------
    st.write("### 4. Menentukan Titik Potong Kedua Garis")
    try:
        A = np.array([[a1, b1], [a2, b2]])
        B = np.array([k1, k2])
        titik_potong = np.linalg.solve(A, B)
        xp, yp = titik_potong[0], titik_potong[1]
        st.success(f"Titik potong antar garis kendala berada di koordinat: **({xp:.2f}, {yp:.2f})**")
    except np.linalg.LinAlgError:
        xp, yp = None, None
        st.warning("Kedua garis sejajar atau tidak memiliki satu titik potong unik.")

    # ----------------------------------------------------
    # Pencarian Titik-Titik Sudut Daerah Penyelesaian (Feasible Region)
    # ----------------------------------------------------
    # Daftar kandidat titik sudut dasar metode grafik untuk kendala <=
    titik_sudut = [[0, 0]]
    
    if x_k1 <= x_k2:
        titik_sudut.append([x_k1, 0])
    else:
        titik_sudut.append([x_k2, 0])
        
    if y_k1 <= y_k2:
        titik_sudut.append([0, y_k1])
    else:
        titik_sudut.append([0, y_k2])
        
    if xp is not None and xp >= 0 and yp >= 0:
        # Validasi apakah titik potong memenuhi semua kendala
        if (a1*xp + b1*yp <= k1 + 1e-5) and (a2*xp + b2*yp <= k2 + 1e-5):
            titik_sudut.append([xp, yp])
            
    # Membersihkan titik duplikat
    titik_sudut = np.unique(titik_sudut, axis=0)

    # ----------------------------------------------------
    # Pembuatan Grafik Menggunakan Matplotlib
    # ----------------------------------------------------
    st.write("### Grafik Daerah Penyelesaian")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Maksimal batas grafik
    max_x = max(x_k1, x_k2, xp if xp else 0) * 1.2
    max_y = max(y_k1, y_k2, yp if yp else 0) * 1.2
    
    # Plot Garis Kendala 1
    ax.plot([x_k1, 0], [0, y_k1], label=f'{a1}x + {b1}y = {k1}', color='blue', linewidth=2)
    # Plot Garis Kendala 2
    ax.plot([x_k2, 0], [0, y_k2], label=f'{a2}x + {b2}y = {k2}', color='orange', linewidth=2)
    
    # Plot Titik Potong
    if xp is not None and xp >= 0 and yp >= 0:
        ax.plot(xp, yp, 'ro', label=f'Titik Potong ({xp:.1f}, {yp:.1f})')
        
    # Mengarsir Daerah Penyelesaian (Feasible Region) menggunakan Polygon pembatas titik sudut
    # Mengurutkan titik secara berputar (polar) agar poligon digambar dengan benar
    center = np.mean(titik_sudut, axis=0)
    angles = np.arctan2(titik_sudut[:,1] - center[1], titik_sudut[:,0] - center[0])
    titik_sudut_sorted = titik_sudut[np.argsort(angles)]
    
    polygon = plt.Polygon(titik_sudut_sorted, closed=True, color='green', alpha=0.3, label='Daerah Layak')
    ax.add_patch(polygon)
    
    # Pengaturan Tampilan Grafik
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_xlabel('Sumbu X')
    ax.set_ylabel('Sumbu Y')
    ax.axhline(0, color='black',linewidth=1)
    ax.axvline(0, color='black',linewidth=1)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    st.pyplot(fig)

    # ----------------------------------------------------
    # Menghitung Nilai Fungsi Tujuan Menggunakan Pandas
    # ----------------------------------------------------
    st.write("### Tabel Perhitungan Nilai Fungsi Tujuan")
    
    data_hasil = []
    for t in titik_sudut:
        tx, ty = t[0], t[1]
        nilai_z = c1 * tx + c2 * ty
        data_hasil.append({"Titik": f"({tx:.2f}, {ty:.2f})", "Nilai X": tx, "Nilai Y": ty, "Nilai Z": nilai_z})
        
    df = pd.DataFrame(data_hasil)
    st.dataframe(df, use_container_width=True)

    # Menentukan Kesimpulan Optimum
    if jenis_optimasi == "Maksimum":
        idx_optimum = df['Nilai Z'].idxmax()
    else:
        idx_optimum = df['Nilai Z'].idxmin()
        
    solusi = df.loc[idx_optimum]

    st.write("### Kesimpulan")
    st.success(f"""
    Berdasarkan hasil perhitungan metode grafik, diperoleh nilai **{jenis_optimasi}** optimum sebesar **{solusi['Nilai Z']:.2f}** yang terletak pada koordinat titik sudut **{solusi['Titik']}** (di mana Nilai X = {solusi['Nilai X']:.2f} dan Nilai Y = {solusi['Nilai Y']:.2f}).
    """)