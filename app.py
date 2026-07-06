import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Metode Grafik - Program Linear", layout="centered")

# --- sedikit css biar rapi di hp, soalnya kolom bawaan streamlit suka dempet ---
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 760px;
    }
    h1 {
        font-size: 1.6rem !important;
    }
    .hasil-box {
        border: 1px solid #333;
        border-left: 4px solid #4a7c59;
        border-radius: 6px;
        padding: 14px 16px;
        margin: 10px 0;
        background-color: rgba(74, 124, 89, 0.07);
    }
    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
        h1 { font-size: 1.3rem !important; }
        h2, h3 { font-size: 1.05rem !important; }
    }
</style>
""", unsafe_allow_html=True)

st.title("Program Linear - Metode Grafik")
st.caption("Isi fungsi tujuan dan dua kendala di bawah, terus klik Hitung.")

st.divider()

optimasi = st.radio("Mau dicari nilai maksimum atau minimum?", ("Maksimum", "Minimum"), horizontal=True)

with st.expander("Fungsi Tujuan", expanded=True):
    c1, c2 = st.columns(2)
    coef_x = c1.number_input("Koefisien x", value=120.0, step=1.0, key="cx")
    coef_y = c2.number_input("Koefisien y", value=150.0, step=1.0, key="cy")

with st.expander("Kendala 1", expanded=True):
    c1, c2, c3 = st.columns(3)
    a1 = c1.number_input("x", value=1.0, step=1.0, key="a1")
    b1 = c2.number_input("y", value=1.0, step=1.0, key="b1")
    k1 = c3.number_input("<= nilai", value=100.0, step=1.0, key="k1")

with st.expander("Kendala 2", expanded=True):
    c1, c2, c3 = st.columns(3)
    a2 = c1.number_input("x", value=2.0, step=1.0, key="a2")
    b2 = c2.number_input("y", value=4.0, step=1.0, key="b2")
    k2 = c3.number_input("<= nilai", value=300.0, step=1.0, key="k2")

hitung = st.button("Hitung", use_container_width=True, type="primary")

if hitung:
    st.divider()
    st.header("Hasil")

    st.write("**Model yang terbentuk:**")
    st.latex(f"Z = {coef_x:g}x + {coef_y:g}y")
    st.latex(f"{a1:g}x + {b1:g}y \\le {k1:g}")
    st.latex(f"{a2:g}x + {b2:g}y \\le {k2:g}")
    st.latex("x \\ge 0,\\ y \\ge 0")

    # titik potong tiap garis kendala ke sumbu x dan y
    x_pot1 = k1 / a1 if a1 != 0 else 0
    y_pot1 = k1 / b1 if b1 != 0 else 0
    x_pot2 = k2 / a2 if a2 != 0 else 0
    y_pot2 = k2 / b2 if b2 != 0 else 0

    st.subheader("Titik potong ke sumbu")
    tab1, tab2 = st.tabs(["Kendala 1", "Kendala 2"])
    with tab1:
        st.write(f"- x = 0 → y = {y_pot1:g}, jadi titik (0, {y_pot1:g})")
        st.write(f"- y = 0 → x = {x_pot1:g}, jadi titik ({x_pot1:g}, 0)")
    with tab2:
        st.write(f"- x = 0 → y = {y_pot2:g}, jadi titik (0, {y_pot2:g})")
        st.write(f"- y = 0 → x = {x_pot2:g}, jadi titik ({x_pot2:g}, 0)")

    # cari titik potong dua garis kendala pakai aljabar linear
    xp, yp = None, None
    try:
        A = np.array([[a1, b1], [a2, b2]])
        B = np.array([k1, k2])
        titik = np.linalg.solve(A, B)
        xp, yp = titik[0], titik[1]
        st.subheader("Titik potong kedua garis kendala")
        st.write(f"Kedua garis bertemu di titik ({xp:.2f}, {yp:.2f})")
    except np.linalg.LinAlgError:
        st.warning("Kedua garis sejajar, jadi tidak ada titik potong tunggal.")

    # kumpulin titik-titik sudut daerah layak
    sudut = [[0.0, 0.0]]
    sudut.append([min(x_pot1, x_pot2), 0.0])
    sudut.append([0.0, min(y_pot1, y_pot2)])

    if xp is not None and xp >= -1e-6 and yp >= -1e-6:
        cek1 = a1 * xp + b1 * yp <= k1 + 1e-5
        cek2 = a2 * xp + b2 * yp <= k2 + 1e-5
        if cek1 and cek2:
            sudut.append([xp, yp])

    sudut = np.unique(np.round(np.array(sudut), 6), axis=0)

    # urutkan titik searah putaran biar polygon-nya kegambar bener
    pusat = sudut.mean(axis=0)
    sudut_urut = sudut[np.argsort(np.arctan2(sudut[:, 1] - pusat[1], sudut[:, 0] - pusat[0]))]

    st.subheader("Daerah penyelesaian")
    batas_x = max(x_pot1, x_pot2, xp if xp else 0) * 1.25 or 10
    batas_y = max(y_pot1, y_pot2, yp if yp else 0) * 1.25 or 10

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([x_pot1, 0], [0, y_pot1], color="#3b6ea5", linewidth=2, label=f"{a1:g}x + {b1:g}y = {k1:g}")
    ax.plot([x_pot2, 0], [0, y_pot2], color="#d68a2c", linewidth=2, label=f"{a2:g}x + {b2:g}y = {k2:g}")

    if xp is not None and xp >= 0 and yp >= 0:
        ax.plot(xp, yp, "o", color="#c0392b", label=f"Potongan ({xp:.1f}, {yp:.1f})")

    poly = plt.Polygon(sudut_urut, closed=True, facecolor="#4a7c59", alpha=0.3, label="Daerah layak")
    ax.add_patch(poly)

    ax.set_xlim(0, batas_x)
    ax.set_ylim(0, batas_y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Nilai Z di tiap titik sudut")
    baris = []
    for (tx, ty) in sudut:
        z = coef_x * tx + coef_y * ty
        baris.append({"Titik": f"({tx:.2f}, {ty:.2f})", "x": round(tx, 2), "y": round(ty, 2), "Z": round(z, 2)})
    df = pd.DataFrame(baris)
    st.dataframe(df, use_container_width=True, hide_index=True)

    idx = df["Z"].idxmax() if optimasi == "Maksimum" else df["Z"].idxmin()
    solusi = df.loc[idx]

    st.markdown(
        f"""<div class="hasil-box">
        Nilai <b>{optimasi.lower()}</b> Z didapat di titik <b>{solusi['Titik']}</b>,
        dengan Z = <b>{solusi['Z']:.2f}</b> (x = {solusi['x']:.2f}, y = {solusi['y']:.2f}).
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.info("Isi datanya dulu di atas, terus tekan Hitung untuk lihat hasilnya.")
