import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Program Linear", layout="centered")

# --- css biar enak dilihat di hp, kolom bawaan streamlit suka dempet ---
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 760px;
    }
    h1 { font-size: 1.6rem !important; }
    .hasil-box {
        border: 1px solid #333;
        border-left: 4px solid #4a7c59;
        border-radius: 6px;
        padding: 14px 16px;
        margin: 10px 0;
        background-color: rgba(74, 124, 89, 0.07);
    }
    .warn-box {
        border-left: 4px solid #c0392b;
        border-radius: 6px;
        padding: 14px 16px;
        margin: 10px 0;
        background-color: rgba(192, 57, 43, 0.07);
    }
    @media (max-width: 640px) {
        .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
        h1 { font-size: 1.3rem !important; }
        h2, h3 { font-size: 1.05rem !important; }
    }
</style>
""", unsafe_allow_html=True)

st.title("Program Linear")
st.caption("Pilih metode penyelesaian, isi datanya, terus klik Hitung.")
st.divider()

metode = st.selectbox("Metode", ["Metode Grafik", "Metode Simplex", "Metode Big M"])
optimasi = st.radio("Tujuan", ("Maksimum", "Minimum"), horizontal=True)

st.subheader("Fungsi Tujuan")
c1, c2 = st.columns(2)
coef_x = c1.number_input("Koefisien x", value=120.0, step=1.0, key="cx")
coef_y = c2.number_input("Koefisien y", value=150.0, step=1.0, key="cy")

# ---------------------------------------------------------------------------
# input kendala - untuk simplex biasa tandanya dikunci <=, big M boleh bebas
# ---------------------------------------------------------------------------
tanda_opsi = ["<=", ">=", "="] if metode == "Metode Big M" else ["<="]

def input_kendala(nama, default_a, default_b, default_k, key_prefix):
    st.subheader(nama)
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2]) if metode == "Metode Big M" else st.columns(3)
    a = c1.number_input("x", value=default_a, step=1.0, key=f"a_{key_prefix}")
    b = c2.number_input("y", value=default_b, step=1.0, key=f"b_{key_prefix}")
    if metode == "Metode Big M":
        tanda = c3.selectbox("tanda", tanda_opsi, key=f"t_{key_prefix}")
        k = c4.number_input("nilai", value=default_k, step=1.0, key=f"k_{key_prefix}")
    else:
        tanda = "<="
        k = c3.number_input("<= nilai", value=default_k, step=1.0, key=f"k_{key_prefix}")
    return a, b, tanda, k

a1, b1, t1, k1 = input_kendala("Kendala 1", 1.0, 1.0, 100.0, "1")
a2, b2, t2, k2 = input_kendala("Kendala 2", 2.0, 4.0, 300.0, "2")

if metode == "Metode Simplex":
    st.caption("Metode ini cuma buat kendala bertanda \u2264. Kalau ada \u2265 atau =, pindah ke Metode Big M.")

hitung = st.button("Hitung", use_container_width=True, type="primary")


# ===========================================================================
# mesin simplex + big M, dipakai bareng oleh dua metode
# ===========================================================================
def bangun_tabel(objective, constraints, maximize):
    """constraints: list of (a, b, tanda, k) untuk 2 variabel x,y"""
    n = 2
    norm = []
    for (a, b, tanda, k) in constraints:
        coef = [a, b]
        if k < 0:
            coef = [-v for v in coef]
            k = -k
            flip = {"<=": ">=", ">=": "<=", "=": "="}
            tanda = flip[tanda]
        norm.append((coef, tanda, k))

    m = len(norm)
    col_defs = []  # (nama, baris, koef, is_artificial)
    for i, (coef, tanda, k) in enumerate(norm):
        if tanda == "<=":
            col_defs.append((f"s{i+1}", i, 1.0, False))
        elif tanda == ">=":
            col_defs.append((f"e{i+1}", i, -1.0, False))
            col_defs.append((f"a{i+1}", i, 1.0, True))
        else:
            col_defs.append((f"a{i+1}", i, 1.0, True))

    extra_names = [d[0] for d in col_defs]
    total_cols = n + len(extra_names) + 1
    tableau = np.zeros((m + 1, total_cols))

    for i, (coef, tanda, k) in enumerate(norm):
        tableau[i, :n] = coef
        tableau[i, -1] = k

    col_index = {}
    for j, (name, row_i, koef, is_art) in enumerate(col_defs):
        pos = n + j
        tableau[row_i, pos] = koef
        col_index[name] = pos

    basis = [None] * m
    for i, (coef, tanda, k) in enumerate(norm):
        basis[i] = col_index[f"s{i+1}"] if tanda == "<=" else col_index[f"a{i+1}"]

    col_names = ["x", "y"] + extra_names + ["RHS"]

    c_use = np.array(objective, dtype=float)
    if not maximize:
        c_use = -c_use

    M = 1e6
    obj_row = np.zeros(total_cols)
    obj_row[:n] = -c_use
    for name in extra_names:
        if name.startswith("a"):
            obj_row[col_index[name]] = M
    tableau[-1, :] = obj_row

    reverse = {v: k for k, v in col_index.items()}
    for i, b_idx in enumerate(basis):
        nm = reverse.get(b_idx, "")
        if nm.startswith("a"):
            tableau[-1, :] -= M * tableau[i, :]

    return tableau, basis, col_index, col_names, extra_names


def jalankan_simplex(tableau, basis, max_iter=60, tol=1e-7):
    history = [tableau.copy()]
    basis_hist = [list(basis)]
    status = "optimal"
    m = tableau.shape[0] - 1
    it = 0
    while it < max_iter:
        obj = tableau[-1, :-1]
        if np.all(obj >= -tol):
            break
        masuk = int(np.argmin(obj))
        kolom = tableau[:-1, masuk]
        if np.all(kolom <= tol):
            status = "unbounded"
            break
        rasio = np.array([tableau[i, -1] / kolom[i] if kolom[i] > tol else np.inf for i in range(m)])
        keluar = int(np.argmin(rasio))
        pivot = tableau[keluar, masuk]
        tableau[keluar, :] = tableau[keluar, :] / pivot
        for i in range(m + 1):
            if i != keluar:
                tableau[i, :] -= tableau[i, masuk] * tableau[keluar, :]
        basis[keluar] = masuk
        history.append(tableau.copy())
        basis_hist.append(list(basis))
        it += 1
    else:
        status = "max_iter"
    return tableau, basis, history, basis_hist, status


def tampilkan_tabel(tableau, basis, col_names, col_index):
    reverse = {v: k for k, v in col_index.items()}
    n = 2
    def nama(idx):
        return col_names[idx] if idx < n else reverse.get(idx, f"c{idx}")
    baris_label = [nama(b) for b in basis] + ["Z"]
    df = pd.DataFrame(np.round(tableau, 3), columns=col_names, index=baris_label)
    st.dataframe(df, use_container_width=True)


def proses_simplex_bigm(judul_langkah_awal):
    constraints = [(a1, b1, t1, k1), (a2, b2, t2, k2)]
    tableau, basis, col_index, col_names, extra_names = bangun_tabel(
        [coef_x, coef_y], constraints, optimasi == "Maksimum"
    )

    st.write(f"**Variabel tambahan yang dipakai:** {', '.join(extra_names) if extra_names else '(tidak ada)'}")
    st.write(judul_langkah_awal)
    tampilkan_tabel(tableau, basis, col_names, col_index)

    tableau_akhir, basis_akhir, history, basis_hist, status = jalankan_simplex(tableau, basis)

    if len(history) > 1:
        st.subheader("Iterasi")
        for i in range(1, len(history)):
            with st.expander(f"Iterasi {i}"):
                tampilkan_tabel(history[i], basis_hist[i], col_names, col_index)

    if status == "unbounded":
        st.markdown('<div class="warn-box">Nilai fungsi tujuan tidak terbatas (unbounded). Cek lagi kendalanya.</div>', unsafe_allow_html=True)
        return
    if status == "max_iter":
        st.markdown('<div class="warn-box">Belum konvergen setelah banyak iterasi, kemungkinan ada siklus.</div>', unsafe_allow_html=True)
        return

    reverse = {v: k for k, v in col_index.items()}
    solusi = {"x": 0.0, "y": 0.0}
    infeasible = False
    for i, b_idx in enumerate(basis_akhir):
        nilai = tableau_akhir[i, -1]
        if b_idx == 0:
            solusi["x"] = nilai
        elif b_idx == 1:
            solusi["y"] = nilai
        else:
            nm = reverse.get(b_idx, "")
            if nm.startswith("a") and nilai > 1e-6:
                infeasible = True

    if infeasible:
        st.markdown('<div class="warn-box">Tidak ada solusi yang memenuhi semua kendala (infeasible) &mdash; masih ada variabel buatan yang bernilai positif.</div>', unsafe_allow_html=True)
        return

    z_internal = tableau_akhir[-1, -1]
    z_final = z_internal if optimasi == "Maksimum" else -z_internal

    st.markdown(
        f"""<div class="hasil-box">
        Solusi optimal: x = <b>{solusi['x']:.2f}</b>, y = <b>{solusi['y']:.2f}</b><br>
        Nilai {optimasi.lower()} Z = <b>{z_final:.2f}</b>
        </div>""",
        unsafe_allow_html=True,
    )


# ===========================================================================
# eksekusi sesuai metode yang dipilih
# ===========================================================================
if hitung:
    st.divider()
    st.header("Hasil")

    st.write("**Model yang terbentuk:**")
    st.latex(f"Z = {coef_x:g}x + {coef_y:g}y")
    st.latex(f"{a1:g}x + {b1:g}y {t1} {k1:g}")
    st.latex(f"{a2:g}x + {b2:g}y {t2} {k2:g}")
    st.latex("x \\ge 0,\\ y \\ge 0")

    if metode == "Metode Grafik":
        if t1 != "<=" or t2 != "<=":
            st.markdown('<div class="warn-box">Metode grafik di sini cuma menghitung untuk kendala \u2264.</div>', unsafe_allow_html=True)
        else:
            x_pot1 = k1 / a1 if a1 != 0 else 0
            y_pot1 = k1 / b1 if b1 != 0 else 0
            x_pot2 = k2 / a2 if a2 != 0 else 0
            y_pot2 = k2 / b2 if b2 != 0 else 0

            st.subheader("Titik potong ke sumbu")
            tab1, tab2 = st.tabs(["Kendala 1", "Kendala 2"])
            with tab1:
                st.write(f"- x = 0 \u2192 y = {y_pot1:g}, titik (0, {y_pot1:g})")
                st.write(f"- y = 0 \u2192 x = {x_pot1:g}, titik ({x_pot1:g}, 0)")
            with tab2:
                st.write(f"- x = 0 \u2192 y = {y_pot2:g}, titik (0, {y_pot2:g})")
                st.write(f"- y = 0 \u2192 x = {x_pot2:g}, titik ({x_pot2:g}, 0)")

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

            sudut = [[0.0, 0.0], [min(x_pot1, x_pot2), 0.0], [0.0, min(y_pot1, y_pot2)]]
            if xp is not None and xp >= -1e-6 and yp >= -1e-6:
                if (a1 * xp + b1 * yp <= k1 + 1e-5) and (a2 * xp + b2 * yp <= k2 + 1e-5):
                    sudut.append([xp, yp])
            sudut = np.unique(np.round(np.array(sudut), 6), axis=0)

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
                Z = <b>{solusi['Z']:.2f}</b> (x = {solusi['x']:.2f}, y = {solusi['y']:.2f}).
                </div>""",
                unsafe_allow_html=True,
            )

    elif metode == "Metode Simplex":
        st.subheader("Tabel awal simplex")
        proses_simplex_bigm("Bentuk baku ditambahkan variabel slack.")

    else:  # Metode Big M
        st.subheader("Tabel awal Big M")
        proses_simplex_bigm("Bentuk baku ditambahkan variabel slack/surplus/buatan sesuai tanda kendala.")

else:
    st.info("Isi datanya dulu di atas, terus tekan Hitung untuk lihat hasilnya.")
