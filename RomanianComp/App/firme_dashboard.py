"""
Dashboard Streamlit - Firme din Romania (bl_bs_sl, strat L2)

Ruleaza cu:
    streamlit run firme_dashboard.py

Utilizatorul alege intreaga tara sau un judet; interfata afiseaza 4 grafice
(histograma cifrei de afaceri, histograma numarului de salariati, bar chart
cu top 10 coduri CAEN, scatter cifra de afaceri vs. salariati) plus un tabel
cu cele 10 coduri CAEN si denumirea activitatii.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

L2_DIR = Path("/Users/tudor/Documents/Data-for-Projects/Cercetare-Research/data.gov.ro/l2_data")
REF_DATA_DIR = Path("/Users/tudor/Documents/Data-for-Projects/Cercetare-Research/data.gov.ro/ref_data")

TOATA_TARA = "Toata tara"


@st.cache_data
def incarca_bl_bs_sl() -> pd.DataFrame:
    df = pd.read_csv(L2_DIR / "bl_bs_sl_l2.csv", encoding="utf-8-sig")
    # cateva coduri CAEN din sursa lipsesc zero-ul de inceput (ex. "154" in loc de "0154")
    df["cod_caen"] = df["caen"].astype(str).str.strip().str.zfill(4)
    return df


@st.cache_data
def incarca_denumiri_caen() -> pd.Series:
    """cod CAEN (4 cifre) -> denumire activitate, preferand cea mai recenta versiune a nomenclatorului
    (acelasi cod poate insemna activitati diferite in versiuni CAEN diferite: 1998/2003/2008/2025)."""
    df_caen = pd.read_csv(REF_DATA_DIR / "N_CAEN.csv", sep="^", encoding="utf-8-sig", dtype=str)
    df_caen["CLASA"] = df_caen["CLASA"].str.strip()
    coduri = df_caen[df_caen["CLASA"].str.fullmatch(r"\d{4}", na=False)]
    return (
        coduri.sort_values("VERSIUNE_CAEN", ascending=False)
        .drop_duplicates("CLASA")
        .set_index("CLASA")["DENUMIRE"]
    )


def histograma_log10(ax, valori: pd.Series, titlu: str, eticheta_x: str, culoare: str) -> None:
    valori_pozitive = valori[valori > 0]
    excluse = len(valori) - len(valori_pozitive)
    ax.hist(np.log10(valori_pozitive), bins=30, color=culoare)
    ax.set_title(f"{titlu}\n({excluse} firme cu valoare <=0 sau lipsa excluse)", fontsize=10)
    ax.set_xlabel(eticheta_x)
    ax.set_ylabel("numar firme")


st.set_page_config(page_title="Firme din Romania - bl_bs_sl", layout="wide")
st.title("Firme din Romania - situatii financiare (bl_bs_sl)")

df_bl_bs_sl = incarca_bl_bs_sl()
denumiri_caen = incarca_denumiri_caen()

judete = [TOATA_TARA] + sorted(df_bl_bs_sl["ADR_JUDET"].dropna().unique().tolist())
scop_selectat = st.sidebar.selectbox("Alege scopul", judete)

if scop_selectat == TOATA_TARA:
    df_scop = df_bl_bs_sl
else:
    df_scop = df_bl_bs_sl[df_bl_bs_sl["ADR_JUDET"] == scop_selectat]

st.subheader(f"{scop_selectat} — {len(df_scop)} firme")

if df_scop.empty:
    st.warning("Nicio firma pentru selectia curenta.")
else:
    top_10_caen = (
        df_scop["cod_caen"]
        .value_counts()
        .head(10)
        .rename_axis("cod_caen")
        .reset_index(name="numar_firme")
    )
    top_10_caen["denumire_activitate"] = top_10_caen["cod_caen"].map(denumiri_caen)

    coloana_stanga, coloana_dreapta = st.columns(2)

    with coloana_stanga:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        histograma_log10(
            ax, df_scop["cifra_de_afaceri_neta"],
            "Cifra de afaceri neta (log10)", "log10(cifra_de_afaceri_neta)", "steelblue",
        )
        st.pyplot(fig)

    with coloana_dreapta:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        histograma_log10(
            ax, df_scop["numar_mediu_de_salariati"],
            "Numar mediu de salariati (log10)", "log10(numar_mediu_de_salariati)", "darkorange",
        )
        st.pyplot(fig)

    coloana_stanga2, coloana_dreapta2 = st.columns(2)

    with coloana_stanga2:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        afisare = top_10_caen.sort_values("numar_firme")
        ax.barh(afisare["cod_caen"], afisare["numar_firme"], color="seagreen")
        ax.set_title("Top 10 coduri CAEN", fontsize=10)
        ax.set_xlabel("numar firme")
        st.pyplot(fig)

    with coloana_dreapta2:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        subset = df_scop[(df_scop["cifra_de_afaceri_neta"] > 0) & (df_scop["numar_mediu_de_salariati"] > 0)]
        ax.scatter(
            np.log10(subset["numar_mediu_de_salariati"]),
            np.log10(subset["cifra_de_afaceri_neta"]),
            s=6, alpha=0.3, color="teal",
        )
        ax.set_title("Cifra de afaceri vs. nr. salariati (log-log)", fontsize=10)
        ax.set_xlabel("log10(numar_mediu_de_salariati)")
        ax.set_ylabel("log10(cifra_de_afaceri_neta)")
        st.pyplot(fig)

    st.subheader("Top 10 coduri CAEN - detaliu")
    st.dataframe(top_10_caen, hide_index=True)
