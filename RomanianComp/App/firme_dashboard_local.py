"""
Dashboard Streamlit - Firme din Romania (bl_bs_sl, strat L2) — COPIE LOCALA DE TESTARE

Aceasta e o copie separata de `firme_dashboard.py`, folosita pentru experimentat local.
Streamlit Cloud ruleaza doar `firme_dashboard.py` (asa e configurat "main file path" in
Cloud) - modificarile facute aici, chiar daca sunt impinse pe `main`, NU ajung pe site-ul
public pana nu le copiezi manual in `firme_dashboard.py`. Un `git push` care atinge doar
acest fisier tot va declansa un rebuild pe Cloud (urmareste tot repo-ul), dar comportamentul
aplicatiei deployate ramane neschimbat, fiindca Cloud citeste tot `firme_dashboard.py`.

Ruleaza cu:
    streamlit run firme_dashboard_local.py

Doua tab-uri clasice, orizontal sus pe pagina principala (`st.tabs()`); fiecare tab isi are
propriul selector chiar sub titlul tab-ului, nu in sidebar:
- "Analiza pe judet": (ca in firme_dashboard.py) utilizatorul alege intreaga tara sau un
  judet; interfata afiseaza 4 grafice (histograma cifrei de afaceri, histograma
  numarului de salariati, bar chart cu top 10 coduri CAEN, scatter cifra de afaceri vs.
  salariati) plus un tabel cu cele 10 coduri CAEN si denumirea activitatii.
- "Analiza pe CAEN - baza": replica analizei din analytics-1-firme-rom.ipynb (celula
  "Top 100 firme per cod CAEN"). Utilizatorul alege un cod CAEN (din top 50
  dupa numarul de firme); tabelul afiseaza primele 100 de firme din acel cod, sortate
  descrescator dupa cifra_de_afaceri_neta, cu identitatea firmei + indicatorii financiari.

Datele L2 sunt Parquet (nu CSV) si sunt citite via DuckDB: filtrarea (pe judet sau pe cod
CAEN) se face direct in fisier (predicate pushdown), fara sa incarcam tabelul intreg in
memorie la fiecare selectie - relevant pentru un deploy cu resurse limitate (Streamlit
Cloud/Replit).

Datele citite aici sunt doar subsetul mic necesar acestui dashboard (bl_bs_sl_l2.parquet
+ N_CAEN.csv, ~7MB), copiat in `data/` (langa acest script) si tinut in git - nu intregul
lac de date de la `data.gov.ro/l2_data/` (vezi data-download-firme-rom.ipynb), care ramane
local, in afara repo-ului. Cand rulezi din nou notebook-ul de download si vrei ca acest
dashboard sa reflecte datele noi, recopiaza cele 2 fisiere din `data.gov.ro/l2_data/` si
`data.gov.ro/ref_data/` peste cele din `data/`.
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

BL_BS_SL_PARQUET = DATA_DIR / "bl_bs_sl_l2.parquet"
N_CAEN_CSV = DATA_DIR / "N_CAEN.csv"

TOATA_TARA = "Toata tara"

COLOANE_TOP_FIRME = [
    "DENUMIRE", "cifra_de_afaceri_neta", "profitul_net", "numar_mediu_de_salariati", "datorii",
    "ADR_JUDET", "profitul_brut", "pierdere_bruta", "pierdere_neta", "venituri_totale",
    "cheltuieli_totale", "DATA_INMATRICULARE", "FORMA_JURIDICA", "ADR_LOCALITATE",
    "creante", "TARA_FIRMA_MAMA",
]

# cateva coduri CAEN din sursa lipsesc zero-ul de inceput (ex. "154" in loc de "0154")
_SELECT_CU_COD_CAEN = f"""
    SELECT *, LPAD(TRIM(CAST(caen AS VARCHAR)), 4, '0') AS cod_caen
    FROM '{BL_BS_SL_PARQUET.as_posix()}'
"""


@st.cache_data
def incarca_judete() -> list[str]:
    query = f"""
        SELECT DISTINCT ADR_JUDET FROM '{BL_BS_SL_PARQUET.as_posix()}'
        WHERE ADR_JUDET IS NOT NULL
        ORDER BY ADR_JUDET
    """
    return duckdb.sql(query).df()["ADR_JUDET"].tolist()


@st.cache_data
def incarca_scop(judet: str | None) -> pd.DataFrame:
    """Incarca doar randurile firmelor din judetul ales (sau toata tara daca judet e None),
    filtrate direct in Parquet - nu incarcam niciodata tabelul intreg doar ca sa il filtram in pandas."""
    if judet is None:
        return duckdb.sql(_SELECT_CU_COD_CAEN).df()
    return duckdb.execute(_SELECT_CU_COD_CAEN + " WHERE ADR_JUDET = ?", [judet]).df()


@st.cache_data
def incarca_denumiri_caen() -> pd.Series:
    """cod CAEN (4 cifre) -> denumire activitate, preferand cea mai recenta versiune a nomenclatorului
    (acelasi cod poate insemna activitati diferite in versiuni CAEN diferite: 1998/2003/2008/2025)."""
    df_caen = pd.read_csv(N_CAEN_CSV, sep="^", encoding="utf-8-sig", dtype=str)
    df_caen["CLASA"] = df_caen["CLASA"].str.strip()
    coduri = df_caen[df_caen["CLASA"].str.fullmatch(r"\d{4}", na=False)]
    return (
        coduri.sort_values("VERSIUNE_CAEN", ascending=False)
        .drop_duplicates("CLASA")
        .set_index("CLASA")["DENUMIRE"]
    )


@st.cache_data
def incarca_top_50_caen() -> pd.DataFrame:
    """Top 50 coduri CAEN dupa numarul de firme (acelasi calcul ca in analytics-1-firme-rom.ipynb)."""
    query = f"""
        SELECT cod_caen, COUNT(*) AS numar_firme
        FROM ({_SELECT_CU_COD_CAEN})
        GROUP BY cod_caen
        ORDER BY numar_firme DESC
        LIMIT 50
    """
    df = duckdb.sql(query).df()
    df["denumire_activitate"] = df["cod_caen"].map(incarca_denumiri_caen()).fillna("denumire necunoscuta")
    return df


@st.cache_data
def incarca_top_companii_caen(cod_caen: str) -> pd.DataFrame:
    """Primele 100 de firme pentru un cod CAEN, sortate descrescator dupa cifra_de_afaceri_neta -
    filtrare si sortare facute direct in DuckDB/Parquet (predicate pushdown), nu in pandas."""
    query = f"""
        SELECT {", ".join(COLOANE_TOP_FIRME)}
        FROM ({_SELECT_CU_COD_CAEN})
        WHERE cod_caen = ?
        ORDER BY cifra_de_afaceri_neta DESC
        LIMIT 100
    """
    return duckdb.execute(query, [cod_caen]).df()


def histograma_log10(ax, valori: pd.Series, titlu: str, eticheta_x: str, culoare: str) -> None:
    valori_pozitive = valori[valori > 0]
    excluse = len(valori) - len(valori_pozitive)
    ax.hist(np.log10(valori_pozitive), bins=30, color=culoare)
    ax.set_title(f"{titlu}\n({excluse} firme cu valoare <=0 sau lipsa excluse)", fontsize=10)
    ax.set_xlabel(eticheta_x)
    ax.set_ylabel("numar firme")
    ax.grid(True, alpha=0.3)


NUME_TAB_JUDET = "Analiza pe judet"
NUME_TAB_CAEN = "Analiza pe CAEN - baza"

st.set_page_config(page_title="Firme din Romania - bl_bs_sl", layout="wide")
st.title("Firme din Romania - situatii financiare (bl_bs_sl)")

denumiri_caen = incarca_denumiri_caen()
top_50_caen = incarca_top_50_caen()

tab_judet, tab_caen = st.tabs([NUME_TAB_JUDET, NUME_TAB_CAEN])

with tab_judet:
    st.header(NUME_TAB_JUDET)
    judete = [TOATA_TARA] + incarca_judete()
    scop_selectat = st.selectbox("Alege scopul", judete)

    df_scop = incarca_scop(None if scop_selectat == TOATA_TARA else scop_selectat)

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
            ax.grid(True, alpha=0.3)
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
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        st.subheader("Top 10 coduri CAEN - detaliu")
        st.dataframe(top_10_caen, hide_index=True)

with tab_caen:
    st.header(NUME_TAB_CAEN)
    optiuni_caen = {
        f"{rand['cod_caen']} — {rand['denumire_activitate']} ({rand['numar_firme']} firme)": rand["cod_caen"]
        for _, rand in top_50_caen.iterrows()
    }
    eticheta_caen_selectata = st.selectbox("Alege codul CAEN", list(optiuni_caen.keys()))
    cod_caen_selectat = optiuni_caen[eticheta_caen_selectata]

    denumire_selectata = top_50_caen.loc[
        top_50_caen["cod_caen"] == cod_caen_selectat, "denumire_activitate"
    ].iloc[0]
    df_top_companii = incarca_top_companii_caen(cod_caen_selectat)

    st.subheader(f"CAEN {cod_caen_selectat} — {denumire_selectata}")
    st.caption(f"Top {len(df_top_companii)} firme dupa cifra de afaceri neta")
    st.dataframe(df_top_companii, hide_index=True)
