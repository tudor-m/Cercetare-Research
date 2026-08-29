"""
Dashboard Streamlit - Firme din Romania (bl_bs_sl, strat L2) — VERSIUNEA DEPLOYATA (Cloud)

Acesta e fisierul pe care il ruleaza Streamlit Cloud ("main file path" configurat acolo).
Pentru experimentat local fara sa afectezi site-ul public, foloseste in schimb
`firme_dashboard_local.py` (copie separata) - si copiaza aici modificarile doar cand esti
sigur ca vrei sa ajunga pe Cloud.

Ruleaza cu:
    streamlit run firme_dashboard.py

Patru tab-uri clasice, orizontal sus pe pagina principala (`st.tabs()`); fiecare tab isi are
propriul selector chiar sub titlul tab-ului, nu in sidebar:
- "Analiza pe judet": utilizatorul alege intreaga tara sau un judet; interfata
  afiseaza 4 grafice (histograma cifrei de afaceri, histograma numarului de salariati, bar
  chart cu top 10 coduri CAEN, scatter cifra de afaceri vs. salariati) plus un tabel cu cele
  10 coduri CAEN si denumirea activitatii.
- "Analiza pe CAEN - baza": replica analizei din analytics-1-firme-rom.ipynb (celula
  "Top 100 firme per cod CAEN"). Utilizatorul alege un cod CAEN (din top 50
  dupa numarul de firme); tabelul afiseaza primele 100 de firme din acel cod, sortate
  descrescator dupa cifra_de_afaceri_neta, cu identitatea firmei + indicatorii financiari.
- "Campionii Cresterii — 2021–2025": firmele bl_bs_sl prezente in ambii ani. Trei sectiuni,
  fiecare cu un Top 10 (tabel + bar chart orizontal, sortate descrescator): (1) crestere
  absoluta a cifrei de afaceri, (2) CAGR cifra de afaceri pe 4 ani (doar firme cu Revenue
  2021 >= 10.000.000 RON si cifra de afaceri pozitiva in ambii ani), (3) crestere absoluta
  a profitului net. Sursa: crestere_2021_2025_l2.parquet.
- "Sectoarele in Ascensiune — 2021–2025": aceleasi firme, agregate pe diviziunea CAEN de 2
  cifre (`cod_caen[:2]`, cod CAEN principal al firmei), cu denumiri lizibile de diviziune.
  Aceleasi trei sectiuni ca mai sus, la nivel de sector; la CAGR se pastreaza doar
  diviziunile cu cifra de afaceri agregata >= PRAG_MINIM_CIFRA_SECTOR in 2021.

Datele L2 sunt Parquet (nu CSV) si sunt citite via DuckDB: filtrarea (pe judet sau pe cod
CAEN) se face direct in fisier (predicate pushdown), fara sa incarcam tabelul intreg in
memorie la fiecare selectie - relevant pentru un deploy cu resurse limitate (Streamlit
Cloud/Replit).

Datele citite aici sunt doar subsetul mic necesar acestui dashboard (bl_bs_sl_l2.parquet +
crestere_2021_2025_l2.parquet + N_CAEN.csv, ~9MB), copiat in `data/` (langa acest script) si
tinut in git - nu intregul lac de date de la `data.gov.ro/l2_data/` (vezi
data-download-firme-rom.ipynb), care ramane local, in afara repo-ului. Cand rulezi din nou
notebook-ul de download si vrei ca acest dashboard sa reflecte datele noi, recopiaza
fisierele din `data.gov.ro/l2_data/` si `data.gov.ro/ref_data/` peste cele din `data/`.
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

BL_BS_SL_PARQUET = DATA_DIR / "bl_bs_sl_l2.parquet"
CRESTERE_PARQUET = DATA_DIR / "crestere_2021_2025_l2.parquet"
N_CAEN_CSV = DATA_DIR / "N_CAEN.csv"

TOATA_TARA = "Toata tara"

# Anii comparati in tab-urile de crestere (vezi crestere_<start>_<end>_l2 in data-download-firme-rom.ipynb).
CRESTERE_AN_START, CRESTERE_AN_END = 2021, 2025

COL_REV_START = f"cifra_de_afaceri_neta_{CRESTERE_AN_START}"
COL_REV_END = f"cifra_de_afaceri_neta_{CRESTERE_AN_END}"
COL_PN_START = f"profitul_net_{CRESTERE_AN_START}"
COL_PN_END = f"profitul_net_{CRESTERE_AN_END}"

# Prag minim pentru cifra de afaceri agregata a unei diviziuni in anul de start, la sectiunea CAGR pe
# sectoare - elimina diviziunile minuscule (cateva firme) unde un raport 2025/2021 devine nereprezentativ.
PRAG_MINIM_CIFRA_SECTOR = 1_000_000_000

# Denumiri lizibile pentru diviziunile CAEN de 2 cifre (CAEN Rev. 2). N_CAEN.csv are doar denumiri
# la nivel de clasa (4 cifre), nu si la nivel de diviziune, asa ca le tinem aici ca referinta statica.
DENUMIRI_DIVIZIUNE_CAEN = {
    "01": "Agricultura, vanatoare si servicii anexe",
    "02": "Silvicultura si exploatare forestiera",
    "03": "Pescuitul si acvacultura",
    "05": "Extractia carbunelui superior si inferior",
    "06": "Extractia petrolului brut si a gazelor naturale",
    "07": "Extractia minereurilor metalifere",
    "08": "Alte activitati extractive",
    "09": "Activitati de servicii anexe extractiei",
    "10": "Industria alimentara",
    "11": "Fabricarea bauturilor",
    "12": "Fabricarea produselor din tutun",
    "13": "Fabricarea produselor textile",
    "14": "Fabricarea articolelor de imbracaminte",
    "15": "Tabacirea pieilor; fabricarea articolelor din piele si a incaltamintei",
    "16": "Prelucrarea lemnului si fabricarea produselor din lemn (exclusiv mobila)",
    "17": "Fabricarea hartiei si a produselor din hartie",
    "18": "Tiparirea si reproducerea pe suporti a inregistrarilor",
    "19": "Fabricarea produselor de cocserie si din prelucrarea titeiului",
    "20": "Fabricarea substantelor si a produselor chimice",
    "21": "Fabricarea produselor farmaceutice de baza si a preparatelor farmaceutice",
    "22": "Fabricarea produselor din cauciuc si mase plastice",
    "23": "Fabricarea altor produse din minerale nemetalice",
    "24": "Industria metalurgica",
    "25": "Constructii metalice si produse din metal (exclusiv masini si utilaje)",
    "26": "Fabricarea calculatoarelor si a produselor electronice si optice",
    "27": "Fabricarea echipamentelor electrice",
    "28": "Fabricarea de masini, utilaje si echipamente n.c.a.",
    "29": "Fabricarea autovehiculelor de transport rutier, a remorcilor si semiremorcilor",
    "30": "Fabricarea altor mijloace de transport",
    "31": "Fabricarea de mobila",
    "32": "Alte activitati industriale n.c.a.",
    "33": "Repararea, intretinerea si instalarea masinilor si echipamentelor",
    "35": "Productia si furnizarea de energie electrica si termica, gaze si aer conditionat",
    "36": "Captarea, tratarea si distributia apei",
    "37": "Colectarea si epurarea apelor uzate",
    "38": "Colectarea, tratarea si eliminarea deseurilor; recuperarea materialelor reciclabile",
    "39": "Activitati si servicii de decontaminare",
    "41": "Constructii de cladiri",
    "42": "Lucrari de geniu civil",
    "43": "Lucrari speciale de constructii",
    "45": "Comert si reparatii de autovehicule si motociclete",
    "46": "Comert cu ridicata (exclusiv autovehicule si motociclete)",
    "47": "Comert cu amanuntul (exclusiv autovehicule si motociclete)",
    "49": "Transporturi terestre si transporturi prin conducte",
    "50": "Transporturi pe apa",
    "51": "Transporturi aeriene",
    "52": "Depozitare si activitati auxiliare pentru transporturi",
    "53": "Activitati de posta si de curier",
    "55": "Hoteluri si alte facilitati de cazare",
    "56": "Restaurante si alte activitati de servicii de alimentatie",
    "58": "Activitati de editare",
    "59": "Productie cinematografica, video si de programe TV; inregistrari audio",
    "60": "Activitati de difuzare si transmitere de programe",
    "61": "Telecomunicatii",
    "62": "Activitati de servicii in tehnologia informatiei",
    "63": "Activitati de servicii informatice",
    "64": "Intermedieri financiare (exclusiv asigurari si fonduri de pensii)",
    "65": "Activitati de asigurari, reasigurari si ale fondurilor de pensii",
    "66": "Activitati auxiliare intermedierilor financiare si asigurarilor",
    "68": "Tranzactii imobiliare",
    "69": "Activitati juridice si de contabilitate",
    "70": "Activitati ale sediilor centrale; consultanta in management",
    "71": "Activitati de arhitectura si inginerie; testari si analize tehnice",
    "72": "Cercetare-dezvoltare",
    "73": "Publicitate si activitati de studiere a pietei",
    "74": "Alte activitati profesionale, stiintifice si tehnice",
    "75": "Activitati veterinare",
    "77": "Activitati de inchiriere si leasing",
    "78": "Activitati de servicii privind forta de munca",
    "79": "Agentii turistice si tur-operatori; servicii de rezervare",
    "80": "Activitati de investigatii si protectie",
    "81": "Activitati de peisagistica si servicii pentru cladiri",
    "82": "Activitati de secretariat si alte servicii suport pentru intreprinderi",
    "84": "Administratie publica si aparare; asigurari sociale din sistemul public",
    "85": "Invatamant",
    "86": "Activitati referitoare la sanatatea umana",
    "87": "Servicii de ingrijire medicala si asistenta sociala, cu cazare",
    "88": "Activitati de asistenta sociala, fara cazare",
    "90": "Activitati de creatie si interpretare artistica",
    "91": "Activitati ale bibliotecilor, arhivelor, muzeelor si alte activitati culturale",
    "92": "Activitati de jocuri de noroc si pariuri",
    "93": "Activitati sportive, recreative si distractive",
    "94": "Activitati asociative diverse",
    "95": "Reparatii de calculatoare si de articole de uz personal si gospodaresc",
    "96": "Alte activitati de servicii",
    "97": "Activitati ale gospodariilor ca angajator de personal casnic",
    "98": "Activitati ale gospodariilor de producere de bunuri si servicii pentru consum propriu",
    "99": "Activitati ale organizatiilor si organismelor extrateritoriale",
}

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


def formateaza_ron(valoare) -> str:
    """RON compact: '1.18 mld RON', '647.3 mil RON', '850 mii RON', '-1 200 RON'."""
    if valoare is None or pd.isna(valoare):
        return "—"
    semn = "-" if valoare < 0 else ""
    x = abs(float(valoare))
    if x >= 1e9:
        return f"{semn}{x / 1e9:.2f} mld RON"
    if x >= 1e6:
        return f"{semn}{x / 1e6:.1f} mil RON"
    if x >= 1e3:
        return f"{semn}{x / 1e3:,.0f} mii RON".replace(",", " ")
    return f"{semn}{x:,.0f} RON".replace(",", " ")


def formateaza_procent(valoare) -> str:
    """Procent cu un zecimal: 0.4231 -> '42.3%'."""
    if valoare is None or pd.isna(valoare):
        return "—"
    return f"{valoare * 100:.1f}%"


def _ron_axa(x, _pozitie) -> str:
    semn = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e9:
        return f"{semn}{x / 1e9:.0f} mld"
    if x >= 1e6:
        return f"{semn}{x / 1e6:.0f} mil"
    if x >= 1e3:
        return f"{semn}{x / 1e3:.0f} mii"
    return f"{semn}{x:.0f}"


def _scurt(nume: str, maxim: int = 34) -> str:
    nume = str(nume)
    return nume if len(nume) <= maxim else nume[: maxim - 1] + "…"


def bar_chart_orizontal(
    nume: pd.Series, valori: pd.Series, eticheta_x: str, culoare: str, formator_axa, maxim_eticheta: int = 34
) -> None:
    """Bar chart orizontal cu top-ul sortat descrescator (cea mai mare valoare sus)."""
    ordine = np.argsort(valori.to_numpy())  # crescator -> cea mai mare bara ajunge in varf
    pozitii = np.arange(len(valori))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(pozitii, valori.to_numpy()[ordine], color=culoare)
    ax.set_yticks(pozitii)
    ax.set_yticklabels([_scurt(n, maxim_eticheta) for n in nume.to_numpy()[ordine]], fontsize=8)
    ax.set_xlabel(eticheta_x)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(formator_axa))
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    st.pyplot(fig)


@st.cache_data
def incarca_crestere() -> pd.DataFrame:
    """Firmele bl_bs_sl prezente in ambii ani comparati, cu cifra de afaceri neta si profitul net
    pentru fiecare an + identitatea firmei + `cod_caen` (codul CAEN principal, 4 cifre)
    (vezi crestere_<start>_<end>_l2 in notebook-ul de download)."""
    df = duckdb.sql(f"SELECT * FROM '{CRESTERE_PARQUET.as_posix()}'").df()
    df["DENUMIRE"] = df["DENUMIRE"].fillna("CUI " + df["cui"].astype(str))
    return df


NUME_TAB_JUDET = "Analiza pe judet"
NUME_TAB_CAEN = "Analiza pe CAEN - baza"
NUME_TAB_CRESTERE = f"Campionii Cresterii — {CRESTERE_AN_START}–{CRESTERE_AN_END}"
NUME_TAB_SECTOARE = f"Sectoarele in Ascensiune — {CRESTERE_AN_START}–{CRESTERE_AN_END}"

st.set_page_config(page_title="Firme din Romania - bl_bs_sl", layout="wide")
st.title("Firme din Romania - situatii financiare (bl_bs_sl)")

denumiri_caen = incarca_denumiri_caen()
top_50_caen = incarca_top_50_caen()

tab_judet, tab_caen, tab_crestere, tab_sectoare = st.tabs(
    [NUME_TAB_JUDET, NUME_TAB_CAEN, NUME_TAB_CRESTERE, NUME_TAB_SECTOARE]
)

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

with tab_crestere:
    st.header(NUME_TAB_CRESTERE)

    df_cr = incarca_crestere()

    st.caption(
        f"{len(df_cr):,}".replace(",", " ")
        + f" firme care au depus situatii financiare bl_bs_sl in ambii ani "
        f"({CRESTERE_AN_START} si {CRESTERE_AN_END}). Fiecare sectiune: Top 10, tabel + bar chart orizontal, "
        f"sortate descrescator."
    )

    # ---- 1. Absolute Revenue Growth ----
    st.subheader("1. Absolute Revenue Growth")
    st.caption(f"revenue_growth = cifra_de_afaceri_neta_{CRESTERE_AN_END} − cifra_de_afaceri_neta_{CRESTERE_AN_START}")

    d1 = (
        df_cr.assign(crestere=df_cr[COL_REV_END] - df_cr[COL_REV_START])
        .nlargest(10, "crestere")
        .reset_index(drop=True)
    )
    st.dataframe(
        pd.DataFrame({
            "Rank": np.arange(1, len(d1) + 1),
            "Company": d1["DENUMIRE"],
            f"Revenue {CRESTERE_AN_START}": d1[COL_REV_START].map(formateaza_ron),
            f"Revenue {CRESTERE_AN_END}": d1[COL_REV_END].map(formateaza_ron),
            "Increase (RON)": d1["crestere"].map(formateaza_ron),
        }),
        hide_index=True,
    )
    bar_chart_orizontal(
        d1["DENUMIRE"], d1["crestere"], "crestere cifra de afaceri (RON)", "seagreen", _ron_axa
    )

    # ---- 2. Revenue CAGR ----
    st.subheader("2. Revenue CAGR")
    st.caption(
        f"CAGR = (Revenue_{CRESTERE_AN_END} / Revenue_{CRESTERE_AN_START})^(1/4) − 1 · doar firme cu "
        f"cifra de afaceri pozitiva in ambii ani si Revenue {CRESTERE_AN_START} ≥ 10.000.000 RON"
    )

    eligibile = df_cr[
        (df_cr[COL_REV_START] > 0)
        & (df_cr[COL_REV_END] > 0)
        & (df_cr[COL_REV_START] >= 10_000_000)
    ].copy()
    eligibile["cagr"] = (eligibile[COL_REV_END] / eligibile[COL_REV_START]) ** (1 / 4) - 1
    d2 = eligibile.nlargest(10, "cagr").reset_index(drop=True)

    st.dataframe(
        pd.DataFrame({
            "Rank": np.arange(1, len(d2) + 1),
            "Company": d2["DENUMIRE"],
            f"Revenue {CRESTERE_AN_START}": d2[COL_REV_START].map(formateaza_ron),
            f"Revenue {CRESTERE_AN_END}": d2[COL_REV_END].map(formateaza_ron),
            "CAGR %": d2["cagr"].map(formateaza_procent),
        }),
        hide_index=True,
    )
    bar_chart_orizontal(
        d2["DENUMIRE"], d2["cagr"], "CAGR", "steelblue", lambda x, _p: f"{x * 100:.0f}%"
    )

    # ---- 3. Absolute Net Profit Growth ----
    st.subheader("3. Absolute Net Profit Growth")
    st.caption(f"profit_growth = profitul_net_{CRESTERE_AN_END} − profitul_net_{CRESTERE_AN_START}")

    d3 = (
        df_cr.assign(crestere=df_cr[COL_PN_END] - df_cr[COL_PN_START])
        .nlargest(10, "crestere")
        .reset_index(drop=True)
    )
    st.dataframe(
        pd.DataFrame({
            "Rank": np.arange(1, len(d3) + 1),
            "Company": d3["DENUMIRE"],
            f"Net Profit {CRESTERE_AN_START}": d3[COL_PN_START].map(formateaza_ron),
            f"Net Profit {CRESTERE_AN_END}": d3[COL_PN_END].map(formateaza_ron),
            "Increase (RON)": d3["crestere"].map(formateaza_ron),
        }),
        hide_index=True,
    )
    bar_chart_orizontal(
        d3["DENUMIRE"], d3["crestere"], "crestere profit net (RON)", "indianred", _ron_axa
    )

with tab_sectoare:
    st.header(NUME_TAB_SECTOARE)

    df_sec = incarca_crestere()
    df_sec = df_sec[df_sec["cod_caen"].notna()].copy()
    df_sec["diviziune"] = df_sec["cod_caen"].str.slice(0, 2)

    sectoare = df_sec.groupby("diviziune", as_index=False)[
        [COL_REV_START, COL_REV_END, COL_PN_START, COL_PN_END]
    ].sum()
    sectoare["Sector"] = sectoare["diviziune"].map(
        lambda d: DENUMIRI_DIVIZIUNE_CAEN.get(d, f"Diviziunea {d}")
    )

    st.caption(
        f"{len(df_sec):,}".replace(",", " ")
        + f" firme bl_bs_sl (prezente in {CRESTERE_AN_START} si {CRESTERE_AN_END}), agregate pe "
        f"{len(sectoare)} diviziuni CAEN de 2 cifre dupa codul CAEN principal al firmei. "
        f"Fiecare sectiune: Top 10, tabel + bar chart orizontal, sortate descrescator."
    )

    # ---- 1. Absolute Revenue Growth by Sector ----
    st.subheader("1. Absolute Revenue Growth by Sector")
    st.caption(
        f"revenue_growth = total_revenue_{CRESTERE_AN_END} − total_revenue_{CRESTERE_AN_START} (pe diviziune)"
    )
    s1 = (
        sectoare.assign(crestere=sectoare[COL_REV_END] - sectoare[COL_REV_START])
        .nlargest(10, "crestere")
        .reset_index(drop=True)
    )
    st.dataframe(
        pd.DataFrame({
            "Rank": np.arange(1, len(s1) + 1),
            "CAEN": s1["diviziune"],
            "Sector": s1["Sector"],
            f"Revenue {CRESTERE_AN_START}": s1[COL_REV_START].map(formateaza_ron),
            f"Revenue {CRESTERE_AN_END}": s1[COL_REV_END].map(formateaza_ron),
            "Increase (RON)": s1["crestere"].map(formateaza_ron),
        }),
        hide_index=True,
    )
    bar_chart_orizontal(
        s1["Sector"], s1["crestere"], "crestere cifra de afaceri (RON)", "seagreen", _ron_axa,
        maxim_eticheta=52,
    )

    # ---- 2. Revenue CAGR by Sector ----
    st.subheader("2. Revenue CAGR by Sector")
    st.caption(
        f"CAGR = (total_revenue_{CRESTERE_AN_END} / total_revenue_{CRESTERE_AN_START})^(1/4) − 1 · doar "
        f"diviziuni cu cifra de afaceri agregata pozitiva in ambii ani si ≥ "
        f"{formateaza_ron(PRAG_MINIM_CIFRA_SECTOR)} in {CRESTERE_AN_START}"
    )
    sectoare_eligibile = sectoare[
        (sectoare[COL_REV_START] > 0)
        & (sectoare[COL_REV_END] > 0)
        & (sectoare[COL_REV_START] >= PRAG_MINIM_CIFRA_SECTOR)
    ].copy()
    sectoare_eligibile["cagr"] = (
        sectoare_eligibile[COL_REV_END] / sectoare_eligibile[COL_REV_START]
    ) ** (1 / 4) - 1
    s2 = sectoare_eligibile.nlargest(10, "cagr").reset_index(drop=True)
    st.dataframe(
        pd.DataFrame({
            "Rank": np.arange(1, len(s2) + 1),
            "CAEN": s2["diviziune"],
            "Sector": s2["Sector"],
            f"Revenue {CRESTERE_AN_START}": s2[COL_REV_START].map(formateaza_ron),
            f"Revenue {CRESTERE_AN_END}": s2[COL_REV_END].map(formateaza_ron),
            "CAGR %": s2["cagr"].map(formateaza_procent),
        }),
        hide_index=True,
    )
    bar_chart_orizontal(
        s2["Sector"], s2["cagr"], "CAGR", "steelblue", lambda x, _p: f"{x * 100:.0f}%",
        maxim_eticheta=52,
    )

    # ---- 3. Absolute Net Profit Growth by Sector ----
    st.subheader("3. Absolute Net Profit Growth by Sector")
    st.caption(
        f"profit_growth = total_net_profit_{CRESTERE_AN_END} − total_net_profit_{CRESTERE_AN_START} (pe diviziune)"
    )
    s3 = (
        sectoare.assign(crestere=sectoare[COL_PN_END] - sectoare[COL_PN_START])
        .nlargest(10, "crestere")
        .reset_index(drop=True)
    )
    st.dataframe(
        pd.DataFrame({
            "Rank": np.arange(1, len(s3) + 1),
            "CAEN": s3["diviziune"],
            "Sector": s3["Sector"],
            f"Net Profit {CRESTERE_AN_START}": s3[COL_PN_START].map(formateaza_ron),
            f"Net Profit {CRESTERE_AN_END}": s3[COL_PN_END].map(formateaza_ron),
            "Increase (RON)": s3["crestere"].map(formateaza_ron),
        }),
        hide_index=True,
    )
    bar_chart_orizontal(
        s3["Sector"], s3["crestere"], "crestere profit net (RON)", "indianred", _ron_axa,
        maxim_eticheta=52,
    )
