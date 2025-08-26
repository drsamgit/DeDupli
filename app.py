# Streamlit Deduplicator (SRA‑like)
# -------------------------------------------------------------
# A local, SRA‑inspired deduplication app for systematic review search results.
# Key ideas mirrored from SRA/TERA Deduplicator (Forbes et al., 2024):
#  - Pre‑processing / mutators to normalize fields
#  - Multi‑step duplicate detection: exact IDs + fuzzy title similarity
#  - Three algorithm profiles (focused / balanced / relaxed)
#  - Buckets by likelihood (extremely likely / highly likely / possible / review)
#  - Human‑review for borderline clusters + full audit exports
#
# Dependencies (install these first):
#   pip install streamlit pandas rapidfuzz lxml unidecode
#
# Optional parsers (uncomment if desired):
#   pip install rispy  # for robust RIS parsing (we also provide a lightweight fallback)
#
# Run:
#   streamlit run app.py
# -------------------------------------------------------------

import io
import re
import json
import math
import hashlib
import zipfile
import unicodedata
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st
from rapidfuzz import fuzz
from unidecode import unidecode

APP_VERSION = "0.2.0"

# Canonical schema we try to map inputs to
CANON_COLS = [
    "record_id", "source_file", "database", "title", "authors", "year", "journal",
    "volume", "issue", "pages", "doi", "pmid", "isbn", "url", "abstract",
]

# Common alias -> canonical col mapping for CSVs
CSV_MAPPINGS = {
    "t": "title", "ti": "title", "article_title": "title", "primary_title": "title",
    "au": "authors", "author": "authors", "authors_list": "authors",
    "py": "year", "y": "year", "publication_year": "year",
    "jo": "journal", "journal_name": "journal", "source": "journal",
    "vl": "volume", "is": "issue", "pg": "pages",
    "do": "doi", "doi_link": "doi",
    "pm": "pmid", "pmid_number": "pmid",
    "sn": "isbn",
    "ur": "url", "link": "url",
    "ab": "abstract", "abstract_text": "abstract",
    "db": "database",
}

ALGO_PRESETS = {
    "focused":   {"extreme": 98, "high": 95, "possible": 90, "block_title_prefix": 18, "year_window": 1},
    "balanced":  {"extreme": 98, "high": 94, "possible": 88, "block_title_prefix": 16, "year_window": 1},
    "relaxed":   {"extreme": 99, "high": 96, "possible": 92, "block_title_prefix": 14, "year_window": 0},
}

# --------- Normalisation helpers (mutators) ---------

def normalize_text(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return ""
    s = unidecode(s)  # strip accents
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^0-9a-z\s]", "", s)  # keep alnum + space
    return s

def normalize_doi(doi: Optional[str]) -> str:
    if not isinstance(doi, str):
        return ""
    doi = doi.strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("doi:", "").strip()
    return doi.lower()

def normalize_authors(raw: Optional[str]) -> str:
    if not isinstance(raw, str):
        return ""
    # Accept semicolon or comma separated; unify to semicolons
    parts = [p.strip() for p in re.split(r";|,\s(?=[A-Z])|\|", raw) if p.strip()]
    return "; ".join(parts)

# --------- Lightweight parsers ---------

# RIS fallback (simple, robust subset). If rispy is installed, you can swap to that.
RIS_TAG_MAP = {
    "TI": "title", "T1": "title",
    "AU": "authors", "A1": "authors",
    "PY": "year", "Y1": "year", "DA": "year",
    "JO": "journal", "JF": "journal", "T2": "journal",
    "VL": "volume", "IS": "issue", "SP": "pages", "EP": "pages_end",
    "DO": "doi", "SN": "isbn", "UR": "url",
    "AB": "abstract",
    "DB": "database",
    "ID": "pmid", "PM": "pmid",
}

def parse_ris_bytes(b: bytes, source_file: str) -> pd.DataFrame:
    records = []
    current = {}

    def commit_current():
        nonlocal current
        if current:
            # Merge SP/EP -> pages
            if current.get("pages_end"):
                sp = str(current.get("pages", "")).split("-", 1)[0]
                ep = str(current.get("pages_end"))
                if sp and ep:
                    current["pages"] = f"{sp}-{ep}"
            records.append(current)
            current = {}

    for raw_line in b.decode(errors="ignore").splitlines():
        if re.match(r"^[A-Z0-9]{2}  - ", raw_line):
            tag, val = raw_line[:2], raw_line[6:]
            key = RIS_TAG_MAP.get(tag)
            if key:
                if key == "authors":
                    current[key] = (current.get(key, "") + ("; " if current.get(key) else "") + val.strip())
                else:
                    current[key] = val.strip()
        elif raw_line.strip() == "ER  -":
            commit_current()
        else:
            # continuation lines: append to last textual field if plausible
            if current:
                if any(k in current for k in ("abstract", "title")):
                    if "abstract" in current:
                        current["abstract"] += " " + raw_line.strip()
                    elif "title" in current:
                        current["title"] += " " + raw_line.strip()

    commit_current()

    df = pd.DataFrame(records)
    if not len(df):
        return pd.DataFrame(columns=CANON_COLS)

    df["source_file"] = source_file
    return to_canonical(df)

# NBIB (PubMed) very simple parser (subset: PMID, TI, JT, DP, AU, LID (DOI), AB, URL)
NBIB_MAP = {
    "PMID": "pmid",
    "TI": "title",
    "JT": "journal",
    "DP": "year",
    "AU": "authors",
    "LID": "doi",
    "AB": "abstract",
    "AID": "doi",  # often contains DOI + [doi]
    "LR": None,
}

def parse_nbib_bytes(b: bytes, source_file: str) -> pd.DataFrame:
    records = []
    current = {}

    def commit_current():
        nonlocal current
        if current:
            # Clean DOI lines that may include suffixes like "[doi]"
            if "doi" in current and isinstance(current["doi"], str):
                current["doi"] = re.sub(r"\s*\[doi\]$", "", current["doi"].strip(), flags=re.I)
            records.append(current)
            current = {}

    for line in b.decode(errors="ignore").splitlines():
        if re.match(r"^[A-Z]{2,4}-\s+", line):
            tag, val = line.split("-", 1)
            tag = tag.strip()
            val = val.strip()
            key = NBIB_MAP.get(tag, None)
            if key is None:
                continue
            if key == "authors":
                current[key] = (current.get(key, "") + ("; " if current.get(key) else "") + val)
            else:
                current[key] = val
        elif line.strip() == "":
            commit_current()
    commit_current()

    df = pd.DataFrame(records)
    if not len(df):
        return pd.DataFrame(columns=CANON_COLS)
    df["source_file"] = source_file
    return to_canonical(df)

# EndNote XML (.xml) parsing (lightweight via lxml)
from lxml import etree

ENDNOTE_XPATHS = {
    "title": ".//title",
    "authors": ".//contributors/authors/author/style",
    "year": ".//dates/year",
    "journal": ".//periodical/full-title",
    "volume": ".//volume",
    "issue": ".//number",
    "pages": ".//pages",
    "doi": ".//electronic-resource-num",
    "url": ".//web-urls/related-urls/url",
    "abstract": ".//abstract",
}

def parse_endnote_xml_bytes(b: bytes, source_file: str) -> pd.DataFrame:
    try:
        root = etree.fromstring(b)
    except Exception:
        return pd.DataFrame(columns=CANON_COLS)
    recs = []
    for rec in root.findall(".//record"):
        row = {}
        for k, xp in ENDNOTE_XPATHS.items():
            if k == "authors":
                vals = [el.text.strip() for el in rec.findall(xp) if el.text]
                row[k] = "; ".join(vals)
            else:
                el = rec.find(xp)
                row[k] = el.text.strip() if el is not None and el.text else None
        recs.append(row)
    df = pd.DataFrame(recs)
    if not len(df):
        return pd.DataFrame(columns=CANON_COLS)
    df["source_file"] = source_file
    return to_canonical(df)

# CSV parser with column auto‑mapping
def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase columns for mapping
    rename = {}
    for c in df.columns:
        key = CSV_MAPPINGS.get(c.strip().lower(), None)
        if key:
            rename[c] = key
    df = df.rename(columns=rename)
    # Ensure required cols exist
    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = None
    # Clean/normalize core fields
    df["title"] = df["title"].astype(str).fillna("")
    df["authors"] = df["authors"].astype(str).fillna("")
    df["year"] = (
        df["year"].astype(str).str.extract(r"(\d{4})")[0]
        .fillna("")
    )
    df["doi"] = df["doi"].apply(normalize_doi)
    df["pmid"] = df["pmid"].astype(str).str.extract(r"(\d+)")[0].fillna("")
    df["authors"] = df["authors"].apply(normalize_authors)

    # Reorder & subset
    return df[CANON_COLS]

# --------- Dedup engine ---------

def fingerprint_title(title: str) -> str:
    return normalize_text(title)

def completeness_score(row: pd.Series) -> int:
    score = 0
    score += 5 if row.get("doi") else 0
    score += 4 if row.get("pmid") else 0
    score += 2 if row.get("abstract") else 0
    score += 1 if row.get("url") else 0
    for k in ("journal", "volume", "issue", "pages"):
        score += 1 if row.get(k) else 0
    score += min(5, len(str(row.get("abstract", ""))) // 300)
    score += min(3, len(str(row.get("title", ""))) // 100)
    return score

def union_find(parents: Dict[int, int]):
    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parents[ry] = rx
    return find, union

def build_clusters(df: pd.DataFrame, algo: str = "balanced"):
    """Return (deduped_df, removed_df, audit_df)."""
    params = ALGO_PRESETS.get(algo, ALGO_PRESETS["balanced"])

    work = df.copy().reset_index(drop=True)
    work["_idx"] = work.index

    # Precompute normalized fields for blocking / similarity
    work["_t_norm"] = work["title"].apply(fingerprint_title)
    work["_a_first"] = work["authors"].apply(lambda s: s.split(";")[0].strip().lower() if s else "")
    work["_year"] = work["year"].astype(str).str.extract(r"(\d{4})")[0].fillna("")

    parents = {i: i for i in range(len(work))}
    find, union = union_find(parents)

    # 1) Exact ID matches
    for col in ("doi", "pmid"):
        key = work[col].astype(str).str.strip().str.lower()
        groups = key[key != ""].groupby(key).groups
        for _, idxs in groups.items():
            idxs = list(idxs)
            if len(idxs) > 1:
                for a in idxs[1:]:
                    union(idxs[0], a)

    # 2) Exact Title+Year+FirstAuthor matches
    key2 = (work["_t_norm"].str[:60] + "|" + work["_year"] + "|" + work["_a_first"])
    groups2 = key2[key2.str.strip() != "||"].groupby(key2).groups
    for _, idxs in groups2.items():
        idxs = list(idxs)
        if len(idxs) > 1:
            for a in idxs[1:]:
                union(idxs[0], a)

    # 3) Fuzzy title within blocks (prefix + year window)
    prefix = params["block_title_prefix"]
    year_w = params["year_window"]

    work["_block"] = work["_t_norm"].str[:prefix]
    blocks = work.groupby("_block").groups

    def year_ok(y1, y2):
        if not y1 or not y2:
            return True
        try:
            return abs(int(y1) - int(y2)) <= year_w
        except Exception:
            return True

    candidates = []
    for _, idxs in blocks.items():
        idxs = list(idxs)
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                if not year_ok(work.loc[a, "_year"], work.loc[b, "_year"]):
                    continue
                s = fuzz.token_set_ratio(work.loc[a, "_t_norm"], work.loc[b, "_t_norm"])
                if s >= params["possible"]:
                    candidates.append((a, b, s))

    for a, b, s in candidates:
        union(a, b)

    # Build clusters
    work["cluster_root"] = work["_idx"].apply(find)
    cluster_map = {root: i for i, root in enumerate(sorted(set(work["cluster_root"]))) }
    work["cluster_id"] = work["cluster_root"].map(cluster_map)

    # Likelihood bucket by max pair score in cluster
    bucket = []
    for cid, sub in work.groupby("cluster_id"):
        if len(sub) == 1:
            bucket.append((cid, "unique", 100))
            continue
        scores = []
        rows = sub.index.tolist()
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                s = fuzz.token_set_ratio(work.loc[a, "_t_norm"], work.loc[b, "_t_norm"])
                scores.append(s)
        max_s = max(scores) if scores else 0
        if max_s >= params["extreme"]:
            lvl = "extremely_likely"
        elif max_s >= params["high"]:
            lvl = "highly_likely"
        elif max_s >= params["possible"]:
            lvl = "possible"
        else:
            lvl = "review"
        bucket.append((cid, lvl, max_s))
    bucket_df = pd.DataFrame(bucket, columns=["cluster_id", "likelihood", "max_title_score"]).set_index("cluster_id")

    # Choose keeper in each non‑singleton cluster
    keep_rows = []
    remove_rows = []
    for cid, sub in work.groupby("cluster_id"):
        if len(sub) == 1:
            keep_rows.append(int(sub.index[0]))
            continue
        sub = sub.copy()
        sub["_comp"] = sub.apply(completeness_score, axis=1)
        sub = sub.sort_values(by=["doi", "pmid", "_comp", "title"], ascending=[False, False, False, False])
        keep = int(sub.index[0])
        keep_rows.append(keep)
        for r in sub.index[1:]:
            remove_rows.append(int(r))

    work["decision"] = np.where(work.index.isin(keep_rows), "keep", "remove")

    audit = work.merge(bucket_df, left_on="cluster_id", right_index=True, how="left")
    out_cols = CANON_COLS + ["cluster_id", "decision", "likelihood", "max_title_score", "source_file"]
    audit = audit.reset_index(drop=True)

    deduped = audit[audit["decision"] == "keep"][CANON_COLS + ["cluster_id"]].reset_index(drop=True)
    removed = audit[audit["decision"] == "remove"][CANON_COLS + ["cluster_id"]].reset_index(drop=True)

    return deduped, removed, audit

# --------- Exporters ---------

RIS_OUT_MAP = {
    "title": "TI",
    "authors": "AU",
    "year": "PY",
    "journal": "JO",
    "volume": "VL",
    "issue": "IS",
    "pages": "SP",
    "doi": "DO",
    "url": "UR",
    "abstract": "AB",
}

def dataframe_to_ris(df: pd.DataFrame, ty: str = "JOUR") -> str:
    lines = []
    for _, r in df.iterrows():
        lines.append(f"TY  - {ty}")
        for col, tag in RIS_OUT_MAP.items():
            val = str(r.get(col) or "").strip()
            if not val:
                continue
            if col == "authors":
                for a in [p.strip() for p in val.split(";") if p.strip()]:
                    lines.append(f"AU  - {a}")
            else:
                lines.append(f"{tag}  - {val}")
        lines.append("ER  -")
        lines.append("")
    return "\n".join(lines)

# --------- UI ---------

st.set_page_config(page_title="Deduplicator (SRA‑like)", layout="wide")
st.title("🧹 Deduplicator (SRA‑like)")
st.caption(
    f"Local, privacy‑friendly duplicate removal for search results. v{APP_VERSION}"
)

with st.sidebar:
    st.header("Algorithm & Settings")
    algo = st.selectbox("Algorithm profile", list(ALGO_PRESETS.keys()), index=1,
                       help="Focused = catch more dups; Relaxed = stricter; Balanced = middle ground.")
    params = ALGO_PRESETS[algo]
    st.markdown(
        f"**Thresholds**  \\\n        Extreme ≥ **{params['extreme']}**, High ≥ **{params['high']}**, Possible ≥ **{params['possible']}** (token_set_ratio)"
    )
    st.markdown(
        f"**Blocking**  \\\n        Title prefix: **{params['block_title_prefix']}** chars; Year window: **±{params['year_window']}**"
    )
    st.divider()
    st.write("**File types**: RIS (.ris), NBIB (.nbib/.txt), EndNote XML (.xml), CSV (.csv)")

uploaded = st.file_uploader("Upload one or more reference files", accept_multiple_files=True,
                            type=["ris", "nbib", "txt", "xml", "csv"])

if "state" not in st.session_state:
    st.session_state.state = {
        "raw_frames": [],
        "combined": pd.DataFrame(columns=CANON_COLS),
        "deduped": None,
        "removed": None,
        "audit": None,
    }

state = st.session_state.state

# Parse
if uploaded:
    raws = []
    for up in uploaded:
        name = up.name
        suf = name.split(".")[-1].lower()
        b = up.read()
        if suf == "ris":
            df = parse_ris_bytes(b, name)
        elif suf in ("nbib", "txt"):
            df = parse_nbib_bytes(b, name)
        elif suf == "xml":
            df = parse_endnote_xml_bytes(b, name)
        elif suf == "csv":
            df = to_canonical(pd.read_csv(io.BytesIO(b)),)
            df["source_file"] = name
        else:
            df = pd.DataFrame(columns=CANON_COLS)
        if len(df):
            raws.append(df)

    if raws:
        combined = pd.concat(raws, ignore_index=True)
        # assign stable record_id
        combined["record_id"] = [f"R{100000+i}" for i in range(len(combined))]
        state["raw_frames"] = raws
        state["combined"] = combined

# Show overview
if len(state["combined"]):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total records", f"{len(state['combined']):,}")
    with c2:
        n_doi = (state["combined"]["doi"] != "").sum()
        st.metric("With DOI", f"{n_doi:,}")
    with c3:
        n_pmid = (state["combined"]["pmid"] != "").sum()
        st.metric("With PMID", f"{n_pmid:,}")
    with c4:
        st.metric("Unique (pre‑dedup est.)", f"~{state['combined']['title'].nunique():,}")

    with st.expander("Preview sample (first 30)"):
        st.dataframe(state["combined"][CANON_COLS].head(30), use_container_width=True, hide_index=True)

    run = st.button("🚀 Run deduplication", type="primary")
    if run:
        with st.spinner("Deduplicating…"):
            deduped, removed, audit = build_clusters(state["combined"], algo=algo)
            state["deduped"], state["removed"], state["audit"] = deduped, removed, audit
        st.success("Done!")

# Results
if state.get("audit") is not None:
    deduped = state["deduped"]
    removed = state["removed"]
    audit = state["audit"]

    total = len(audit)
    kept = len(deduped)
    rem = len(removed)

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Kept (deduplicated)", f"{kept:,}")
    with c2:
        st.metric("Removed as duplicates", f"{rem:,}")
    with c3:
        st.metric("Reduction", f"{(rem/total*100):.1f}%")

    # Likelihood buckets breakdown
    bucket_counts = audit["likelihood"].value_counts().reindex(
        ["extremely_likely", "highly_likely", "possible", "review", "unique"], fill_value=0
    )
    st.write("**Buckets** (by max title similarity within cluster):")
    st.table(bucket_counts.rename("records").to_frame())

    with st.expander("View deduplicated records (first 100)"):
        st.dataframe(deduped.head(100), use_container_width=True, hide_index=True)

    with st.expander("View removed duplicates (first 100)"):
        st.dataframe(removed.head(100), use_container_width=True, hide_index=True)

    with st.expander("Audit table (sample 200)"):
        show_cols = CANON_COLS + ["cluster_id", "decision", "likelihood", "max_title_score", "source_file"]
        st.dataframe(audit[show_cols].head(200), use_container_width=True, hide_index=True)

    # Downloads
    st.subheader("Downloads")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "⬇️ Deduplicated CSV",
            data=deduped.to_csv(index=False).encode("utf-8"),
            file_name="deduplicated.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "⬇️ Removed (duplicates) CSV",
            data=removed.to_csv(index=False).encode("utf-8"),
            file_name="duplicates_removed.csv",
            mime="text/csv",
        )
    with c3:
        st.download_button(
            "⬇️ Audit CSV",
            data=audit.to_csv(index=False).encode("utf-8"),
            file_name="dedup_audit.csv",
            mime="text/csv",
        )
    with c4:
        ris_txt = dataframe_to_ris(deduped)
        st.download_button("⬇️ Deduplicated RIS", data=ris_txt.encode("utf-8"), file_name="deduplicated.ris", mime="application/x-research-info-systems")

    st.info(
        "Tip: for large libraries (> ~50k records), consider splitting by first letter of title or year to keep memory low."
    )

st.markdown("")
st.caption(
    "Inspired by the SRA/TERA Deduplicator algorithm design and buckets (Forbes et al., 2024)."
)
