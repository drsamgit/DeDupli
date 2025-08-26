# Deduplicator 

A local, privacy‑friendly duplicate remover for reference libraries inspired by the SR‑Accelerator / TERA Deduplicator workflow.

## Features
- Import **RIS**, **NBIB** (PubMed), **EndNote XML**, or **CSV**
- Normalization (“mutators”), exact ID matches (DOI/PMID), blocked **fuzzy** title matching
- Three algorithm profiles: **focused**, **balanced**, **relaxed**
- Likelihood buckets: **extremely likely / highly likely / possible / review / unique**
- Auto‑keeper heuristic + **full audit table**
- Export **CSV** (kept/removed/audit) and **RIS**

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL Streamlit prints (e.g., `http://localhost:8501`), upload one or more files, click **Run deduplication**, and download results.

## File formats
- `.ris` (generic RIS)  
- `.nbib` / `.txt` (PubMed)  
- `.xml` (EndNote XML)  
- `.csv` (mapped to canonical columns automatically when possible)

## Notes
- The fuzzy matcher uses RapidFuzz’s `token_set_ratio` on normalized titles, within **title‑prefix blocks** and an optional **±year** window.
- The default **balanced** profile is a good start. Switch to **focused** to catch more duplicates or **relaxed** to be conservative.

## License
MIT
