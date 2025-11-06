---
title: "DeDupli: A Streamlit-based, Privacy-Friendly Deduplication Tool for Systematic Reviews"
tags:
  - Python
  - Streamlit
  - Deduplication
  - Systematic Review
  - Research Software
authors:
  - name: Satchit Sagar
    affiliation: 1
  - name: Dikhra Khan
    affiliation: 2
  - name: Rajashekhar C H T
    affiliation: 3
  - name: Sambit Sagar
    affiliation: 2
  - name: Pooja Tiwari
    affiliation: 4
  - name: Nivedita Kundu
    affiliation: 2
  - name: Arup Roy
    affiliation: 2
  - name: Lalitha Goriparthi
    affiliation: 2
  - name: Jagatti Krishna
    affiliation: 2
  - name: Aparna Mahalik
    affiliation: 2
  - name: Ayan Dhiman
    affiliation: 2
  - name: Avnish Singh
    affiliation: 5
  - name: Amlan Rout
    affiliation: 6
  - name: Hemant Khairwa
    affiliation: 6
affiliations:
  - index: 1
    name: Department of English Literature, English and Foreign Languages University (EFLU), Hyderabad, India
  - index: 2
    name: Department of Nuclear Medicine, All India Institute of Medical Sciences (AIIMS), New Delhi, India
  - index: 3
    name: Department of Radiation Oncology, AIIMS–CAPFIMS, New Delhi, India
  - index: 4
    name: Department of Endocrinology, AIIMS–CAPFIMS, New Delhi, India
  - index: 5
    name: Department of Orthopedics, AIIMS, Bhubaneswar, India
  - index: 6
    name: Department of Anaesthesia, AIIMS, Bhubaneswar, India
date: 6 November 2025
bibliography: paper.bib
---

# Summary

**DeDupli** is an open-source, Streamlit-based tool designed to accelerate and simplify **duplicate detection in systematic reviews**. Duplicate records frequently arise from overlapping database searches and inconsistent citation formats, leading to inflated screening workloads and potential bias. Existing tools such as *EndNote* and *Rayyan* offer partial automation but involve trade-offs between sensitivity, specificity, and user oversight.

DeDupli combines **speed**, **accuracy**, and **privacy-friendly design** by offering both **automatic** and **manual** modes. It can process hundreds of records within seconds, identify duplicates using multi-field similarity (title, author, year, DOI), and allow reviewers to confirm matches interactively—all within a local environment, ensuring data confidentiality. The software targets systematic reviewers, librarians, and evidence-synthesis researchers seeking transparent, reproducible, and efficient deduplication.

# Statement of need

Deduplication is an essential early step in evidence synthesis workflows. Manual identification of duplicate records is time-consuming and error-prone. While proprietary tools such as *EndNote* and *Rayyan* support basic duplicate removal, evaluations show that they often **miss true duplicates (false negatives)** or **remove unique records (false positives)**, compromising reproducibility.

DeDupli was developed to address three methodological gaps:

1. **Speed and scalability:** Ability to deduplicate thousands of citations within seconds, reducing screening burden.  
2. **Transparency and user control:** Dual-mode interface—automatic for rapid processing and manual for reviewer verification.  
3. **Privacy-friendly local execution:** Unlike cloud-based platforms, DeDupli runs entirely offline, suitable for sensitive or unpublished datasets.

In benchmarking using three synthetic RIS datasets (total = 782 records; 428 true duplicates), DeDupli achieved **perfect sensitivity (1.000)**, identifying all duplicates, and processed data in under 5 seconds in automatic mode. Manual mode improved **precision (0.934)** and **F1-score (0.966)** by reducing false positives, while *EndNote* and *Rayyan* showed lower sensitivity (0.939) and took ~40 minutes per dataset. These findings highlight that DeDupli achieves a pragmatic balance between automation and accuracy.

# Software description

DeDupli is implemented in **Python** and deployed via **Streamlit**, ensuring easy web-app deployment without local setup complexity.  
Key modules include:

- **`dedupli_core.py`** – parses RIS files, performs field-wise similarity matching using Levenshtein distance and DOI normalization.  
- **`app.py`** – Streamlit frontend offering upload, deduplication, and export interfaces.  
- **`utils.py`** – helper functions for logging, timing, and result visualization.  
- **Output** – generates CSV/Excel exports of deduplicated and flagged records.

The tool supports modular integration, enabling researchers to embed DeDupli into existing review pipelines (e.g., PRISMA workflows or automation scripts).  
Documentation and demo available at: [https://github.com/sambitsagar/DeDupli](https://github.com/sambitsagar/DeDupli)

# Validation and use

Performance testing used synthetic datasets modeled on real systematic review searches. Three independent reviewers compared DeDupli against *EndNote* and *Rayyan* in crossover trials.  
Metrics included sensitivity, specificity, precision, and F1-score from true/false positive/negative counts.

| Tool | Sensitivity | Specificity | Precision | F1 | Mean Time |
|------|--------------|-------------|------------|----|-----------|
| DeDupli-Auto | 1.000 | 0.956 | 0.899 | 0.947 | 5 s |
| DeDupli-Manual | 1.000 | 0.973 | 0.934 | 0.966 | 320 s |
| EndNote | 0.939 | 1.000 | 1.000 | 0.969 | 2636 s |
| Rayyan | 0.939 | 1.000 | 1.000 | 0.969 | 2580 s |

These results confirm DeDupli’s superior recall and substantial time savings.

# Acknowledgements

We thank the reviewers who participated in the performance evaluation.  
No external funding was received.  
**Corresponding author:**  
Dr Sambit Sagar (*drsambitofficial@gmail.com*, +91 7008552759)  
Assistant Professor, Department of Nuclear Medicine, AIIMS Delhi

# References

[^1]: Rathbone J., Carter M., Hoffmann T., & Glasziou P. (2015). *Better duplicate detection for systematic reviewers: evaluation of Systematic Review Assistant–Deduplication Module.* Systematic Reviews, 4(6). https://doi.org/10.1186/2046-4053-4-6  

[^2]: Borissov N., Haas Q., Minder B., Kopp-Heim D., von Gernler M., et al. (2022). *Reducing systematic review burden using Deduklick: a novel, automated, reliable, and explainable deduplication algorithm.* Systematic Reviews, 11(172). https://doi.org/10.1186/s13643-022-02045-9  

[^3]: McKeown S., & Mir Z. M. (2021). *Comparison of methods for identifying duplicate records in systematic reviews.* Systematic Reviews, 10(150). https://doi.org/10.1186/s13643-021-01583-y  

[^4]: Forbes C., Greenwood H., Carter M., & Clark J. (2024). *Automation of duplicate record detection for systematic reviews: Deduplicator.* Systematic Reviews, 13(206). https://doi.org/10.1186/s13643-024-02619-9  
