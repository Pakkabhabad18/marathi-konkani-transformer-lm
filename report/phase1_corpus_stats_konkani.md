# Phase 1 — Corpus Statistics: konkani

- Documents: **323,112**
- Words: **266,211,363**
- Manual: **64,435,242** (24.2%)
- Downloaded (real): **116,071,660** (43.6%)
- Synthetic (MT / LLM-generated): **85,704,461** (32.2%)
- Real text (manual + downloaded): **180,506,902** (67.8%)

`synthetic` is reported as its own bucket and is never counted toward the 20% manual floor.

## Sources

| Source | Documents | Words | Words/doc | Type |
|---|---:|---:|---:|---|
| archive_org_konkani_books | 53,843 | 62,997,759 | 1,170 | manual |
| hf_konkani_raw_machine_translated | 60,843 | 59,295,762 | 975 | synthetic |
| hf_konkani_books_corpus_v1 | 41,764 | 49,682,672 | 1,190 | downloaded |
| hf_konkani_books_corpus_v2 | 40,597 | 47,016,182 | 1,158 | downloaded |
| hf_bulk_anag007_instruct | 19,154 | 7,016,222 | 366 | synthetic |
| hf_bulk_anag007_alpaca | 13,150 | 5,038,553 | 383 | synthetic |
| ai4bharat_indiccorp_v2_gom | 12,696 | 4,319,751 | 340 | downloaded |
| hf_bulk_saillab_cleaned | 11,235 | 4,311,915 | 384 | synthetic |
| hf_madlad400_gom_noisy | 7,222 | 4,254,586 | 589 | downloaded |
| hf_bulk_telugu_labs_alpaca | 10,689 | 4,179,067 | 391 | synthetic |
| bpcc_gom_deva | 10,807 | 3,319,633 | 307 | downloaded |
| hf_sangraha_verified_gom | 9,827 | 3,266,816 | 332 | downloaded |
| hf_madlad400_gom_clean | 4,188 | 2,787,400 | 666 | downloaded |
| hf_bulk_devarshee_v2 | 7,716 | 2,470,036 | 320 | synthetic |
| konkani_wikipedia_selfcollected | 2,459 | 1,395,235 | 567 | manual |
| hf_glotcc_v1_gom_deva | 2,020 | 1,325,434 | 656 | downloaded |
| hf_bulk_anag007_wiki | 1,221 | 1,013,811 | 830 | synthetic |
| hf_bulk_gpteacher | 2,563 | 867,359 | 338 | synthetic |
| hf_konkani_instruct_100k_synthetic | 5,439 | 631,270 | 116 | synthetic |
| hf_bulk_saillab_taco | 1,344 | 409,166 | 304 | synthetic |
| hf_bulk_devarshee_v1 | 767 | 296,198 | 386 | synthetic |
| mt_marathi_to_konkani_indictrans2 | 407 | 128,519 | 316 | synthetic |
| hf_cfilt_roundtripocr_konkani | 2,966 | 87,497 | 30 | downloaded |
| hf_bulk_predictionguard | 149 | 46,583 | 313 | synthetic |
| news_goanews | 31 | 41,728 | 1,346 | manual |
| hf_konkani_raw_scrape | 13 | 11,689 | 899 | downloaded |
| news_vishwakonkani | 2 | 520 | 260 | manual |

## Splits

| Split | Documents | Words |
|---|---:|---:|
| train | 261,186 | 200,293,343 |
| val | 2,655 | 2,042,816 |
| test | 2,655 | 2,031,644 |
