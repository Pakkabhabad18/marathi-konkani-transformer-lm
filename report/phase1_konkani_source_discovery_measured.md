# Phase 1 - Konkani source discovery: measured results

Generated 2026-08-16T12:15:19.003812+00:00 by `konkani/scripts/discover_sources.py`.
This file is produced by the script. Do not hand-edit it; the narrative
inventory lives in `report/phase1_konkani_source_discovery.md`.

## Enumeration

| query | items returned |
|---|---:|
| `language:kok AND mediatype:texts` | 5,093 |
| `language:Konkani AND mediatype:texts` | 18 |
| `language:Konknni AND mediatype:texts` | 0 |
| `language:Concani AND mediatype:texts` | 0 |
| `language:"Konkani (Devanagari)" AND mediatype:texts` | 0 |
| `language:gom AND mediatype:texts` | 3 |
| `subject:Konkani AND mediatype:texts` | 848 |

**Unique candidates after deduplication and exclusions: 5,178** (4 excluded by prefix rule)

### Largest parent collections

| collection | items |
|---|---:|
| `JaiGyan` | 4,825 |
| `ServantsOfKnowledge` | 4,626 |
| `Vishwakonkani` | 4,610 |
| `fav-lucke133` | 853 |
| `newspapers_miscellaneous` | 188 |
| `newspapers` | 188 |
| `fav-urmiksha_naik` | 152 |
| `fav-smita_rao` | 132 |
| `fav-sanjana_shetkar749` | 111 |
| `fav-meghashri_gaonkar` | 98 |
| `fav-vailanki_pednekar` | 83 |
| `fav-laichar` | 79 |
| `opensource` | 61 |
| `folkscanomy` | 58 |
| `newsstand` | 55 |
| `folkscanomy_religion` | 53 |
| `fav-sanjana_shetkar` | 52 |
| `christian-tracts-literature-society` | 52 |
| `fav-spandan_kochi` | 49 |
| `fav-user_27097` | 44 |

## Probe

- items fetched and measured: **120**
- accepted (Devanagari body, langid not Marathi): **58**
- measured Devanagari accept share: **48.3%**
- median words per accepted item: **16,776**
- mean words per accepted item: **28,852**

### Verdict breakdown

| verdict | items |
|---|---:|
| rejected_not_devanagari | 59 |
| accepted | 58 |
| rejected_langid_marathi | 3 |

## Extrapolated yield

- estimated usable items: **2,502** of 5,178 candidates
- estimated words (median-based, headline): **41,985,295**
- estimated words (mean-based, optimistic): **72,210,144**

The median-based figure is the one to quote. Item sizes are strongly
right-skewed: a four-volume encyclopedia is ~700k words while a poetry
booklet is ~5k, so the mean overstates a typical item.

Sample size is 120 items. The Devanagari
share is the dominant uncertainty in this estimate; re-run with a larger
`--sample` to tighten it before committing to the full collection.
