<!--
Reference copy (added 2026-11-11). This is the `usace-landuse` method text, v1, as given by Jack.
The text below is quoted unchanged; only the Markdown headings and tables are new.
It describes the `usace_veg` pipeline of `usace-landuse`, not the code of this repository (D3).
The rules of this repository are in docs/DATA_STANDARD.md; where the two differ, DATA_STANDARD.md wins here.
-->

# Method v1: class scheme, condition score, change categories and measurements

Superseded by METHOD_v2.md (2026-10-29), which folds the pilot findings and the implementation log below into a revised specification. Keep this file for the record; where the two differ, v2 wins.

Draft for review, 2026-10-29. Inputs are the datasets in data_sources/DATA_SOURCES.md and the USACE form document/VegetationClass (1).xls. Codes and rules build on Modified_USACE_classification_rules.xlsx; where this file differs, this file wins.

Items marked [Decide] were settled on 2026-10-29; the decisions are in §6.

## 0. Design in one paragraph

Each year 1985–2025, every 30 m pixel inside a USACE project boundary gets one class from the USACE-M scheme (24 classes, 3 flags, a confidence tier). Classes are then grouped into stands (contiguous patches at the NVC class level), and each stand is scored on the measurable metrics of the USACE condition form (1–5 points), rolled up with the form's own cut-offs to Sustainable / Transitioning / Degraded, or Not assessed where the form does not apply (water, developed, agriculture). Acres by class × rating reproduce the USACE OMBIL table; an area-weighted score gives one project condition index per year. For any two years, a class × class cross-tab is summarized into change categories (succession, woody loss, inundation, exposure, development, agricultural conversion, recovery, wetland change, refinement, unconfirmed), with gain/loss/net, annual rates, intensity analysis, landscape-metric change and condition-rating transitions.

## 1. Class scheme: USACE-M (v2)

### 1.1 Why not the 23 USACE subclasses as they stand

| USACE practice | Problem for a reproducible method | USACE-M answer |
| --- | --- | --- |
| One "Non-Vegetated" class (39% of reported acres) | Hides reservoir pool change and development | Split: M01 water, M02 developed, M03 extractive/ice |
| No land-use classes | Pasture and cropland get forced into "graminoid" | M12 pasture/hay, M13 cropland |
| No class for 10–25% tree cover | Savanna/thinned stands fall arbitrarily into woodland or herbaceous | M18 sparse tree canopy |
| Forb vs. graminoid; three dwarf-shrub leaf types | Not separable in any national dataset | Merged (M10, M17); crosswalk keeps the USACE codes |
| Algae, bryophyte, lichen; floating/submerged beds | No national data | M07 not mapped; M08 aquatic bed candidate from NWI + NDVI |
| Wetland identity lost in woody classes; introduced species invisible | Both matter for stewardship | Flags F1 (woody wetland), F2 (introduced) |
| No record of uncertainty | Numbers cannot be qualified | Confidence tier 1–3 per pixel |

### 1.2 Classes

NVC class = the 1997 NVC level USACE uses for stands (Section 2 scoring unit). U = USACE subclass code in NLCD_LANDFIRE_USACE_crosswalk.xlsx.

| Code | USACE-M class | NVC class | USACE code | Lifeform group |
| --- | --- | --- | --- | --- |
| M01 | Open water | — | U01 (part) | W |
| M02 | Developed | — | U01 (part) | D |
| M03 | Other non-vegetated (extractive, ice/snow) | — | U01 (part) | X |
| M04 | Unconsolidated material sparse vegetation | Sparse | U02 | SP |
| M05 | Consolidated rock sparse vegetation | Sparse | U03 | SP |
| M06 | Boulder/gravel/cobble/talus sparse vegetation | Sparse | U04 | SP |
| M04-06u | Sparse vegetation, substrate unresolved | Sparse | U02–U04 | SP |
| M07 | Lichen and bryophyte vegetation | Nonvascular | U05, U06 | — (not mapped) |
| M08 | Aquatic bed (candidate) | Hydromorphic | U07, U10 (part) | AB |
| M09 | Emergent hydromorphic vegetation | Herbaceous | U10 (part) | EM |
| M10 | Perennial herbaceous (graminoid + forb) | Herbaceous | U08, U09 | H |
| M11 | Annual graminoid/forb vegetation | Herbaceous | U11 | H |
| M12 | Pasture/hay | — | — (new) | AG |
| M13 | Cultivated cropland (incl. orchard, vineyard) | — | — (new) | AG |
| M14 | Mixed evergreen-deciduous shrubland | Shrubland | U12 | S |
| M15 | Evergreen shrubland | Shrubland | U13 | S |
| M16 | Deciduous shrubland | Shrubland | U14 | S |
| M17 | Dwarf shrubland (< 0.5 m) | Dwarf-shrubland | U15–U17 | S |
| M18 | Sparse tree canopy (10–25% cover) | Open tree canopy (nearest) | — (new) | T |
| M19 | Mixed open tree canopy (25–60%) | Open tree canopy | U18 | T |
| M20 | Evergreen open tree canopy | Open tree canopy | U19 | T |
| M21 | Deciduous open tree canopy | Open tree canopy | U20 | T |
| M22 | Mixed closed tree canopy (> 60%) | Closed tree canopy | U21 | T |
| M23 | Evergreen closed tree canopy | Closed tree canopy | U22 | T |
| M24 | Deciduous closed tree canopy | Closed tree canopy | U23 | T |

Flags (recorded with the class, never change it): F1 woody wetland (NLCD 90 or NWI FO/SS or LANDFIRE riparian tree/shrub), F2 introduced vegetation (LANDFIRE EVT group 701–709), F3 drawdown zone (reporting stratum: not permanent water, and JRC occurrence ≥ 5% since 1984 or water in ≥ 5% of growing-season months 2022–2024; permanent water = water in ≥ 90% of May–October JRC monthly observations 2022–2024, falling back to occurrence ≥ 90% where the monthly history is missing. Occurrence alone fails on northern lakes: ice and cloud pull Raystown's lake interior to 75–95% occurrence, which the original 5–95% rule read as drawdown).

Confidence tier: 1 = all inputs from the map year and in agreement; 2 = one input borrowed from another year, or a disagreement settled by a rule; 3 = value inferred (neighbourhood leaf type, substrate fallback) or an input borrowed by more than 3 years.

### 1.3 Pixel rules (one pass per year, first match wins)

W = water this year (JRC yearly class 2–3 to 2021; Landsat MNDWI > 0 water fraction ≥ 50% from 2022). TCC = tree canopy cover % (USFS Science TCC 1985–2023; LANDFIRE EVC tree cover for 2024–2025, tier 2). LF = LANDFIRE layer assigned to the year (§1.4).

| # | Condition | Class | Tier |
| --- | --- | --- | --- |
| 1 | W and inside NWI aquatic bed (AB) and growing-season NDVI > 0.2 | M08 (pass 1: counted as candidate in T3 only, no NDVI yet) | 3 |
| 2 | W | M01 | 1 if NLCD 11, else 2 |
| 3 | Not W but NLCD 11 (exposed lakebed) | NDVI ≥ 0.3 → M09; else M04 (or NWI shore code → M05/M06) | 2 |
| 4 | NLCD 12 | M03 | 1 |
| 5 | NLCD 21–24 | M02 | 1 if LF developed, else 2 |
| 6 | NLCD 81 / 82 | M12 / M13 | 1 if CDL agrees (2008+), else 2 (pass 1: LANDFIRE agriculture lifeform stands in for CDL) |
| 6b | NLCD 31/52/71/95 with TCC > 60 (TCC guard, added 2026-10-29 after the NAIP check: these cells are closed canopy in the imagery) | M22/M23/M24, leaf type from LF tree subclass, else 5×5 NLCD tree-leaf neighbourhood, else mixed; flag TCC conflict; F1 kept for NLCD 95 | 2 if leaf from LF, else 3 |
| 7 | NLCD 31, substrate in order: 7a LF EVT 7295 → M03; 7b NWI RS1 → M05, RS2/US1 → M06, US2–4 → M04; 7c LF sparse EVT with one substrate → its class; 7d gSSURGO Rock outcrop → M05, Rubble land → M06, Riverwash/Beach/Dune/Badland → M04; 7e slope > 45° → M05, 25–40° below M05 → M06; 7f none → M04-06u | M03–M06 | 7a–b: 1; 7c–e: 2; 7f: 3 (pass 1 runs 7a–7c and 7f; 7d gSSURGO and 7e slope are the D6 second pass) |
| 8 | NLCD 95 | M09 | 1 if NWI EM or LF riparian herb, else 2 |
| 9 | NLCD 71 | NWI EM or LF riparian herb or F3 → M09; LF annual subclass → M11; else M10 | 1 if LF herbaceous, else 2 |
| 10 | NLCD 52 | LF dwarf subclass or EVH < 0.5 m → M17; else leaf type from LF shrub subclass → M14/M15/M16 | 1 if LF shrub; 3 if leaf type from 5×5 neighbourhood |
| 11 | NLCD 41/42/43 | TCC > 60 → M24/M23/M22; 25–60 → M21/M20/M19; 10–25 → M18; < 10 → rule 10 and flag conflict | 1 if LF subclass agrees, else 2 |
| 12 | NLCD 90 | TCC ≥ 25 → rule 11 with leaf type from LF (NLCD-leaf type if no LF); TCC < 25 → rule 10; set F1 | 2 |

Leaf type for trees comes from NLCD itself (41/42/43), so M18–M24 need only NLCD + TCC and are available 1985–2023 at tier 1–2 without LANDFIRE. Shrub leaf type, annual vs. perennial herbs, dwarf shrubs and the F2 flag need LANDFIRE.

### 1.4 Year coverage and borrowing

| Input | Native years | Rule for other years | Tier effect |
| --- | --- | --- | --- |
| Annual NLCD | 1985–2025 (1991 to download; all years to be Collection 1.2) | — | — |
| TCC (NLCD TCC v2025.6) | 1985–2025 | — | — |
| LANDFIRE EVT/EVH/EVC | 2016, 2022, 2023, 2024 | ≤ 2018 → LF2016; 2019–2022 → LF2022; 2023 → LF2023; 2024–2025 → LF2024 | 2 if ≤ 3 years away, else 3 |
| Water (W) | JRC v1.5 occurrence 1984–2024 (drawdown zone); JRC monthly water 2022–2024 | Yearly W for every year = NLCD 11 in pilot pass 1; pass 2 replaces it with a Landsat MNDWI growing-season composite calibrated on JRC monthly 2022–2024 | 1 (pass 2) / 2 (pass 1) |
| LCMS | 1985–2025 | — | — |
| NWI, gSSURGO, 3DEP, roads | static | used every year | — |
| BPS | static (LF2016/LF2020) | used every year | — |

[Decide] Pre-2016 years borrow LF2016 for shrub leaf type and annual/perennial herbs (tier 3 for 1985–2012). The alternative is to report those years at NVC-class level only (shrubland, herbaceous) with no leaf-type split. Recommendation: borrow and flag, so every table has the same columns; the tier-3 share tells the reader how much is borrowed.

### 1.5 Stability filter (applied before any change measure)

A class change at pixel p in year t is kept if (a) the class in t+1 equals the class in t, or (b) LCMS records change at p in t (Change band ≠ Stable, or any loss/gain probability ≥ 50). Otherwise p reverts to its t−1 class and the change is logged as unconfirmed. Changes to or from M01 are exempt (pool level is a real yearly signal). The last year (2025) is provisional.

## 2. Condition score (per year)

### 2.1 Principle

Reproduce the USACE form with the metrics a 30 m time series can measure; omit the field-only ones and say so. Scores use the form's own point values and cut-offs so a USACE reader can follow them.

### 2.2 Scoring unit

Stand = contiguous patch (8-neighbour) of one NVC class (Closed tree, Open tree, Shrubland, Dwarf-shrubland, Herbaceous, Sparse) inside the project boundary. A patch of closed deciduous forest touching closed evergreen forest is one stand, as on the form. Each stand gets metric values, a score and a rating; acres then roll up by USACE-M class × rating.

Condition is computed only for natural vegetation: M04–M06, M09–M11, M14–M24. M01–M03, M08, M12, M13 and M04-06u are Not assessed (USACE's own fourth category). [Decide] The form excludes lawn/pasture/cropland; agreed to keep M12/M13 unassessed.

### 2.3 Metrics

Window for landscape metrics: focal circle of 1,000 ha (radius 1,784 m) around each pixel; 100 ha (radius 564 m) for wetland stands (M09, F1); stand value = mean over its pixels. (The form's "2× patch area for patches > 500 ha" is applied when a stand exceeds 500 ha.)

Natural land for L1 = M01, M04–M11, M14–M24 without F2. Land Use Index coefficients (form Table 1, mapped to data):

| Cover in window | Coefficient | Source |
| --- | --- | --- |
| NLCD 22–24, M03 extractive, paved roads (TIGER S1100–S1400, buffered 12 m) | 0.0 | NLCD, LANDFIRE, TIGER |
| Unpaved roads/trails (TIGER S1500, S1740, 6 m) | 0.1 | TIGER |
| M13 cropland; NLCD 21 developed open space (lawns, golf) | 0.2 | NLCD, CDL |
| Recent clearing: tree in t−3..t−1, now S/H/SP with LCMS fast loss | 0.4 | NLCD, LCMS |
| M12 pasture/hay; CDL orchard/vineyard | 0.4 | NLCD, CDL |
| F2 introduced vegetation; drawdown zone F3 (not water) | 0.5 | LANDFIRE, JRC |
| All other natural cover and water | 1.0 | — |

| Metric | Measured as | Points (form) | Status |
| --- | --- | --- | --- |
| L1 Landscape connectivity | % natural land in window | ≥ 90: 5; 60–89: 4; 20–59: 3; < 20: 1 | Measured |
| L2 Surrounding land use | Σ coefficient × share, in window | 0.95–1.0: 5; 0.80–0.94: 4; 0.40–0.79: 3; < 0.40: 1 | Measured |
| L3 Landscape stressors | Count present in window of: developed, primary/secondary road, row crop, pasture, mining, introduced plants (F2); "significant" if > 10% of window | 0: 5; 1–3 none significant: 4; 2–4 with 1–2 significant: 3; > 4 with ≥ 2 significant: 1 | Proxy (6 of 17 checklist items) |
| Z1 Patch size | Stand area, ha | > 1,000: 5; 100–999: 4; 10–99: 3; < 10: 1 | Measured |
| Z2 Patch size condition | Share of the stand's NVC class that BPS expects at that lifeform still present (BPS-expected area now other lifeform = loss) | change < 5%: 5; 5–20: 4; 20–50: 3; > 50: 1 | Proxy, optional (BPS predates the reservoir) |
| V1 Vegetation structure | By NVC class variant (form "Vegetation Structure" sheet): total vegetation cover (EVC in LANDFIRE years, Landsat fractional cover otherwise), woody cover (TCC for trees, EVC shrub cover for shrubs), structural diversity = CV of EVH/TCC in stand | Forest: cover > 80 & woody > 40 & diverse: 5; cover > 80 & woody > 10: 4; cover > 50 & woody > 10: 3; else 1. Woodland: woody > 25: 5; > 10: 4; cover > 50: 3; else 1. Shrubland (closed): shrub > 40: 5; 25–39: 3; < 25: 1. Grassland: cover > 80 & woody < 10: 5/4; cover > 50: 3; else 1 | Measured (pixel → stand mean) |
| V2 Invasive exotic plants | F2 share of stand area | 0: 5; 1–2%: 4; 3–5%: 3; 5–25%: 2; > 25%: 1 | Partial (dominance-level only) |
| V3 Composition | LANDFIRE Vegetation Departure (VDEP) mean in stand, if downloaded | VDEP < 34: 5; 34–66: 3; > 66: 1 | Optional, future |
| V5 Vegetation stressors | LCMS slow loss (last 5 yr), fast loss (last 5 yr), fire cause class, F2, woody encroachment in herbaceous stands | same count rule as L3 | Proxy |
| S2 On-site land use | Land Use Index inside the stand and its 100 m ring | same as L2 | Measured |
| L4, V4, S1, S3, H1–H4 | Buffer condition, % native, soil disturbance, hydrology | — | Not measurable at 30 m, or constant for reservoir lands (dams, managed levels) |

[Decide] Hydrology: every USACE project has a dam and managed pool, so H-metrics would rate all projects alike. Recommendation: report the drawdown-zone and shoreline strata and CWMS pool statistics as context, but leave Hydrology out of the v1 rollup.

### 2.4 Roll-up (form arithmetic)

- Attribute score = mean of its measured metrics: Landscape (L1, L2, L3), Size (Z1 [, Z2]), Vegetation (V1, V2 [, V3, V5]), Soil (S2).
- Stand score = mean of attribute scores (form weights are all 1; equal weights adopted 2026-10-29).
- Rating (state): 4.5–5.0 A Sustainable+; 3.5–4.4 B Sustainable; 2.0–3.4 C Transitioning; 1.0–1.9 D Degraded. A and B both report as Sustainable, as on the form's lookup sheet. In OMBIL-format outputs the C band keeps USACE's name "Transitioning"; in the method's own tables it is called Intermediate, so that "transitioning" is never used for both a state and a direction.
- Acres by USACE-M class × {Sustainable, Transitioning, Degraded, Not assessed}: the OMBIL-format table.
- Project condition index = area-weighted mean stand score over assessed acres; project rating by the same cut-offs. Also report the share of assessed acres in each rating and the share of project acres not assessed.
- Confidence: every stand carries the area-weighted mean tier of its pixels; tables report the tier-1 share beside each number.

### 2.5 Trajectory (decided 2026-10-29)

The state answers "how is this stand?"; the trajectory answers "is it getting better or worse?". Every assessed stand, every year, carries both. The trajectory has three values, Recovering / Stable / Declining, and is reported beside the state, never merged into it, so the OMBIL S/T/D totals are unchanged.

- Trajectory score. A second score is computed from the metrics that are observed every year, do not depend on LANDFIRE's stepwise versions, and have no look-back: L1 (natural land), L2 (land-use index), Z1 (patch size), the TCC-based part of V1 (canopy cover and its variation; for herbaceous stands the woody-encroachment share), and S2. It uses the same points and attribute arithmetic as §2.4 (Landscape, Size, Vegetation, Soil). V5 is deliberately excluded: its five-year disturbance memory would make an event "expire" five years later and read as recovery for every affected stand at once (seen as a 21 % Recovering spike at Grapevine in 2020, five years after the 2015 flood). Disturbance history stays in the state score and in the change categories with LCMS cause classes. The full score is kept for the state; the trajectory score exists only to measure direction.
- Window and test. For year t the trajectory score of the stand's pixels is traced back over the five-year window t−5 … t (area-weighted mean of the pixels that were in the same NVC class in all six years). The Theil–Sen slope of those six values is the trend. Recovering if slope ≥ +0.1 points per year, Declining if ≤ −0.1 per year, otherwise Stable. ±0.1 per year equals ±0.5 points, half a rating band, over the window. Years before 1990 have no full window and are reported as "not available".
- Precedence of class change. The trajectory is defined only for stands that persisted in their NVC class through the window. A stand that converted (for example tree → developed, marsh → water) takes the class-change direction of §3.2 / RESEARCH_METHOD §3.4 instead. A stand on ground that was Not assessed five years earlier (newly vegetated lakebed, old field) is Recovering by definition.
- Reading the pair. Sustainable + Declining is the early-warning cell ("Sustainable, at risk"); Intermediate + Recovering is a recovering stand; Intermediate + Declining is a stand heading for Degraded; Degraded + Recovering is partial recovery. These readings are derived in the report, not stored as extra categories.
- Roll-up. Acres by class × state × trajectory (T5), and for the project the share of assessed acres Recovering / Stable / Declining with the acre-weighted mean slope (T6).

## 3. Change between two years (t1 → t2)

### 3.1 Cross-tabs

Pixel-level class map at t1 and t2 (after the stability filter), area-weighted by the fraction of each cell inside the boundary.

- C1 Transition matrix at USACE-M level (24 × 24, acres) and at NVC-class level (11 × 11).
- C2 Per class: persistence, gross gain, gross loss, net change, swap (Pontius), annual rate r = (A_t2 / A_t1)^(1/(t2−t1)) − 1.

Repeated by stratum: whole project, permanent water, drawdown zone (F3), shoreline band (≤ 100 m from permanent water), upland. Permanent water is defined under the F3 flag in §1.2 (strata.tif).

### 3.2 Change categories

Lifeform groups from §1.2: W water, D developed, X extractive, SP sparse, AB aquatic bed, EM emergent, H herbaceous, AG agriculture, S shrub, T tree. "veg" = SP, EM, H, S, T.

| Category | From → To | Reading for management |
| --- | --- | --- |
| Persistence | same class | stable |
| Refinement: canopy closure | T open/sparse → T closed (M18–21 → M22–24) | growth, infill |
| Refinement: canopy opening | T closed → T open/sparse | thinning, mortality, partial harvest |
| Refinement: other | same lifeform, other subclass (leaf type, annual/perennial, dwarf) | usually map noise unless persistent |
| Succession / woody encroachment | H → S, H → T, S → T, SP → veg, EM → S/T | natural dynamics; a concern in managed grassland and wetland |
| Woody loss | T → S, T → H, S → H, veg → SP | harvest, fire, storm, clearing; check LCMS cause |
| Loss to development | veg → D | land-use conversion, usually permanent |
| Loss to extractive / barren | veg → X | disturbance |
| Loss to agriculture | veg → AG | land-use conversion |
| Agricultural recovery | AG → veg | old-field succession |
| Recovery from developed/barren | D/X → veg | restoration or map error; verify |
| Inundation | non-W → W | pool rise; read with CWMS elevation |
| Exposure | W → non-W (split: → SP bare, → EM/H vegetated) | drawdown; vegetated lakebed is a real habitat change |
| Wetland gain | H/SP → EM; W → AB | wetter conditions or colonization |
| Wetland loss | EM → H/SP; AB → W | drying, succession, or loss of beds |
| Unconfirmed | any change failed §1.5 | reported separately, excluded from totals above |

Outputs C3: acres and share by category, by stratum; the three largest class-to-class transitions inside each category.

### 3.3 Intensity analysis (Aldwaik & Pontius 2012)

Interval level (is change in t1→t2 fast or slow relative to other intervals), category level (which classes gain/lose faster than uniform), transition level (which from-classes a gaining class targets). Keeps the user's existing Step-2 product; chi-square is dropped as uninformative at pixel counts.

### 3.4 Condition change

- C5 Stand-rating cross-tab S/T/D/Not assessed at t1 × t2 (acres) with the RESEARCH_METHOD §3.5 labels; per-class mean score delta; project index delta with the tier-1 share of both years.
- Trajectory shares at t1 and t2 (Recovering / Stable / Declining acres by class), so a pair can be read as "more acres were declining in t2 than in t1".
- Metric-level deltas (L1, L2, Z1, V1, V2) on pixels that kept their class, so the narrative can say why a rating moved.
- Rating transitions are reported as "improved / unchanged / declined" (score change ≥ ±0.5), never relabelled as S/T/D of the change itself.

### 3.5 Landscape-metric change

Metrics computed per year (§4.2) and differenced: patch number and mean size, largest patch index, edge density, aggregation index, forest-interior share.

## 4. Measurements per year (tables per project)

### 4.1 Classification tables

| Table | Rows | Columns |
| --- | --- | --- |
| T1 class_area | class × stratum | year, acres, share, tier-1 share, tier-3 share, F1 acres, F2 acres |
| T2 usace_crosswalk | USACE subclass U01–U23 (+ "not mapped") | acres from M classes; NLCD-year ↔ FY mapping (FY = NLCD year, since NLCD is dated ~1 July) |
| T3 qa | one row | boundary acres, mapped acres, nodata acres, unresolved sparse acres, M08 candidate acres, unconfirmed-change acres, inputs borrowed (years), LANDFIRE/TCC source years |

### 4.2 Landscape metrics (pylandstats, 8-neighbour, NVC-class map)

Class level: number of patches, patch density, mean and median patch area, largest patch index, edge density, area-weighted mean shape index, aggregation index, cohesion. Landscape level: Shannon diversity, contagion, total edge. Forest interior (Riitters & Wickham 2012): share of tree pixels with forest area density ≥ 0.9 in 7×7 (4.4 ha) and 27×27 (65.6 ha) windows. Connectivity: tree patches within 100 / 500 / 1,000 m linked (the user's existing thresholds).

### 4.3 Condition tables

| Table | Rows | Columns |
| --- | --- | --- |
| T4 condition_stand | stand | class, NVC class, acres, L1, L2, L3, Z1, V1, V2, V5, S2 values and points, attribute scores, score, rating (state), trajectory score, 5-year slope, trajectory, mean tier |
| T5 condition_acres (OMBIL format) | class | Sustainable, Transitioning, Degraded, Not assessed acres; total; tier-1 share; and within each state the Recovering / Stable / Declining acres (T5_condition_trajectory) |
| T6 project_summary | one row | project index, rating, assessed acres, % S/T/D, % Recovering / Stable / Declining, mean 5-year slope, % natural land, % water, % drawdown zone, F2 acres, tier-1 share |

### 4.4 Series and standard comparisons

- Every year 1985–2025: T1, T5, T6 → time series plots and the all-projects stack.
- Change tables for any user-chosen pair; defaults run automatically: consecutive years (for the trend), 1985→1995→2005→2015→2025, and 2016→2023 as the high-confidence pair (NLCD, TCC, LANDFIRE and LCMS all native in both years).
- USACE comparison: T2 and T5 for 2023, 2024, 2025 against the FY23–25 report rows for the same ombilSiteId; differences by subclass with the USACE row's internal consistency flag (S+T+D = Total?).

## 5. Validation

- Classification accuracy: stratified random sample of 30 m cells per USACE-M class (≥ 50 per class present, ≥ 500 per site), labelled on the nearest NAIP flight (Raystown 2010–2022, Grapevine 2012–2022) by an interpreter who did not build the rules; Claude proposes labels, the reviewer confirms. Error matrix, class accuracies and error-adjusted areas with 95% CI (Olofsson et al. 2014).
- Threshold tuning on the same sample: TCC 25/60%, EVH 0.5 m, NDVI 0.2/0.3, MNDWI, slope.
- Comparison with USACE reports: §4.4; agreement is informative, not a truth test, given the report errors documented in DATA_SOURCES.md §1.
- Condition plausibility: reviewer reads the T4 stands with the lowest and highest scores against NAIP; the checkpoint records accept / adjust / request explanation.

## 6. Decisions (all recommendations adopted by the project lead, 2026-10-29)

| # | Question | Decision |
| --- | --- | --- |
| D1 | Years without LANDFIRE: borrow nearest and flag, or report NVC-class level only | Borrow and flag (tier 2/3) |
| D2 | Scoring unit | Stand = contiguous NVC-class patch; roll up to class and project |
| D3 | Hydrology attribute in the rollup | Exclude in v1; report pool context separately |
| D4 | Attribute weights | Equal (as the form); revisit after pilot |
| D5 | Z2 (patch size vs. BPS) and V3 (VDEP) | Optional; compute but do not include in the rollup until checked |
| D6 | Sparse substrate (M04–M06) in the pilot | Run 7a–7c now (NWI, LANDFIRE); add gSSURGO and slope in a second pass |
| D7 | High-confidence change pair | 2016 → 2023 |

## 7. Implementation log

2026-10-29, pass 1 run on Raystown (392) and Grapevine (77), 1985–2025. Code: code/usace_veg/ (build_stack → classify → tables → compare_usace). Water = NLCD 11; M08 counted as candidate only; CDL, gSSURGO and slope not yet used (see rule notes above). Tier 1 reaches 64–70 % of project area in 2013–2025; in 1985–2012 tree classes are tier 2 because LANDFIRE 2016 is more than three years away and cannot confirm them (§1.4), so tier 1 there is only water and NWI-confirmed classes (3–5 %). Stability filter reverts 0.02–0.2 % of pixels per year; the last year (2025) has no following year and reverts ~0.5 %, so it is provisional.

Strata and F3 redefined with the JRC growing-season water fraction (§1.2) after the occurrence-only rule put 7,300 ac of Raystown's 8,300 ac lake into the drawdown zone.

First USACE comparison (T7) for FY2025: Raystown non-vegetated 9,142 (USACE) vs 9,230 (mapped) ac and closed tree canopy 18,554 vs 19,669 ac agree within 1–6 %; USACE reports no open tree canopy (mapped 446 ac) and 1,330 ac perennial herbaceous (mapped 269 ac plus 751 ac pasture/cropland that the USACE list has no class for). Grapevine: USACE 6,403 ac closed vs 3,233 ac mapped, 1,073 vs 1,452 ac open, 300 vs 1,541 ac hydromorphic; the USACE closed/open split there does not follow the 60 % canopy threshold.

2026-10-29, change, condition and landscape modules written and run (change, condition, landscape). Decisions taken in code, to confirm: (a) the 11 change groups split tree into open (M18–M21) and closed (M22–M24) and put lichen (M07) with sparse; the full group × group category and direction table is C0_direction_map.csv, identical to RESEARCH_METHOD §3.4, with per-project overrides read from config.yaml (change.overrides). (b) "Unconfirmed" change = pixels whose raw class in t2 differs from the filtered class, reported as its own row and raster code. (c) Stands are built from cells at least half inside the boundary; single-pixel stands are kept (Z1 = 1 point, as the form prescribes) and are ~60 % of stands but < 3 % of assessed acres at Raystown. (d) V1 total vegetation cover = max(LANDFIRE EVC cover, TCC); shrub cover falls back to total woody cover where LANDFIRE does not call the pixel shrub; structural diversity = coefficient of variation of TCC (tree stands) or EVH (others) > 0.25; sparse stands get no V1. (e) L3/V5 "present" = ≥ 0.5 % of the window or stand, "significant" = > 10 %; LCMS codes 1–9 and 13 count as fast loss, 10–12 slow loss, 6–7 fire. (f) Roll-up uses equal attribute weights (Landscape, Size, Vegetation, Soil) with the available metrics only; Hydrology, Z2 and the field-only metrics are omitted. (g) Condition change (C5) uses the §3.5 rating matrix plus a score-change class (±0.5), and metric deltas are computed on pixels that kept the same class in both years so conversions do not confound them. (h) Same day, after discussion: the state (S/T/D) is kept as USACE defines it and a separate trajectory (Recovering / Stable / Declining, §2.5) is added from a five-year Theil–Sen slope of a LANDFIRE-free score; the two are reported side by side and never merged. (i) V5 removed from the trajectory score after the first run showed window-edge spikes (Grapevine 2020: 21 % Recovering as the 2015 flood left the five-year stressor window); the stack was rebuilt on Annual NLCD C1V2 for all years (nlcd.version in config; C1V1 outputs kept in stack_C1V1/, tables_C1V1/). Result: C1V2 and C1V1 are pixel-identical in both pilot windows for every year checked, so every table is unchanged and the version boundary was never the cause; the Raystown 2025 Recovering share (6 % → 0.4 %) was the V5 artefact. The remaining Grapevine 2020 Recovering share of 21 % is the newly-vegetated rule working as defined: 1,343 ac, mostly emergent wetland (M09), that was under water in the 2015 flood year and vegetated by 2020. It is a real signal, confined to the one year whose window starts in 2015; from 2021 those stands are judged by their slope (399 ac of M09 still Recovering). (j) USACE FY23–25 report Raystown 41 % Degraded (12,046 ac, mostly closed-canopy forest, rank 25 of 400 projects) where the method finds none, and Grapevine 100 % Sustainable with zero T/D, the default pattern shared by 163 of 400 projects. T8 (compare_usace) now reports the S/T/D comparison per group with flags for the default pattern and FY-to-FY repetition. Two sensitivity variants were run beside the base roll-up rather than changing it: Vegetation weight ×2 (_vw2) and a forest-health stressor set (_fh: LCMS insect/disease/drought over ten years, TCC decline ≥ 5 points over ten years), and both together (_vw2fh). The field-only metrics (composition, regeneration, understory invasives) that most likely drove Raystown's rating are outside a 30 m Level-1 assessment; the Raystown NRM staff should be asked what the D rating rests on. Sensitivity result (2025): neither variant moves toward the USACE figures. Vegetation ×2 raises Raystown's Sustainable share from 95.8 to 97.2 % (closed forest scores high on structure) and Grapevine's from 35.4 to 40.3 %; the forest-health stressors change the shares by < 1 % because LCMS records insect/disease/drought on 0.1 % of Raystown's tree acres and mean canopy cover rose 0.4 points over 2015–2025 (Grapevine: 1 %, −3.7 points). Degraded stays at 0 % (Raystown) and 0.3–0.4 % (Grapevine) in every variant. The base roll-up (equal weights, no forest-health set) is kept; the variants are reported as _vw2, _fh, _vw2fh tables. First results: Raystown project index ≈ 4.3 (Sustainable) with ~96 % of assessed acres Sustainable and no Degraded acres; 1985→1995 canopy closure 3,081 ac (the early-TCC rise, to validate); Grapevine 2005→2015 inundation 1,883 ac (the 2015 flood) and 2015→2025 exposure 1,081 ac vegetated.

2026-10-29, NAIP validation sample drawn (validate). Reference year 2022 at both sites (NLCD, TCC, LANDFIRE 2022 and LCMS all native; growing-season flights: Raystown 28 Apr – 29 Jun, 529 of 838 cells on 28 Jun; Grapevine 3–13 Jul). Stratified by mapped class, 50 cells per class present (all cells for rarer classes): Raystown 838 cells in 18 classes, Grapevine 710 in 17. Chips (180 m, RGB and CIR, cell outlined) streamed from Planetary Computer; 70 + 60 contact sheets; offline labelling page V0_labelling.html. Labels are not yet entered. A first look at one Raystown sheet of twelve M21 (open deciduous, TCC 29–59) cells shows most of them under closed canopy in NAIP, which points to open-canopy commission at the 60 % TCC threshold in eastern deciduous forest; the full sample will test this (§5.2 threshold tuning). Note also cells mapped M10 with TCC > 80 inside stands of standing dead trees with regrowth (NLCD herbaceous under a dead canopy), which may be the defoliation USACE staff rated as degraded.

2026-10-29, reference labels proposed for the classes in doubt (option 3). Claude proposed a label, a confidence and a note for every sampled cell in the herbaceous, shrub, open-canopy and sparse classes (Raystown 510 cells, Grapevine 468), written to V0_proposals.csv and, in the labelling page's import format, V0_labels_proposed.csv; the labelling page gained an "Import labels CSV" button so proposals can be confirmed or corrected in place. The dominant, visually clear classes (closed canopy, water, developed, cropland, pasture) are left for the reviewer. Before review, agreement with the map in these classes is low at both sites (Raystown M21 13/50, M10 6/50, all other doubt classes ≤ 1/50; Grapevine M21 23/50, M10 20/50, M09 9/50, M16 5/50, others 0/50). The disagreements point in the same direction at both sites: tier 2/3 LANDFIRE and NWI overrides place wetland, shrubland and annual-herbaceous labels on cells that are open or closed woodland in NAIP; the sparse-vegetation class M04-06u is dam riprap, beaches and developed shore; the 60 % TCC closed-canopy threshold is too high in eastern deciduous forest (Raystown) while in the Cross Timbers (Grapevine) the open/closed split is closer to the imagery. The numbers are proposals only and will be replaced by the reviewed labels before any accuracy is reported.

2026-10-29, TCC guard (rule 6b) added and the pipeline rerun. Motivation: in the proposed reference labels, NLCD herbaceous, shrub and emergent-wetland cells with TCC > 60 were closed canopy in NAIP at Raystown (M09 43/50, M16 45/50, M10 20/50). Rule 6b now sends NLCD 31/52/71/95 with TCC above classify.tcc_guard (60) to the closed tree classes before rules 7–10 run. Pre-guard outputs are kept in tables_noguard/ and stack_noguard/ (2022 rasters and classify logs only). The Grapevine shrubland and wetland errors are not touched by the guard because TCC is 0 there; they come from LANDFIRE tier 3 leaf and lifeform on post-oak woodland and from the drawdown rule, and are left for the threshold/tier review. Note for §5: the validation sample was drawn from the pre-guard map, so the stratification no longer equals the map being assessed; the estimator must use the sampling strata as strata and the new map as the map (Stehman 2014 style), which validate analyze does not yet do.

2026-10-29, METHOD_v2 steps 3–5 run provisionally against the proposed labels (user: "run steps 3 to 5 now against the proposals, mark provisional"). Everything in this entry is tuned on Claude's proposed labels, not reviewed ones, and is to be repeated on the reviewed set. Code: classify_year now reads every threshold from config.yaml classify: (tcc_closed/open/sparse, tcc_guard, tcc_guard_open, leaf_infer, leaf_nb_min, leaf_nb_share, dam_slope_deg) and takes an out_dir; load_static() factored out; new classes 26 M14-16u, 27 M19-21u, 28 M22-24u in scheme.py (ASSESSED/WOODY/TREE/SHRUB/UNRESOLVED sets, change GROUP11, T3 unresolved_leaf_acres); shrub leaf and tree leaf for NLCD 90 / guard cells come only from a LANDFIRE subclass within 3 years or a near-unanimous 5×5 NLCD neighbourhood (≥ 6 tree cells, ≥ 67 % one leaf type, tier 3), else unresolved; the 5×5 shrub-leaf inference and the "mixed" default are gone; landfire.evt_lookup marks annual only for non-agriculture EVTs (the old M11 was LANDFIRE pasture/row crop); rule 9 drops the drawdown term; rule 5b (NLCD 31 in dam_zone.tif with slope ≥ 8° → M02), 7d (ssurgo_substrate.tif: pits → M03, rock outcrop → M05, rubble → M06, riverwash/beach/dune/badland → M04, urban land → M02), 7e (slope_deg.tif: > 45° → M05, 25–40° next to M05 → M06), 7f (shoreline or drawdown barren → M04). build_stack --only ssurgo slope dams writes the three layers from the drive (gSSURGO gpkg substrate field; 3DEP tiles mosaicked, slope from the gradient on the 30 m grid; NID CSV with its metadata first row skipped, federal-owner dams only, radius max(300 m, half the dam length + 150 m); Grapevine's window holds 26 private pond dams that were excluded). Sweep (tables/threshold_sweep_2026-10-29.csv, METHOD_v2 §5.4): 2022 only, TCC_CLOSED {35,40,45,50,60} × TCC_OPEN {10,15,20,25} × guard open band on/off, scored against the 510 + 468 proposals at M / NVC / group level. TCC_CLOSED is the only threshold that moves the NVC score (Raystown 0.49 at 60 → 0.54 at 45, Grapevine 0.42 → 0.47, flat below 45); TCC_OPEN only moves the M score (15 beats 25 by 4–6 points); TCC_SPARSE 5 vs 10 is identical; the open-band guard gains Grapevine 1.5–2 points and costs Raystown 0.5. Chosen, provisional: 45 / 15 / 10, guard 45 with the open band on. Full 1985–2025 rerun (tables/v2_rerun_2026-10-29.log, 4.5 min); guard-only tables kept in tables_guard60/, v1 in tables_noguard/. Boundary acres 2022 guard-only → v2: Raystown M21 466 → 228, M24 18,765 → 18,980, M14 33 → 0 (M14-16u 33), M09 25 → 0, M10 94 → 26, M18 20 → 1, M04-06u 33 → 6; Grapevine M24 3,194 → 3,850, M21 1,303 → 831, M09 1,449 → 877, M14/M16 290 → 10 (M14-16u 263), M19-21u 333, M22-24u 389, M04-06u 48 → 3 (M04 37). 1985 Raystown open deciduous 3,743 → 1,216 ac and closed 14,993 → 17,617 ac, so the early-TCC "canopy closure" signal shrinks: 2016 → 2023 canopy closure 252 → 115 ac, opening 77 → 49 ac. Condition 2025: Raystown 4.33 → 4.35, S 97.0 → 98.2 %; Grapevine 3.21 → 3.22, S 36.8 → 40.7 %, tier-1 share of assessed acres 0.29 → 0.19 (unresolved classes are tier 2). Unresolved leaf acres: Raystown 52 (2022) – 276 (1985); Grapevine 984 (2022) – 2,535 (2000), i.e. 14–35 % of Grapevine's assessed acres carry no leaf type, the honest cost of §1.7 until a better canopy-type source exists. Remaining disagreements with the proposals are not threshold problems: NLCD shrub with TCC < 15 that NAIP shows as woodland (Grapevine 57 cells, Raystown 22) and the Grapevine drawdown-zone grass mapped as wetland from NWI/NLCD 95.

2026-10-30, Stehman (2014) estimator added to validate analyze. Strata = V0_strata.csv (map at sampling time); map assessed = classf_<year>.tif of the current stack or --map-stack <dir>, read at the sample cells; estimates are stratified totals and ratio estimators with per-stratum variances (eq. 2–5 of Stehman 2014); strata not fully labelled are dropped (--partial overrides) and the domain share is printed; --tag suffixes the output files; at USACE-M level an unresolved parent class counts as correct for any of its subclasses. The old Olofsson function is kept for strata = map. Before the run, the proposed labels were harmonised to the v2 dam convention (M03 "rock dam face / dam crest / dam toe / embankment" at Raystown, M05 "riprap dam face" at Grapevine → M02; 56 + 5 cells; note appended in V0_proposals.csv and V0_labels_proposed.csv). Provisional results on the proposals (doubtful strata only: 11 strata = 3.0 % of Raystown's mapped area, 10 strata = 25.3 % of Grapevine's; tags _proposed, _proposed_v1): overall accuracy in the domain, v1 → v2, Raystown 0.16 → 0.35 (M), 0.19 → 0.46 (NVC, group); Grapevine 0.29 → 0.44 (M), 0.42 → 0.54 (NVC), 0.32 → 0.44 (group); ± 0.07. Error-adjusted areas inside the Raystown domain: closed canopy 411 ± 71 ac against 553 mapped, open canopy 256 ± 65 vs 283, herbaceous 145 ± 56 vs 18 (the herb that the guard and lowered thresholds turned into canopy is partly real grass: M10 producer's accuracy 0.03), developed 91 ± 39 vs 27 (dam structures still mapped as M03/M04-06u/M22 where TCC is high on the dam face). Grapevine: open canopy 1,510 ± 277 vs 1,069 mapped, closed 1,144 ± 242 vs 864, herbaceous 1,000 ± 226 vs 1,713 (M09 user's accuracy 0.16: the NWI-driven wetland is mostly grass and woodland), shrubland 30 ± 19 vs 269 (M14-16u user's accuracy 0.09), sparse 205 ± 129 vs 39. These are domain figures from proposed labels and must be replaced after review.

2026-10-30, all-projects batch prepared and launched (user: run the routine for all projects without review, to understand how the method works). Three changes made the batch feasible: (a) the national NLCD and LANDFIRE zips are deflate-compressed, so a window read through /vsizip decompresses most of a 1.4–3 GB file (Raystown's 41 NLCD years took 12 min); they were extracted once to <SOURCE>/extracted/ on the drive and common.zip_path / landfire.read_attr_csv prefer the extracted file (TCC and LCMS zips are stored, not deflated, and stay as they are); (b) the NRM boundary .gdb takes minutes per read because the river projects are polygons with hundreds of rings; common.load_boundaries caches the dissolved layer as Project Site Boundary/projects_5070.gpkg, reads it with OGR_ORGANIZE_POLYGONS=ONLY_CCW and an id filter (1–2 s; pilot geometries identical, 11 self-touching polygons unchanged in area); (c) a driver usace_veg.batch runs the nine steps per project as subprocesses (build_stack → SSURGO download → SSURGO layer → classify → tables → change → condition → landscape → compare_usace), 5 projects in parallel, at most 2 above 8 Mpx, smallest window first, resumable through tables/<project>/batch_status.json, per-project logs in tables/batch_2026-10-30/. Scope: 339 projects, 807 Mpx of window in total (median 0.47 Mpx; pilots 0.6 and 1.1; largest the Apalachicola–Chattahoochee–Flint, Illinois Waterway and Upper Mississippi river systems at 45–70 Mpx). usace_veg.summary aggregates the per-project tables into tables/ALL_summary/ (projects.csv, class_acres.csv, group_acres_years.csv, change_2016_2023.csv, usace_compare.csv, overview.txt). Everything in this round uses the provisional v2 rules and thresholds; no project has reviewed labels.

2026-10-30, batch run: fixes made while it ran (all provisional, no change to any number). By 11:30 the 321 windows up to 6 Mpx were through (about 280 s per project at 1–1.5 Mpx, 200 s/Mpx at 3–5 Mpx). Failures and their fixes: Chena River Lakes (AK) is outside NLCD and stays excluded; Ball Mountain Lake (32 cells) and John H. Kerr (18.7 ac reference area spread over a 4.4 Mpx window, no cell half inside) had no assessable stand in any year, so condition.run_project and landscape.run_project now write empty tables / zero-patch rows instead of failing; Union City Lake's only county (Erie PA) has no readable TIGER file (Census server rejects the download), so build_stack writes an empty, tagged roads.tif and the road metrics there are not valid; Grapevine's window had been widened by one column by buffered_bounds and build_stack now keeps an existing window.json. Memory for the river windows: condition kept ten per-year rasters for all 41 years in RAM (≈1.4 kB/px, 100 GB at 70 Mpx); they now go to a disk-backed PixStore (stack/<project>/_condition_tmp/<name>_<year>.npy, memory-mapped on read, removed at the end), and the per-year function frees its intermediates as soon as they are used — outputs byte-identical on DeGray Lake and Salt Creek. Measured peak resident memory of condition is still ≈0.2 GB per Mpx (4.2 GB at 20 Mpx), so a memory gate (common.wait_for_memory) now holds a classify / change / condition / landscape step on a window above 8 Mpx until MemAvailable covers 1 GB + 0.22 (condition) or 0.12 GB per Mpx, and the batch timeouts scale with window size (× Mpx/10 above 10 Mpx). The driver got a third concurrency tier (--huge-mpx 30 --huge-workers 2) and was restarted (resumable) with 3 workers / 3 big / 2 huge once the small windows were done. Per-step timings at 10–14 Mpx: build_stack 300–360 s, classify 190–650 s, condition 1,700–2,600 s, landscape 400–1,200 s; the 70 Mpx Apalachicola window is expected to take about 4 h in condition alone.
