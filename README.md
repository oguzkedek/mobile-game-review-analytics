# ML-Based Game Review Analytics

End-to-end machine learning pipeline for analyzing mobile game App Store reviews. Covers exploratory analysis, topic discovery, sentiment classification, and aspect-based sentiment analysis (ABSA) across three languages (EN, TR, DE) and four games.

## Games Analyzed

| Game | Role |
|------|------|
| **Royal Match** | Primary focus |
| Candy Crush Saga | Competitor |
| Toon Blast | Competitor |
| Gardenscapes | Competitor |

**Markets:** US, UK, TR, DE, CA, JP &nbsp;|&nbsp; **Period:** Jan 2025 – Jan 11, 2026 &nbsp;|&nbsp; **EDA scope:** 9,630 reviews (6 markets) &nbsp;|&nbsp; **NLP scope:** 7,256 reviews → 14,108 segments (EN/TR/DE only — JP excluded due to language support)

---

## Pipeline Overview

```
┌─────────────┐    ┌──────────────────┐    ┌───────────────────┐    ┌─────────┐    ┌─────────────────────┐
│   1 - EDA   │───▶│ 2 - BERTopic     │───▶│ 3 - Sentiment     │───▶│ 4 - ABSA│───▶│ 5 - Synthesis &     │
│             │    │   Topic Modeling  │    │   Classification   │    │         │    │   Hypothesis Testing│
└─────────────┘    └──────────────────┘    └───────────────────┘    └─────────┘    └─────────────────────┘
   9,630 reviews      7,256 → 14,108          7,236 reviews        14,108 seg.       Unified analysis
   6 markets           segments                3 language-          Topic ×            7 stat tests
   4 games             114 topics              specific models      Sentiment          Product insights
```

---

## Notebooks

### 1 — Exploratory Data Analysis

> **Goal:** Establish a clean, structured, and well-documented analytical foundation for Royal Match App Store review data.

**Coverage & data quality:**
- Raw data loaded, parsed, and validated with an immutable raw table preserved throughout
- **2025+ analysis scope** yielding **9,630 reviews** across 4 games and 6 countries
- **Language detection** (FastText `lid.176`) cleanly separated EN/TR/DE reviews from Japanese and other unsupported languages, enabling language-aware downstream work

**Text-based exploration:**
- Rule-based theme detection surfaced **directional friction signals** — ads intrusion (16.2%), monetization (15.1%), rewards (14.1%), difficulty (13.9%)
- **Loyal players who complain** most frequently cite difficulty (34%) and rewards (31%) — a product-relevant finding
- A **preliminary mixed sentiment proxy** (~9.4%) identified high-rating reviews with friction keywords, a crude approximation awaiting proper sentiment classification

**Rating landscape:**
- All games show **strongly bimodal distributions** (1★ and 5★ dominant), confirming the polarized nature of App Store written reviews
- **Toon Blast** is the competitive benchmark — leading in nearly every market
- **UK** is the closest competitive parity zone for Royal Match; **Germany and Japan** are structurally challenging markets
- Within each market, **lower ratings correlate with longer reviews** — a consistent behavioral signal

**Temporal dynamics:**
- Daily signals are too noisy for direct attribution; a controlled diagnostic stack (7d volume-weighted rolling + 14d past-only baseline + Δ vs baseline) was built to surface meaningful movements
- **Critical drop detection** identified event candidates with conservative thresholds (Δ ≤ −0.30, country-adaptive volume gates, local minima filtering)
- Several update windows show clear event-shaped dynamics: Oct 27 (multi-market uplift), Dec 8 (multi-market deterioration), Dec 22 (heterogeneous)

**Update impact:**
- Pre/post event-level analysis confirmed **Dec 8 (v32672)** as the strongest negative update and **Oct 27 (v32131)** as the strongest positive, both with multi-market evidence
- Cross-market heterogeneity is a recurring pattern — the same version can produce opposite effects in different markets

**Downstream handoff** to three pipelines: (1) BERTopic topic modeling via `reviews_nlp_scope_2025.csv`, (2) Sentiment analysis via `reviews_sentiment_scope_2025.csv`, (3) Hypothesis tests to formalize observed patterns

---

### 2 — Topic Modeling (BERTopic)

> **Goal:** Discover latent themes in user feedback and build a clean, interpretable three-tier topic taxonomy.

This notebook built an end-to-end BERTopic topic modeling pipeline on 7,256 app store reviews across three languages. The pipeline followed a deliberate sequence: **text preprocessing and segmentation** broke multi-topic reviews into 14,108 theme-pure segments using contrast splitting, theme-aware conjunction splitting, noise filtering, and micro-merge repair. Segments were embedded with OpenAI `text-embedding-3-large` (3,072d), then clustered per language with **UMAP + HDBSCAN**, followed by per-language hyperparameter tuning, manual similarity-based merging (6 merges total across all languages), and cosine-similarity outlier reduction.

The final **114 topics** (EN: 49, TR: 32, DE: 33) were named via GPT-5.4 with a structured three-tier taxonomy (main_topic → subtopic → subtopic_detail), producing PM-actionable labels with 66 high-confidence and 48 medium-confidence assignments, zero errors.

**Taxonomy:**

| Main Topic | Segments | Reviews Affected | Key Subtopics |
|-----------|:--------:|:---------------:|---------------|
| MONETIZATION_ECONOMY | 3,935 (27.9%) | 2,122 (36.0%) | P2W & Refunds, Reward Balance, Purchase Issues |
| GAMEPLAY | 2,954 (20.9%) | 1,607 (27.2%) | Difficulty & Balance, Game Design, Content & Progression, Accessibility |
| ADS_EXPERIENCE | 2,483 (17.6%) | 1,487 (25.2%) | Scam & Deception, Frequency & Intrusiveness, Ad-Free Praise |
| TECHNICAL_SUPPORT | 1,625 (11.5%) | 1,097 (18.6%) | Bugs & Stability, Customer Service, UI/UX, Account & Data |
| OTHER | 472 (3.3%) | — | General Feedback |
| COMMUNITY_SOCIAL | 289 (2.0%) | 136 (2.3%) | Team & Social Features |
| *UNASSIGNED* | *2,349 (16.7%)* | — | *Outlier segments the model couldn't classify* |

*Segment %: share of all 14,107 segments. Reviews Affected: share of all reviews — a single review can contain multiple topics, so percentages exceed 100%.*

**Cross-language patterns:** Results are strong but not uniform. EN benefits from the largest corpus (9,324 segments) and delivers the most granular topic landscape. TR achieved the lowest outlier rate (14.9%) but the highest ambiguity — its 32 topics show fuzzier boundaries, particularly in the monetization and difficulty space where Turkish review language tends to blend complaints. DE clusters are the tightest by cohesion (~0.69 avg_intra_sim) but carry a higher outlier rate (19.2%).

**What could be improved:** The most impactful lever would be **more data** — TR and DE corpora are 3–4× smaller than EN, limiting cluster granularity. The segmentation lexicon was built iteratively from the reviews themselves; as the review landscape evolves, **periodic lexicon refresh** against fresh review batches would keep the pipeline aligned with emerging user concerns.

**Output:** `segments_topic.parquet`, `segments_absa.parquet`, `segment_embeddings_fit.parquet`

---

### 3 — Review-Based Sentiment Analysis

> **Goal:** Train language-specific binary sentiment classifiers using GPT-4.1 pseudo-labels as ground truth, enabling cost-efficient daily batch inference.

GPT-4.1 pseudo-labels are used as ground truth instead of rating-based heuristics, enabling the models to capture cases where sentiment diverges from star ratings (e.g., 5★ reviews with negative tone). The neutral class was excluded from training due to insufficient samples (209 out of 7,236 — just 17 in German) and is instead handled post-hoc via confidence thresholds in the dashboard layer.

**Models:**

| Language | Model | Test Macro F1 | Zero-shot Baseline | Gain |
|----------|-------|:------------:|:------------------:|:----:|
| EN | `roberta-large` (355M) | **0.957** | 0.814 | +14.3 |
| TR | `savasy/bert-turkish-text-classification` (110M) | **0.919** | 0.707 | +21.2 |
| DE | `deepset/gbert-base` (110M) | **0.898** | 0.729 | +16.9 |

- **Training:** Weighted cross-entropy loss (positive class upweighted 1.5–2.5×), cosine LR scheduling with 10% warmup, best checkpoint by validation F1
- **5-Fold CV:** TR 0.915 ± 0.016, DE 0.898 ± 0.008 — confirms stability
- **Full dataset:** **98.4% ML-GPT agreement** (7,119/7,236). The 117 disagreements are predominantly sarcastic, mixed-sentiment, or ambiguous edge cases. These models are ready for daily batch inference, delivering GPT-level sentiment quality at a fraction of the cost

**Output:** `sentiment_results.parquet`

---

### 4 — Aspect-Based Sentiment Analysis (ABSA)

> **Goal:** Classify sentiment at the segment level — the final piece of the ABSA pipeline. Combined with BERTopic topics, this enables analysis of *how users feel about each specific aspect*.

Each segment already has a topic from BERTopic; this project adds sentiment, enabling full Topic × Sentiment analysis. A 3-class approach was tested first but neutral performance was poor (DE macro F1=0.695, TR neutral F1=0.625) due to limited samples. Switching to 2-class improved all languages by +3 to +9 points — neutral is handled post-hoc via confidence thresholds instead.

| Language | ZS Macro F1 | FT Macro F1 | Δ | ML-GPT Agreement |
|----------|:----------:|:----------:|:---:|:----------------:|
| EN | 0.641 | **0.909** | +0.268 | — |
| TR | 0.584 | **0.908** | +0.325 | — |
| DE | 0.601 | **0.793** | +0.192 | — |
| **All** | — | — | — | **97.8%** |

- Fine-tuning gains are **even larger** than review-level (+19 to +33 points vs. +14 to +21) — short, context-stripped clauses are harder for general multilingual models
- **TR standout:** 0.908 F1 despite 10:1 class imbalance with only 185 positive samples — combination of language-specific pre-training and 5.6× class weights
- 5-fold CV confirms stability: TR 0.830 ± 0.032, DE 0.831 ± 0.018. Full dataset inference shows 97.8% ML-GPT agreement with only 11 low-confidence predictions (0.1%)

**How it will be used:** These segment-level predictions complete the ABSA pipeline — each segment now has both a **topic** (from BERTopic) and a **sentiment** (from this model). This enables product-level insight: which aspects drive negative vs positive sentiment, how aspect sentiment shifts across app updates, and where competitors differ. The ML models replace GPT for daily batch inference at near-zero marginal cost, while GPT remains available for edge cases and deeper analysis.

**What could be improved:** The main limitation is **data scarcity**: DE has only 475 positive segments (F1=0.661, weakest result), TR has 185. More training data, upgrading to larger models (e.g., `gbert-large`), or data augmentation via back-translation would help. With 1,000+ neutral samples per language, a viable 3-class model could also become feasible.

**Output:** `segments_sentiment_final.csv`

---

### 5 — Synthesis & Hypothesis Testing

> **Goal:** Unify all pipeline outputs, convert model outputs into actionable product insights, and statistically validate patterns.

**Unified data:** 7,236 reviews × 14,107 segments, 86.3% segment-review sentiment agreement.

#### Key Findings

**1. Star ratings systematically overestimate satisfaction.** 50.7% of 4★ and 19.1% of 5★ reviews carry negative sentiment. The reverse is negligible (1.4% of 1-2★ are positive). This asymmetry means any product intelligence system relying solely on star ratings will significantly undercount dissatisfied users — particularly in Turkey, where 48.4% of high-rating reviews are negative.

**2. MONETIZATION is the #1 product pain point by both intensity and reach.** 92.7% negative rate with 36.0% impact (2,122 reviews affected). At subtopic level, MONETIZATION_P2W_REFUNDS alone accounts for 28.2% of all negative review volume. TECHNICAL_SUPPORT is more intense (93.2% neg rate) but affects fewer reviews (18.6% impact).

**3. Hidden dissatisfaction reveals "silent" product risks invisible to ratings.** ABSA adds 13.7% more signal than review-level sentiment. Inside positive reviews, the dominant complaint topic shifts from MONETIZATION to **GAMEPLAY** (38.2%) — specifically DIFFICULTY_BALANCE. Users who like the game overall still complain about difficulty tuning and bugs, but these frustrations don't surface in star ratings. Statistical testing (H4) confirms this concentration is significant (p=2.15e-9).

**4. Each game has a distinct competitive "pain fingerprint".** Topic profiles differ significantly across games (H3: Cramer's V=0.208). Royal Match's defining issue is ADS_EXPERIENCE (+10.8 std. residual), Candy Crush Saga's is TECHNICAL_SUPPORT (+22.4), Toon Blast's is COMMUNITY_SOCIAL (+14.5), Gardenscapes' is GAMEPLAY (+8.0). Aggregate sentiment benchmarking misses these structural differences.

**5. Market-level sentiment varies significantly.** Turkey and Germany produce systematically more negative sentiment than US and UK (H5: p=3.12e-32). This is not noise — it reflects either cultural expression patterns or genuine market-specific product experiences. Localized response strategies are warranted.

**6. Version-level sentiment impact could not be confirmed.** Neither the strongest negative (v32672) nor positive (v32131) update candidate showed a statistically significant sentiment shift (H1a: p=0.52, H1b: p=0.31). This is likely a data resolution limitation — App Store version fields reflect store-level versions, not user-installed versions.

**7. Hidden dissatisfaction is a category-level phenomenon.** Royal Match's hidden dissatisfaction rate (26.2%) is not statistically different from the competitor average (25.0%, H6: p=0.53). This suggests the pattern is structural to the mobile puzzle genre, not specific to any single product.

#### Hypothesis Tests

| # | Hypothesis | Result | p-value | Effect Size | Interpretation |
|---|-----------|--------|---------|-------------|----------------|
| H1a | v32672 (Dec 8) caused sentiment drop | Fail to reject | 0.52 | — | Small post-window samples and store-level version fields limit statistical power |
| H1b | v32131 (Oct 27) caused sentiment improvement | Fail to reject | 0.31 | — | Pre/post neg rates barely changed (79.5→79.1% and 69.3→65.4%) |
| H2 | Topic affects rating | **Reject** | ~0 | η² = 0.022 | MONETIZATION and TECHNICAL pull ratings to floor; GAMEPLAY sits one tier higher |
| H3 | Games have distinct topic fingerprints | **Reject** | ~0 | V = 0.208 | Strongest result — competitive benchmarking must be topic-aware |
| H4 | Hidden dissatisfaction concentrates in specific topics | **Reject** | 2.15e-9 | V = 0.064 | TECHNICAL_SUPPORT (+3.54) and GAMEPLAY (+2.83) over-represented; MONETIZATION (-2.65) under-represented |
| H5 | Sentiment differs across countries | **Reject** | 3.12e-32 | V = 0.146 | TR (+3.93) and DE (+2.87) more negative; US (-3.25) and UK (-3.07) more positive |
| H6 | RM hidden dissatisfaction ≠ competitor average | Fail to reject | 0.53 | V = 0.012 | A useful null: hidden dissatisfaction is a category-level phenomenon, not RM-specific |

**Methods:** Mann-Whitney U, Kruskal-Wallis + Dunn post-hoc, Chi-square, proportion z-test, Cramer's V. All non-parametric, α = 0.05.

#### Pipeline Summary

| Step | Focus | Key Metric |
|:----:|-------|------------|
| 01 | Data integration | 7,236 reviews → 14,107 segments, 86.3% segment-review agreement |
| 02 | Sentiment × Rating | 50.7% of 4★ reviews are negative; TR hidden dissatisfaction at 48.4% |
| 03 | Topic × Sentiment | MONETIZATION: 36.0% impact; each game has distinct topic fingerprint |
| 04 | ABSA value-add | 13.7% disagreement; hidden negativity led by GAMEPLAY not MONETIZATION |
| 05 | Hypothesis testing | 4/7 rejected; structural patterns confirmed, event-level effects not |

---

## Data Files (`data/`)

| File | Source | Description |
|------|--------|-------------|
| `appstore_reviews.csv` | Raw | Raw App Store reviews |
| `reviews_nlp_scope_2025.csv` | NB 1 | EN/TR/DE reviews for topic modeling |
| `reviews_sentiment_scope_2025.csv` | NB 1 | Reviews with temporal/version context |
| `sentiment_results.parquet` | NB 3 | Review-level ML predictions + confidence |
| `segment_sentiment_results.parquet` | NB 4 | Segment-level ML predictions |
| `segments_sentiment_final.csv` | NB 4 | Final segment-level results |

## Tech Stack

| Component | Tools |
|-----------|-------|
| **Embeddings** | OpenAI `text-embedding-3-large` |
| **Topic Modeling** | BERTopic, UMAP, HDBSCAN, c-TF-IDF |
| **Sentiment Models** | `roberta-large` (EN), `savasy/bert-turkish-text-classification` (TR), `deepset/gbert-base` (DE) |
| **LLM Labeling** | GPT-4.1 (pseudo-labels), GPT-5.4 (topic naming) |
| **Training** | HuggingFace Transformers, PyTorch, scikit-learn |
| **Language Detection** | FastText `lid.176` |
| **Statistics** | scipy, scikit-posthocs (Dunn test) |
| **Data** | pandas, DuckDB, Parquet |
| **Visualization** | matplotlib, seaborn (dark theme) |

## Limitations

- **App Store only.** Google Play data was not included. Cross-store patterns may differ.
- **RSS feed sampling bias.** App Store RSS prioritizes recent and extreme reviews — the dataset is not a random sample of all user sentiment.
- **Version field is approximate.** App Store does not expose the user's actual installed version — it shows the current store version at the time of the review. A user on v1.0 who writes a review after v2.0 is published will appear under v2.0. We used this field for update impact analysis despite this limitation, which likely dilutes version-specific signals and may explain why hypothesis tests H1a/H1b failed to reach significance.
- **Segment-level ABSA covers 81.6% of reviews.** 1,335 reviews (18.4%) were too short for segmentation and are excluded from topic-level analyses.
- **BERTopic coverage.** UNASSIGNED segments (2,349) represent topics the model couldn't classify — a non-trivial blind spot.
- **Temporal scope.** ~12 months of data (Jan 2025 – Jan 11, 2026). Seasonal effects and long-term trends may require a longer observation window.
