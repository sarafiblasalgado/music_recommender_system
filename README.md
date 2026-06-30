# Last.fm Multi-Layer Recommender System

A recommender system prototype built on the **Last.fm HetRec2011** dataset
(1,892 users, 17,632 artists, 92,834 listening interactions, 11,946 tags,
186,479 tag assignments, and a 25,434-edge social friend graph).

Unlike a ratings-only project (e.g. MovieLens), this dataset has **three
independent data layers**, and this project uses all of them:

| Layer | File | Used by |
|---|---|---|
| Listening behaviour (implicit play counts) | `user_artists.dat` | baselines, CF, matrix factorization |
| Folksonomy tags | `tags.dat`, `user_taggedartists.dat` | content-based recommender |
| Social friend graph | `user_friends.dat` | friend-based recommender, hybrid |

## Project structure

```
.
├── data/raw/                  Last.fm HetRec2011 dataset (.dat files)
├── src/
│   ├── config.py               paths & shared constants
│   ├── data_loading.py         loading, EDA, train/test split, relevance
│   ├── baselines.py             MostPopular, MostPlayed, Random
│   ├── content_based.py         TF-IDF tag-based recommender
│   ├── collaborative_filtering.py  item-item & user-user CF
│   ├── matrix_factorization.py   implicit-feedback ALS
│   ├── social_filtering.py       friend-based + graph-diffusion (Personalized PageRank) + hybrid rank-fusion
│   ├── backfill.py               cold-start: top up short lists with Most Popular
│   ├── diversification.py        MMR diversity-aware re-ranking (wraps any fitted model)
│   └── evaluation.py             Precision/Recall/NDCG/MRR/coverage/novelty/diversity
├── main.py                     end-to-end analysis pipeline (EDA → train → evaluate)
├── robustness_check.py         multi-seed variance check (run separately, see below)
├── temporal_split_check.py     random- vs temporal-split comparison (run separately, see below)
├── relevance_threshold_check.py  does the headline model ranking depend on the relevance definition? (run separately, see below)
├── backend/                    FastAPI service for the consumer-facing prototype
│   ├── recommender_service.py   fits models on the full data, builds each listener's home feed
│   ├── artist_images.py         real artist artwork lookup (iTunes Search) + disk cache
│   ├── feedback_store.py        like/skip persistence, feeds the Made-For-You re-ranker
│   └── main.py                  the API: GET /api/users, /api/users/random, /api/home/{id}, POST /api/feedback
├── frontend/                   React + Vite + Tailwind prototype UI ("Wavelength")
│   └── src/                     components, API client -- no model names or metrics in the UI
├── results/                    metrics.csv, figures/, example recommendations
├── REPORT.md                   full write-up (the assignment's final report)
└── requirements.txt
```

The analysis pipeline (`main.py`, `src/`) and the product prototype
(`backend/`, `frontend/`) are deliberately separate: the former trains on
an 80/20 split to measure offline accuracy (everything in `REPORT.md`),
the latter trains on the *full* interaction history to serve the best
live recommendations for a demo, and shows the user nothing about which
algorithm produced what -- no model names, scores, or comparison tables,
just a normal-looking music app. Both reuse the same `src/` recommender
classes.

## Setup

```bash
pip install -r requirements.txt
```

## Run the full pipeline (EDA, training, evaluation, examples)

```bash
python main.py
```

This grid-searches ALS latent factors, ALS regularization (`reg`) and
confidence scaling (`α`), CF neighbourhood size `k`, the CF+Social hybrid
blend weight, the MMR re-ranker's relevance/diversity weight `λ`, and the
ALS+Graph-Diffusion hybrid blend weight (see below) on an inner
train/validation split (never the test set), trains all 14 recommenders,
evaluates them with an identical protocol, runs paired significance tests
between key model pairs, and writes:

- `results/metrics.csv` — the full model-comparison table (precision,
  recall, NDCG, MRR, hit-rate, coverage, novelty, diversity, popularity bias)
- `results/significance_tests.csv` — paired Wilcoxon signed-rank tests on
  per-user NDCG@10 between key model pairs (is model A *significantly*
  better than model B, or could the gap be noise?)
- `results/metrics_backfilled.csv` — before/after accuracy for the models
  that have cold-start gaps, once wrapped with the Most-Popular backfill
- `results/mmr_lambda_sweep.csv` — the MMR relevance/diversity trade-off
  sweep used to pick `λ` (see `REPORT.md` §4.7 for the selection rule and
  §6 for what it found: diversity improves, but popularity bias and
  coverage barely move — they're more independent than they look)
- The ALS+Graph-Diffusion hybrid weight sweep is printed during
  `# STEP 3d` rather than written to its own CSV — see `REPORT.md` §6 for
  why its validation-set "win" (+1.1% NDCG) is the project's clearest
  example of a tuned hyperparameter that doesn't generalize: it's
  significantly *worse* than plain ALS on the held-out test set
- `results/figures/*.png` — EDA charts, ALS convergence curve, model
  comparison bar chart, coverage-vs-novelty scatter, popularity-bias
  chart, and the MMR relevance-vs-diversity trade-off curve
- `results/examples/recommendation_examples.txt` — side-by-side
  recommendations for a few example users

Separately, `python robustness_check.py` re-runs the split/fit/evaluate
cycle across 5 random seeds and reports mean ± std per model
(`results/metrics_multiseed_summary.csv`) — kept out of the main pipeline
since it repeats the most expensive step (fitting ALS) 5x.

Also separately, `python temporal_split_check.py` compares the usual
random per-user split against a genuine chronological split, on the
~22% of interactions where a timestamp proxy exists at all (see
`REPORT.md` §8 for why the dataset only supports this partially, and what
it found: the random split is meaningfully optimistic, especially for
Implicit ALS).

Also separately, `python relevance_threshold_check.py` re-evaluates a
representative subset of models under three different definitions of
"relevant" (not just the project's default median-play-count bar) and
writes `results/relevance_threshold_check.csv` — checks that the headline
model ranking in `REPORT.md` §6 isn't an artifact of that one threshold
choice (it isn't: the ranking is identical at every threshold tested).

## Run the interactive prototype ("Wavelength")

Two processes: the API backend (Python) and the React frontend (Node).

```bash
# Terminal 1 -- backend (fits all models once at startup, ~15s)
uvicorn backend.main:app --reload --port 8000

# Terminal 2 -- frontend
cd frontend
npm install   # first time only
npm run dev
```

Open **http://localhost:5173**. Switch listeners with the search box or
the "Shuffle" button in the header (there's no real login -- HetRec users
are anonymized IDs, so this simulates "logging in" as different people).
Each listener's home feed has four shelves: **Made For You** (personalized
picks), **Friends Are Listening To** (their actual Last.fm friends' plays),
**Because You Listened To X** (similar artists to whatever they play most),
and **Trending Now**. Real artist photos are fetched live from the iTunes
Search API (keyless) and cached to disk; any artist without a match falls
back to a generated gradient avatar, never a broken image. Cold-start
gaps (e.g. a thin-data listener) are backfilled with Most-Popular picks
invisibly -- unlike the research pipeline, a real product doesn't show
its seams.

Hover any card for a ❤️ / ✕. **Skip** removes that artist from every
shelf immediately and permanently for that listener. **Like** nudges
"Made For You" towards content-similar artists (a lightweight re-ranking
pass, not a retrain -- see `REPORT.md` §10) -- give it a few seconds after
clicking and the feed updates with the new picks.

This UI intentionally shows **no model names, scores, or metrics** --
that analysis lives entirely in `REPORT.md`. The prototype is for
*using* the recommender, not explaining it.

The layout is responsive (verified down to a 375px mobile viewport, no
horizontal overflow) and keyboard-accessible: every like/skip control has
an `aria-label`, focusing a card with Tab reveals its controls the same
way hovering does, the listener search is a labelled combobox with
Escape-to-close, and there's a skip-to-content link for screen-reader and
keyboard users. The loading state is a layout-matching skeleton rather
than a bare "loading" string.

## Models implemented

1. **Most Popular** / **Most Played** (non-personalized baselines)
2. **Random** (sanity-check lower bound)
3. **Content-Based**: TF-IDF *and* raw tag-count variants over Last.fm
   folksonomy tags (`ContentBasedRecommender(use_tfidf=...)`), user profile
   = listening-weighted centroid of artists played — counter-intuitively,
   raw counts win on accuracy here, see `REPORT.md` §6
4. **Item-Item CF**: cosine similarity over log-compressed play counts,
   with shared-listener shrinkage + a hard minimum-co-occurrence floor,
   neighbourhood size `k` chosen by grid search (not hand-picked)
5. **User-User CF**: same idea, over users instead of artists
6. **Implicit ALS Matrix Factorization**: confidence-weighted Alternating
   Least Squares (Hu, Koren & Volinsky 2008) — the model built specifically
   for implicit feedback, rather than adapting an explicit-rating SGD model;
   latent factor count, regularization (`reg`), and confidence scaling
   (`α`) all chosen by grid search, not hand-picked
7. **Friend-Based** (social layer): recommends what a user's actual Last.fm
   friends listen to
8. **Graph Diffusion (Personalized PageRank)**: extends Friend-Based past
   one hop -- a random walk over the (symmetrized) friend graph that
   restarts at the query user with probability 0.15, computed by power
   iteration (`src/social_filtering.py`), giving every user in the
   network a continuous relevance weight instead of a hard one-hop cutoff
9. **Hybrid (CF + Social)**: weighted reciprocal-rank fusion of item-item
   CF and the friend-based recommender, with the blend weight chosen by
   grid search on held-out validation NDCG (not hand-picked)
10. **Implicit ALS + MMR re-ranking**: wraps the tuned ALS model with
    Maximal Marginal Relevance (`src/diversification.py`) to directly
    target the popularity-bias/low-diversity weakness ALS's own evaluation
    numbers expose — the relevance/diversity trade-off weight `λ` is
    chosen by sweeping a grid and picking the most diversification that
    keeps validation precision within a 10% budget, not by hand
11. **Hybrid (ALS + Graph Diffusion)**: reciprocal-rank fusion of this
    project's two most accurate individual models, with the blend weight
    chosen by grid search on held-out validation NDCG -- included despite
    *not* beating plain ALS on the test set, because that gap (real,
    measured, and statistically significant) is itself the clearest
    evidence in this project that an inner-validation grid search can
    overfit its own validation split

Evaluation includes Precision/Recall/NDCG/MRR/Hit-Rate@10 plus four
beyond-accuracy metrics: catalog coverage, novelty, intra-list diversity
(TF-IDF tag-vector dissimilarity within each user's list), and popularity
bias (mean popularity percentile of recommended items — distinct from the
other three, see `REPORT.md` §5). Paired Wilcoxon significance tests, a
5-seed variance check (`robustness_check.py`), and a relevance-threshold
robustness check (`relevance_threshold_check.py`) back up the headline
accuracy comparisons. Cold-start gaps (a user with no friends, or whose CF
neighbourhood never clears the minimum-support floor) are backfilled with
Most-Popular picks at the app layer, while the headline evaluation table
keeps each algorithm's raw, unmodified accuracy for a fair comparison.

See `REPORT.md` for the full methodology, the evaluation protocol, results,
a worked debugging story (why naive item-item CF initially failed on this
dataset and how it was fixed), the hybrid-weight/parameter tuning results,
several counter-intuitive findings (ALS's accuracy win comes with the
highest popularity bias of any personalized model; raw tag counts beat
TF-IDF; MMR re-ranking fixes ALS's diversity but not its popularity bias;
Graph Diffusion replays the same accuracy/popularity-bias trade-off as ALS
vs. CF, one layer up in the social graph; tuning ALS's `reg`/`α` is a real
but mixed result, and the ALS+Graph-Diffusion hybrid's validation-set win
reverses, significantly, on the held-out test set), and a discussion of
limitations.
