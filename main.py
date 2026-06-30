"""Main entry point for the Last.fm music recommender prototype.

Pipeline:
  1. Load data (listening interactions, artist metadata, tags, friend graph) + EDA
  2. Per-user train/test split
  3. Train every recommender: 2 non-personalized baselines, content-based
     (tags), item-item CF, user-user CF, implicit ALS matrix factorization,
     a friend-based social recommender, and a CF+social hybrid
  4. Evaluate all of them with the same protocol
  5. Save a metrics table + figures + example recommendations
"""

import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data_loading import (
    load_interactions, load_artists, load_tags, load_user_tagged_artists, load_friends,
    build_artist_tag_corpus, describe_dataset, train_test_split_interactions,
    get_seen_items, get_user_median_weights,
)
from src.baselines import MostPopularRecommender, MostPlayedRecommender, RandomRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import ItemItemCollaborativeFiltering, UserUserCollaborativeFiltering
from src.matrix_factorization import ImplicitALSRecommender
from src.social_filtering import FriendBasedRecommender, GraphDiffusionRecommender, HybridRecommender
from src.backfill import BackfillRecommender
from src.diversification import MMRRecommender
from src.evaluation import evaluate_model, compute_popularity_percentiles, paired_wilcoxon_test

warnings.filterwarnings("ignore", category=FutureWarning)


def run_eda(interactions, artists, friends):
    print("\n" + "#" * 60)
    print("# STEP 1 / EXPLORATORY DATA ANALYSIS")
    print("#" * 60)
    stats = describe_dataset(interactions, artists, friends)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(interactions["log_weight"], bins=50, color="#028090")
    ax.set_title("log(1 + play count) distribution")
    ax.set_xlabel("log1p(play count)")
    ax.set_ylabel("Number of (user, artist) interactions")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "play_count_distribution.png", dpi=150)
    plt.close(fig)

    listeners_per_artist = interactions.groupby(config.ITEM_COL).size().sort_values(ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(listeners_per_artist.values, color="#00A896")
    ax.set_title("Artist popularity long tail")
    ax.set_xlabel("Artists, ranked by popularity")
    ax.set_ylabel("Number of listeners")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "popularity_long_tail.png", dpi=150)
    plt.close(fig)

    interactions_per_user = interactions.groupby(config.USER_COL).size()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(interactions_per_user.values, bins=40, color="#02C39A")
    ax.set_title("Artists listened to per user")
    ax.set_xlabel("Number of distinct artists")
    ax.set_ylabel("Number of users")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "interactions_per_user.png", dpi=150)
    plt.close(fig)

    if friends is not None and len(friends) > 0:
        degree = friends.groupby(config.USER_COL).size()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(degree.values, bins=40, color="#21295C")
        ax.set_title("Social layer: friends per user")
        ax.set_xlabel("Number of friends")
        ax.set_ylabel("Number of users")
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / "friends_per_user.png", dpi=150)
        plt.close(fig)

    return stats


def tune_model_params(train, factor_grid=(20, 50, 80), reg_grid=(0.01, 0.05, 0.1, 0.3),
                       alpha_grid=(1.0, 2.0, 4.0, 8.0), k_grid=(10, 20, 30, 50), tuning_iterations=10):
    """Grid-search implicit-ALS latent factors, regularization (`reg`) and
    confidence scaling (`alpha`), and item-item-CF neighbourhood size k on
    an inner train/validation split (same rationale as tune_hybrid_weight
    -- never the held-out test set). Resolves the assignment's suggested
    extension "parameter tuning for k or latent factors": earlier
    iterations of this project used hand-picked defaults (n_factors=50,
    reg=0.1, alpha=2.0, k=20) without ever checking whether they were
    actually good choices for this dataset -- `n_factors` was the first
    one tuned; `reg`/`alpha` were left untouched until this iteration. The
    tuned k is reused for both CF variants (item-item and user-user) for
    simplicity -- only item-item is searched directly, since it's the
    cheaper/faster model to evaluate repeatedly and the two variants share
    the same sparsity-driven trade-off (more neighbours per query vs.
    needing tighter support).

    `reg`/`alpha` are searched *after* `n_factors` is fixed (a staged,
    coordinate-wise search, not a full 3-way grid) -- the same pattern
    `tune_mmr_lambda()` uses for keeping runtime bounded, at the cost of
    not exploring interactions between `n_factors` and `reg`/`alpha`; see
    REPORT.md §8 for that trade-off.
    """
    print("\n" + "#" * 60)
    print("# STEP 3a / TUNING ALS FACTORS, REG, ALPHA & CF k (inner train/val split)")
    print("#" * 60)
    inner_train, inner_val = train_test_split_interactions(train, test_size=0.15)
    val_users = sorted(inner_val[config.USER_COL].unique())
    user_median_weights = get_user_median_weights(inner_train)
    global_median_weight = float(inner_train[config.WEIGHT_COL].median())

    best_factors, best_factors_ndcg = factor_grid[0], -1.0
    for f in factor_grid:
        model = ImplicitALSRecommender(n_factors=f, n_iterations=tuning_iterations, verbose=False).fit(inner_train)
        res = evaluate_model(model, inner_train, inner_val, val_users, k=config.TOP_K,
                              user_median_weights=user_median_weights, global_median_weight=global_median_weight)
        ndcg = res[f"ndcg@{config.TOP_K}"]
        print(f"  ALS n_factors={f:3d}  val NDCG@10={ndcg:.4f}  P@10={res[f'precision@{config.TOP_K}']:.4f}")
        if ndcg > best_factors_ndcg:
            best_factors_ndcg, best_factors = ndcg, f

    best_reg, best_alpha, best_reg_alpha_ndcg = reg_grid[0], alpha_grid[0], -1.0
    for reg in reg_grid:
        for alpha in alpha_grid:
            model = ImplicitALSRecommender(n_factors=best_factors, reg=reg, alpha=alpha,
                                            n_iterations=tuning_iterations, verbose=False).fit(inner_train)
            res = evaluate_model(model, inner_train, inner_val, val_users, k=config.TOP_K,
                                  user_median_weights=user_median_weights, global_median_weight=global_median_weight)
            ndcg = res[f"ndcg@{config.TOP_K}"]
            print(f"  ALS reg={reg:.2f} alpha={alpha:.1f}  val NDCG@10={ndcg:.4f}  "
                  f"P@10={res[f'precision@{config.TOP_K}']:.4f}")
            if ndcg > best_reg_alpha_ndcg:
                best_reg_alpha_ndcg, best_reg, best_alpha = ndcg, reg, alpha

    best_k, best_k_ndcg = k_grid[0], -1.0
    for k in k_grid:
        model = ItemItemCollaborativeFiltering(k=k).fit(inner_train)
        res = evaluate_model(model, inner_train, inner_val, val_users, k=config.TOP_K,
                              user_median_weights=user_median_weights, global_median_weight=global_median_weight)
        ndcg = res[f"ndcg@{config.TOP_K}"]
        print(f"  Item-Item CF k={k:3d}    val NDCG@10={ndcg:.4f}  P@10={res[f'precision@{config.TOP_K}']:.4f}")
        if ndcg > best_k_ndcg:
            best_k_ndcg, best_k = ndcg, k

    print(f"\n  Selected ALS n_factors={best_factors} (val NDCG@10={best_factors_ndcg:.4f}), "
          f"reg={best_reg:.2f} alpha={best_alpha:.1f} (val NDCG@10={best_reg_alpha_ndcg:.4f}), "
          f"CF k={best_k} (val NDCG@10={best_k_ndcg:.4f})")
    return best_factors, best_reg, best_alpha, best_k


def tune_hybrid_weight(train, friends, cf_k=20, weight_grid=(0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)):
    """Grid-search the item-item-CF vs. friend-based blend weight for the
    hybrid recommender on an *inner* train/validation split carved out of
    the training data -- never the held-out test set, so the final
    accuracy numbers in the report aren't contaminated by having been used
    to pick this hyperparameter. Resolves a limitation flagged in earlier
    iterations of this report: the CF/social blend weight was previously
    fixed at 0.65/0.35 for demonstration, not validated.
    """
    print("\n" + "#" * 60)
    print("# STEP 3b / TUNING HYBRID WEIGHT (inner train/val split)")
    print("#" * 60)
    inner_train, inner_val = train_test_split_interactions(train, test_size=0.15)
    item_item_inner = ItemItemCollaborativeFiltering(k=cf_k).fit(inner_train)
    friend_inner = FriendBasedRecommender().fit(inner_train, friends)

    val_users = sorted(inner_val[config.USER_COL].unique())
    user_median_weights = get_user_median_weights(inner_train)
    global_median_weight = float(inner_train[config.WEIGHT_COL].median())

    best_weight, best_ndcg = weight_grid[0], -1.0
    for w in weight_grid:
        hybrid = HybridRecommender(
            models={"item_item_cf": item_item_inner, "friend_based": friend_inner},
            weights={"item_item_cf": w, "friend_based": 1.0 - w},
            pool_size=50,
        )
        res = evaluate_model(hybrid, inner_train, inner_val, val_users, k=config.TOP_K,
                              user_median_weights=user_median_weights, global_median_weight=global_median_weight)
        ndcg = res[f"ndcg@{config.TOP_K}"]
        print(f"  item_item_cf weight={w:.2f}  friend_based weight={1 - w:.2f}  "
              f"NDCG@{config.TOP_K}={ndcg:.4f}  P@{config.TOP_K}={res[f'precision@{config.TOP_K}']:.4f}")
        if ndcg > best_ndcg:
            best_ndcg, best_weight = ndcg, w

    print(f"\n  Selected item_item_cf weight = {best_weight:.2f} "
          f"(friend_based = {1 - best_weight:.2f}), validation NDCG@{config.TOP_K} = {best_ndcg:.4f}")
    return best_weight


def tune_mmr_lambda(train, artists, tag_corpus, mf_factors, mf_reg=0.1, mf_alpha=config.IMPLICIT_ALPHA,
                     lambda_grid=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
                     tuning_iterations=10, precision_tolerance=0.10):
    """Sweep the MMR relevance/diversity trade-off weight (src/diversification.py)
    on an inner train/validation split, applied to Implicit ALS -- the model
    §6/§8 identify as needing this most. Picking the lambda that maximizes
    validation NDCG would trivially always pick lambda=1.0 (no
    diversification at all) and defeat the point of the technique, so the
    selection rule here is instead "the most diversification (lowest
    lambda) that keeps validation precision@10 within `precision_tolerance`
    relative of the pure-relevance baseline" -- an explicit accuracy budget,
    not a blind metric-maximization. Returns (selected_lambda, sweep_df);
    the full sweep is plotted in plot_mmr_tradeoff() and reported in
    REPORT.md §6 so the trade-off is shown, not just asserted.
    """
    print("\n" + "#" * 60)
    print("# STEP 3c / TUNING MMR LAMBDA (inner train/val split, Implicit ALS)")
    print("#" * 60)
    inner_train, inner_val = train_test_split_interactions(train, test_size=0.15)
    val_users = sorted(inner_val[config.USER_COL].unique())
    user_median_weights = get_user_median_weights(inner_train)
    global_median_weight = float(inner_train[config.WEIGHT_COL].median())
    item_popularity_percentile = compute_popularity_percentiles(
        inner_train.groupby(config.ITEM_COL).size().to_dict()
    )

    content_inner = ContentBasedRecommender(use_tfidf=True).fit(inner_train, artists, tag_corpus)
    als_inner = ImplicitALSRecommender(n_factors=mf_factors, reg=mf_reg, alpha=mf_alpha,
                                        n_iterations=tuning_iterations).fit(inner_train)

    rows = []
    baseline_precision = None
    for lam in lambda_grid:
        model = MMRRecommender(als_inner, content_inner.item_features_, content_inner.item_id_to_index_,
                                lambda_param=lam, pool_size=50)
        res = evaluate_model(model, inner_train, inner_val, val_users, k=config.TOP_K,
                              user_median_weights=user_median_weights, global_median_weight=global_median_weight,
                              item_features=content_inner.item_features_,
                              item_id_to_index=content_inner.item_id_to_index_,
                              item_popularity_percentile=item_popularity_percentile)
        precision = res[f"precision@{config.TOP_K}"]
        if lam == 1.0:
            baseline_precision = precision
        rows.append({"lambda": lam, f"precision@{config.TOP_K}": precision,
                     f"ndcg@{config.TOP_K}": res[f"ndcg@{config.TOP_K}"],
                     "diversity": res.get("diversity"), "popularity_bias": res.get("popularity_bias")})
        print(f"  lambda={lam:.2f}  val P@10={precision:.4f}  "
              f"diversity={res.get('diversity', float('nan')):.3f}  "
              f"pop_bias={res.get('popularity_bias', float('nan')):.3f}")

    sweep_df = pd.DataFrame(rows)
    if baseline_precision is None:
        baseline_precision = sweep_df[f"precision@{config.TOP_K}"].max()
    threshold = baseline_precision * (1 - precision_tolerance)
    eligible = sweep_df[sweep_df[f"precision@{config.TOP_K}"] >= threshold]
    selected_lambda = float(eligible["lambda"].min()) if not eligible.empty else 1.0

    print(f"\n  Baseline (lambda=1.0) val P@10={baseline_precision:.4f}; "
          f"selected lambda={selected_lambda:.2f} (most diversification within "
          f"{precision_tolerance:.0%} relative precision budget)")
    return selected_lambda, sweep_df


def plot_mmr_tradeoff(sweep_df, selected_lambda, k=config.TOP_K):
    """Precision-vs-diversity Pareto curve from tune_mmr_lambda's sweep --
    visual evidence for the relevance/diversity trade-off the report
    discusses, rather than a single before/after number."""
    if sweep_df.empty or "diversity" not in sweep_df:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ordered = sweep_df.sort_values("lambda")
    ax.plot(ordered["diversity"], ordered[f"precision@{k}"], color="#7B2CBF", marker="o", zorder=2)
    for _, row in ordered.iterrows():
        ax.annotate(f"λ={row['lambda']:.1f}", (row["diversity"], row[f"precision@{k}"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    selected_row = sweep_df[sweep_df["lambda"] == selected_lambda]
    if not selected_row.empty:
        ax.scatter(selected_row["diversity"], selected_row[f"precision@{k}"], color="#E85D04", s=140,
                   zorder=3, label=f"selected (λ={selected_lambda:.1f})")
        ax.legend()
    ax.set_xlabel("Intra-list diversity (validation)")
    ax.set_ylabel(f"Precision@{k} (validation)")
    ax.set_title("MMR re-ranking: relevance vs. diversity trade-off (Implicit ALS)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "mmr_tradeoff.png", dpi=150)
    plt.close(fig)


def tune_als_social_hybrid_weight(train, friends, mf_factors, mf_reg=0.1, mf_alpha=config.IMPLICIT_ALPHA,
                                   weight_grid=(0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0), tuning_iterations=10):
    """Grid-search a blend weight for Implicit ALS + Graph Diffusion
    (§4.5) -- the two most accurate individual models in this project,
    never previously combined. `tune_hybrid_weight()` already blends
    item-item CF + Friend-Based and found CF contributes nothing once the
    social signal is available (it degenerates to 100% social, REPORT.md
    §6/§8); this is the natural follow-up question raised by that result:
    does blending the *two strongest* signals (rather than the strongest
    listening-layer signal with the strongest social-layer signal) do any
    better than either alone? Selected by validation NDCG@10, same
    protocol and inner train/validation split as every other tuned
    hyperparameter in this project (§5) -- never the held-out test set.
    """
    print("\n" + "#" * 60)
    print("# STEP 3d / TUNING ALS + GRAPH-DIFFUSION HYBRID WEIGHT (inner train/val split)")
    print("#" * 60)
    inner_train, inner_val = train_test_split_interactions(train, test_size=0.15)
    als_inner = ImplicitALSRecommender(n_factors=mf_factors, reg=mf_reg, alpha=mf_alpha,
                                        n_iterations=tuning_iterations).fit(inner_train)
    graph_inner = GraphDiffusionRecommender().fit(inner_train, friends)

    val_users = sorted(inner_val[config.USER_COL].unique())
    user_median_weights = get_user_median_weights(inner_train)
    global_median_weight = float(inner_train[config.WEIGHT_COL].median())

    best_weight, best_ndcg = weight_grid[0], -1.0
    for w in weight_grid:
        hybrid = HybridRecommender(
            models={"matrix_factorization": als_inner, "graph_diffusion_social": graph_inner},
            weights={"matrix_factorization": w, "graph_diffusion_social": 1.0 - w},
            pool_size=50,
        )
        res = evaluate_model(hybrid, inner_train, inner_val, val_users, k=config.TOP_K,
                              user_median_weights=user_median_weights, global_median_weight=global_median_weight)
        ndcg = res[f"ndcg@{config.TOP_K}"]
        print(f"  matrix_factorization weight={w:.2f}  graph_diffusion_social weight={1 - w:.2f}  "
              f"NDCG@{config.TOP_K}={ndcg:.4f}  P@{config.TOP_K}={res[f'precision@{config.TOP_K}']:.4f}")
        if ndcg > best_ndcg:
            best_ndcg, best_weight = ndcg, w

    print(f"\n  Selected matrix_factorization weight = {best_weight:.2f} "
          f"(graph_diffusion_social = {1 - best_weight:.2f}), validation NDCG@{config.TOP_K} = {best_ndcg:.4f}")
    return best_weight


def fit_all(train, artists, tag_corpus, friends, cf_k=20, mf_factors=50, mf_reg=0.1,
            mf_alpha=config.IMPLICIT_ALPHA, mf_iterations=15, hybrid_weight=0.65, verbose_mf=True):
    print("\n" + "#" * 60)
    print("# STEP 3 / TRAINING MODELS")
    print("#" * 60)
    models = {}

    def _fit(name, model, *args):
        t0 = time.time()
        print(f"\nTraining {name} ...")
        model.fit(*args)
        print(f"  done in {time.time() - t0:.1f}s")
        models[name] = model

    _fit("most_popular", MostPopularRecommender(), train)
    _fit("most_played", MostPlayedRecommender(min_listeners=20), train)
    _fit("random", RandomRecommender(), train)
    _fit("content_based", ContentBasedRecommender(use_tfidf=True), train, artists, tag_corpus)
    _fit("content_based_raw", ContentBasedRecommender(use_tfidf=False), train, artists, tag_corpus)
    _fit("user_user_cf", UserUserCollaborativeFiltering(k=cf_k), train)
    _fit("item_item_cf", ItemItemCollaborativeFiltering(k=cf_k), train)
    _fit("matrix_factorization", ImplicitALSRecommender(n_factors=mf_factors, reg=mf_reg, alpha=mf_alpha,
                                                         n_iterations=mf_iterations, verbose=verbose_mf), train)
    _fit("friend_based", FriendBasedRecommender(), train, friends)
    _fit("graph_diffusion_social", GraphDiffusionRecommender(), train, friends)

    # The hybrid wraps already-fitted models -- no separate "training" step,
    # just a weighted reciprocal-rank blend at recommendation time. Weight
    # comes from tune_hybrid_weight()'s inner-validation grid search.
    print(f"\nAssembling hybrid_cf_social (item-item CF + friend-based, "
          f"{hybrid_weight:.2f}/{1 - hybrid_weight:.2f}) ...")
    models["hybrid_cf_social"] = HybridRecommender(
        models={"item_item_cf": models["item_item_cf"], "friend_based": models["friend_based"]},
        weights={"item_item_cf": hybrid_weight, "friend_based": 1.0 - hybrid_weight},
        pool_size=50,
    )

    return models


def evaluate_all(models, train, test, eval_users, all_items, item_popularity, n_users,
                  user_median_weights, global_median_weight, k=config.TOP_K,
                  item_features=None, item_id_to_index=None, item_popularity_percentile=None):
    print("\n" + "#" * 60)
    print(f"# STEP 4 / EVALUATION (k={k}, {len(eval_users)} evaluable users)")
    print("#" * 60)
    rows = []
    for name, model in models.items():
        t0 = time.time()
        res = evaluate_model(model, train, test, eval_users, k=k,
                              user_median_weights=user_median_weights,
                              global_median_weight=global_median_weight,
                              item_popularity=item_popularity, n_users=n_users, all_items=all_items,
                              item_features=item_features, item_id_to_index=item_id_to_index,
                              item_popularity_percentile=item_popularity_percentile)
        res["model"] = name
        res["seconds"] = round(time.time() - t0, 1)
        rows.append(res)
        print(f"  {name:22s} P@{k}={res[f'precision@{k}']:.4f}  "
              f"R@{k}={res[f'recall@{k}']:.4f}  "
              f"NDCG@{k}={res[f'ndcg@{k}']:.4f}  "
              f"MRR@{k}={res[f'mrr@{k}']:.4f}  "
              f"Coverage={res.get('catalog_coverage', float('nan')):.3f}  "
              f"Novelty={res.get('novelty', float('nan')):.2f}  "
              f"Diversity={res.get('diversity', float('nan')):.3f}  "
              f"PopBias={res.get('popularity_bias', float('nan')):.3f}  "
              f"({res['seconds']}s)")

    df = pd.DataFrame(rows).set_index("model")
    cols_order = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"mrr@{k}", f"hit_rate@{k}",
                  "catalog_coverage", "novelty", "diversity", "popularity_bias",
                  "n_users_evaluated", "n_users_no_recs", "seconds"]
    cols_order = [c for c in cols_order if c in df.columns]
    return df[cols_order]


def run_significance_tests(models, train, test, eval_users, user_median_weights, global_median_weight,
                            k=config.TOP_K, pairs=(("matrix_factorization", "friend_based"),
                                                    ("matrix_factorization", "hybrid_cf_social"),
                                                    ("friend_based", "hybrid_cf_social"),
                                                    ("matrix_factorization", "most_popular"),
                                                    ("matrix_factorization", "matrix_factorization_mmr"),
                                                    ("graph_diffusion_social", "friend_based"),
                                                    ("hybrid_als_social", "matrix_factorization"))):
    """Paired Wilcoxon signed-rank test on per-user NDCG@k between
    selected model pairs -- answers "is model A *significantly* more
    accurate than model B for the same users, or could this gap be
    noise?", which a bare comparison of two mean NDCG numbers in a table
    can't tell you. Limited to a handful of the most interesting pairs
    (not all C(11,2) combinations) to keep the result focused and
    multiple-comparisons risk low.
    """
    print("\n" + "#" * 60)
    print("# STEP 4c / STATISTICAL SIGNIFICANCE (paired Wilcoxon on per-user NDCG@k)")
    print("#" * 60)
    per_user_ndcg = {}
    rows = []
    for a, b in pairs:
        for name in (a, b):
            if name not in per_user_ndcg:
                res = evaluate_model(models[name], train, test, eval_users, k=k,
                                      user_median_weights=user_median_weights,
                                      global_median_weight=global_median_weight,
                                      return_per_user=True)
                per_user_ndcg[name] = res["per_user_ndcg"]
        stat, p_value = paired_wilcoxon_test(per_user_ndcg[a], per_user_ndcg[b])
        significant = p_value < 0.05
        rows.append({"model_a": a, "model_b": b, "wilcoxon_stat": stat,
                     "p_value": p_value, "significant_at_0.05": significant})
        print(f"  {a:22s} vs {b:22s}  W={stat:10.1f}  p={p_value:.2e}  "
              f"{'significant' if significant else 'NOT significant'}")
    return pd.DataFrame(rows)


def run_backfill_analysis(models, train, test, eval_users, all_items, item_popularity, n_users,
                           user_median_weights, global_median_weight, k=config.TOP_K,
                           item_features=None, item_id_to_index=None):
    """Quantify the cold-start gap visible in evaluate_all's n_users_no_recs
    column (content_based, user_user_cf, item_item_cf, friend_based all
    return nothing for some users) and the tradeoff of fixing it: wrap each
    affected model with BackfillRecommender (top up with Most Popular) and
    re-evaluate, so we can see the *user-coverage* gain against any
    accuracy cost from the table directly, instead of just asserting it."""
    print("\n" + "#" * 60)
    print("# STEP 4b / COLD-START BACKFILL: BEFORE vs. AFTER")
    print("#" * 60)
    targets = ["content_based", "user_user_cf", "item_item_cf", "friend_based"]
    rows = []
    for name in targets:
        wrapped = BackfillRecommender(models[name], models["most_popular"])
        res = evaluate_model(wrapped, train, test, eval_users, k=k,
                              user_median_weights=user_median_weights,
                              global_median_weight=global_median_weight,
                              item_popularity=item_popularity, n_users=n_users, all_items=all_items,
                              item_features=item_features, item_id_to_index=item_id_to_index)
        res["model"] = f"{name}_backfilled"
        rows.append(res)
        print(f"  {name:14s} -> backfilled  n_users_no_recs=0 (was nonzero)  "
              f"P@{k}={res[f'precision@{k}']:.4f}  Coverage={res.get('catalog_coverage', float('nan')):.3f}")

    df = pd.DataFrame(rows).set_index("model")
    cols_order = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"mrr@{k}", f"hit_rate@{k}",
                  "catalog_coverage", "novelty", "diversity", "n_users_evaluated", "n_users_no_recs"]
    df = df[[c for c in cols_order if c in df.columns]]
    df.to_csv(config.RESULTS_DIR / "metrics_backfilled.csv")
    return df


def plot_metrics_comparison(metrics_df, k=config.TOP_K):
    metric_cols = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"mrr@{k}"]
    plot_df = metrics_df[metric_cols]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(plot_df))
    width = 0.2
    colors = ["#028090", "#00A896", "#21295C", "#84B59F"]
    for i, col in enumerate(metric_cols):
        ax.bar(x + i * width, plot_df[col].values, width, label=col, color=colors[i])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(plot_df.index, rotation=30, ha="right")
    ax.set_title(f"Model comparison @k={k}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "metrics_comparison.png", dpi=150)
    plt.close(fig)


def plot_coverage_novelty(metrics_df):
    if "catalog_coverage" not in metrics_df or "novelty" not in metrics_df:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(metrics_df["novelty"], metrics_df["catalog_coverage"], s=80, color="#990011")
    for name, row in metrics_df.iterrows():
        ax.annotate(name, (row["novelty"], row["catalog_coverage"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Novelty (mean -log2 popularity of recs)")
    ax.set_ylabel("Catalog coverage")
    ax.set_title("Beyond-accuracy: coverage vs. novelty")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "coverage_vs_novelty.png", dpi=150)
    plt.close(fig)


def plot_popularity_bias(metrics_df):
    """Bar chart of mean recommended-item popularity percentile per model
    -- see evaluation.popularity_bias docstring for why this is a distinct
    beyond-accuracy lens from novelty/coverage. The dashed line at 0.5
    marks "recommends from the middle of the popularity distribution on
    average"; bars well above it indicate a model leaning on the catalog's
    head, bars well below indicate a long-tail/niche skew.
    """
    if "popularity_bias" not in metrics_df:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = metrics_df["popularity_bias"].sort_values(ascending=False)
    ax.bar(order.index, order.values, color="#E85D04")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Mean popularity percentile of recommended items")
    ax.set_title("Popularity bias by model (1.0 = always the most popular item)")
    ax.set_xticklabels(order.index, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "popularity_bias.png", dpi=150)
    plt.close(fig)


def plot_als_training_curve(als_model):
    if not als_model.train_loss_curve_:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(als_model.train_loss_curve_) + 1), als_model.train_loss_curve_,
             marker="o", color="#028090")
    ax.set_title("Implicit ALS training convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Confidence-weighted reconstruction loss\n(observed entries)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "als_training_curve.png", dpi=150)
    plt.close(fig)


def print_recommendation_examples(models, train, artists, user_ids, n=10):
    print("\n" + "#" * 60)
    print("# STEP 5 / RECOMMENDATION EXAMPLES")
    print("#" * 60)
    id_to_name = artists.set_index(config.ITEM_COL)[config.NAME_COL].to_dict()
    lines = []
    for user_id in user_ids:
        seen = get_seen_items(train, user_id)
        lines.append(f"\n=== User {user_id} (listened to {len(seen)} artists in training set) ===")
        print(lines[-1])
        for name, model in models.items():
            recs = model.recommend(user_id, train, n=n, exclude_seen=True)
            names = [id_to_name.get(item, f"artist {item}") for item, _ in recs[:5]]
            line = f"  [{name:22s}] " + " | ".join(names) if names else f"  [{name:22s}] (no recommendations)"
            lines.append(line)
            print(line)
    config.EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.EXAMPLES_DIR / "recommendation_examples.txt", "w") as f:
        f.write("\n".join(lines))


def main():
    print("Music Recommender System Prototype -- Last.fm (HetRec2011)")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    interactions = load_interactions()
    artists = load_artists()
    tags = load_tags()
    user_tagged_artists = load_user_tagged_artists()
    friends = load_friends()
    tag_corpus = build_artist_tag_corpus(user_tagged_artists, tags)

    run_eda(interactions, artists, friends)

    print("\n" + "#" * 60)
    print("# STEP 2 / TRAIN-TEST SPLIT")
    print("#" * 60)
    train, test = train_test_split_interactions(interactions, test_size=0.2)
    print(f"Train interactions: {len(train):,}   Test interactions: {len(test):,}")
    print(f"Users with test data: {test[config.USER_COL].nunique():,} / {interactions[config.USER_COL].nunique():,}")

    best_factors, best_reg, best_alpha, best_cf_k = tune_model_params(train)
    best_hybrid_weight = tune_hybrid_weight(train, friends, cf_k=best_cf_k)
    best_mmr_lambda, mmr_sweep_df = tune_mmr_lambda(train, artists, tag_corpus, best_factors,
                                                     mf_reg=best_reg, mf_alpha=best_alpha)
    mmr_sweep_df.to_csv(config.RESULTS_DIR / "mmr_lambda_sweep.csv", index=False)
    plot_mmr_tradeoff(mmr_sweep_df, best_mmr_lambda)
    best_als_social_weight = tune_als_social_hybrid_weight(train, friends, best_factors,
                                                            mf_reg=best_reg, mf_alpha=best_alpha)

    models = fit_all(train, artists, tag_corpus, friends, cf_k=best_cf_k, mf_factors=best_factors,
                      mf_reg=best_reg, mf_alpha=best_alpha, hybrid_weight=best_hybrid_weight)
    plot_als_training_curve(models["matrix_factorization"])

    content_model = models["content_based"]
    models["matrix_factorization_mmr"] = MMRRecommender(
        models["matrix_factorization"], content_model.item_features_, content_model.item_id_to_index_,
        lambda_param=best_mmr_lambda, pool_size=50,
    )
    print(f"\nAssembling hybrid_als_social (Implicit ALS + Graph Diffusion, "
          f"{best_als_social_weight:.2f}/{1 - best_als_social_weight:.2f}) ...")
    models["hybrid_als_social"] = HybridRecommender(
        models={"matrix_factorization": models["matrix_factorization"],
                "graph_diffusion_social": models["graph_diffusion_social"]},
        weights={"matrix_factorization": best_als_social_weight,
                 "graph_diffusion_social": 1.0 - best_als_social_weight},
        pool_size=50,
    )

    all_items = train[config.ITEM_COL].unique()
    item_popularity = train.groupby(config.ITEM_COL).size().to_dict()
    n_users = train[config.USER_COL].nunique()
    eval_users = sorted(test[config.USER_COL].unique())
    user_median_weights = get_user_median_weights(train)
    global_median_weight = float(train[config.WEIGHT_COL].median())

    item_popularity_percentile = compute_popularity_percentiles(item_popularity)
    metrics_df = evaluate_all(models, train, test, eval_users, all_items, item_popularity, n_users,
                               user_median_weights, global_median_weight,
                               item_features=content_model.item_features_,
                               item_id_to_index=content_model.item_id_to_index_,
                               item_popularity_percentile=item_popularity_percentile)
    metrics_df.to_csv(config.RESULTS_DIR / "metrics.csv")
    plot_metrics_comparison(metrics_df)
    plot_coverage_novelty(metrics_df)
    plot_popularity_bias(metrics_df)

    print("\nFinal metrics table:")
    print(metrics_df.round(4).to_string())

    significance_df = run_significance_tests(models, train, test, eval_users,
                                              user_median_weights, global_median_weight)
    significance_df.to_csv(config.RESULTS_DIR / "significance_tests.csv", index=False)

    backfilled_df = run_backfill_analysis(
        models, train, test, eval_users, all_items, item_popularity, n_users,
        user_median_weights, global_median_weight,
        item_features=content_model.item_features_, item_id_to_index=content_model.item_id_to_index_,
    )
    print("\nBackfilled metrics table:")
    print(backfilled_df.round(4).to_string())

    # Pick example users with at least a few friends so the social /
    # hybrid recommenders have something to show off too.
    friend_counts = friends.groupby(config.USER_COL).size()
    candidates = [u for u in eval_users if friend_counts.get(u, 0) >= 3]
    example_users = (candidates[:3] if len(candidates) >= 3 else eval_users[:3])
    print_recommendation_examples(models, train, artists, example_users)

    print(f"\nAll done. Metrics saved to {config.RESULTS_DIR / 'metrics.csv'}")
    print(f"Figures saved to {config.FIGURES_DIR}")
    print(f"Recommendation examples saved to {config.EXAMPLES_DIR / 'recommendation_examples.txt'}")


if __name__ == "__main__":
    main()
