"""Cold-start backfill: a generic wrapper that tops up a recommender's
output with a fallback model's picks whenever the base model can't fill
the full list.

This addresses a gap visible in `results/examples/recommendation_examples.txt`
and in `n_users_no_recs` in `results/metrics.csv`: item-item/user-user CF
return nothing for a user whose neighbours all fail the minimum-support
floor, and the friend-based recommender returns nothing for a user with no
friends. A real product would never show an empty results page -- it would
chain to a generic fallback (typically Most Popular) instead. Wrapping
*after* evaluation lets us still report the unmodified per-algorithm
numbers for the accuracy comparison (see REPORT.md S6), while using the
backfilled version wherever recommendations are actually shown to a user
(the Streamlit app).
"""


class BackfillRecommender:
    """Wraps `base_model`; if it returns fewer than `n` recommendations for
    a user, tops up the list with `fallback_model`'s picks (skipping
    anything already returned or already seen) until there are `n` items
    or the fallback itself runs out.
    """

    def __init__(self, base_model, fallback_model):
        self.base_model = base_model
        self.fallback_model = fallback_model

    def recommend(self, user_id, interactions_train, n=10, exclude_seen=True):
        recs = list(self.base_model.recommend(user_id, interactions_train, n=n, exclude_seen=exclude_seen))
        if len(recs) >= n:
            return recs

        already = {item_id for item_id, _ in recs}
        fallback_recs = self.fallback_model.recommend(
            user_id, interactions_train, n=n + len(already), exclude_seen=exclude_seen
        )
        for item_id, score in fallback_recs:
            if item_id in already:
                continue
            recs.append((item_id, score))
            already.add(item_id)
            if len(recs) >= n:
                break
        return recs
