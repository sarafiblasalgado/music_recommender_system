"""Matrix factorization for implicit feedback: confidence-weighted ALS.

The explicit-rating version of this algorithm (predict r_hat = mu + b_u +
b_i + p_u.q_i, trained with plain SGD on observed ratings) doesn't fit
implicit play-count data well: there are no negative examples to learn
from -- every observed entry is "the user listened", and everything else
is just *unknown*, not "the user dislikes this". Treating unobserved
entries as if they were negative ratings (as plain SGD on missing-as-zero
would) overwhelms the signal, since 17,632 artists x 1,892 users is >99%
unobserved.

We instead implement the standard implicit-feedback model of
Hu, Koren & Volinsky (2008), "Collaborative Filtering for Implicit
Feedback Datasets", trained with Alternating Least Squares (ALS):

    preference   p_ui = 1 if the user played artist i at all, else 0
    confidence   c_ui = 1 + alpha * log1p(play_count_ui)
    model        p_hat_ui = x_u . y_i

ALS alternately fixes one factor matrix and solves a *weighted* ridge
regression in closed form for the other -- every unobserved (u, i) pair
still contributes to the loss (with confidence 1, preference 0), so the
model is explicitly told "probably not interested" for everything a user
hasn't played, weighted much more lightly than the things they did play.
This is exactly the implicit-feedback case the assignment's matrix
factorization placeholder gestures at with "Option B: SGD" -- ALS is the
closed-form sibling that turns out to be both more appropriate for
implicit data and faster to train here, since each per-user/per-item
update is an exact small linear solve, not a noisy gradient step.

The closed-form update exploits the standard trick from the paper:
  Y^T C^u Y = Y^T Y + Y^T (C^u - I) Y,   where C^u - I is zero outside the
items user u actually played -- so each update only touches the (small)
set of items/users that were actually observed, not the full catalog.
"""

import numpy as np

from . import config
from .data_loading import get_seen_items


class ImplicitALSRecommender:
    """Confidence-weighted ALS for implicit listening data."""

    def __init__(self, n_factors=50, n_iterations=15, reg=0.1,
                 alpha=config.IMPLICIT_ALPHA, random_state=config.RANDOM_STATE, verbose=False):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg = reg
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose

        self.X_ = None  # user factors, (n_users, n_factors)
        self.Y_ = None  # item factors, (n_items, n_factors)
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None
        self.train_loss_curve_ = []

    def fit(self, interactions):
        rng = np.random.RandomState(self.random_state)

        self.user_ids_ = np.sort(interactions[config.USER_COL].unique())
        self.item_ids_ = np.sort(interactions[config.ITEM_COL].unique())
        self.user_id_to_index_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.item_id_to_index_ = {m: i for i, m in enumerate(self.item_ids_)}
        n_users, n_items, f = len(self.user_ids_), len(self.item_ids_), self.n_factors

        rows = interactions[config.USER_COL].map(self.user_id_to_index_).to_numpy()
        cols = interactions[config.ITEM_COL].map(self.item_id_to_index_).to_numpy()
        confidence = 1.0 + self.alpha * interactions["log_weight"].to_numpy()

        # CSR (user-major) for user updates, CSR-of-transpose (item-major) for item updates.
        from scipy.sparse import csr_matrix
        R = csr_matrix((confidence, (rows, cols)), shape=(n_users, n_items))
        RT = R.T.tocsr()

        self.X_ = rng.normal(0, 0.1, size=(n_users, f))
        self.Y_ = rng.normal(0, 0.1, size=(n_items, f))
        reg_I = self.reg * np.eye(f)
        self.train_loss_curve_ = []

        for iteration in range(self.n_iterations):
            # --- update user factors, item factors fixed ---
            YtY = self.Y_.T @ self.Y_
            for u in range(n_users):
                start, end = R.indptr[u], R.indptr[u + 1]
                idx, c = R.indices[start:end], R.data[start:end]
                if idx.size == 0:
                    self.X_[u] = 0.0
                    continue
                Yu = self.Y_[idx]
                A = YtY + Yu.T @ ((c - 1.0)[:, None] * Yu) + reg_I
                b = Yu.T @ c
                self.X_[u] = np.linalg.solve(A, b)

            # --- update item factors, user factors fixed ---
            XtX = self.X_.T @ self.X_
            for i in range(n_items):
                start, end = RT.indptr[i], RT.indptr[i + 1]
                idx, c = RT.indices[start:end], RT.data[start:end]
                if idx.size == 0:
                    self.Y_[i] = 0.0
                    continue
                Xi = self.X_[idx]
                A = XtX + Xi.T @ ((c - 1.0)[:, None] * Xi) + reg_I
                b = Xi.T @ c
                self.Y_[i] = np.linalg.solve(A, b)

            pred = np.einsum("ij,ij->i", self.X_[rows], self.Y_[cols])
            # Loss on observed entries only (cheap diagnostic, not the full
            # implicit objective which also sums over all unobserved pairs).
            loss = float(np.mean(confidence * (1.0 - pred) ** 2))
            self.train_loss_curve_.append(loss)
            if self.verbose:
                print(f"  [ALS] iteration {iteration + 1}/{self.n_iterations}  "
                      f"observed-entry loss = {loss:.4f}")

        return self

    def predict_score(self, user_id, item_id):
        u = self.user_id_to_index_.get(user_id)
        i = self.item_id_to_index_.get(item_id)
        if u is None or i is None:
            return 0.0
        return float(self.X_[u].dot(self.Y_[i]))

    def recommend(self, user_id, interactions_train, n=10, exclude_seen=True):
        if self.X_ is None:
            raise RuntimeError("Call fit() before recommend().")
        u = self.user_id_to_index_.get(user_id)
        if u is None:
            return []

        scores = self.Y_.dot(self.X_[u])
        seen = get_seen_items(interactions_train, user_id) if exclude_seen else set()
        order = np.argsort(-scores)

        recs = []
        for idx in order:
            item_id = int(self.item_ids_[idx])
            if item_id in seen:
                continue
            recs.append((item_id, float(scores[idx])))
            if len(recs) >= n:
                break
        return recs
