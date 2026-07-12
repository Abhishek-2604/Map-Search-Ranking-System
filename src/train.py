"""
Trains a LambdaMART model (gradient-boosted trees optimized directly for
ranking metrics like NDCG) using LightGBM's LGBMRanker.

Why LambdaMART and not plain regression/classification?
  - Pointwise models (regress on relevance) don't optimize ranking order directly.
  - LambdaMART is a pairwise/listwise method: it learns from pairs of
    documents *within the same query* ("A should rank above B") and its
    gradients are weighted by how much swapping A/B would change NDCG.
    This is the same family of algorithm used in real search/maps ranking.
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import ndcg_score
import json

FEATURES = [
    "text_match", "distance_km", "rating", "num_ratings",
    "popularity", "is_open", "price_match", "hour",
]


def load_data():
    df = pd.read_csv("/home/claude/map_search_ltr/data/real_search_sessions.csv")
    return df


def split_by_query(df, train_frac=0.8):
    qids = df["query_id"].unique()
    rng = np.random.default_rng(0)
    rng.shuffle(qids)
    n_train = int(len(qids) * train_frac)
    train_q, test_q = set(qids[:n_train]), set(qids[n_train:])
    return df[df.query_id.isin(train_q)].copy(), df[df.query_id.isin(test_q)].copy()


def to_group_sizes(df):
    # LightGBM ranker needs group sizes = number of candidates per query,
    # and rows for the same query must be contiguous.
    df = df.sort_values("query_id")
    groups = df.groupby("query_id").size().values
    return df, groups


def evaluate_ndcg(model, df, k=5):
    scores_all, labels_all = [], []
    ndcgs = []
    for qid, g in df.groupby("query_id"):
        X = g[FEATURES].values
        y = g["relevance"].values
        if len(g) < 2 or y.max() == 0:
            continue
        preds = model.predict(X)
        ndcgs.append(ndcg_score([y], [preds], k=k))
    return float(np.mean(ndcgs))


def main():
    df = load_data()
    train_df, test_df = split_by_query(df)
    train_df, train_groups = to_group_sizes(train_df)
    test_df, test_groups = to_group_sizes(test_df)

    X_train, y_train = train_df[FEATURES], train_df["relevance"]
    X_test, y_test = test_df[FEATURES], test_df["relevance"]

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=120,
        num_leaves=15,
        max_depth=5,
        learning_rate=0.08,
        min_child_samples=15,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        group=train_groups,
        eval_set=[(X_test, y_test)],
        eval_group=[test_groups],
        eval_at=[3, 5, 10],
    )

    print("\n=== Test NDCG ===")
    for k in [3, 5, 10]:
        print(f"NDCG@{k}: {evaluate_ndcg(model, test_df, k=k):.4f}")

    print("\n=== Feature importance (gain) ===")
    importances = dict(zip(FEATURES, model.booster_.feature_importance(importance_type="gain")))
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:15s} {imp:10.1f}")

    # Save the model in LightGBM's native text format (used by inference / demo)
    model.booster_.save_model("/home/claude/map_search_ltr/models/lambdamart.txt")

    # Dump full tree structure as JSON so we can run inference anywhere,
    # including in a browser for the UI demo.
    dumped = model.booster_.dump_model()
    with open("/home/claude/map_search_ltr/models/lambdamart.json", "w") as f:
        json.dump(dumped, f)

    with open("/home/claude/map_search_ltr/models/features.json", "w") as f:
        json.dump(FEATURES, f)

    print("\nSaved model to models/lambdamart.txt and models/lambdamart.json")


if __name__ == "__main__":
    main()
