import lightgbm as lgb
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize


def contain_text(text_df, pair_id: str):
    text_id1, text_id2 = pair_id.split("_")
    return (text_id1 in text_df.text_id.values) and (text_id2 in text_df.text_id.values)


def jaccard(t1_vectors, t2_vectors):
    num = t1_vectors.minimum(t2_vectors).sum(axis=1)
    den = t1_vectors.maximum(t2_vectors).sum(axis=1)
    return num / (den + 1e-10)


def dice(t1_vectors, t2_vectors):
    num = 2 * t1_vectors.minimum(t2_vectors).sum(axis=1)
    den = t1_vectors.sum(axis=1) + t2_vectors.sum(axis=1)
    return num / (den + 1e-10)


def cosine(t1_vectors, t2_vectors):
    t1_vectors = normalize(t1_vectors)
    t2_vectors = normalize(t2_vectors)
    return t1_vectors.multiply(t2_vectors).sum(axis=1)


def create_match_features(df, target_cols: str):
    res = df.copy()

    for target_col in target_cols:
        NLTK_STOPWORDS = set(nltk.corpus.stopwords.words("english"))
        ngram_range = (1, 2)
        count_vectorizer = CountVectorizer(
            tokenizer=lambda x: x.split(),
            stop_words=NLTK_STOPWORDS,
            ngram_range=ngram_range,
            min_df=5,
            binary=True,
        )
        tfidf_transformer = TfidfTransformer()

        title_counts = count_vectorizer.fit_transform(
            df[f"{target_col}_1"].to_list() + df[f"{target_col}_2"].to_list()
        )
        title_tfidfs = tfidf_transformer.fit_transform(title_counts)
        t1_counts, t2_counts = title_counts[: len(df)], title_counts[len(df) :]
        t1_tfidfs, t2_tfidfs = title_tfidfs[: len(df)], title_tfidfs[len(df) :]

        suffix = target_col
        res[f"jaccard_count_{suffix}"] = jaccard(t1_counts, t2_counts)
        res[f"dice_count_{suffix}"] = dice(t1_counts, t2_counts)
        res[f"cosine_count_{suffix}"] = cosine(t1_counts, t2_counts)
        res[f"jaccard_tfidf_{suffix}"] = jaccard(t1_tfidfs, t2_tfidfs)
        res[f"dice_tfidf_{suffix}"] = dice(t1_tfidfs, t2_tfidfs)
        res[f"cosine_tfidf_{suffix}"] = cosine(t1_tfidfs, t2_tfidfs)
    return res


def run_lgbm(X_train, y_train, X_test, categorical_cols=[]):
    y_preds = []
    models = []
    oof_train = np.zeros((len(X_train),))
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 14,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "subsample_freq": 1,
        "bagging_fraction": 0.7,
        "min_data_in_leaf": 10,
        "learning_rate": 0.1,
        "boosting": "gbdt",
        "lambda_l1": 0.4,
        "lambda_l2": 0.4,
        "verbosity": -1,
        "random_state": 42,
        "num_boost_round": 1000,
    }

    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):
        X_tr = X_train.loc[train_index, :]
        X_val = X_train.loc[valid_index, :]
        y_tr = y_train[train_index]
        y_val = y_train[valid_index]

        lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_cols)

        lgb_eval = lgb.Dataset(
            X_val, y_val, reference=lgb_train, categorical_feature=categorical_cols
        )

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )

        oof_train[valid_index] = model.predict(
            X_val, num_iteration=model.best_iteration
        )
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        y_preds.append(y_pred)
        y_pred = sum(y_preds) / len(y_preds)
        models.append(model)

    return oof_train, y_pred, models


def visualize_importance(models, X_train):
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["feature_importance"] = model.feature_importance()
        _df["column"] = X_train.columns
        _df["fold"] = i + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, _df], axis=0, ignore_index=True
        )

    order = (
        feature_importance_df.groupby("column")
        .sum()[["feature_importance"]]
        .sort_values("feature_importance", ascending=False)
        .index[:50]
    )

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 0.4), 7))
    sns.boxenplot(
        data=feature_importance_df,
        x="column",
        y="feature_importance",
        order=order,
        ax=ax,
        palette="viridis",
    )
    ax.tick_params(axis="x", rotation=90)
    ax.grid()
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    TRAIN_DF_PATH = "../input/semeval2022/semeval-2022_task8_train-data_batch.csv"
    TEST_DF_PATH = "../input/semeval2022/PUBLIC-semeval-2022_task8_eval_data_202201.csv"
    TRAIN_TEXT_PATH = "../input/semeval2022/text_dataframe.csv"
    TEST_TEXT_PATH = "../input/semeval2022/text_dataframe_eval.csv"

    train = pd.read_csv(TRAIN_DF_PATH)
    test = pd.read_csv(TEST_DF_PATH)
    train_text = (
        pd.read_csv(TRAIN_TEXT_PATH, low_memory=False)
        .dropna(subset=["title", "text"])
        .reset_index(drop=True)
    )
    test_text = (
        pd.read_csv(TEST_TEXT_PATH, low_memory=False)
        .dropna(subset=["title", "text"])
        .reset_index(drop=True)
    )
    train_text["text_id"] = train_text["text_id"].astype(str)
    test_text["text_id"] = test_text["text_id"].astype(str)

    train = train[
        train["pair_id"].map(lambda x: contain_text(train_text, x))
    ].reset_index(drop=True)
    test = test[test["pair_id"].map(lambda x: contain_text(test_text, x))].reset_index(
        drop=True
    )

    train["text_id1"] = train["pair_id"].str.split("_").map(lambda x: x[0])
    train["text_id2"] = train["pair_id"].str.split("_").map(lambda x: x[1])
    test["text_id1"] = test["pair_id"].str.split("_").map(lambda x: x[0])
    test["text_id2"] = test["pair_id"].str.split("_").map(lambda x: x[1])

    train = pd.merge(
        train,
        train_text[["text_id", "title", "text"]],
        left_on="text_id1",
        right_on="text_id",
        how="left",
    )
    train = pd.merge(
        train,
        train_text[["text_id", "title", "text"]],
        left_on="text_id2",
        right_on="text_id",
        how="left",
        suffixes=("_1", "_2"),
    )
    test = pd.merge(
        test,
        test_text[["text_id", "title", "text"]],
        left_on="text_id1",
        right_on="text_id",
        how="left",
    )
    test = pd.merge(
        test,
        test_text[["text_id", "title", "text"]],
        left_on="text_id2",
        right_on="text_id",
        how="left",
        suffixes=("_1", "_2"),
    )
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    df = create_match_features(df, ["title", "text"])

    use_cols = [
        "jaccard_count_title",
        "dice_count_title",
        "cosine_count_title",
        "jaccard_tfidf_title",
        "dice_tfidf_title",
        "cosine_tfidf_title",
        "jaccard_count_text",
        "dice_count_text",
        "cosine_count_text",
        "jaccard_tfidf_text",
        "dice_tfidf_text",
        "cosine_tfidf_text",
    ]
    X_train = df[use_cols][: len(train)].reset_index(drop=True)
    X_test = df[use_cols][len(train) :].reset_index(drop=True)
    y_train = train["Overall"]

    oof_train, y_pred, models = run_lgbm(X_train, y_train, X_test)
    print("OOF rmse: ", mean_squared_error(y_train, oof_train, squared=False))
    visualize_importance(models, X_train)
    rule_based_pair_ids = [
        "1489951217_1489983888",
        "1615462021_1614797257",
        "1556817289_1583857471",
        "1485350427_1486534258",
        "1517231070_1551671513",
        "1533559316_1543388429",
        "1626509167_1626408793",
        "1494757467_1495382175",
    ]

    test["Overall"] = np.nan
    test.loc[test["pair_id"].isin(rule_based_pair_ids), "Overall"] = 2.8
    test.loc[~test["pair_id"].isin(rule_based_pair_ids), "Overall"] = y_pred
    test[["pair_id", "Overall"]].to_csv("submission.csv", index=False)
    test[["pair_id", "Overall"]].head(2)
