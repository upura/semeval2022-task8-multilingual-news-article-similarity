import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

if __name__ == "__main__":

    oof_bert = pd.concat(
        [
            pd.read_csv("./bert/oof_fold0.csv"),
            pd.read_csv("./bert/oof_fold1.csv"),
            pd.read_csv("./bert/oof_fold2.csv"),
            pd.read_csv("./bert/oof_fold3.csv"),
            pd.read_csv("./bert/oof_fold4.csv"),
        ]
    )
    print(oof_bert.corr())

    oof_mbert = pd.concat(
        [
            pd.read_csv("./mbert/oof_fold0.csv"),
            pd.read_csv("./mbert/oof_fold1.csv"),
            pd.read_csv("./mbert/oof_fold2.csv"),
            pd.read_csv("./mbert/oof_fold3.csv"),
            pd.read_csv("./mbert/oof_fold4.csv"),
        ]
    )
    print(oof_mbert.corr())

    oof_mbert_cased = pd.concat(
        [
            pd.read_csv("./mbert_cased/oof_fold0.csv"),
            pd.read_csv("./mbert_cased/oof_fold1.csv"),
            pd.read_csv("./mbert_cased/oof_fold2.csv"),
            pd.read_csv("./mbert_cased/oof_fold3.csv"),
            pd.read_csv("./mbert_cased/oof_fold4.csv"),
        ]
    )
    print(oof_mbert_cased.corr())

    y_test_pred_bert = (
        (
            np.load("./bert/y_test_pred_fold0.npy")
            + np.load("./bert/y_test_pred_fold1.npy")
            + np.load("./bert/y_test_pred_fold2.npy")
            + np.load("./bert/y_test_pred_fold3.npy")
            + np.load("./bert/y_test_pred_fold4.npy")
        )
        / 5
    ).reshape(-1)
    y_test_pred_mbert = (
        (
            np.load("./mbert/y_test_pred_fold0.npy")
            + np.load("./mbert/y_test_pred_fold1.npy")
            + np.load("./mbert/y_test_pred_fold2.npy")
            + np.load("./mbert/y_test_pred_fold3.npy")
            + np.load("./mbert/y_test_pred_fold4.npy")
        )
        / 5
    ).reshape(-1)

    y_test_pred_mbert_cased = (
        (
            np.load("./mbert_cased/y_test_pred_fold0.npy")
            + np.load("./mbert_cased/y_test_pred_fold1.npy")
            + np.load("./mbert_cased/y_test_pred_fold2.npy")
            + np.load("./mbert_cased/y_test_pred_fold3.npy")
            + np.load("./mbert_cased/y_test_pred_fold4.npy")
        )
        / 5
    ).reshape(-1)

    X_train = pd.DataFrame(
        {
            "oof_bert": oof_bert["y_pred"],
            "oof_mbert": oof_mbert["y_pred"],
            "oof_mbert_cased": oof_mbert_cased["y_pred"],
        }
    ).reset_index(drop=True)
    X_test = pd.DataFrame(
        {
            "oof_bert": y_test_pred_bert,
            "oof_mbert": y_test_pred_mbert,
            "oof_mbert_cased": y_test_pred_mbert_cased,
        }
    ).reset_index(drop=True)
    y_train = oof_bert["Overall"].reset_index(drop=True)

    y_preds = []
    models = []
    oof_train = np.zeros((len(X_train)))
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):
        X_tr = X_train.loc[train_index, :]
        X_val = X_train.loc[valid_index, :]
        y_tr = y_train[train_index]
        y_val = y_train[valid_index]
        params_r = {"alpha": 10, "random_state": 0}
        model_r = Ridge(**params_r)
        model_r.fit(X_tr, y_tr)
        oof_train[valid_index] = model_r.predict(X_val)
        y_pred = model_r.predict(X_test)
        y_preds.append(y_pred)
        models.append(model_r)

    oof_sub = oof_bert.copy()
    oof_sub["y_pred"] = oof_train
    print(oof_sub.corr())
    y_test_pred = sum(y_preds) / len(y_preds)

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

    sub = pd.read_csv(
        "../input/semeval2022/PUBLIC-semeval-2022_task8_eval_data_202201.csv"
    )
    sub["Overall"] = np.nan
    sub.loc[sub["pair_id"].isin(rule_based_pair_ids), "Overall"] = 2.8
    sub.loc[~sub["pair_id"].isin(rule_based_pair_ids), "Overall"] = y_test_pred
    sub[["pair_id", "Overall"]].to_csv("submission.csv", index=False)
    sub[["pair_id", "Overall"]].head(2)
