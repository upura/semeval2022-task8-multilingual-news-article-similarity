import numpy as np
import pandas as pd

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

    oof_sub = oof_bert.copy()
    oof_sub["y_pred"] = oof_bert["y_pred"] * 0.4 + oof_mbert["y_pred"] * 0.6
    print(oof_sub.corr())

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
    y_test_pred = y_test_pred_bert * 0.4 + y_test_pred_mbert * 0.6

    sub = pd.read_csv(
        "../input/semeval2022/PUBLIC-semeval-2022_task8_eval_data_202201.csv"
    )
    sub["Overall"] = np.nan
    sub.loc[sub["pair_id"].isin(rule_based_pair_ids), "Overall"] = 2.8
    sub.loc[~sub["pair_id"].isin(rule_based_pair_ids), "Overall"] = y_test_pred
    sub[["pair_id", "Overall"]].to_csv("submission.csv", index=False)
    sub[["pair_id", "Overall"]].head(2)
