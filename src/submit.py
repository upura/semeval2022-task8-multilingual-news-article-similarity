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

    oof_mbert_split20 = pd.concat(
        [
            pd.read_csv("./split20/oof_fold0.csv"),
            pd.read_csv("./split20/oof_fold1.csv"),
            pd.read_csv("./split20/oof_fold2.csv"),
            pd.read_csv("./split20/oof_fold3.csv"),
            pd.read_csv("./split20/oof_fold4.csv"),
            pd.read_csv("./split20/oof_fold5.csv"),
            pd.read_csv("./split20/oof_fold6.csv"),
            pd.read_csv("./split20/oof_fold7.csv"),
            pd.read_csv("./split20/oof_fold8.csv"),
            pd.read_csv("./split20/oof_fold9.csv"),
            pd.read_csv("./split20/oof_fold10.csv"),
            pd.read_csv("./split20/oof_fold11.csv"),
            pd.read_csv("./split20/oof_fold12.csv"),
            pd.read_csv("./split20/oof_fold13.csv"),
            pd.read_csv("./split20/oof_fold14.csv"),
            pd.read_csv("./split20/oof_fold15.csv"),
            pd.read_csv("./split20/oof_fold16.csv"),
            pd.read_csv("./split20/oof_fold17.csv"),
            pd.read_csv("./split20/oof_fold18.csv"),
            pd.read_csv("./split20/oof_fold19.csv"),
        ]
    )
    print(oof_mbert_split20.corr())

    X_train = pd.DataFrame()
    X_train["pair_id"] = oof_bert["pair_id"]
    X_train["oof_bert"] = oof_bert["y_pred"]
    X_train["oof_mbert"] = oof_mbert["y_pred"]
    X_train["oof_mbert_cased"] = oof_mbert_cased["y_pred"]
    print(X_train.shape)
    X_train = pd.merge(
        X_train.drop_duplicates(subset=["pair_id"]).reset_index(drop=True),
        oof_mbert_split20[["pair_id", "y_pred"]]
        .drop_duplicates(subset=["pair_id"])
        .reset_index(drop=True),
        on="pair_id",
        how="inner",
    ).drop("pair_id", axis=1)
    print(X_train.shape)
    print(X_train.corr())

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

    y_test_pred_mbert_split20 = (
        (
            np.load("./split20/y_test_pred_fold0.npy")
            + np.load("./split20/y_test_pred_fold1.npy")
            + np.load("./split20/y_test_pred_fold2.npy")
            + np.load("./split20/y_test_pred_fold3.npy")
            + np.load("./split20/y_test_pred_fold4.npy")
            + np.load("./split20/y_test_pred_fold5.npy")
            + np.load("./split20/y_test_pred_fold6.npy")
            + np.load("./split20/y_test_pred_fold7.npy")
            + np.load("./split20/y_test_pred_fold8.npy")
            + np.load("./split20/y_test_pred_fold9.npy")
            + np.load("./split20/y_test_pred_fold10.npy")
            + np.load("./split20/y_test_pred_fold11.npy")
            + np.load("./split20/y_test_pred_fold12.npy")
            + np.load("./split20/y_test_pred_fold13.npy")
            + np.load("./split20/y_test_pred_fold14.npy")
            + np.load("./split20/y_test_pred_fold15.npy")
            + np.load("./split20/y_test_pred_fold16.npy")
            + np.load("./split20/y_test_pred_fold17.npy")
            + np.load("./split20/y_test_pred_fold18.npy")
            + np.load("./split20/y_test_pred_fold19.npy")
        )
        / 20
    ).reshape(-1)

    y_train = oof_bert.drop_duplicates(subset=["pair_id"]).reset_index(drop=True)[
        "Overall"
    ]
    X_train["y_pred_avg"] = (
        X_train["oof_bert"] * 0.2
        + X_train["oof_mbert"] * 0.2
        + X_train["oof_mbert_cased"] * 0.3
        + X_train["y_pred"] * 0.3
    )
    X_train["y_true"] = y_train
    print(X_train.corr())

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

    y_test_pred = (
        y_test_pred_bert * 0.2
        + y_test_pred_mbert * 0.2
        + y_test_pred_mbert_cased * 0.3
        + y_test_pred_mbert_split20 * 0.3
    )

    sub = pd.read_csv(
        "../input/semeval2022/PUBLIC-semeval-2022_task8_eval_data_202201.csv"
    )
    sub["Overall"] = np.nan
    sub.loc[sub["pair_id"].isin(rule_based_pair_ids), "Overall"] = 2.8
    sub.loc[~sub["pair_id"].isin(rule_based_pair_ids), "Overall"] = y_test_pred
    # Because the labels of training data are reversed at the initial release
    sub["Overall"] = sub["Overall"] * -1
    sub[["pair_id", "Overall"]].to_csv("submission.csv", index=False)
    sub[["pair_id", "Overall"]].head(2)
