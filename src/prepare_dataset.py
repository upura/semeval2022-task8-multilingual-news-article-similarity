import json
import pathlib

import pandas as pd
from tqdm import tqdm


def fetch_dateframe(text_id: str, meta):
    search_results = list(meta.glob(f"*/{text_id}.json"))
    text_path = str(search_results[0])
    with open(text_path) as f:
        text_json = f.read()
    text_json = json.loads(text_json)
    res_df = pd.json_normalize(text_json)
    res_df["text_id"] = text_id
    return res_df


if __name__ == "__main__":
    phase = ["train", "test"]

    if "train" in phase:
        train = pd.read_csv(
            "../input/semeval2022/semeval-2022_task8_train-data_batch.csv"
        )
        text_ids = set(sum(train["pair_id"].str.split("_"), []))
        meta = pathlib.Path("../input/semeval2022")

        text_dfs = []
        for text_id in tqdm(text_ids):
            try:
                res = fetch_dateframe(text_id, meta)
                text_dfs.append(res)
            except IndexError:
                pass
        text_df = pd.concat(text_dfs).reset_index(drop=True)
        text_df.to_csv("../input/semeval2022/text_dataframe.csv", index=False)

    if "test" in phase:
        train = pd.read_csv(
            "../input/eval/PUBLIC-semeval-2022_task8_eval_data_202201.csv"
        )
        text_ids = set(sum(train["pair_id"].str.split("_"), []))
        meta = pathlib.Path("../input/eval")

        text_dfs = []
        for text_id in tqdm(text_ids):
            try:
                res = fetch_dateframe(text_id, meta)
                text_dfs.append(res)
            except IndexError:
                pass
        text_df = pd.concat(text_dfs).reset_index(drop=True)
        text_df.to_csv("../input/eval/text_dataframe_eval.csv", index=False)
