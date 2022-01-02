import json
import pathlib

import pandas as pd
from tqdm import tqdm


def fetch_dateframe(text_id: str, meta):
    search_results = list(meta.glob(f"*/{text_id}.json"))
    if len(search_results) == 0:
        res_df = pd.DataFrame(
            columns=[
                "source_url",
                "url",
                "title",
                "top_image",
                "meta_img",
                "images",
                "movies",
                "text",
                "keywords",
                "meta_keywords",
                "tags",
                "authors",
                "publish_date",
                "summary",
                "article_html",
                "meta_description",
                "meta_lang",
                "meta_favicon",
                "canonical_link",
                "meta_data.oath.guce.consent-host",
                "meta_data.msapplication-TileColor",
                "meta_data.msapplication-TileImage",
                "meta_data.msvalidate.01",
                "meta_data.referrer",
                "meta_data.theme-color",
                "meta_data.twitter.dnt",
                "meta_data.twitter.site",
                "meta_data.twitter.player.identifier",
                "meta_data.twitter.player.height",
                "meta_data.twitter.player.width",
                "meta_data.twitter.card",
                "meta_data.twitter.description",
                "meta_data.twitter.image.identifier",
                "meta_data.twitter.image.src",
                "meta_data.twitter.title",
                "meta_data.summary",
                "meta_data.robots.",
                "meta_data.news_keywords",
                "meta_data.apple-itunes-app",
                "meta_data.description",
                "meta_data.og.type",
                "meta_data.og.image",
                "meta_data.og.description",
                "meta_data.og.title",
                "meta_data.og.url",
                "meta_data.fb.pages",
                "meta_data.al.android.app_name",
                "meta_data.al.android.package",
                "meta_data.al.android.url",
                "meta_data.al.ios.app_name",
                "meta_data.al.ios.app_store_id",
                "meta_data.al.ios.url",
                "text_id",
            ]
        )
        return
    else:
        text_path = str(search_results[0])
        with open(text_path) as f:
            text_json = f.read()
        text_json = json.loads(text_json)
        res_df = pd.json_normalize(text_json)
        res_df["text_id"] = text_id
        return res_df


if __name__ == "__main__":
    train = pd.read_csv("../input/semeval2022/semeval-2022_task8_train-data_batch.csv")
    text_ids = set(sum(train["pair_id"].str.split("_"), []))
    meta = pathlib.Path("../input/semeval2022")

    text_dfs = []
    for text_id in tqdm(text_ids):
        text_dfs.append(fetch_dateframe(text_id, meta))
    text_df = pd.concat(text_dfs).reset_index(drop=True)
    text_df.to_csv("../input/semeval2022/text_dataframe.csv", index=False)
