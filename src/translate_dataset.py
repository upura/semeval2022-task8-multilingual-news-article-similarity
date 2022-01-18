import pandas as pd
from googletrans import Translator
from tqdm import tqdm


def trans(s):
    translator = Translator()
    return translator.translate(s).text


def merge_df_and_text(df, text_dataframe):
    text_dataframe = text_dataframe.dropna(subset=["title", "text"]).reset_index(
        drop=True
    )
    text_dataframe["title"] = (
        text_dataframe["title"].fillna("") + "[SEP]" + text_dataframe["text"].fillna("")
    )
    text_dataframe["text_id"] = text_dataframe["text_id"].astype(str)
    df = df[df["pair_id"].map(lambda x: contain_text(text_dataframe, x))].reset_index(
        drop=True
    )
    df["text_id1"] = df["pair_id"].str.split("_").map(lambda x: x[0])
    df["text_id2"] = df["pair_id"].str.split("_").map(lambda x: x[1])

    df = pd.merge(
        df,
        text_dataframe[["text_id", "title"]],
        left_on="text_id1",
        right_on="text_id",
        how="left",
    )
    df = pd.merge(
        df,
        text_dataframe[["text_id", "title"]],
        left_on="text_id2",
        right_on="text_id",
        how="left",
        suffixes=("_1", "_2"),
    )
    return df


def contain_text(text_df, pair_id: str):
    text_id1, text_id2 = pair_id.split("_")
    return (text_id1 in text_df.text_id.values) and (text_id2 in text_df.text_id.values)


def tranlatation(df, col):
    trans_texts = []
    cnt = 0
    res = df.copy()
    lang_col = "url1_lang" if "1" in col else "url2_lang"
    for text, lang in tqdm(zip(df[col], df[lang_col]), total=len(df)):
        if lang == "en":
            trans_texts.append(text)
        try:
            trans_texts.append(trans(text))
        except TypeError:
            cnt += 1
            trans_texts.append(text)
    print(cnt, len(trans_texts))
    res[col] = trans_texts
    return res


if __name__ == "__main__":
    train = pd.read_csv("../input/semeval2022/semeval-2022_task8_train-data_batch.csv")
    test = pd.read_csv(
        "../input/semeval2022/PUBLIC-semeval-2022_task8_eval_data_202201.csv"
    )
    text_dataframe = pd.read_csv(
        "../input/semeval2022/text_dataframe.csv", low_memory=False
    )
    text_dataframe_eval = pd.read_csv(
        "../input/semeval2022/text_dataframe_eval.csv", low_memory=False
    )

    train = merge_df_and_text(train, text_dataframe)
    test = merge_df_and_text(test, text_dataframe_eval)
    # train_trans = train.query("url1_lang!='en' or url2_lang!='en'").copy()
    # test_trans = test.query("url1_lang!='en' or url2_lang!='en'").copy()
    print(train.shape, test.shape)
    train = tranlatation(train, "title_1")
    train = tranlatation(train, "title_2")
    test = tranlatation(test, "title_1")
    test = tranlatation(test, "title_2")
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)
