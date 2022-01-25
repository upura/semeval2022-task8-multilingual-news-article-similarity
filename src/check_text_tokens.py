import pandas as pd
from transformers import AutoTokenizer

if __name__ == "__main__":
    text_dataframe = pd.read_csv(
        "../input/semeval2022/text_dataframe.csv", low_memory=False
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")

    encoded1 = tokenizer.batch_encode_plus(
        text_dataframe["title"].fillna("").tolist(),
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    encoded2 = tokenizer.batch_encode_plus(
        text_dataframe["text"].fillna("").tolist(),
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
    )

    tmp = []
    for idx in range(len(text_dataframe)):
        tmp.append(len(encoded2[idx]))
    pd.Series(tmp).hist()
