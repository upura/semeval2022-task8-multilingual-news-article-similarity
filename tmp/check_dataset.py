import pandas as pd

if __name__ == "__main__":
    text_df = pd.read_csv("../input/semeval2022/text_dataframe.csv", low_memory=False)
    print(text_df.shape)
    print(text_df.isnull().sum())
    print(text_df.head(2))
    print(text_df.iloc[0])
