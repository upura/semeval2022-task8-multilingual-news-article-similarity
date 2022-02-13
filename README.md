# Exploring Bert-based Bi-Encoder Approach for Pairwise Multilingual News Article Similarity

12th place solution for SemEval 2022 Task 8: Multilingual News Article Similarity.

## Dataset preparation

1. Download csv files from [CodaLab](https://competitions.codalab.org/competitions/33835).
1. Fetch meta data of news articles by official provided [downloader](https://github.com/euagendas/semeval_8_2022_ia_downloader).
1. Run `src/prepare_dataset.py`.
1. Run `src/translate_dataset.py`.
1. Run `src/extract_features.py` to create some hand-craft features like Jaccard Index, Dice Index, and cosine similarity.

## Training

## Submission
