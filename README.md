# Exploring BERT-based Bi-Encoder Approach for Pairwise Multilingual News Article Similarity

12th place solution for SemEval 2022 Task 8: Multilingual News Article Similarity.
The final prediction is calculated by a weighted average of the output of the four neural networks.
![proposed](proposed.png)

The following figure illustrates the base architecture of each neural network.
![base_architecture](base_architecture.png)

## Dataset preparation

1. Download csv files from [CodaLab](https://competitions.codalab.org/competitions/33835).
1. Fetch meta data of news articles by official provided [downloader](https://github.com/euagendas/semeval_8_2022_ia_downloader).
1. Run `src/prepare_dataset.py`.
1. Run `src/translate_dataset.py`.
1. Run `src/extract_features.py` to create some hand-craft features like Jaccard Index, Dice Index, and cosine similarity.

## Training

Run `src/run_model_XXX.sh`.
When you do `sh run_model_001.sh`, the following command is executed.

```bash
python train_nn.py \
    --fold 0 \
    --max_len 512 \
    --num_folds 5 \
    --model bert-base-multilingual-cased \
    --custom_header concat \
    --lr 1e-5
```

## Submission

Run `src/submit.py`.

```bash
python submit.py
```

## Citation

```
@inproceedings{ishihara-shirai-2022-nikkei,
    title = "{N}ikkei at {S}em{E}val-2022 Task 8: Exploring {BERT}-based Bi-Encoder Approach for Pairwise Multilingual News Article Similarity",
    author = "Ishihara, Shotaro and Shirai, Hono",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.171",
    pages = "1208--1214",
    abstract = "This paper describes our system in SemEval-2022 Task 8, where participants were required to predict the similarity of two multilingual news articles. In the task of pairwise sentence and document scoring, there are two main approaches: Cross-Encoder, which inputs pairs of texts into a single encoder, and Bi-Encoder, which encodes each input independently. The former method often achieves higher performance, but the latter gave us a better result in SemEval-2022 Task 8. This paper presents our exploration of BERT-based Bi-Encoder approach for this task, and there are several findings such as pretrained models, pooling methods, translation, data separation, and the number of tokens. The weighted average ensemble of the four models achieved the competitive result and ranked in the top 12.",
}
```
https://aclanthology.org/2022.semeval-1.171/
