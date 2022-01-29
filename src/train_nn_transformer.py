import argparse
import dataclasses
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer


def contain_text(text_df, pair_id: str):
    text_id1, text_id2 = pair_id.split("_")
    return (text_id1 in text_df.text_id.values) and (text_id2 in text_df.text_id.values)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat.flatten(), y))


class MyLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = MyModel(
            model_path=cfg.MODEL_PATH,
            num_classes=cfg.NUM_CLASSES,
            transformer_params=cfg.TRANSFORMER_PARAMS,
            custom_header=cfg.CUSTOM_HEADER,
        )
        self.criterion = RMSELoss()
        self.lr = cfg.LEARNING_RATE

    def forward(self, x):
        ids1 = x["ids1"]
        ids2 = x["ids2"]
        attention_mask1 = x["attention_mask1"]
        attention_mask2 = x["attention_mask2"]
        token_type_ids1 = x["token_type_ids1"]
        token_type_ids2 = x["token_type_ids2"]
        features = x["features"]
        output = self.backbone(
            input_ids1=ids1,
            attention_mask1=attention_mask1,
            token_type_ids1=token_type_ids1,
            input_ids2=ids2,
            attention_mask2=attention_mask2,
            token_type_ids2=token_type_ids2,
            features=features,
        )
        return output

    def training_step(self, batch, batch_idx):
        ids1 = batch["ids1"]
        ids2 = batch["ids2"]
        attention_mask1 = batch["attention_mask1"]
        attention_mask2 = batch["attention_mask2"]
        token_type_ids1 = batch["token_type_ids1"]
        token_type_ids2 = batch["token_type_ids2"]
        features = batch["features"]
        targets = batch["targets"]
        output = self.backbone(
            input_ids1=ids1,
            attention_mask1=attention_mask1,
            token_type_ids1=token_type_ids1,
            input_ids2=ids2,
            attention_mask2=attention_mask2,
            token_type_ids2=token_type_ids2,
            features=features,
        )
        loss = self.criterion(output, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        ids1 = batch["ids1"]
        ids2 = batch["ids2"]
        attention_mask1 = batch["attention_mask1"]
        attention_mask2 = batch["attention_mask2"]
        token_type_ids1 = batch["token_type_ids1"]
        token_type_ids2 = batch["token_type_ids2"]
        features = batch["features"]
        targets = batch["targets"]
        output = self.backbone(
            input_ids1=ids1,
            attention_mask1=attention_mask1,
            token_type_ids1=token_type_ids1,
            input_ids2=ids2,
            attention_mask2=attention_mask2,
            token_type_ids2=token_type_ids2,
            features=features,
        )
        loss = self.criterion(output, targets)
        output = OrderedDict(
            {
                "targets": targets.detach(),
                "preds": output.detach(),
                "loss": loss.detach(),
            }
        )
        return output

    def validation_epoch_end(self, outputs):
        d = dict()
        d["epoch"] = int(self.current_epoch)
        d["v_loss"] = torch.stack([o["loss"] for o in outputs]).mean().item()

        targets = torch.cat([o["targets"].view(-1) for o in outputs]).cpu().numpy()
        preds = torch.cat([o["preds"].view(-1) for o in outputs]).cpu().numpy()

        score = pd.DataFrame({"targets": targets, "preds": preds}).corr()["targets"][
            "preds"
        ]
        d["v_score"] = score
        self.log_dict(d, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]


class TextDataset(Dataset):
    def __init__(
        self,
        df,
        text_col: str,
        target_col: str,
        tokenizer_name: str,
        max_len: int,
        is_train: bool = True,
    ):
        super().__init__()

        self.df = df
        self.is_train = is_train

        if self.is_train:
            self.target = torch.tensor(self.df[target_col].values, dtype=torch.float32)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encoded1 = tokenizer.batch_encode_plus(
            (
                self.df["title_1"].fillna("") + "[SEP]" + self.df["title_2"].fillna("")
            ).tolist(),
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        self.encoded2 = tokenizer.batch_encode_plus(
            (
                self.df[f"{text_col}_1"].str[:256].fillna("")
                + "[SEP]"
                + self.df[f"{text_col}_2"].str[:256].fillna("")
            ).tolist(),
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        self.features = (
            pd.read_csv("../input/semeval2022/X_train.csv")
            if is_train
            else pd.read_csv("../input/semeval2022/X_test.csv")
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        input_ids1 = torch.tensor(self.encoded1["input_ids"][index])
        attention_mask1 = torch.tensor(self.encoded1["attention_mask"][index])
        token_type_ids1 = torch.tensor(self.encoded1["token_type_ids"][index])
        input_ids2 = torch.tensor(self.encoded2["input_ids"][index])
        attention_mask2 = torch.tensor(self.encoded2["attention_mask"][index])
        token_type_ids2 = torch.tensor(self.encoded2["token_type_ids"][index])
        features = torch.tensor(self.features.loc[index].values, dtype=torch.float32)
        if self.is_train:
            target = self.target[index]
            return {
                "ids1": input_ids1,
                "attention_mask1": attention_mask1,
                "token_type_ids1": token_type_ids1,
                "ids2": input_ids2,
                "attention_mask2": attention_mask2,
                "token_type_ids2": token_type_ids2,
                "features": features,
                "targets": target,
            }
        else:
            return {
                "ids1": input_ids1,
                "attention_mask1": attention_mask1,
                "token_type_ids1": token_type_ids1,
                "ids2": input_ids2,
                "attention_mask2": attention_mask2,
                "token_type_ids2": token_type_ids2,
                "features": features,
            }


class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.test_df = None
        self.train_df = None
        self.valid_df = None
        self.cfg = cfg

    def merge_df_and_text(self, df, text_dataframe):

        text_dataframe["text_id"] = text_dataframe["text_id"].astype(str)
        df["text_id1"] = df["pair_id"].str.split("_").map(lambda x: x[0])
        df["text_id2"] = df["pair_id"].str.split("_").map(lambda x: x[1])

        text_dataframe = text_dataframe.dropna(subset=["title", "text"]).reset_index(
            drop=True
        )

        df = pd.merge(
            df,
            text_dataframe[["text_id", "title"]],
            left_on="text_id1",
            right_on="text_id",
            how="left",
        ).reset_index(drop=True)
        df = pd.merge(
            df,
            text_dataframe[["text_id", "title"]],
            left_on="text_id2",
            right_on="text_id",
            how="left",
            suffixes=("_1", "_2"),
        ).reset_index(drop=True)
        return df

    def get_test_df(self):
        df = pd.read_csv(self.cfg.TEST_DF_PATH)
        text_dataframe = pd.read_csv(
            "../input/semeval2022/text_dataframe_eval.csv", low_memory=False
        )
        df = self.merge_df_and_text(df, text_dataframe)
        return df

    def split_train_valid_df(self):
        if int(self.cfg.debug):
            df = pd.read_csv(self.cfg.TRAIN_DF_PATH, nrows=100)
        else:
            df = pd.read_csv(self.cfg.TRAIN_DF_PATH)

        text_dataframe = pd.read_csv(
            "../input/semeval2022/text_dataframe.csv", low_memory=False
        )
        df = self.merge_df_and_text(df, text_dataframe)
        cv = KFold(n_splits=self.cfg.NUM_FOLDS, shuffle=True, random_state=42)
        for n, (train_index, val_index) in enumerate(cv.split(df)):
            df.loc[val_index, "fold"] = int(n)
        df["fold"] = df["fold"].astype(int)

        train_df = df[df["fold"] != self.cfg.fold].reset_index(drop=True)
        valid_df = df[df["fold"] == self.cfg.fold].reset_index(drop=True)
        return train_df, valid_df

    def setup(self, stage):
        self.test_df = self.get_test_df()
        train_df, valid_df = self.split_train_valid_df()
        self.train_df = train_df
        self.valid_df = valid_df

    def get_dataframe(self, phase):
        assert phase in {"train", "valid", "test"}
        if phase == "train":
            return self.train_df
        elif phase == "valid":
            return self.valid_df
        elif phase == "test":
            return self.test_df

    def get_ds(self, phase):
        assert phase in {"train", "valid", "test"}
        ds = TextDataset(
            df=self.get_dataframe(phase=phase),
            text_col=self.cfg.TEXT_COL,
            target_col=self.cfg.TARGET_COL,
            tokenizer_name=self.cfg.TOKENIZER_PATH,
            max_len=self.cfg.MAX_LEN,
            is_train=(phase != "test"),
        )
        return ds

    def get_loader(self, phase):
        dataset = self.get_ds(phase=phase)
        return DataLoader(
            dataset,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=(phase == "train"),
            num_workers=self.cfg.NUM_WORKERS,
            drop_last=(phase == "train"),
        )

    # Trainer.fit() 時に呼び出される
    def train_dataloader(self):
        return self.get_loader(phase="train")

    # Trainer.fit() 時に呼び出される
    def val_dataloader(self):
        return self.get_loader(phase="valid")

    def test_dataloader(self):
        return self.get_loader(phase="test")


class MyModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        num_classes: List[int],
        transformer_params: Dict[str, Any] = {},
        custom_header: str = "max_pool",
    ):
        super().__init__()
        self.num_classes = num_classes
        model_config = AutoConfig.from_pretrained(model_path)
        model_config.update(transformer_params)
        self.net1 = AutoModel.from_pretrained(model_path, config=model_config)
        self.net2 = AutoModel.from_pretrained(model_path, config=model_config)
        self.out_shape = model_config.hidden_size * 2
        self.fc = nn.Linear(self.out_shape + 12, num_classes)
        self.custom_header = custom_header
        if self.custom_header == "cnn":
            self.cnn1 = nn.Conv1d(self.out_shape, 256, kernel_size=2, padding=1)
            self.cnn2 = nn.Conv1d(256, 1, kernel_size=2, padding=1)
            self.relu = nn.ReLU()
        elif self.custom_header == "lstm":
            self.lstm = nn.LSTM(self.out_shape, self.out_shape, batch_first=True)
        elif self.custom_header == "concat":
            self.fc = nn.Linear(self.out_shape * 4 + 12, num_classes)

    def forward(
        self,
        input_ids1,
        attention_mask1,
        token_type_ids1,
        input_ids2,
        attention_mask2,
        token_type_ids2,
        features,
    ):
        outputs1 = self.net1(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
            token_type_ids=token_type_ids1,
        )
        outputs2 = self.net2(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            token_type_ids=token_type_ids2,
        )
        if self.custom_header == "max_pool":
            sequence_output1, _ = outputs1["last_hidden_state"].max(1)
            sequence_output2, _ = outputs2["last_hidden_state"].max(1)
        elif self.custom_header == "cnn":
            last_hidden_state1 = outputs1["last_hidden_state"].permute(0, 2, 1)
            cnn_embeddings1 = self.relu(self.cnn1(last_hidden_state1))
            cnn_embeddings1 = self.cnn2(cnn_embeddings1)
            outputs1, _ = torch.max(cnn_embeddings1, 2)
            last_hidden_state2 = outputs2["last_hidden_state"].permute(0, 2, 1)
            cnn_embeddings2 = self.relu(self.cnn1(last_hidden_state2))
            cnn_embeddings2 = self.cnn2(cnn_embeddings2)
            outputs2, _ = torch.max(cnn_embeddings2, 2)
        elif self.custom_header == "lstm":
            out1, _ = self.lstm(outputs1["last_hidden_state"], None)
            sequence_output1 = out1[:, -1, :]
            outputs1 = self.fc(sequence_output1)
            out2, _ = self.lstm(outputs2["last_hidden_state"], None)
            sequence_output2 = out2[:, -1, :]
            outputs2 = self.fc(sequence_output2)
        elif self.custom_header == "concat":
            sequence_output1 = torch.cat(
                [outputs1["hidden_states"][-1 * i][:, 0] for i in range(1, 4 + 1)],
                dim=1,
            )
            sequence_output2 = torch.cat(
                [outputs2["hidden_states"][-1 * i][:, 0] for i in range(1, 4 + 1)],
                dim=1,
            )
        sequence_output = torch.cat(
            [
                sequence_output1,
                sequence_output2,
                features,
                # torch.abs(sequence_output1 - sequence_output2),
                # sequence_output1 * sequence_output2,
            ],
            dim=1,
        )
        outputs = self.fc(sequence_output)
        return outputs


@dataclasses.dataclass
class Cfg:
    PROJECT_NAME = "semeval2022"
    RUN_NAME = "exp003"
    NUM_FOLDS = 5
    NUM_CLASSES = 1
    NUM_EPOCHS = 10
    NUM_WORKERS = 2
    NUM_GPUS = 1
    MAX_LEN = 512
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-6
    MODEL_PATH = "bert-base-multilingual-uncased"
    TOKENIZER_PATH = "bert-base-multilingual-uncased"
    TRANSFORMER_PARAMS = {
        "output_hidden_states": True,
        "hidden_dropout_prob": 0.0,
        "layer_norm_eps": 1e-7,
    }
    CUSTOM_HEADER = "concat"
    OUTPUT_PATH = "."
    TRAIN_DF_PATH = "../input/semeval2022/semeval-2022_task8_train-data_batch.csv"
    TEST_DF_PATH = "../input/semeval2022/PUBLIC-semeval-2022_task8_eval_data_202201.csv"
    TEXT_COL = "title"
    TARGET_COL = "Overall"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold")
    args = parser.parse_args()

    fold = int(args.fold)
    debug = False
    cfg = Cfg()
    cfg.fold = fold
    cfg.debug = debug

    seed_everything(777)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if "google.colab" in sys.modules:
        secret_value = "YOUR_SECRET"
    else:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        secret_value = user_secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=secret_value)

    logger = CSVLogger(save_dir=str(cfg.OUTPUT_PATH), name=f"fold_{fold}")
    wandb_logger = WandbLogger(name=f"{cfg.RUN_NAME}_{fold}", project=cfg.PROJECT_NAME)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(cfg.OUTPUT_PATH),
        filename=f"{cfg.RUN_NAME}_fold_{fold}",
        save_weights_only=True,
        monitor=None,
    )
    trainer = Trainer(
        max_epochs=cfg.NUM_EPOCHS,
        gpus=cfg.NUM_GPUS,
        callbacks=[checkpoint_callback],
        logger=[logger, wandb_logger],
    )

    model = MyLightningModule(cfg)
    datamodule = MyDataModule(cfg)
    trainer.fit(model, datamodule=datamodule)

    y_val_pred = torch.cat(trainer.predict(model, datamodule.val_dataloader()))
    y_test_pred = torch.cat(trainer.predict(model, datamodule.test_dataloader()))
    np.save(f"y_val_pred_fold{fold}", y_val_pred.to("cpu").detach().numpy())
    np.save(f"y_test_pred_fold{fold}", y_test_pred.to("cpu").detach().numpy())

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

    y_val_pred = np.load(f"y_val_pred_fold{fold}.npy")
    y_test_pred = np.load(f"y_test_pred_fold{fold}.npy")

    oof = datamodule.valid_df[["pair_id", cfg.TARGET_COL]].copy()
    oof["y_pred"] = y_val_pred.reshape(-1)
    oof.to_csv(f"oof_fold{fold}.csv", index=False)

    sub = pd.read_csv(cfg.TEST_DF_PATH)
    sub[cfg.TARGET_COL] = np.nan
    sub.loc[sub["pair_id"].isin(rule_based_pair_ids), cfg.TARGET_COL] = 2.8
    sub.loc[
        ~sub["pair_id"].isin(rule_based_pair_ids), cfg.TARGET_COL
    ] = y_test_pred.reshape(-1)
    sub[["pair_id", cfg.TARGET_COL]].to_csv("submission.csv", index=False)
    sub[["pair_id", cfg.TARGET_COL]].head(2)
