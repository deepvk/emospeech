import json
import torch
import numpy as np

from typing import Any
from pathlib import Path
from torch.utils.data import DataLoader

from config.config import TrainConfig
from src.utils.fastspeech_utils import pad_1d, pad_2d


def get_dataloader(
    config: TrainConfig, mode: str, size: int = None
) -> torch.utils.data.DataLoader:
    txt_name = f"{mode}.txt"
    shuffle = True if "train" in txt_name else False
    sort = True if "train" in txt_name else False
    batch_size = config.train_batch_size if mode == "train" else config.val_batch_size
    dataset = Dataset(
        filename=txt_name, cfg=config, sort=sort, batch_size=batch_size, size=size
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filename: str,
        cfg: TrainConfig,
        batch_size: int,
        size=None,
        sort=False,
        drop_last=False,
    ):
        self.preprocessed_path = cfg.preprocessed_data_path
        self.batch_size = batch_size
        self.speaker_id, self.file_id, self.emotion_id, self.text = self.process_meta(
            filename
        )
        with open(Path(self.preprocessed_path) / "phones.json", "r") as f:
            self.phones_mapping = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        self.multi_speaker = cfg.multi_speaker
        self.multi_emotion = cfg.multi_emotion
        self.n_emotions = cfg.n_emotions
        self.n_egemap_features = cfg.n_egemap_features
        self.size = size

    def __len__(self) -> int:
        if self.size is not None:
            return self.size
        return len(self.text)

    def __getitem__(self, idx: int) -> dict:
        speaker_id = self.speaker_id[idx]
        file_id = self.file_id[idx]
        emotion_id = self.emotion_id[idx]
        phone = np.array(
            [self.phones_mapping[i] for i in self.text[idx][1:-1].split(" ")]
        )
        basename = f"{speaker_id}_{file_id}_{emotion_id}"

        if self.n_egemap_features > 0:
            egemap_feature = np.load(
                str(Path(self.preprocessed_path) / "egemap" / f"{basename}.npy"),
                allow_pickle=True,
            )[: self.n_egemap_features]
        else:
            egemap_feature = None
        mel = np.load(
            str(Path(self.preprocessed_path) / "mel" / f"{basename}.npy"),
            allow_pickle=True,
        )
        pitch = np.load(
            str(Path(self.preprocessed_path) / "pitch" / f"{basename}.npy"),
            allow_pickle=True,
        )
        energy = np.load(
            str(Path(self.preprocessed_path) / "energy" / f"{basename}.npy"),
            allow_pickle=True,
        )
        duration = np.load(
            str(Path(self.preprocessed_path) / "duration" / f"{basename}.npy"),
            allow_pickle=True,
        )

        assert duration.shape == phone.shape, (
            f"Duration and phone shapes do not match. Phone shape {phone.shape}, "
            f"duration: {duration.shape} for sample: {basename}."
        )

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "emotion": emotion_id,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "text": phone,
            "egemap_feature": egemap_feature,
        }

        return sample

    def process_meta(self, filename: str) -> tuple[np.ndarray, ...]:
        with open(Path(self.preprocessed_path) / filename, "r", encoding="utf-8") as f:
            speaker, name, emotion, text = [], [], [], []
            for line in f.readlines():
                s, n, e, t, _ = line.strip("\n").split("|")
                name.append(int(n))
                speaker.append(int(s))
                emotion.append(int(e))
                text.append(t)
            return np.array(speaker), np.array(name), np.array(emotion), np.array(text)

    @staticmethod
    def reprocess(data: list[dict], indexes: list) -> tuple[Any, ...]:
        """
        :param: data List[dict] (len = batch_size, dict = __getitem__ dictionary)
        :indexes: indexes
        :return:
            ids: List[str], len of bs, filenames for txt / wav / mel etc.
            speakers: TorchLong Tensor, len of bs, speaker ids (will be used for embedding extraction)
            emotions: TorchLong Tensor, len of bs, ids for emotion (will be used for embedding extraction)
            texts: TorchLong Tensor, padded to max_seq_len, shape [bs, max_seq_len] ids of phones used for the sentence
            text_lens: TorchLong Tensor, len of bs, number of phones in samples before padding
            mels: TorchFloat Tensor, padded to max mel_len, shape [bs, max_mel_len, n_mels], precomputed mels
            mel_lens: TorchLong Tensor, len of bs, original len of mels before padding
            pitches: TorchFloat Tensor, shape [bs, max_text_len], pitch for each phone in a sequence padded to the max_text_len
            energies: TorchFloat Tensor, shape [bs, max_text_len], energy for each phone in a sequence padded to the max_text_len
            durations: TorchFloat Tensor, shape [bs, max_text_len], duration for each phone in a sequence padded to the max_text_len
        """
        ids = [data[idx]["id"] for idx in indexes]
        texts = [data[idx]["text"] for idx in indexes]
        text_lens = torch.Tensor([text.shape[0] for text in texts]).long()
        texts = torch.from_numpy(pad_1d(texts)).long()

        mels = [data[idx]["mel"] for idx in indexes]
        mel_lens = torch.Tensor([mel.shape[0] for mel in mels]).long()
        mels = torch.from_numpy(pad_2d(mels)).float()

        pitches = torch.from_numpy(
            pad_1d([data[idx]["pitch"] for idx in indexes])
        ).float()
        energies = torch.from_numpy(
            pad_1d([data[idx]["energy"] for idx in indexes])
        ).float()
        durations = torch.from_numpy(
            pad_1d([data[idx]["duration"] for idx in indexes])
        ).float()
        egemap_features = np.array([data[idx]["egemap_feature"] for idx in indexes])
        egemap_features = (
            torch.Tensor(egemap_features).float() if egemap_features.any() else None
        )
        speakers = torch.Tensor([data[idx]["speaker"] for idx in indexes]).long()
        emotions = torch.Tensor([data[idx]["emotion"] for idx in indexes]).long()
        return (
            ids,
            speakers,
            emotions,
            texts,
            text_lens,
            mels,
            mel_lens,
            pitches,
            energies,
            durations,
            egemap_features,
        )

    def collate_fn(self, data: list[dict]) -> list[tuple[Any]]:
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr.extend([tail.tolist()])
        output = [self.reprocess(data, idx) for idx in idx_arr]

        return output
