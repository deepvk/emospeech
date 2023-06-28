import glob
import json
import re
import string
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import opensmile
import pyrallis
import pyworld as pw
import tgt
import torchaudio
from loguru import logger
from scipy.interpolate import interp1d
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler

from config.config import TrainConfig
from src.dataset.compute_mel import ComputeMelEnergy, PAD_MEL_VALUE
from src.utils.multiprocess_utils import run_pool
from src.utils.utils import write_txt, set_up_logger, crash_with_msg


class Preprocessor:
    def __init__(self, config: TrainConfig):
        """
        Method 'run' of the following class creates:
        - manifests for model training and evaluation: train.txt, val.txt, test.txt
        - computes Mel-Spectrograms
        - computes: durations, pitch, energy (one value per phoneme)
        - computes listed in config eGeMaps features (one value per wav)
        - creates phones_dictionary.json
        :param config: DataClass, see TrainConfig in config.config.py
        """
        self.config = config
        self.phones_mapping = {"": 1}
        self.phones_id_counter = 1
        for punc_symbol in string.punctuation:
            self.phones_id_counter += 1
            self.phones_mapping[punc_symbol] = self.phones_id_counter
        self.compute_mel_energy = ComputeMelEnergy(**asdict(config))
        self.compiled_regular_expression = re.compile(r"[\w']+|[.,!?;]")
        self.val_ids = np.genfromtxt(config.val_ids_path, delimiter="|", dtype=str)
        self.test_ids = np.genfromtxt(config.test_ids_path, delimiter="|", dtype=str)
        self.opensmile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        Path(self.config.preprocessed_data_path, "trimmed_wav").mkdir(
            parents=True, exist_ok=True
        )
        Path(self.config.preprocessed_data_path, "mel").mkdir(
            parents=True, exist_ok=True
        )
        Path(self.config.preprocessed_data_path, "pitch").mkdir(
            parents=True, exist_ok=True
        )
        Path(self.config.preprocessed_data_path, "energy").mkdir(
            parents=True, exist_ok=True
        )
        Path(self.config.preprocessed_data_path, "duration").mkdir(
            parents=True, exist_ok=True
        )
        Path(self.config.preprocessed_data_path, "egemap").mkdir(
            parents=True, exist_ok=True
        )

    def _run(self, filename: str):
        """
        Method is called in multi-processes pool and calls process_utterance for each sample;
        :param filename: str, .TextGrid filename
        :return: None
        """
        basename = filename[:-9].split("/")[-1]
        speaker_idx, filename_idx, emotion_id = basename.split("_")

        tg_path = Path(self.config.raw_data_path, basename).with_suffix(".TextGrid")
        wav_path = Path(self.config.raw_data_path, basename).with_suffix(".wav")
        txt_path = Path(self.config.raw_data_path, basename).with_suffix(".txt")

        if tg_path.exists() and wav_path.exists() and txt_path.exists():
            return self.process_utterance(basename, tg_path, wav_path, txt_path)

    def run(self) -> None:
        """
        This method:
            1) calls iterator through all TextGrids in multiprocess mode;
            2) normalize feature stats
            3) saves final manifests and feature stats.
        :return: None
        """
        data = glob.glob(f"{self.config.raw_data_path}/*.TextGrid")
        results = run_pool(self._run, data, n_threads=self.config.n_threads)

        manifest_data, n_frames = [], 0
        pitch_scaler, energy_scaler, egemap_scaler = (
            StandardScaler(),
            StandardScaler(),
            StandardScaler(),
        )

        logger.info(f"Fitting normalizing feature scalers...")
        for result in results:
            if result:
                result_string, pitch, energy, egemap, n = result
                manifest_data.append(result_string)
                try:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))
                    egemap_scaler.partial_fit(egemap.reshape((-1, 1)).T)
                    n_frames += n
                except Exception:
                    logger.info(f"Pitch scaler exception: {pitch.shape}")
                    logger.info(f"Pitch scaler exception: {result_string}")

        pitch_mean = pitch_scaler.mean_[0]
        pitch_std = pitch_scaler.scale_[0]

        energy_mean = energy_scaler.mean_[0]
        energy_std = energy_scaler.scale_[0]

        egemap_means = egemap_scaler.mean_
        egemap_stds = egemap_scaler.scale_

        logger.info(f"Running feature normalization...")
        pitch_min, pitch_max = self.normalize(
            Path(self.config.preprocessed_data_path, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            Path(self.config.preprocessed_data_path, "energy"), energy_mean, energy_std
        )
        egemap_mins, egemap_maxs = self.normalize_egemap(
            Path(self.config.preprocessed_data_path, "egemap"),
            egemap_means,
            egemap_stds,
        )

        with open(Path(self.config.preprocessed_data_path, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
                "egemap": [
                    [float(i) for i in egemap_mins],
                    [float(i) for i in egemap_maxs],
                    [float(i) for i in egemap_means],
                    [float(i) for i in egemap_stds],
                ],
            }
            f.write(json.dumps(stats))

        # Write metadata
        logger.info(f"Writing data to manifests...")
        train_data, val_data, test_data = self.train_test_val_split(manifest_data)
        with open(
            Path(self.config.preprocessed_data_path, "phones.json"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(self.phones_mapping))
        write_txt(Path(self.config.preprocessed_data_path, "train.txt"), train_data)
        write_txt(Path(self.config.preprocessed_data_path, "val.txt"), val_data)
        write_txt(Path(self.config.preprocessed_data_path, "test.txt"), test_data)
        logger.info(f"Total time: {n_frames * self.config.hop_in_ms / 3600} hours")

    def train_test_val_split(self, metadata: List[str]) -> Tuple[List, List, List]:
        """
        Method:
        1) filters Nones from the list of strings
        2) classifies the strings by ids on train / val / test samples
        3) returns 3 lists of strings that follows the split
        :param metadata: List of str anf 4 numpy arrays
        :return: 3 lists with filenames for training, validating and testing
        """
        metadata = [r for r in metadata if r is not None]
        train_set, val_set, test_set = [], [], []
        for sample in metadata:
            speaker_idx, filename_idx, emotion_idx, text, raw_text = sample.split("|")
            text = text[1:-1].split(" ")
            for phone in text:
                if phone not in self.phones_mapping:
                    self.phones_id_counter += 1
                    self.phones_mapping[phone] = self.phones_id_counter
            if f"{speaker_idx}_{filename_idx}_{emotion_idx}" in self.val_ids:
                val_set.append(sample)
            elif f"{speaker_idx}_{filename_idx}_{emotion_idx}" in self.test_ids:
                test_set.append(sample)
            else:
                train_set.append(sample)
        if len(train_set) + len(val_set) + len(test_set) != len(metadata):
            logger.warning(f"Divided samples don't add up to {len(metadata)}")
        return train_set, val_set, test_set

    def process_utterance(
        self, basename: str, tg_path: Path, wav_path: Path, txt_path: Path
    ) -> Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Method:
        1) extracts phones from .TextGrid file
        2) extracts and save eGeMAPS features listed in config
        3) computes and saves:
            3.1 pitch
            3.2 mel-spectrogram
            3.3 energy
            3.4 durations
        :param basename: str, filename without extension
        :param tg_path: Path, path to .TextGrid file
        :param wav_path: Path, path to .wav file
        :param txt_path: Path, path to .txt file
        :return: List:
                    - str, speaker_idx|filename_idx|emotion_idx|phones|text
                    - 4 np.ndarray (pitch, egemap features, energy, mel spectrogram)
                or None in case couldn't open one of the files
        """

        # eGeMaps Features Processing
        try:
            wav_df = self.opensmile.process_file(str(wav_path))
        except Exception:
            # Handle broken audio files
            logger.info(f"Couldn't process {wav_path}, file may be broken")
            return None
        egemap_features = np.array(
            wav_df[list(self.config.egemap_feature_names)].iloc[0, :]
        ).astype(np.float32)
        if len(egemap_features) != len(self.config.egemap_feature_names):
            message = (
                f"Expected eGeMaps to have {len(self.config.egemap_feature_names)} features, but "
                f"got {len(egemap_features)} for sample {basename}."
            )
            crash_with_msg(message)
        if np.isnan(egemap_features).any():
            logger.info(f"{basename} sample egemap_features contains nan")
            return None

        # Get alignments
        textgrid = tgt.io.read_textgrid(
            tg_path, include_empty_intervals=self.config.include_empty_intervals
        )
        txt_phones = textgrid.get_tier_by_name("phones")
        sentence = " ".join(np.loadtxt(txt_path, dtype="U"))
        clean_text = self.compiled_regular_expression.findall(sentence.lower())

        try:
            phone, duration, start, end = self.get_alignment(
                textgrid.get_tier_by_name("phones"),
                textgrid.get_tier_by_name("words"),
                clean_text,
            )
            if np.isnan(duration).any():
                crash_with_msg("{basename} sample duration contains nan")
            if phone is None or start >= end:
                return None
        except TypeError:
            logger.info(f"Couldn't get alignment of {basename}")
            return None
        brackets = "{}"
        text = f"{brackets[0]}{' '.join(phone)}{brackets[-1]}"

        # Read and trim wav files
        wav = torchaudio.load(wav_path)[0].cpu().numpy().squeeze(0)
        wav = wav[
            int(self.config.sample_rate * start) : int(self.config.sample_rate * end)
        ].astype(np.float32)
        speaker_idx, filename_idx, emotion_idx = basename.split("_")

        # Extract min speaker id from speakers so that ids starts from 0
        trimmed_wav_filename = Path(
            self.config.preprocessed_data_path,
            "trimmed_wav",
            f"{speaker_idx}_{filename_idx}_{emotion_idx}.wav",
        )
        wavfile.write(trimmed_wav_filename, self.config.sample_rate, wav)

        # Read raw text
        with open(txt_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute pitch
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.config.sample_rate,
            frame_period=self.config.hop_in_ms * 1000,
        )

        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.config.sample_rate)
        if np.isnan(pitch).any():
            logger.info(f"{basename} sample pitch contains nan")
            return None

        # Compute mel-spectrogram and energy
        mel_spectrogram, energy = self.compute_mel_energy(wav)
        if np.isnan(mel_spectrogram).any():
            logger.info(f"{basename} sample mel_spectrogram contains nan")
            return None
        if np.isnan(energy).any():
            logger.info(f"{basename} sample energy contains nan")
            return None
        mel_count = mel_spectrogram.shape[1]
        if np.sum(pitch != 0) <= 1:
            logger.info(f"Audio might be silent (pitch == 0), {pitch}")
            return None
        pitch = pitch[: sum(duration)]

        # Duration check
        duration_sum = sum(duration)
        if duration_sum - mel_count == 1:
            mel_spectrogram = np.pad(
                mel_spectrogram,
                ((0, 0), (0, duration_sum - mel_count)),
                mode="constant",
                constant_values=PAD_MEL_VALUE,
            )
        if mel_count - duration_sum == 1:
            mel_spectrogram = mel_spectrogram[:, :-1]
        mel_count = mel_spectrogram.shape[1]
        if mel_count != duration_sum:
            message = f"Mels and durations mismatch, mel count: {mel_count}, duration count: {duration_sum}."
            crash_with_msg(message)
        if mel_spectrogram.shape[0] != self.config.n_mel_channels:
            message = f"Incorrect padding, supposed to have: {self.config.n_mel_channels}, got {mel_spectrogram.shape[0]}."
            crash_with_msg(message)
        if pitch.shape[0] > mel_count:
            pitch = pitch[:mel_count]
        if pitch.shape[0] < mel_count:
            pitch = np.pad(
                pitch, (0, mel_count - pitch), mode="constant", constant_values=0
            )
        if pitch.shape[0] != mel_count:
            crash_with_msg(
                f"Pitch isn't count for each mel. Mel count: {mel_count}, pitch count {pitch.shape[0]}"
            )
        if mel_count - energy.shape[0] == 1:
            energy = np.pad(energy, (0, 1), mode="constant", constant_values=0)
        energy = energy[: sum(duration)]
        if energy.shape[0] != mel_count:
            message = f"Energy isn't count for each mel. Mel count: {mel_count}, energy count {energy.shape[0]}"
            crash_with_msg(message)

        # Perform linear interpolation to smooth unvoiced pitch contour (following FastSpeech2 paper, page 14, C.2)
        nonzero_ids = np.where(pitch != 0)[0]
        interp_fn = interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
            bounds_error=False,
        )
        pitch = interp_fn(np.arange(0, len(pitch)))

        # Pitch phoneme-level averaging
        pitch = self.phoneme_level_averaging(duration, pitch)
        energy = self.phoneme_level_averaging(duration, energy)
        # Save files
        speaker_idx, file_idx, emotion_idx = basename.split("_")
        numpy_filename = f"{speaker_idx}_{file_idx}_{emotion_idx}.npy"
        np.save(
            str(Path(self.config.preprocessed_data_path, "duration", numpy_filename)),
            duration,
        )
        np.save(
            str(Path(self.config.preprocessed_data_path, "pitch", numpy_filename)),
            pitch,
        )
        np.save(
            str(Path(self.config.preprocessed_data_path, "energy", numpy_filename)),
            energy,
        )
        np.save(
            str(Path(self.config.preprocessed_data_path, "mel", numpy_filename)),
            mel_spectrogram.T,
        )
        np.save(
            str(Path(self.config.preprocessed_data_path, "egemap", numpy_filename)),
            egemap_features,
        )
        res_string = "|".join([speaker_idx, filename_idx, emotion_idx, text, raw_text])
        removed_outlier_pitch = self.remove_outlier(pitch)
        removed_outlier_energy = self.remove_outlier(energy)
        return (
            res_string,
            removed_outlier_pitch,
            removed_outlier_energy,
            egemap_features,
            mel_spectrogram.shape[1],
        )

    @staticmethod
    def phoneme_level_averaging(
        duration: List[int], feature_vector: np.ndarray
    ) -> np.ndarray:
        """
        Method creates from n-dimension feature vector, m-dimension feature vector, where m is a number of
        phones (with punctuation symbols) in a sequence
        :param duration: List, len() = m, each integer stands for number of mel-spec frames
        :param feature_vector: 1d np.ndarray, len() = n, all integers refers to one phone to be averaged.
        :return: np.ndarray, averaged feature vector
        """
        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                feature_vector[i] = np.mean(feature_vector[pos : pos + d])
            else:
                feature_vector[i] = 0
            pos += d
        return feature_vector[: len(duration)]

    def get_alignment(
        self,
        tier_phones: tgt.core.IntervalTier,
        tier_words: tgt.core.IntervalTier,
        clean_text: List,
    ) -> Optional[Tuple[List, List, int, int]]:
        """
        Method extracts phones from tier_phones and punctuation symbols from tier_words and returns a sequence
        of phones with punctuation symbols along with the sequence of durations
        :param tier_phones: tgt.core.IntervalTier,
        :param tier_words: tgt.core.IntervalTier,
        :param clean_text: List,
        :return: Tuple [List, List, int, int] sequence of phones, sequence of durations, start and end of the
        original audio to trim the silence
        """
        punctuation_symbols = string.punctuation
        words = tier_words._objects
        sil_phones = ["sil", ""]
        start_time, end_time, end_idx, word_idx, text_idx, phones, durations = (
            0,
            0,
            0,
            0,
            0,
            [],
            [],
        )
        punctuation_symbol = None
        for t in tier_phones._objects:
            s, e, p = t.start_time, t.end_time, t.text
            ws, we, w = (
                words[word_idx].start_time,
                words[word_idx].end_time,
                words[word_idx].text,
            )
            if e >= ws:
                if len(words) - word_idx > 1:
                    word_idx += 1
                if w != "":
                    if (
                        words[word_idx].text != clean_text[text_idx]
                        and clean_text[text_idx] in punctuation_symbols
                    ):
                        punctuation_symbol = [clean_text[text_idx]]
                        if len(clean_text) - (text_idx + 1) > 0:
                            text_idx += 1
                        while (
                            len(clean_text) - (text_idx + 1) > 0
                            and clean_text[text_idx] in punctuation_symbols
                        ):
                            text_idx += 1
                            punctuation_symbol.append(clean_text[text_idx])

                    elif len(clean_text) - (text_idx + 1) > 0:
                        text_idx += 1
            # Trim leading silences
            if len(phones) == 0:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            # Skip samples where words weren't align
            if p == "spn":
                logger.info(f"Skip sample contained <spn> non aligned word")
                return None

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
            else:
                # For silent phones
                phones.append(p)

            end_idx = np.round(e * self.config.sample_rate / self.config.hop_length)
            start_idx = np.round(s * self.config.sample_rate / self.config.hop_length)
            durations.append(int(end_idx - start_idx))

            if punctuation_symbol is not None:
                phones.extend(punctuation_symbol)
                durations.extend([0] * len(punctuation_symbol))
                punctuation_symbol = None
        if phones[-1] == "":
            # Trim tailing silences
            phones = phones[:-1]
            durations = durations[:-1]
        if clean_text[text_idx] in punctuation_symbols:
            punctuation_symbol = [clean_text[text_idx]]
            while (
                len(clean_text) - (text_idx + 1) > 0
                and clean_text[text_idx] in punctuation_symbols
            ):
                text_idx += 1
                punctuation_symbol.append(clean_text[text_idx])
            phones.extend(punctuation_symbol)
            durations.extend([0] * len(punctuation_symbol))
        if len(phones) != len(durations):
            message = f"Phones and durations mismatch phones count {len(phones)} durations count {len(durations)}"
            crash_with_msg(message)
        return phones, durations, start_time, end_time

    @staticmethod
    def normalize(in_dir: Path, mean: float, std: float) -> Tuple[float, float]:
        """
        Method iterates through all files saved to the directory and applies normalization them
        :param in_dir: Path, path to the directory with saved feature-arrays
        :param mean: float
        :param std: float
        :return: Tuple[float, float], min and max of normalized values for the feature
        """
        max_value = np.finfo(np.float32).min
        min_value = np.finfo(np.float32).max
        for filename in in_dir.iterdir():
            filename = str(Path(in_dir, filename))
            values = (np.load(filename) - mean) / std
            np.save(filename, values)
            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))
        return min_value, max_value

    def normalize_egemap(
        self, in_dir: Path, means: List[float], stds: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Method
        :param in_dir: Path, path where egemap features are saved
        :param means: List of float values, means for each feature
        :param stds:  List of float values, stds for each feature
        :return: Tuple[List[float], List[float]], returns 2 lists with min and max values for each egemap feature
        """
        max_values = [
            np.finfo(np.float32).min
            for i in range(len(self.config.egemap_feature_names))
        ]
        min_values = [
            np.finfo(np.float32).max
            for i in range(len(self.config.egemap_feature_names))
        ]
        for filename in in_dir.iterdir():
            filename = Path(in_dir, filename)
            norm_values = [0.0] * len(self.config.egemap_feature_names)
            values = np.load(str(Path(in_dir, filename)))
            if len(values) != len(means) != len(stds):
                message = (
                    f"eGeMap of sample {filename} has a length {len(values)} "
                    f"but means: {len(means)} and stds: {len(stds)}"
                )
                crash_with_msg(message)
            for i, v in enumerate(values):
                norm_values[i] = (v - means[i]) / stds[i]
                max_values[i] = max(max_values[i], norm_values[i])
                min_values[i] = min(min_values[i], norm_values[i])
            np.save(str(Path(in_dir, filename)), norm_values)
        return min_values, max_values

    @staticmethod
    def remove_outlier(values: np.ndarray) -> np.ndarray:
        """
        Method removes all values that are not in 25th or 75th percentile of the array
        :param values: values to be filtered
        :return: truncated input values
        """
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)
        return values[normal_indices]


if __name__ == "__main__":
    set_up_logger("preprocess.log")
    cfg = pyrallis.parse(config_class=TrainConfig)
    preprocessor = Preprocessor(cfg)
    preprocessor.run()
