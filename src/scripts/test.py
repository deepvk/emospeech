from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm
import torchaudio
from loguru import logger
from lightning import seed_everything

from config.config import TrainConfig
from src.dataset.dataset import get_dataloader
from src.metrics.NISQA.tts_predict_mos import get_mos_scores
from src.models import Generator, TorchSTFT
from src.models.acoustic_model.fastspeech.lightning_model import FastSpeechLightning
from src.utils.utils import set_up_logger, write_wav, compute_overall_mos
from src.utils.vocoder_utils import load_checkpoint, synthesize_wav_from_mel


def create_directories(paths: list):
    for path in paths:
        Path(path).mkdir(exist_ok=True, parents=True)


def create_paths(audio_save_path):
    original_audio_path = Path(audio_save_path) / Path("original")
    logger.info(f"Path for original audios: {original_audio_path}")
    reconstructed_audio_path = Path(audio_save_path) / Path("reconstructed")
    logger.info(f"Path for reconstructed audios: {reconstructed_audio_path}")
    generated_audio_path = Path(audio_save_path) / Path("generated")
    logger.info(f"Path for generated audios: {generated_audio_path}")
    return original_audio_path, reconstructed_audio_path, generated_audio_path


def compute_nisqa_scores(original_path, reconstructed_path, generated_path, nisqa_path):
    original_mos = get_mos_scores(
        str(original_path), str(Path(nisqa_path) / "original")
    )
    original_mos_score = compute_overall_mos(original_mos)
    reconstructed_mos = get_mos_scores(
        str(reconstructed_path), str(Path(nisqa_path) / "reconstructed")
    )
    reconstructed_mos_score = compute_overall_mos(reconstructed_mos)
    generated_mos = get_mos_scores(
        str(generated_path), str(Path(nisqa_path) / "generated")
    )
    generated_mos_score = compute_overall_mos(generated_mos)
    return original_mos_score, reconstructed_mos_score, generated_mos_score


@torch.no_grad()
def test(config: TrainConfig) -> None:
    seed_everything(config.seed)
    vocoder = Generator(**asdict(config))
    stft = TorchSTFT(**asdict(config))
    vocoder_state_dict = load_checkpoint(config.vocoder_checkpoint_path)
    vocoder.load_state_dict(vocoder_state_dict["generator"])
    vocoder.remove_weight_norm()
    vocoder.eval()
    test_loader = get_dataloader(config, "test")
    model = FastSpeechLightning.load_from_checkpoint(
        config.testing_checkpoint,
        strict=True,
        config=config,
        vocoder=vocoder,
        stft=stft,
        train=False,
    )
    model.eval()
    model = model.to(config.device)
    torch.set_float32_matmul_precision(config.matmul_precision)
    original_audio_path, reconstructed_audio_path, generated_audio_path = create_paths(
        config.audio_save_path
    )
    create_directories(
        [original_audio_path, reconstructed_audio_path, generated_audio_path]
    )

    current_sample = 0
    if config.limit_generation is not None:
        logger.info(
            f"Will generate only first {config.limit_generation} from test loader."
        )
    for batch in tqdm(
        test_loader,
        total=config.limit_generation
        if config.limit_generation is not None
        else len(test_loader),
    ):
        batch_dict_no_tf = model._get_batch_dict_from_dataloader(batch, validation=True)
        batch_dict_tf = model._get_batch_dict_from_dataloader(batch, validation=False)
        model_output = model.model(model.device, batch_dict_no_tf)

        for i, tag in enumerate(batch_dict_no_tf["ids"]):
            current_sample += 1
            if (
                config.limit_generation is not None
                and current_sample > config.limit_generation
            ):
                break
            # original audio
            source_audio_path = (
                Path(config.preprocessed_data_path)
                / Path("trimmed_wav")
                / Path(tag).with_suffix(".wav")
            )
            original_audio = (
                torchaudio.load(source_audio_path)[0].squeeze(0).cpu().numpy()
            )
            original_audio_sample_path = original_audio_path / Path(tag).with_suffix(
                ".wav"
            )
            write_wav(
                str(original_audio_sample_path), original_audio, config.sample_rate
            )
            # reconstructed audio
            gt_mel_no_padding = batch_dict_tf["mels"][i, : batch_dict_tf["mel_lens"][i]]
            reconstructed_wav = synthesize_wav_from_mel(
                gt_mel_no_padding, model.vocoder, model.stft
            )
            reconstructed_audio_sample_path = reconstructed_audio_path / Path(
                tag
            ).with_suffix(".wav")
            write_wav(
                str(reconstructed_audio_sample_path),
                reconstructed_wav,
                config.sample_rate,
            )
            # generated audio
            predicted_mel_len = model_output["mel_len"][i]
            predicted_mel_no_padding = model_output["predicted_mel"][
                i, :predicted_mel_len
            ]
            generated_wav = synthesize_wav_from_mel(
                predicted_mel_no_padding, model.vocoder, model.stft
            )
            generated_audio_sample_path = generated_audio_path / Path(tag).with_suffix(
                ".wav"
            )
            write_wav(
                str(generated_audio_sample_path), generated_wav, config.sample_rate
            )

    if config.compute_nisqa_on_test:
        create_directories(list(create_paths(config.nisqa_save_path)))
        (
            original_mos_score,
            reconstructed_mos_score,
            generated_mos_score,
        ) = compute_nisqa_scores(
            original_audio_path,
            reconstructed_audio_path,
            generated_audio_path,
            config.nisqa_save_path,
        )
        logger.info(f"Original audios NISQA TTS MOS / std, {original_mos_score}")
        logger.info(
            f"Reconstructed audios NISQA TTS MOS / std, {reconstructed_mos_score}"
        )
        logger.info(
            f"Generated_mos_score audios NISQA TTS MOS / std, {generated_mos_score}"
        )


if __name__ == "__main__":
    set_up_logger("test.log")
    config = TrainConfig()
    test(config)
