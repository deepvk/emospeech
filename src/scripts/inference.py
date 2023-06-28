import argparse
import json
from dataclasses import asdict

import torch
from lightning import seed_everything

from config.config import TrainConfig
from src.models import Generator, TorchSTFT
from src.models.acoustic_model.fastspeech.lightning_model import FastSpeechLightning
from src.utils.utils import set_up_logger, write_wav, crash_with_msg
from src.utils.vocoder_utils import load_checkpoint, synthesize_wav_from_mel


def get_input_dict(config, args):
    with open(config.phones_path, "r") as f:
        phones_mapping = json.load(f)
    phone_ids = []
    for p in args.phone_sequence.split(" "):
        try:
            phone_ids.append(phones_mapping[p])
        except KeyError:
            crash_with_msg(
                f"Couldn't map input sequence: {args.phone_sequence} into phone ids. \n"
                f"Supported phones: {phones_mapping} \n"
                f"Phone: {p} is not in a dictionary."
            )
    texts = torch.tensor(phone_ids).long().unsqueeze(0)
    text_lens = torch.tensor([texts.shape[1]]).long()
    ids = [f"{args.speaker_id}_0_{args.emotion_id}"]
    speakers = torch.tensor([args.speaker_id])
    emotions = torch.tensor([args.emotion_id])
    mels, mel_lens, pitches, energies, durations, egemap_features = [None] * 6
    batch_dict = {
        "ids": ids,
        "speakers": speakers,
        "emotions": emotions,
        "texts": texts,
        "text_lens": text_lens,
        "mels": mels,
        "mel_lens": mel_lens,
        "pitches": pitches,
        "energies": energies,
        "durations": durations,
        "egemap_features": egemap_features,
    }
    return batch_dict


@torch.no_grad()
def inference(config: TrainConfig, args: argparse.Namespace) -> None:
    seed_everything(config.seed)
    vocoder = Generator(**asdict(config))
    stft = TorchSTFT(**asdict(config))
    vocoder_state_dict = load_checkpoint(config.vocoder_checkpoint_path)
    vocoder.load_state_dict(vocoder_state_dict["generator"])
    vocoder.remove_weight_norm()
    vocoder.eval()
    model = FastSpeechLightning.load_from_checkpoint(
        config.testing_checkpoint,
        config=config,
        vocoder=vocoder,
        stft=stft,
        train=False,
    )
    model.eval()
    torch.set_float32_matmul_precision(config.matmul_precision)
    input_dict = get_input_dict(config, args)
    model_output = model.model(model.device, input_dict)
    predicted_mel_len = model_output["mel_len"][0]
    predicted_mel_no_padding = model_output["predicted_mel"][0, :predicted_mel_len]
    generated_wav = synthesize_wav_from_mel(
        predicted_mel_no_padding, model.vocoder, model.stft
    )
    write_wav(args.generated_audio_path, generated_wav, config.sample_rate)


if __name__ == "__main__":
    set_up_logger("inference.log")
    config = TrainConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sq", "--phone_sequence", help="Sequence of phones for synthesis"
    )
    parser.add_argument(
        "-sp",
        "--speaker_id",
        help="Speaker id from 1 to 10, default is 5",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-emo",
        "--emotion_id",
        help="Emotion id from 0 to 4, default is 1 (angry)",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-p",
        "--generated_audio_path",
        help="Path ended with .wav where to save generated audio ",
        type=str,
        default="generation_from_phoneme_sequence.wav",
    )
    args = parser.parse_args()
    inference(config, args)
