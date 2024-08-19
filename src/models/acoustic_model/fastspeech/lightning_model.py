import glob
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torchaudio
import wandb
from lightning import LightningModule

from config.config import TrainConfig
from src.metrics.NISQA.tts_predict_mos import get_mos_scores
from src.models import FastSpeech2, FastSpeech2Loss, Generator, TorchSTFT
from src.models.acoustic_model.discriminator.jcu_discriminator import JCU
from src.models.acoustic_model.discriminator.loss import AdversarialLoss
from src.utils.utils import (compute_mos_per_speaker, compute_overall_mos,
                             write_wav)
from src.utils.vocoder_utils import synthesize_wav_from_mel


class FastSpeechLightning(LightningModule):
    def __init__(
        self,
        config: TrainConfig,
        vocoder: Generator,
        stft: TorchSTFT,
        train: bool = True,
    ):
        super().__init__()
        if train:
            self.val_mos_files_directory = Path(config.test_mos_files_directory) / "val"
            self.val_wav_files_directory = Path(config.test_wav_files_directory) / "val"
            self.test_mos_files_directory = (
                Path(config.test_mos_files_directory) / "test"
            )
            self.test_wav_files_directory = (
                Path(config.test_wav_files_directory) / "test"
            )

            Path(self.val_mos_files_directory).mkdir(exist_ok=True, parents=True)
            Path(self.val_wav_files_directory).mkdir(exist_ok=True, parents=True)
            Path(self.test_mos_files_directory).mkdir(exist_ok=True, parents=True)
            Path(self.test_wav_files_directory).mkdir(exist_ok=True, parents=True)

        self.stft = stft
        self.config = config
        self.vocoder = vocoder
        self.model = FastSpeech2(config=config)
        self.discriminator = JCU(config=config)
        self.loss = FastSpeech2Loss(config=config)
        self.adversarial_loss = AdversarialLoss(config=config)
        self.automatic_optimization = False if config.compute_adversarial_loss else True

        self.anneal_rate = config.optimizer_anneal_rate
        self.anneal_steps = config.optimizer_anneal_steps
        self.n_warmup_steps = config.optimizer_warm_up_step
        self.init_lr = np.power(config.transformer_encoder_hidden, -0.5)

        self.save_hyperparameters(ignore=["vocoder", "stft"])
        self.ground_truth_audio_path = (
            Path(config.preprocessed_data_path) / "trimmed_wav"
        )

        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.cur_epoch_original_wav_dir = None

    def _scheduler(
        self, optimizer: torch.optim.Adam
    ) -> torch.optim.lr_scheduler.LambdaLR:
        def lr_lambda(current_step: int) -> float:
            current_step += 1
            lr = np.min(
                [
                    np.power(current_step, -0.5),
                    np.power(self.n_warmup_steps, -1.5) * current_step,
                ]
            )
            for s in self.config.optimizer_anneal_steps:
                if current_step > s:
                    lr = lr * self.anneal_rate
            return self.init_lr * lr

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def _disc_scheduler(optimizer):
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1)

    def configure_optimizers(self) -> tuple[list[torch.optim.Adam], list[dict]]:
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1,
            betas=self.config.fastspeech_optimizer_betas,
            eps=self.config.fastspeech_optimizer_eps,
            weight_decay=self.config.fastspeech_optimizer_weight_decay,
        )
        scheduler = {
            "scheduler": self._scheduler(self.optimizer),
            "interval": "step",
            "frequency": 1,
        }

        self.optimizer_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=float(self.config.optimizer_lrate_d),
            betas=self.config.optimizer_betas_d,
        )

        scheduler_disc = {
            "scheduler": self._disc_scheduler(self.optimizer_disc),
            "interval": "step",
            "frequency": 1,
        }

        if self.config.compute_adversarial_loss:
            return [self.optimizer, self.optimizer_disc], [scheduler, scheduler_disc]
        else:
            return [self.optimizer], [scheduler]

    def _fs_step(self, input_dict, output_dict):
        losses = self.loss(self.device, input_dict, output_dict)
        if self.config.compute_adversarial_loss:
            generator_adversarial_loss, fm_loss = self.adversarial_loss.generator_loss(
                input_dict, output_dict, self.discriminator.to(self.device)
            )
            if self.config.compute_fm_loss and fm_loss > 0:
                fm_alpha = (losses["total_loss"] / fm_loss).detach()
                fm_loss = fm_alpha * fm_loss
            losses["total_loss"] += generator_adversarial_loss + fm_loss
            losses["adv_g_loss"] = generator_adversarial_loss
            losses["fm_loss"] = fm_loss
        gen_log_dict = losses
        gen_log_dict["optimizer_rate/optimizer"] = self.optimizer.param_groups[0]["lr"]
        self.log_dict(gen_log_dict, on_step=True, on_epoch=False)

        return losses["total_loss"]

    def _ds_step(self, input_dict, output_dict):
        loss = self.adversarial_loss.discriminator_loss(
            input_dict, output_dict, self.discriminator.to(self.device)
        )
        disc_log_dict = {
            "optimizer_rate/discriminator": self.optimizer_disc.param_groups[0]["lr"],
            "adv_d_loss": loss,
        }
        self.log_dict(disc_log_dict, on_step=True, on_epoch=False)

        return loss

    @staticmethod
    def _get_batch_dict_from_dataloader(batch: torch.Tensor, validation: bool) -> dict:
        # dataset returns tuple of tensors and dataloader returns tensor object, convert it to a dict for fs model
        ids, speakers, emotions, texts, text_lens = batch[0][:5]
        batch_dict = {
            "ids": ids,
            "speakers": speakers,
            "emotions": emotions,
            "texts": texts,
            "text_lens": text_lens,
        }
        if validation:
            mels, mel_lens, pitches, energies, durations, egemap_features = [None] * 6
        else:
            mels, mel_lens, pitches, energies, durations, egemap_features = batch[0][5:]
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

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        batch_dict = self._get_batch_dict_from_dataloader(batch, validation=False)
        if not self.automatic_optimization:
            # generator
            g_opt, d_opt = self.optimizers()
            g_sch, d_sch = self.lr_schedulers()

            output_dict = self.model(self.device, batch_dict)
            generator_loss = self._fs_step(batch_dict, output_dict)
            g_opt.zero_grad()
            self.manual_backward(generator_loss)
            g_opt.step()
            g_sch.step()

            # discriminator
            output_dict = self.model(self.device, batch_dict)
            discriminator_loss = self._ds_step(batch_dict, output_dict)
            d_opt.zero_grad()
            self.manual_backward(discriminator_loss)
            d_opt.step()
            d_sch.step()
        else:
            output_dict = self.model(self.device, batch_dict)
            return self._fs_step(batch_dict, output_dict)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        val_logs_dict = self._val_test_shared_step(batch, mode="val")
        self.validation_step_outputs.append(val_logs_dict)
        return val_logs_dict

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        test_logs_dict = self._val_test_shared_step(batch, mode="test")
        self.test_step_outputs.append(test_logs_dict)
        return test_logs_dict

    def on_validation_epoch_end(self) -> None:
        log_dict = self._val_test_shared_epoch_end(self.validation_step_outputs)

        Path((Path(self.cur_mos_files_directory_path) / "original")).mkdir(
            exist_ok=True
        )
        original_mos = get_mos_scores(
            str(self.cur_epoch_original_wav_dir),
            str(Path(self.cur_mos_files_directory_path) / "original"),
        )
        self.original_mos_score_val = compute_overall_mos(original_mos)
        self.original_mos_scores_val_per_speaker = compute_mos_per_speaker(original_mos)

        Path((Path(self.cur_mos_files_directory_path) / "reconstructed")).mkdir(
            exist_ok=True
        )
        reconstructed_mos = get_mos_scores(
            str(self.cur_epoch_reconstructed_wav_dir),
            str(Path(self.cur_mos_files_directory_path) / "reconstructed"),
        )
        self.reconstructed_mos_score_val = compute_overall_mos(reconstructed_mos)
        self.reconstructed_mos_scores_val_per_speaker = compute_mos_per_speaker(
            reconstructed_mos
        )

        Path((Path(self.cur_mos_files_directory_path) / "generated")).mkdir(
            exist_ok=True
        )

        if (
            len(glob.glob(str(self.cur_epoch_generated_wav_dir / "*.wav"))) > 0
            and self.global_step > 100
        ):
            generated_mos = get_mos_scores(
                str(self.cur_epoch_generated_wav_dir),
                str(Path(self.cur_mos_files_directory_path) / "generated"),
            )
            generated_mos_score = compute_overall_mos(generated_mos)
            generated_mos_scores_per_speaker = compute_mos_per_speaker(generated_mos)
        else:
            generated_mos_score, generated_mos_scores_per_speaker = None, None

        log_dict[f"val_mos/gt_audio_mos_mean"] = torch.FloatTensor(
            [self.original_mos_score_val[0]]
        )
        log_dict[f"val_mos/gt_audio_mos_speaker_std"] = torch.FloatTensor(
            [self.original_mos_score_val[1]]
        )
        for speaker in list(self.original_mos_scores_val_per_speaker.keys()):
            log_dict[f"val_mos_per_speaker/{speaker}/gt"] = torch.FloatTensor(
                [self.original_mos_scores_val_per_speaker[speaker]]
            )

        log_dict[f"val_mos/reconstructed_audio_mos_mean"] = torch.FloatTensor(
            [self.reconstructed_mos_score_val[0]]
        )
        log_dict[f"val_mos/reconstructed_audio_mos_speaker_std"] = torch.FloatTensor(
            [self.reconstructed_mos_score_val[1]]
        )
        for speaker in list(self.reconstructed_mos_scores_val_per_speaker.keys()):
            log_dict[f"val_mos_per_speaker/{speaker}/reconstructed"] = (
                torch.FloatTensor(
                    [self.reconstructed_mos_scores_val_per_speaker[speaker]]
                )
            )

        if generated_mos_score and generated_mos_scores_per_speaker:
            log_dict[f"val_mos/generated_audio_mos_mean"] = torch.FloatTensor(
                [generated_mos_score[0]]
            )
            log_dict[f"val_mos/generated_audio_mos_speaker_std"] = torch.FloatTensor(
                [generated_mos_score[1]]
            )
            for speaker in list(generated_mos_scores_per_speaker.keys()):
                log_dict[f"val_mos_per_speaker/{speaker}/generated"] = (
                    torch.FloatTensor([generated_mos_scores_per_speaker[speaker]])
                )
        else:
            # as we save best models monitoring val mos, write 0 to the dict if no audios were generated
            log_dict[f"val_mos/generated_audio_mos_mean"] = torch.FloatTensor([0.0])

        self.log_dict(log_dict, sync_dist=True)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        log_dict = self._val_test_shared_epoch_end(self.test_step_outputs)
        Path((Path(self.cur_mos_files_directory_path) / "original")).mkdir(
            exist_ok=True
        )
        original_mos = get_mos_scores(
            str(self.cur_epoch_original_wav_dir),
            str(Path(self.cur_mos_files_directory_path) / "original"),
        )
        original_mos_score = compute_overall_mos(original_mos)
        original_mos_scores_per_speaker = compute_mos_per_speaker(original_mos)

        Path((Path(self.cur_mos_files_directory_path) / "reconstructed")).mkdir(
            exist_ok=True
        )
        reconstructed_mos = get_mos_scores(
            str(self.cur_epoch_reconstructed_wav_dir),
            str(Path(self.cur_mos_files_directory_path) / "reconstructed"),
        )
        reconstructed_mos_score = compute_overall_mos(reconstructed_mos)
        reconstructed_mos_scores_per_speaker = compute_mos_per_speaker(
            reconstructed_mos
        )

        Path((Path(self.cur_mos_files_directory_path) / "generated")).mkdir(
            exist_ok=True
        )
        generated_mos = get_mos_scores(
            str(self.cur_epoch_generated_wav_dir),
            str(Path(self.cur_mos_files_directory_path) / "generated"),
        )
        generated_mos_score = compute_overall_mos(generated_mos)
        generated_mos_scores_per_speaker = compute_mos_per_speaker(generated_mos)

        log_dict[f"test_mos/gt_audio_mos_mean"] = torch.FloatTensor(
            [original_mos_score[0]]
        )
        log_dict[f"test_mos/gt_audio_mos_speaker_std"] = torch.FloatTensor(
            [original_mos_score[1]]
        )
        for speaker in list(original_mos_scores_per_speaker.keys()):
            log_dict[f"test_mos_per_speaker/{speaker}/gt"] = torch.FloatTensor(
                [original_mos_scores_per_speaker[speaker]]
            )

        log_dict[f"test_mos/reconstructed_audio_mos_mean"] = torch.FloatTensor(
            [reconstructed_mos_score[0]]
        )
        log_dict[f"test_mos/reconstructed_audio_mos_speaker_std"] = torch.FloatTensor(
            [reconstructed_mos_score[1]]
        )
        for speaker in list(reconstructed_mos_scores_per_speaker.keys()):
            log_dict[f"test_mos_per_speaker/{speaker}/reconstructed"] = (
                torch.FloatTensor([reconstructed_mos_scores_per_speaker[speaker]])
            )

        if generated_mos_score and generated_mos_scores_per_speaker:
            log_dict[f"test_mos/generated_audio_mos_mean"] = torch.FloatTensor(
                [generated_mos_score[0]]
            )
            log_dict[f"test_mos/generated_audio_mos_speaker_std"] = torch.FloatTensor(
                [generated_mos_score[1]]
            )
            for speaker in list(generated_mos_scores_per_speaker.keys()):
                log_dict[f"test_mos_per_speaker/{speaker}/generated"] = (
                    torch.FloatTensor([generated_mos_scores_per_speaker[speaker]])
                )

        self.log_dict(log_dict, sync_dist=True)
        self.test_step_outputs.clear()

    @staticmethod
    def _val_test_shared_epoch_end(outputs: list[dict]) -> dict:
        log_dict = dict()
        keys = list(set(chain.from_iterable(sub.keys() for sub in outputs)))
        for key in keys:
            log_dict[key] = torch.stack([x[key] for x in outputs]).mean()
        return log_dict

    def _val_test_shared_step(self, batch: torch.Tensor, mode: str):
        with torch.no_grad():
            logs_dict = dict()

            batch_dict_no_tf = self._get_batch_dict_from_dataloader(
                batch, validation=True
            )
            output_dict_no_tf = self.model(self.device, batch_dict_no_tf)
            # use tf (gt durations, pitch, energies) to compute val mel loss
            batch_dict_with_tf = self._get_batch_dict_from_dataloader(
                batch, validation=False
            )

            # predict durations on val -> target & pred mel have different shapes -> don't compute mel_loss
            no_mel_loss_dict = self.loss(
                self.device,
                batch_dict_with_tf,
                output_dict_no_tf,
                compute_mel_loss=False,
            )
            logs_dict[f"{mode}/pitch_loss"] = no_mel_loss_dict["pitch_loss"]
            logs_dict[f"{mode}/energy_loss"] = no_mel_loss_dict["energy_loss"]
            logs_dict[f"{mode}/duration_loss"] = no_mel_loss_dict["duration_loss"]
            if "egemap_loss" in no_mel_loss_dict:
                logs_dict[f"{mode}/egemap_loss"] = no_mel_loss_dict["egemap_loss"]

            # use tf (gt durations, pitch, energies) to compute val mel loss
            output_dict_with_tf = self.model(self.device, batch_dict_with_tf)
            mel_loss_dict = self.loss(
                self.device,
                batch_dict_with_tf,
                output_dict_with_tf,
                compute_mel_loss=True,
            )
            logs_dict[f"{mode}/total_loss"] = mel_loss_dict["total_loss"]
            logs_dict[f"{mode}/mel_loss"] = mel_loss_dict["mel_loss"]
            # logs_dict[f"{mode}/postnet_mel_loss"] = mel_loss_dict["postnet_mel_loss"]
            if "lpips_loss" in mel_loss_dict:
                logs_dict[f"{mode}/lpips_loss"] = mel_loss_dict["lpips_loss"]

            # compute adversarial & fm loss if needed
            if self.config.compute_adversarial_loss:
                adv_g_loss, fm_loss = self.adversarial_loss.generator_loss(
                    batch_dict_with_tf,
                    output_dict_with_tf,
                    self.discriminator.to(self.device),
                )
                logs_dict[f"{mode}/adv_g_loss"] = adv_g_loss

                if self.config.compute_fm_loss and fm_loss > 0:
                    fm_alpha = (mel_loss_dict["total_loss"] / fm_loss).detach()
                    fm_loss = fm_alpha * fm_loss
                # print(f"VAL FM LOSS: {fm_loss}")
                logs_dict[f"{mode}/fm_loss"] = fm_loss
                logs_dict[f"{mode}/total_loss"] += adv_g_loss + fm_loss
                adv_d_loss = self.adversarial_loss.discriminator_loss(
                    batch_dict_with_tf,
                    output_dict_with_tf,
                    self.discriminator.to(self.device),
                )
                logs_dict[f"{mode}/adv_d_loss"] = adv_d_loss

            logs_dict = self._log_wav(
                logs_dict, batch_dict_with_tf, output_dict_no_tf, mode
            )

        return logs_dict

    def _log_wav(
        self, logs_dict: dict, batch: dict, output_dict_no_tf: dict, mode: str
    ) -> dict:
        if mode == "val":
            wav_files_directory_path = (
                Path(self.val_wav_files_directory)
                / f"{self.current_epoch} / {self.global_rank}"
            )
            self.cur_mos_files_directory_path = (
                Path(self.val_mos_files_directory)
                / f"{self.current_epoch} / {self.global_rank}"
            )
        else:
            wav_files_directory_path = (
                Path(self.test_wav_files_directory)
                / f"{self.current_epoch} / {self.global_rank}"
            )
            self.cur_mos_files_directory_path = (
                Path(self.test_mos_files_directory)
                / f"{self.current_epoch} / {self.global_rank}"
            )

        Path(wav_files_directory_path).mkdir(exist_ok=True, parents=True)
        Path(self.cur_mos_files_directory_path).mkdir(exist_ok=True, parents=True)

        for i, tag in enumerate(batch["ids"]):
            speaker_id = batch["speakers"][i]
            # Log Reconstructed (by vocoder) and GT speech just once
            if (
                self.current_epoch == 0
                or mode == "test"
                or not self.cur_epoch_original_wav_dir
            ):
                ground_truth_audio_path = Path(self.ground_truth_audio_path) / Path(
                    tag
                ).with_suffix(".wav")
                ground_truth_wav = torchaudio.load(ground_truth_audio_path)[0].squeeze(
                    0
                )
                self.cur_epoch_original_wav_dir = Path(
                    Path(wav_files_directory_path) / "original"
                )
                self.cur_epoch_original_wav_dir.mkdir(exist_ok=True)
                ground_truth_audio_path = self.cur_epoch_original_wav_dir / Path(
                    tag
                ).with_suffix(".wav")
                write_wav(
                    ground_truth_audio_path,
                    ground_truth_wav.detach().cpu().numpy(),
                    self.config.sample_rate,
                )
                self.logger.experiment.log(
                    {
                        f"{mode}_audio/{speaker_id}/original/{tag}": wandb.Audio(
                            ground_truth_wav,
                            caption=f"original_{tag}",
                            sample_rate=self.config.sample_rate,
                        )
                    }
                )

                gt_mel_no_padding = batch["mels"][i, : batch["mel_lens"][i]]
                reconstructed_wav = synthesize_wav_from_mel(
                    gt_mel_no_padding, self.vocoder, self.stft
                )
                self.cur_epoch_reconstructed_wav_dir = Path(
                    Path(wav_files_directory_path) / "reconstructed"
                )
                self.cur_epoch_reconstructed_wav_dir.mkdir(exist_ok=True)
                reconstructed_audio_path = self.cur_epoch_reconstructed_wav_dir / Path(
                    tag
                ).with_suffix(".wav")
                write_wav(
                    str(reconstructed_audio_path),
                    reconstructed_wav,
                    self.config.sample_rate,
                )
                self.logger.experiment.log(
                    {
                        f"{mode}_audio/{speaker_id}/reconstructed/{tag}": wandb.Audio(
                            reconstructed_wav,
                            caption=f"reconstructed_{tag}",
                            sample_rate=self.config.sample_rate,
                        )
                    }
                )

            predicted_mel_len = output_dict_no_tf["mel_len"][i]
            predicted_mel_no_padding = output_dict_no_tf["predicted_mel"][
                i, :predicted_mel_len
            ]
            # skip first training steps as predicted mel is too short for generating by vocoder
            self.cur_epoch_generated_wav_dir = Path(
                Path(wav_files_directory_path) / "generated"
            )
            self.cur_epoch_generated_wav_dir.mkdir(exist_ok=True)
            if predicted_mel_no_padding.shape[0] > max(
                self.config.istft_upsample_kernel_sizes
            ):
                generated_wav = synthesize_wav_from_mel(
                    predicted_mel_no_padding, self.vocoder, self.stft
                )
                if generated_wav.shape[0] > self.config.sample_rate:
                    generated_audio_path = self.cur_epoch_generated_wav_dir / Path(
                        tag
                    ).with_suffix(".wav")
                    write_wav(
                        str(generated_audio_path),
                        generated_wav,
                        self.config.sample_rate,
                    )
                    if (
                        mode == "test"
                        or self.global_step % self.config.val_audio_log_each_step == 0
                    ):
                        self.logger.experiment.log(
                            {
                                f"{mode}_audio/{speaker_id}/generated/{tag}": wandb.Audio(
                                    generated_wav,
                                    caption=f"generated_{tag}",
                                    sample_rate=self.config.sample_rate,
                                )
                            }
                        )

        return logs_dict
