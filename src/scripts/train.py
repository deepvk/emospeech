from dataclasses import asdict
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

from config.config import TrainConfig
from src.dataset.dataset import get_dataloader
from src.models import Generator, TorchSTFT
from src.models.acoustic_model.fastspeech.lightning_model import \
    FastSpeechLightning
from src.utils.utils import set_up_logger
from src.utils.vocoder_utils import load_checkpoint


def train(config: TrainConfig) -> None:
    seed_everything(config.seed)
    vocoder = Generator(**asdict(config))
    stft = TorchSTFT(**asdict(config))
    vocoder_state_dict = load_checkpoint(config.vocoder_checkpoint_path)
    vocoder.load_state_dict(vocoder_state_dict["generator"])
    vocoder.remove_weight_norm()
    vocoder.eval()
    train_loader = get_dataloader(config, "train")
    val_loader = get_dataloader(config, "val")
    test_loader = get_dataloader(config, "test")
    model = FastSpeechLightning(config, vocoder, stft)

    wandb_logger = WandbLogger(
        project=config.wandb_project,
        log_model=config.wandb_log_model,
        offline=config.wandb_offline,
        config=config,
        resume=config.resume_wandb_run,
        id=config.wandb_run_id,
    )
    Path(config.lightning_checkpoint_path).mkdir(exist_ok=True, parents=True)
    callbacks = ModelCheckpoint(
        dirpath=config.lightning_checkpoint_path,
        monitor="val_mos/generated_audio_mos_mean",
        save_top_k=config.save_top_k_model_weights,
        mode=config.metric_monitor_mode,
    )

    progress_bar = TQDMProgressBar(refresh_rate=config.wandb_progress_bar_refresh_rate)
    wandb_logger.watch(model.model, log_graph=False)

    trainer = Trainer(
        max_steps=config.total_training_steps,
        check_val_every_n_epoch=config.val_each_epoch,
        log_every_n_steps=config.wandb_log_every_n_steps,
        logger=wandb_logger,
        accelerator="gpu" if config.device == "cuda" else "cpu",
        devices=list(config.devices) if config.devices else "auto",
        callbacks=[callbacks, progress_bar],
        limit_val_batches=config.limit_val_batches,
        limit_test_batches=config.limit_test_batches,
        num_sanity_val_steps=config.num_sanity_val_steps,
        strategy=config.strategy,
        deterministic=True,
        enable_checkpointing=True,
        precision=config.precision,
    )
    torch.set_float32_matmul_precision(config.matmul_precision)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=(
            Path(config.lightning_checkpoint_path) / config.train_from_checkpoint
            if config.train_from_checkpoint
            else None
        ),
    )
    trainer.validate(model, dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    set_up_logger("train.log")
    config = TrainConfig()
    train(config)
