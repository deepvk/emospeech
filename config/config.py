from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union


@dataclass
class TrainConfig:
    # Preprocess
    n_threads: int = 16  # n_threads to parallel process utterance
    include_empty_intervals: bool = (
        True  # if True silence will be loaded from .TextGrid
    )

    mel_fmin: int = 0
    mel_fmax: int = 8000
    hop_length: int = 192
    stft_length: int = 768
    sample_rate: int = 16000
    window_length: int = 768
    n_mel_channels: int = 80

    raw_data_path: Path = "/app/data/data/ssw_esd"
    val_ids_path: Path = "/app/data/val_ids.txt"
    test_ids_path: Path = "/app/data/test_ids.txt"
    preprocessed_data_path: Path = Path("/app/data/preprocessed")

    egemap_feature_names: Tuple[str] = (
        "F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
        "F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
        "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
        "spectralFlux_sma3_amean",
        "HNRdBACF_sma3nz_amean",
        "mfcc1V_sma3nz_amean",
        "equivalentSoundLevel_dBp",
    )

    # Vocoder
    vocoder_checkpoint_path: str = "/app/data/g_01800000"
    istft_resblock_kernel_sizes: Tuple[int] = (3, 7, 11)
    istft_upsample_rates: Tuple[int] = (6, 8)
    istft_upsample_initial_channel: int = 512
    istft_upsample_kernel_sizes: Tuple[int] = (16, 16)
    istft_resblock_dilation_sizes: Tuple[Tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    gen_istft_n_fft: int = 16
    gen_istft_hop_size: int = 4

    # Transformer Encoder
    padding_index: int = 0
    max_seq_len: int = 2000
    phones_mapping_path: Path = Path("/app/data/preprocessed/phones.json")
    transformer_encoder_hidden: int = 512
    transformer_encoder_layer: int = 9
    transformer_encoder_head: int = 2
    transformer_conv_filter_size: int = 512
    transformer_conv_kernel_size: tuple = (9, 1)
    transformer_encoder_dropout: float = 0.2

    # Transformer Decoder
    transformer_decoder_hidden: int = 512
    transformer_decoder_layer: int = 9
    transformer_decoder_head: int = 2
    transformer_decoder_dropout: float = 0.2

    # Emotion Conditioning
    emotion_emb_hidden_size: int = 256
    stack_speaker_with_emotion_embedding: bool = (
        True  # if True speaker and emotion embedding would be concatenated
    )
    n_egemap_features: int = 2  # 0, ... 7, could be more than 7 if adjust preprocessing
    conditional_layer_norm: bool = True  # if False Layer Norm applied
    conditional_cross_attention: bool = (
        True  # if False emotion and speaker embeddings added to the encoder output
    )

    # Discriminator
    compute_adversarial_loss: bool = True
    compute_fm_loss: bool = True
    optimizer_lrate_d: float = 1e-4
    optimizer_betas_d: tuple[float, float] = (0.5, 0.9)
    kernels_d: tuple[float, ...] = (3, 5, 5, 5, 3)
    strides_d: tuple[float, ...] = (1, 2, 2, 1, 1)

    # FastSpeech2, Variance Predictor
    speaker_emb_hidden_size: int = 256
    variance_embedding_n_bins: int = 256
    variance_predictor_kernel_size: int = 3
    variance_predictor_filter_size: int = 256
    variance_predictor_dropout: float = 0.5

    # Dataset
    multi_speaker: bool = True
    multi_emotion: bool = True
    n_emotions: int = 5
    n_speakers: int = 10
    train_batch_size: int = 64
    val_batch_size: int = 32
    device: str = "cuda"

    # Train
    seed: int = 3
    precision: str = 32
    matmul_precision: str = "high"
    lightning_checkpoint_path: str = (
        "/app/data/checkpoints/emospeech.ckpt"  # directory to save checkpoints
    )
    train_from_checkpoint: Optional[str] = (
        None  # filename in <lightning_checkpoint_path> directory
    )
    num_workers: int = 1
    test_wav_files_directory: str = "/app/data/wav"
    test_mos_files_directory: str = "/app/data/mos"
    total_training_steps: int = 100000
    val_each_epoch: int = 20
    val_audio_log_each_step: int = (
        1  # if greater than one will log audio each <n> step, set to save storage
    )

    # Test / Inference
    testing_checkpoint: str = (
        "/app/data/emospeech.ckpt"  # "data/deepvk_large_checkpoint/epoch=1079-step=127440.ckpt"
    )
    audio_save_path: str = (
        "/app/data/deepvk_test"  # directory where synthesised wav files will be saved
    )
    nisqa_save_path: str = (
        "/app/data/deepvk_test"  # directory where nisqa output files will be saved
    )
    limit_generation: int = (
        None  # if specified, will stop and do not iterate through all samples in testing loader
    )
    compute_nisqa_on_test: bool = (
        True  # is True will write NISQA scores and stds to test.log file
    )
    phones_path: str = "/app/data/phones.json"  # path to phones dictionary

    # Optimizer
    optimizer_grad_clip_val: float = 1.0
    optimizer_warm_up_step: float = 4000
    optimizer_anneal_steps: tuple[float, ...] = (300000, 400000, 500000)
    optimizer_anneal_rate: float = 0.3
    fastspeech_optimizer_betas: tuple[float, float] = (0.9, 0.98)
    fastspeech_optimizer_eps: float = 1e-9
    fastspeech_optimizer_weight_decay: float = 0.0

    # Wandb
    wandb_log_model: bool = False
    wandb_project: str = "EmoSpeech"
    wandb_run_id: str = (
        None  # if specified, continue to log into existing charts, use for training interrupted cases
    )
    resume_wandb_run: bool = (
        False  # if true will log data to the last wandb run in the specified project
    )
    strategy: str = "ddp_find_unused_parameters_true"
    wandb_offline: bool = False
    wandb_progress_bar_refresh_rate: int = 1
    wandb_log_every_n_steps: int = 1
    devices: Union[tuple, int] = (0, 1, 2, 3)
    limit_val_batches: Optional[int] = (
        4  # val_batch_size * limit_val_batches samples will be logged to wandb and saved locally each val step
    )
    limit_test_batches: Optional[int] = (
        4  # test_batch_size * limit_test_batches samples will be logged to wandb and saved locally during test
    )
    num_sanity_val_steps: int = 4
    save_top_k_model_weights: int = 5
    metric_monitor_mode: str = "max"  # 'min' or 'max'

    def __post_init__(self):
        self.hop_in_ms = self.hop_length / self.sample_rate
        if self.stack_speaker_with_emotion_embedding:
            self.emb_size_dis = (
                self.speaker_emb_hidden_size + self.emotion_emb_hidden_size
            )
        else:
            self.emb_size_dis = self.emotion_emb_hidden_size
