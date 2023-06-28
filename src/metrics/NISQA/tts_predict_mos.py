import numpy as np

from src.metrics.NISQA.nisqa.NISQA_model import nisqaModel


def get_mos(audio_path: str, output_dir_path: str):
    args = {
        "mode": "predict_file",
        "pretrained_model": "src/metrics/NISQA/weights/nisqa_tts.tar",
        "ms_channel": 1,
        "deg": audio_path,
        "output_dir": output_dir_path,
    }
    nisqa = nisqaModel(args)
    mos_score = nisqa.predict()["mos_pred"].item()
    return mos_score


def get_mos_scores(audio_dir_path: str, output_dir_path: str):
    args = {
        "mode": "predict_dir",
        "pretrained_model": "src/metrics/NISQA/weights/nisqa_tts.tar",
        "ms_channel": 1,
        "data_dir": audio_dir_path,
        "output_dir": output_dir_path,
    }
    nisqa = nisqaModel(args)
    mos_scores_df = nisqa.predict()
    filenames = [i[:-4] for i in np.array(mos_scores_df["deg"])]
    scores = np.array(mos_scores_df["mos_pred"])
    return dict(zip(filenames, scores))
