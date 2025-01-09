import os
import sys
import traceback

import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

exp_dir = sys.argv[1]

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

f = open(f"{exp_dir}/logfile.log", "a+")


def printt(strr):
    print(strr)
    f.write(f"{strr}\n")
    f.flush()


model_path = "assets/hubert/hubert_base.pt"

wavPath = f"{exp_dir}/1_16k_wavs"
outPath = f"{exp_dir}/3_feature768"
os.makedirs(outPath, exist_ok=True)


def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats

if os.access(model_path, os.F_OK) == False:
    printt(f"Error: Extracting is shut down because {model_path} does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main")
    exit(0)
models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path], suffix="")
model = models[0]
model = model.to(device)
model.eval()

todo = sorted(list(os.listdir(wavPath)))[0::1]
n = max(1, len(todo) // 10)
if len(todo) == 0:
    error_message = (
        "ОШИБКА: Не найдено ни одного признака для обработки.\n"
        "Возможные причины:\n"
        "1. Датасет не имеет звука.\n"
        "2. Датасет слишком тихий.\n"
        "3. Датасет слишком короткий."
    )
    printt(error_message)
    sys.exit(1)
else:
    printt(f"Фрагментов готовых к обработке - {len(todo)}")
    printt("Извлечение признаков...")
    for idx, file in enumerate(todo):
        try:
            if file.endswith(".wav"):
                wav_path = f"{wavPath}/{file}"
                out_path = f"{outPath}/{file.replace('wav', 'npy')}"

                if os.path.exists(out_path):
                    continue

                feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                inputs = {
                    "source": feats.to(device),
                    "padding_mask": padding_mask.to(device),
                    "output_layer": 12,
                }
                with torch.no_grad():
                    logits = model.extract_features(**inputs)
                    feats = logits[0]

                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    printt(f"Ошибка: Файл {file} содержит некорректные значения (NaN).")
                if idx % n == 0:
                    printt(f"{idx}/{len(todo)} | {feats.shape}")
            printt("Все признаки извлечены!")
        except:
            printt(traceback.format_exc())
