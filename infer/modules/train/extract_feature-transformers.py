import os
import sys
import traceback
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import HubertModel, HubertConfig, Wav2Vec2FeatureExtractor

n_part = int(sys.argv[1])
i_part = int(sys.argv[2])
exp_dir = sys.argv[3]
version = sys.argv[4]
is_half = sys.argv[5].lower() == "true"

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


model_path = "assets/hubert"  # Local directory containing model files
config_path = os.path.join(model_path, "config.json")  # Path to config.json
model_file_path = os.path.join(model_path, "pytorch_model.bin")  # Path to pytorch_model.bin
preprocessor_config_path = os.path.join(model_path, "preprocessor_config.json")
wavPath = f"{exp_dir}/1_16k_wavs"
outPath = (
    f"{exp_dir}/3_feature256"
    if version == "v1"
    else f"{exp_dir}/3_feature768"
)
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


if not os.path.exists(config_path) or not os.path.exists(model_file_path):
    printt(
        f"Error: Model files not found in {model_path}. "
        "Please ensure the directory contains 'config.json' and 'pytorch_model.bin'."
    )
    exit(0)

# Load the HuBERT model and feature extractor from Transformers
config = HubertConfig.from_pretrained(model_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
# Determine which model to load based on the configuration
if "HubertModelWithFinalProj" in config.architectures:
    class HubertModelWithFinalProj(HubertModel):
        def __init__(self, config):
            super().__init__(config)

        # The final projection layer is only used for backward compatibility.
        # Following https://github.com/auspicious3000/contentvec/issues/6
        # Remove this layer is necessary to achieve the desired outcome.
            self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)
    model = HubertModelWithFinalProj.from_pretrained(model_path)
else:
    model = HubertModel.from_pretrained(model_path)
model = model.to(device)

if is_half and device not in ["mps", "cpu"]:
    model = model.half()
model.eval()

todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
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

                feats = readwave(wav_path, normalize=False)  # Normalize input
                inputs = feature_extractor(
                    feats.squeeze(0).numpy(),
                    sampling_rate=16000,
                    return_tensors="pt",
                ).to(device)

                if is_half and device not in ["mps", "cpu"]:
                    inputs["input_values"] = inputs["input_values"].half()

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    if version == "v1":
                        feats = outputs.hidden_states[9]  # Use the 9th layer for v1
                    else:
                        feats = outputs.last_hidden_state  # Use the last layer for others

                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    printt(f"Ошибка: Файл {file} содержит некорректные значения (NaN).")
                if idx % n == 0:
                    printt(f"{idx}/{len(todo)} | {feats.shape}")
        except:
            printt(traceback.format_exc())
    printt("Все признаки извлечены!")
