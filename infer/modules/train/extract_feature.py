import os
import sys
import traceback
import argparse

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

model_name = sys.argv[4]
device = sys.argv[1]
n_part = int(sys.argv[2])
i_part = int(sys.argv[3])
if len(sys.argv) == 7:
    exp_dir = sys.argv[4]
    version = sys.argv[5]
    is_half = sys.argv[6].lower() == "true"
else:
    i_gpu = sys.argv[4]
    exp_dir = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    version = sys.argv[6]
    is_half = sys.argv[7].lower() == "true"
import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

# Определение устройства
if "privateuseone" not in device:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
else:
    import torch_directml
    device = torch_directml.device(torch_directml.default_device())

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

f = open(f"{model_name}/logfile.log".format(exp_dir), "a+")

def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()

printt(" ".join(sys.argv))
model_path = "assets/hubert/hubert_base.pt"

printt("exp_dir: " + exp_dir)
wavPath = f"{model_name}/1_16k_wavs".format(exp_dir)
outPath = (
    f"{model_name}/3_feature256".format(exp_dir) if version == "v1" else f"{model_name}/3_feature768".format(exp_dir)
)
os.makedirs(outPath, exist_ok=True)

# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats

# HuBERT model
printt("load model(s) from {}".format(model_path))
# if hubert model is exist
if os.access(model_path, os.F_OK) == False:
    printt(
        "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
        % model_path
    )
    exit(0)
models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
model = model.to(device)
printt("move model to %s" % device)
if is_half:
    if device not in ["mps", "cpu"]:
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
    printt(f"Признаков готовых к обработке - {len(todo)}")
    printt("Извлечение признаков...")
    for idx, file in enumerate(todo):
        try:
            if file.endswith(".wav"):
                wav_path = "%s/%s" % (wavPath, file)
                out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

                if os.path.exists(out_path):
                    continue

                feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                inputs = {
                    "source": (
                        feats.half().to(device)
                        if is_half and device not in ["mps", "cpu"]
                        else feats.to(device)
                    ),
                    "padding_mask": padding_mask.to(device),
                    "output_layer": 9 if version == "v1" else 12,
                }
                with torch.no_grad():
                    logits = model.extract_features(**inputs)
                    feats = (
                        model.final_proj(logits[0]) if version == "v1" else logits[0]
                    )

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