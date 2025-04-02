import os
import traceback
from collections import OrderedDict

import torch


def extract_model(hps, ckpt, name, epoch, step, sample_rate, model_dir, final_save):
    weights_dir = os.path.join(model_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    filename = f"{name}_e{epoch}_s{step}.pth" if not final_save else f"{name}.pth"
    filepath = os.path.join(weights_dir, filename)

    try:
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sampling_rate,
        ]

        opt["model_name"] = name
        opt["epoch"] = epoch
        opt["step"] = step
        opt["sr"] = sample_rate
        opt["f0"] = True
        opt["version"] = "v2"

        torch.save(opt, filepath)

        return f"Модель '{filename}' успешно сохранена!"
    except Exception as e:
        return f"Ошибка при сохранении модели: {traceback.format_exc()}"
