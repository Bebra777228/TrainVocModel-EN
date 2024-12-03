import hashlib
import datetime
import traceback
from collections import OrderedDict

import torch


def extract_model(
    sr,
    hps,
    ckpt,
    name,
    step,
    epoch,
    if_f0,
    version,
    save_path,
):
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
            hps.data.sample_rate,
        ]

        opt["epoch"] = f"e{epoch}"
        opt["step"] = f"s{step}"
        opt["sr"] = sr
        opt["f0"] = if_f0
        opt["version"] = version
        opt["creation_date"] = datetime.datetime.now().isoformat()

        hash_input = f"{str(ckpt)} {epoch} {step} {datetime.datetime.now().isoformat()}"
        model_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        opt["model_hash"] = model_hash
        opt["model_name"] = name
        opt["learning_environment"] = "PolTrain"

        torch.save(opt, save_path)

        return "Модель успешно сохранена!"
    except:
        return traceback.format_exc()
