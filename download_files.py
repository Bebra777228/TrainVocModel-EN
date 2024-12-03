import os
import subprocess
from urllib.parse import urlparse


# Функция для установки rmvpe и hubert_base
def install_rmvpe_and_hubert_base(assets_folder, embedder):
    hugg_link = "https://huggingface.co/Politrees/RVC_resources/resolve/main"
    file_links = {
        "rmvpe/rmvpe.pt": f"{hugg_link}/predictors/rmvpe.pt",
        "hubert/hubert_base.pt": f"{hugg_link}/embedders/{embedder}"
    }

    os.makedirs(assets_folder, exist_ok=True)

    for file, link in file_links.items():
        file_path = os.path.join(assets_folder, file)
        if not os.path.exists(file_path):
            try:
                subprocess.run(['wget', '-O', file_path, link], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Ошибка установки {file}: {e}")


# Функция для установки претрейнов
def install_pretrains(pretrain_outpath, pretrain, custom_pretrained, d_pretrained_link, g_pretrained_link, sample_rate):
    hugg_link = "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2"
    param_aria = "--console-log-level=error -c -x 16 -s 16 -k 1M"

    MODELS = {
        "* Default —> (Дискретизация — ВСЕ)": [
            (f"{sample_rate}/Default/f0D{sample_rate}.pth", f"default_D.pth"),
            (f"{sample_rate}/Default/f0G{sample_rate}.pth", f"default_G.pth"),
        ],
        "* Snowie —> (Дискретизация — 40k)": [
            (f"40k/Snowie/D_Snowie_40k.pth", f"Snowie_D.pth"),
            (f"40k/Snowie/G_Snowie_40k.pth", f"Snowie_G.pth"),
        ],
        "* Snowie v2 —> (Дискретизация — 40k и 48k)": [
            (f"{sample_rate}/Snowie/D_SnowieV2_{sample_rate}.pth", f"SnowieV2_D.pth"),
            (f"{sample_rate}/Snowie/G_SnowieV2_{sample_rate}.pth", f"SnowieV2_G.pth"),
        ],
        "* Snowie v3 —> (Дискретизация — ВСЕ)": [
            (f"{sample_rate}/Snowie/D_SnowieV3.1_{sample_rate}.pth", f"SnowieV3_D.pth"),
            (f"{sample_rate}/Snowie/G_SnowieV3.1_{sample_rate}.pth", f"SnowieV3_G.pth"),
        ],
        "* Ov2Super —> (Дискретизация — 40k)": [
            (f"40k/Ov2/f0Ov2Super40kD.pth", f"Ov2Super_D.pth"),
            (f"40k/Ov2/f0Ov2Super40kG.pth", f"Ov2Super_G.pth"),
        ],
        "* RIN_E3 —> (Дискретизация — 40k)": [
            (f"40k/RIN_E/D_RIN_E3.pth", f"RinE3_D.pth"),
            (f"40k/RIN_E/G_RIN_E3.pth", f"RinE3_G.pth"),
        ],
        "* TITAN-Medium —> (Дискретизация — ВСЕ)": [
            (f"{sample_rate}/TITAN/D-f0{sample_rate}-TITAN-Medium.pth", f"TITAN_D.pth"),
            (f"{sample_rate}/TITAN/G-f0{sample_rate}-TITAN-Medium.pth", f"TITAN_G.pth"),
        ],
        "* Snowie + RIN_E3 —> (Дискретизация — 40k)": [
            (f"40k/Snowie/D_Snowie-X-Rin_40k.pth", f"SnowieV3_x_RinE3_D.pth"),
            (f"40k/Snowie/G_Snowie-X-Rin_40k.pth", f"SnowieV3_x_RinE3_G.pth"),
        ],
        "* Rigel —> (Дискретизация — 32k)": [
            (f"32k/Rigel/D_Rigel_32k.pth", f"Rigel_D.pth"),
            (f"32k/Rigel/G_Rigel_32k.pth", f"Rigel_G.pth"),
        ],
    }

    os.makedirs(pretrain_outpath, exist_ok=True)

    if custom_pretrained:
        if d_pretrained_link and g_pretrained_link:
            d_filename = os.path.basename(urlparse(d_pretrained_link).path)
            g_filename = os.path.basename(urlparse(g_pretrained_link).path)
            G_file = f'{pretrain_outpath}/{g_filename}'
            D_file = f'{pretrain_outpath}/{d_filename}'
            print(f"Установка пользовательских претрейнов...\nG_file - {g_filename}\nD_file - {d_filename}")
            subprocess.run(['aria2c', param_aria, g_pretrained_link, '-d', pretrain_outpath, '-o', g_filename], check=True)
            subprocess.run(['aria2c', param_aria, d_pretrained_link, '-d', pretrain_outpath, '-o', d_filename], check=True)
        else:
            raise ValueError("Для custom_pretrained необходимо указать ссылки на D и G файлы претрейна.")
    else:
        print(f"Установка претрейна {pretrain}...")
        for f in MODELS[pretrain]:
            subprocess.run(['aria2c', param_aria, f"{hugg_link}/{f[0]}", '-d', pretrain_outpath, '-o', f[1]], check=True)

        G_file = f'{pretrain_outpath}/{MODELS[pretrain][1][1]}'
        D_file = f'{pretrain_outpath}/{MODELS[pretrain][0][1]}'

    return G_file, D_file
