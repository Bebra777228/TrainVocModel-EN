import logging
import os
import sys
import traceback

import numpy as np
import parselmouth
import pyworld

sys.path.append(os.getcwd())

from infer.lib.audio import load_audio
from infer.lib.rmvpe import RMVPE

logging.getLogger("numba").setLevel(logging.WARNING)

n_part = int(sys.argv[1])
i_part = int(sys.argv[2])
i_gpu = sys.argv[3]
exp_dir = sys.argv[4]
is_half = sys.argv[5]
f0_method = sys.argv[6]

os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)

f = open(f"{exp_dir}/logfile.log", "a+")


def printt(strr):
    print(strr)
    f.write(f"{strr}\n")
    f.flush()


class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size
        self.window_size = 5
        self.thred = 0.03
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, f0_method):
        x = load_audio(path, self.fs)
        rmvpe_path = "assets/rmvpe/rmvpe.pt"

        if not hasattr(self, "model_rmvpe"):
            self.model_rmvpe = RMVPE(rmvpe_path, is_half=is_half, device="cuda")

        if f0_method == "harvest":
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)

        elif f0_method == "rmvpe":
            f0 = self.model_rmvpe.infer_from_audio(x, self.thred)
        elif f0_method == "rmvpe+":
            f0 = self.model_rmvpe.infer_from_audio_modified(x, self.thred, self.f0_min, self.f0_max, self.window_size)

        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def go(self, paths, f0_method):
        if len(paths) == 0:
            error_message = (
                "ОШИБКА: Не найдено ни одного фрагмента для обработки.\n"
                "Возможные причины:\n"
                "1. Датасет не имеет звука.\n"
                "2. Датасет слишком тихий.\n"
                "3. Датасет слишком короткий."
            )
            printt(error_message)
            sys.exit(1)
        else:
            printt(f"Фрагментов готовых к обработке - {len(paths)}")
            printt(f"Извлечение тона методом '{f0_method}'...")
            n = max(len(paths) // 5, 1)
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if idx % n == 0:
                        printt(f"{idx}/{len(paths)}")
                    if (
                        os.path.exists(opt_path1 + ".npy") == True
                        and os.path.exists(opt_path2 + ".npy") == True
                    ):
                        continue
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    np.save(
                        opt_path2,
                        featur_pit,
                        allow_pickle=False,
                    )
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(
                        opt_path1,
                        coarse_pit,
                        allow_pickle=False,
                    )
                except:
                    printt(f"Ошибка извлечения тона!\nФрагмент - {idx}\nФайл - {inp_path}\n{traceback.format_exc()}")


if __name__ == "__main__":
    featureInput = FeatureInput()
    paths = []
    inp_root = f"{exp_dir}/1_16k_wavs"
    opt_root1 = f"{exp_dir}/2a_f0"
    opt_root2 = f"{exp_dir}/2b-f0nsf"

    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)
    for name in sorted(list(os.listdir(inp_root))):
        inp_path = f"{inp_root}/{name}"
        if "spec" in inp_path:
            continue
        opt_path1 = f"{opt_root1}/{name}"
        opt_path2 = f"{opt_root2}/{name}"
        paths.append([inp_path, opt_path1, opt_path2])
    try:
        featureInput.go(paths[i_part::n_part], f0_method)
    except:
        printt(f"Ошибка извлечения тона!\n{traceback.format_exc()}")
    printt("Тон извлечен!")
    printt("\n\n")
