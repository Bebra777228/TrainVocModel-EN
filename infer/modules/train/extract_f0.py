import os
import sys
import traceback

import parselmouth

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging

import numpy as np
import pyworld

from infer.lib.audio import load_audio

logging.getLogger("numba").setLevel(logging.WARNING)

n_part = int(sys.argv[1])
i_part = int(sys.argv[2])
i_gpu = sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
exp_dir = sys.argv[4]
is_half = sys.argv[5]
f0_method = sys.argv[6]

f = open("%s/extract_f0_feature.log" % exp_dir, "a+")
def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def compute_f0(self, path, f0_method):
        x = load_audio(path, self.fs)
        rmvpe_path = "assets/rmvpe/rmvpe.pt"

        if not hasattr(self, "model_rmvpe"):
            print(f"Загрузка {f0_method} модели...")
            self.model_rmvpe = RMVPE(rmvpe_path, is_half=is_half, device="cuda")

        if f0_method == "rmvpe":
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        elif f0_method == "rmvpe+":
            f0 = self.model_rmvpe.infer_from_audio_modified(
                x, thred=0.03, f0_min=50, f0_max=1100, window_size=5
            )

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
            printt("Извлечение тона...")
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
    printt(" ".join(sys.argv))
    featureInput = FeatureInput()
    paths = []
    inp_root = "%s/1_16k_wavs" % (exp_dir)
    opt_root1 = "%s/2a_f0" % (exp_dir)
    opt_root2 = "%s/2b-f0nsf" % (exp_dir)

    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)
    for name in sorted(list(os.listdir(inp_root))):
        inp_path = "%s/%s" % (inp_root, name)
        if "spec" in inp_path:
            continue
        opt_path1 = "%s/%s" % (opt_root1, name)
        opt_path2 = "%s/%s" % (opt_root2, name)
        paths.append([inp_path, opt_path1, opt_path2])
    try:
        featureInput.go(paths[i_part::n_part], f0_method)
    except:
        printt(f"Ошибка извлечения тона!\n{traceback.format_exc()}")
