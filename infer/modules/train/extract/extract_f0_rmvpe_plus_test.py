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

    def get_f0_rmvpe(self, x, f0_min=1, f0_max=40000, *args, **kwargs):
        if not hasattr(self, "model_rmvpe"):
            self.model_rmvpe = RMVPE0Predictor(RMVPE_DIR, is_half=self.is_half, device=self.device)
        f0 = self.model_rmvpe.infer_from_audio_with_pitch(x, thred=0.03, f0_min=f0_min, f0_max=f0_max)
        return f0
    
    def compute_f0(self, path, f0_method):
        x = load_audio(path, self.fs)
        time_step = 160 / 16000 * 1000
        p_len = x.shape[0] // self.hop
        if f0_method == "rmvpe+":
            params = {
                'x': x,
                'p_len': p_len,
                'f0_min': f0_min,
                'f0_max': f0_max,
                'time_step': time_step,
                'hop_length': self.hop,
                'model': "full"
            }
            f0 = self.get_f0_rmvpe(**params)
        else:
            raise ValueError("Unsupported f0 method: {}".format(f0_method))
        
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
            printt("no-f0-todo")
        else:
            printt("todo-f0-%s" % len(paths))
            n = max(len(paths) // 5, 1)  # каждое n-е сообщение
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if idx % n == 0:
                        printt("f0ing, now-%s, all-%s, -%s" % (idx, len(paths), inp_path))
                    if (
                        os.path.exists(opt_path1 + ".npy") 
                        and os.path.exists(opt_path2 + ".npy")
                    ):
                        continue
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    np.save(opt_path2, featur_pit, allow_pickle=False)  # nsf
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(opt_path1, coarse_pit, allow_pickle=False)  # ori
                except Exception as e:
                    printt("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))

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
        featureInput.go(paths[i_part::n_part], "rmvpe+")
    except Exception as e:
        printt("f0_all_fail-%s" % (traceback.format_exc()))
