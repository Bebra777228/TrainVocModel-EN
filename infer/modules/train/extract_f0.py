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


# Класс для извлечения и обработки F0
class FeatureInput(object):
    def __init__(self):
        self.sample_rate = 16000  # Частота дискретизации
        self.hop_size = 160  # Размер шага (в сэмплах)
        self.f0_bin = 256  # Количество бинов для F0
        self.f0_min = 50.0  # Минимальное значение F0
        self.f0_max = 1100.0  # Максимальное значение F0

        # Преобразование F0 в мел-шкалу
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        # Инициализация модели RMVPE
        self.model_rmvpe = RMVPE("assets/rmvpe/rmvpe.pt", False, "cuda")

    # Метод для вычисления F0
    def compute_f0(self, path, f0_method):
        audio = load_audio(path, self.sample_rate)  # Загрузка аудио

        # Извлечение F0 в зависимости от метода
        if f0_method == "harvest":
            f0, t = pyworld.harvest(
                audio.astype(np.double),
                fs=self.sample_rate,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop_size / self.sample_rate,
            )
            f0 = pyworld.stonemask(audio.astype(np.double), f0, t, self.sample_rate)

        elif f0_method == "rmvpe":
            f0 = self.model_rmvpe.infer_from_audio(audio, 0.03)
        elif f0_method == "rmvpe+":
            f0 = self.model_rmvpe.infer_from_audio_modified(audio, 0.02)

        return f0

    # Функция для квантования F0
    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)  # Преобразование F0 в мел-шкалу
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1

        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)  # Квантование F0
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
        return f0_coarse

    # Основная функция для обработки всех путей
    def go(self, paths, f0_method):
        if len(paths) == 0:
            error_message = (
                "ОШИБКА: Не найдено ни одного фрагмента для обработки.\n"
                "Возможные причины:\n"
                "1. Датасет не имеет звука.\n"
                "2. Датасет слишком тихий.\n"
                "3. Датасет слишком короткий (менее 3 секунд).\n"
                "4. Датасет слишком длинный (более 1 часа одним файлом).\n\n"
                "Попробуйте увеличить громкость или объем датасета. Если у вас один большой файл, можно разделить его на несколько более мелких."
            )
            raise FileNotFoundError(error_message)
        else:
            printt(f"Фрагментов готовых к обработке - {len(paths)}")
            printt(f"Извлечение тона методом '{f0_method}'...")
            n = max(len(paths) // 5, 1)  # Определяем шаг для вывода прогресса
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    if idx % n == 0:
                        printt(f"{idx}/{len(paths)}")  # Вывод прогресса
                    # Пропускаем, если файлы уже существуют
                    if (
                        os.path.exists(opt_path1 + ".npy") == True
                        and os.path.exists(opt_path2 + ".npy") == True
                    ):
                        continue
                    featur_pit = self.compute_f0(inp_path, f0_method)  # Извлечение F0
                    np.save(opt_path2, featur_pit, allow_pickle=False)  # Сохранение F0
                    coarse_pit = self.coarse_f0(featur_pit)  # Квантование F0
                    np.save(opt_path1, coarse_pit, allow_pickle=False)  # Сохранение квантованного F0
                except:
                    printt(f"Ошибка извлечения тона!\nФрагмент - {idx}\nФайл - {inp_path}\n{traceback.format_exc()}")


if __name__ == "__main__":
    featureInput = FeatureInput()
    paths = []
    inp_root = f"{exp_dir}/1_16k_wavs"  # Директория с входными аудиофайлами
    opt_root1 = f"{exp_dir}/2a_f0"  # Директория для сохранения квантованного F0
    opt_root2 = f"{exp_dir}/2b-f0nsf"  # Директория для сохранения F0

    # Создаем директории, если они не существуют
    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)

    # Собираем пути к файлам
    for name in sorted(list(os.listdir(inp_root))):
        inp_path = f"{inp_root}/{name}"
        if "spec" in inp_path:
            continue
        opt_path1 = f"{opt_root1}/{name}"
        opt_path2 = f"{opt_root2}/{name}"
        paths.append([inp_path, opt_path1, opt_path2])
    try:
        featureInput.go(paths[i_part::n_part], f0_method)  # Обработка файлов
        printt("Тон извлечен!")
        printt("\n\n")
    except:
        printt(f"Ошибка извлечения тона!\n{traceback.format_exc()}")
