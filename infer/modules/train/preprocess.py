import multiprocessing
import os
import sys
import traceback

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile

sys.path.append(os.getcwd())

from infer.lib.audio import load_audio
from infer.lib.slicer import Slicer

n_p = int(sys.argv[1])
exp_dir = sys.argv[2]
inp_root = sys.argv[3]
per = float(sys.argv[4])
sr = int(sys.argv[5])
normalize = sys.argv[6] == "True"

SR_TARGET = sr
RES_TYPE = "soxr_vhq"
SAMPLE_RATE_16K = 16000

f = open(f"{exp_dir}/logfile.log", "a+")


# Функция для вывода и логирования сообщений
def printt(strr):
    print(strr)
    f.write(f"{strr}\n")
    f.flush()


class PreProcess:
    def __init__(self, sr, sr_trgt, exp_dir, per=3.0, normalize=True):
        # Директории для сохранения обработанных аудиофайлов
        self.gt_wavs_dir = os.path.join(exp_dir, "0_gt_wavs")
        self.wavs16k_dir = os.path.join(exp_dir, "1_16k_wavs")

        # Создаем директории, если они не существуют
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

        # Инициализация Slicer для нарезки аудио
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr  # Частота дискретизации
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)  # Фильтр высоких частот
        self.per = per  # Длина сегмента
        self.overlap = 0.3  # Перекрытие между сегментами
        self.tail = self.per + self.overlap  # Хвост для обработки
        self.max = 0.9  # Максимальное значение для нормализации
        self.alpha = 0.75  # Коэффициент для нормализации
        self.sr_trgt = sr_trgt  # Целевая частота дискретизации
        self.normalize = normalize  # Флаг для включения/выключения нормализации

    def norm_write(self, tmp_audio, idx0, idx1):
        # Проверка на превышение максимального уровня сигнала
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            printt(f"{idx0}-{idx1}-{tmp_max}-filtered")
            return

        # Ресемплирование аудио до целевой частоты дискретизации
        tmp_audio = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=self.sr_trgt, res_type=RES_TYPE)

        # Применение нормализации
        if self.normalize:
            tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (1 - self.alpha) * tmp_audio

        # Сохранение аудио в формате WAV
        wavfile.write(f"{self.gt_wavs_dir}/{idx0}_{idx1}.wav", self.sr_trgt, tmp_audio.astype(np.float32))

        # Ресемплирование аудио до 16 кГц
        tmp_audio = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=SAMPLE_RATE_16K, res_type=RES_TYPE)

        # Сохранение аудио в формате WAV (16 кГц)
        wavfile.write(f"{self.wavs16k_dir}/{idx0}_{idx1}.wav", SAMPLE_RATE_16K, tmp_audio.astype(np.float32))

    def pipeline(self, path, idx0):
        try:
            # Загрузка аудио
            audio = load_audio(path, self.sr_trgt)
            # Применение фильтра высоких частот
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            # Нарезка аудио на сегменты
            for audio in self.slicer.slice(audio):
                i = 0
                while 1:
                    # Вычисление начальной точки сегмента
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    # Проверка, остался ли хвост аудио
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                        break
            printt(f"{path}\t-> Success")
        except Exception as e:
            printt(f"{path}\t-> {traceback.format_exc()}")

    def pipeline_mp(self, infos):
        # Обработка списка файлов
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        printt("Обработка датасета...")
        try:
            # Сбор информации о файлах в директории
            infos = [(os.path.join(inp_root, name), idx) for idx, name in enumerate(sorted(list(os.listdir(inp_root))))]

            # Параллельная обработка
            ps = []
            for i in range(n_p):
                p = multiprocessing.Process(target=self.pipeline_mp, args=(infos[i::n_p]))
                ps.append(p)
                p.start()
            for p in ps:
                p.join()
            printt("Обработка успешно завершена!")
        except Exception as e:
            printt(f"Ошибка! {traceback.format_exc()}")
            sys.exit(1)


def preprocess_trainset(inp_root, sr, n_p, exp_dir, per, normalize):
    # Инициализация и запуск обработки
    pp = PreProcess(sr, SR_TARGET, exp_dir, per, normalize)
    pp.pipeline_mp_inp_dir(inp_root, n_p)


if __name__ == "__main__":
    # Запуск препроцессинга
    preprocess_trainset(inp_root, sr, n_p, exp_dir, per, normalize)
    printt("\n\n")
