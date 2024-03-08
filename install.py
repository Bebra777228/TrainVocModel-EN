import os
import subprocess
from tqdm import tqdm

# Установка зависимостей
print('Установка зависимостей...')
subprocess.check_call(['apt', 'install', '-y', '-qq', 'aria2', 'wget'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

requirements_file = 'requirements.txt'
with open(requirements_file, 'r') as f:
    packages = f.read().split('\n')

progress_bar = tqdm(total=len(packages))
for package in packages:
    if package:
        subprocess.check_call(['pip', 'install', package], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        progress_bar.update(1)
progress_bar.close()

# Установка моделей
print('Установка моделей...')

# Загрузка предобученных моделей
pretrained_folder = "/content/pretrained_models"
if not os.path.exists(pretrained_folder):
    os.makedirs(pretrained_folder)

files = {
    "f0D40k.pth":"https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth",
    "f0G40k.pth":"https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth",
    "f0Ov2Super40kD.pth":"https://huggingface.co/ORVC/Ov2Super/resolve/main/f0Ov2Super40kD.pth",
    "f0Ov2Super40kG.pth":"https://huggingface.co/ORVC/Ov2Super/resolve/main/f0Ov2Super40kG.pth",
    "f0SnowieRuPre40kD.pth":"https://huggingface.co/MUSTAR/SnowyRuPretrain_EnP_40k/resolve/main/D_Snowie_RuPretrain_EnP.pth",
    "f0SnowieRuPre40kG.pth":"https://huggingface.co/MUSTAR/SnowyRuPretrain_EnP_40k/resolve/main/G_Snowie_RuPretrain_EnP.pth",
    "f0_Rin_E3_40kD.pth":"https://huggingface.co/ORVC/RIN_E3/resolve/main/RIN_E3_D.pth",
    "f0_Rin_E3_40kG.pth":"https://huggingface.co/ORVC/RIN_E3/resolve/main/RIN_E3_G.pth"
}

progress_bar = tqdm(total=len(files))
for file, link in files.items():
    file_path = os.path.join(pretrained_folder, file)
    if not os.path.exists(file_path):
        try:
            # Запускаем команду загрузки файла с выводом в консоль
            process = subprocess.Popen(['aria2c', '--console-log-level=info', '-c', '-x', '16', '-s', '16', '-k', '1M', link, '-d', pretrained_folder, '-o', file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            # Читаем вывод команды и обновляем полосу прогресса
            progress = None
            for line in iter(process.stdout.readline, b''):
                line = line.decode().strip()
                if progress is None and 'Progress' in line:
                    progress = line.split('=')[1].strip()
                elif progress is not None:
                    new_progress = line.split('=')[1].strip()
                    if new_progress != progress:
                        progress = new_progress
                        progress_bar.set_description(f"Downloading {file}: {progress}")
                        progress_bar.update(1)

            # Дожидаемся завершения команды
            process.wait()

        except subprocess.CalledProcessError as e:
            print(f"Error downloading {file}: {e}")
    else:
        progress_bar.update(1)
progress_bar.close()

# Загрузка дополнительных файлов
assets_folder = "./assets/"
os.makedirs(assets_folder, exist_ok=True)

file_links = {
    "rmvpe/rmvpe.pt": "https://huggingface.co/Rejekts/project/resolve/main/rmvpe.pt",
    "hubert/hubert_base.pt": "https://huggingface.co/Rejekts/project/resolve/main/hubert_base.pt"
}

progress_bar = tqdm(total=len(file_links))
for file, link in file_links.items():
    file_path = os.path.join(assets_folder, file)
    if not os.path.exists(file_path):
        try:
            # Запускаем команду загрузки файла с выводом в консоль
            process = subprocess.Popen(['wget', '-O', file_path, link], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            # Читаем вывод команды и обновляем полосу прогресса
            progress = None
            for line in iter(process.stdout.readline, b''):
                line = line.decode().strip()
                if progress is None and '[' in line and ']' in line:
                    progress = line.split('[')[1].split(']')[0].strip()
                elif progress is not None:
                    new_progress = line.split('[')[1].split(']')[0].strip()
                    if new_progress != progress:
                        progress = new_progress
                        progress_bar.set_description(f"Downloading {file}: {progress}")
                        progress_bar.update(1)

            # Дожидаемся завершения команды
            process.wait()

        except subprocess.CalledProcessError as e:
            print(f"Error downloading {file}: {e}")
    else:
        progress_bar.update(1)
progress_bar.close()

print("\u2714 Готово")
