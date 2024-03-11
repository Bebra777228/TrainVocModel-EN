import os, subprocess
import gradio as gr
import shutil
from mega import Mega

import pandas as pd
import os

# Class to handle caching model urls from a spreadsheet
class CachedModels:
    def __init__(self):
        csv_url = "https://docs.google.com/spreadsheets/d/1tAUaQrEHYgRsm1Lvrnj14HFHDwJWl0Bd9x0QePewNco/export?format=csv&gid=1977693859"
        if os.path.exists("spreadsheet.csv"):
            self.cached_data = pd.read_csv("spreadsheet.csv") 
        else:
            self.cached_data = pd.read_csv(csv_url)
            self.cached_data.to_csv("spreadsheet.csv", index=False)
        # Cache model urls        
        self.models = {}
        for _, row in self.cached_data.iterrows():
            filename = row['Filename']
            url = None
            for value in row.values:
                if isinstance(value, str) and "huggingface" in value:
                    url = value
                    break
            if url:
                self.models[filename] = url
    # Get cached model urls    
    def get_models(self):
        return self.models
        
def show(path,ext,on_error=None):
    try:
        return list(filter(lambda x: x.endswith(ext), os.listdir(path)))
    except:
        return on_error
    
def run_subprocess(command):
    try:
        subprocess.run(command, check=True)
        return True, None
    except Exception as e:
        return False, e
        
def download_from_url(url=None, model=None):
    if not url:
        try:
            url = model[f'{model}']
        except:
            gr.Warning("Failed")
            return ''
    if model == '':
        try:
            model = url.split('/')[-1].split('?')[0]
        except:
            gr.Warning('Please name the model')
            return
    model = model.replace('.pth', '').replace('.index', '').replace('.zip', '')
    url = url.replace('/blob/main/', '/resolve/main/').strip()

    for directory in ["downloads", "unzips","zip"]:
        #shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory, exist_ok=True)

    try:
        if url.endswith('.pth'):
            subprocess.run(["wget", url, "-O", f'assets/weights/{model}.pth'])
        elif url.endswith('.index'):
            os.makedirs(f'logs/{model}', exist_ok=True)
            subprocess.run(["wget", url, "-O", f'logs/{model}/added_{model}.index'])
        elif url.endswith('.zip'):
            subprocess.run(["wget", url, "-O", f'downloads/{model}.zip'])
        else:
            if "drive.google.com" in url:
                url = url.split('/')[0]
                subprocess.run(["gdown", url, "--fuzzy", "-O", f'downloads/{model}'])
            elif "mega.nz" in url:
                Mega().download_url(url, 'downloads')
            else:
                subprocess.run(["wget", url, "-O", f'downloads/{model}'])

        downloaded_file = next((f for f in os.listdir("downloads")), None)
        if downloaded_file:
            if downloaded_file.endswith(".zip"):
                shutil.unpack_archive(f'downloads/{downloaded_file}', "unzips", 'zip')
                for root, _, files in os.walk('unzips'):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file.endswith(".index"):
                            os.makedirs(f'logs/{model}', exist_ok=True)
                            shutil.copy2(file_path, f'logs/{model}')
                        elif file.endswith(".pth") and "G_" not in file and "D_" not in file:
                            shutil.copy(file_path, f'assets/weights/{model}.pth')
            elif downloaded_file.endswith(".pth"):
                shutil.copy(f'downloads/{downloaded_file}', f'assets/weights/{model}.pth')
            elif downloaded_file.endswith(".index"):
                os.makedirs(f'logs/{model}', exist_ok=True)
                shutil.copy(f'downloads/{downloaded_file}', f'logs/{model}/added_{model}.index')
            else:
                gr.Warning("Failed to download file")
                return 'Failed'

        gr.Info("Done")
    except Exception as e:
        gr.Warning(f"There's been an error: {str(e)}")
    finally:
        shutil.rmtree("downloads", ignore_errors=True)
        shutil.rmtree("unzips", ignore_errors=True)
        shutil.rmtree("zip", ignore_errors=True)
        return 'Done'
        
def speak(audio, text):
    print(f"({audio}, {text})")
    current_dir = os.getcwd()
    os.chdir('./gpt_sovits_demo')
    process = subprocess.Popen([
        "python", "./zero.py",
        "--input_file", audio,
        "--audio_lang", "English", 
        "--text", text,
        "--text_lang", "English"
    ], stdout=subprocess.PIPE, text=True)
    
    for line in process.stdout:
        line = line.strip()
        if "All keys matched successfully" in line:
            continue
        if line.startswith("(") and line.endswith(")"):
            path, finished = line[1:-1].split(", ")
            if finished:
                os.chdir(current_dir)
                return path
    os.chdir(current_dir)
    return None
