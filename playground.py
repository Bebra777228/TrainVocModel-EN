import gradio as gr
import os, shutil
import subprocess, glob
from datetime import datetime
from tools.useftools import *
os.environ["rmvpe_root"] = "assets/rmvpe"
os.environ['index_root']="logs"
os.environ['weight_root']="assets/weights"
from infer.modules.vc.modules import VC
from configs.config import Config
import torch
os.makedirs(os.path.join(".", "audios"), exist_ok=True)
config = Config()
vc = VC(config)

def warn(text):
    try: gr.Warning(text)
    except: pass

def load_model(model_picker,index_picker):
    logs = show_available("logs")
    if model_picker.replace(".pth","") in logs:
        log = model_picker.replace(".pth","")
    else:
        log = index_picker
        warn("Could not find a matching index file.")
    vc.get_vc(model_picker,0,0)
    return {"choices":logs,"value":log,"__type__": "update"}

def convert(audio_picker,model_picker,index_picker,index_rate,pitch,method):
    warn("Your audio is being converted. Please wait.")
    now = datetime.now().strftime("%d%m%Y%H%M%S")
    index_files = glob.glob(f"logs/{index_picker}/*.index")
    if index_files:
        print(f"Found index: {index_files[0]}")
    else:
        warn("Sorry, I couldn't find your .index file.")
        print("Did not find a matching .index file")
        index_files = [f'logs/{model_picker}/fake_index.index']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    command = [
        "python",
        "tools/infer_cli.py",
        "--f0up_key", str(pitch),
        "--input_path", f"audios/{audio_picker}",
        "--index_path", index_files[0],
        "--f0method", method,
        "--opt_path", f"audios/cli_output_{now}.wav",
        "--model_name", f"{model_picker}",
        "--index_rate", str(float(index_rate)),
        "--device", device,
        "--filter_radius", "3",
        "--resample_sr", "0",
        "--rms_mix_rate", "0.0",
        "--protect", "0"
    ]

    try:
        process = subprocess.run(command, check=True)
        print("Script executed successfully.")
        return {"choices":show_available("audios"),"__type__":"update","value":f"cli_output_{now}.wav"},f"audios/cli_output_{now}.wav"
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return {"choices":show_available("audios"),"__type__":"update"}, None

assets_folder = "assets"
if not os.path.exists(assets_folder):
    os.makedirs(assets_folder)
files = {
    "rmvpe/rmvpe.pt":"https://huggingface.co/Rejekts/project/resolve/main/rmvpe.pt",
    "hubert/hubert_base.pt":"https://huggingface.co/Rejekts/project/resolve/main/hubert_base.pt",
    "pretrained_v2/D40k.pth":"https://huggingface.co/Rejekts/project/resolve/main/D40k.pth",
    "pretrained_v2/G40k.pth":"https://huggingface.co/Rejekts/project/resolve/main/G40k.pth",
    "pretrained_v2/f0D40k.pth":"https://huggingface.co/Rejekts/project/resolve/main/f0D40k.pth",
    "pretrained_v2/f0G40k.pth":"https://huggingface.co/Rejekts/project/resolve/main/f0G40k.pth"
}
for file, link in files.items():
    file_path = os.path.join(assets_folder, file)
    if not os.path.exists(file_path):
        try:
            subprocess.run(['wget', link, '-O', file_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {file}: {e}")

def download_from_url(url, model):
    if model =='':
        try:
            model = url.split('/')[-1].split('?')[0]
        except:
            return "You need to name your model. For example: My-Model", {"choices":show_available("assets/weights"),"__type__":"update"}
    url=url.replace('/blob/main/','/resolve/main/')
    model=model.replace('.pth','').replace('.index','').replace('.zip','')
    print(f"Model name: {model}")
    if url == '':
        return "URL cannot be left empty.", {"choices":show_available("assets/weights"),"__type__":"update"}
    url = url.strip()
    zip_dirs = ["zips", "unzips"]
    for directory in zip_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    os.makedirs("zips", exist_ok=True)
    os.makedirs("unzips", exist_ok=True)
    zipfile = model + '.zip'
    zipfile_path = './zips/' + zipfile
    try:
        if url.endswith('.pth'):
            subprocess.run(["wget", url, "-O", f'./assets/weights/{model}.pth'])
            return f"Sucessfully downloaded as {model}.pth", {"choices":show_available("assets/weights"),"__type__":"update"}
        elif url.endswith('.index'):
            if not os.path.exists(f'./logs/{model}'): os.makedirs(f'./logs/{model}')
            subprocess.run(["wget", url, "-O", f'./logs/{model}/added_{model}.index'])
            return f"Successfully downloaded as added_{model}.index", {"choices":show_available("assets/weights"),"__type__":"update"}
        if "drive.google.com" in url:
            subprocess.run(["gdown", url, "--fuzzy", "-O", zipfile_path])
        elif "mega.nz" in url:
            m = Mega()
            m.download_url(url, './zips')
        else:
            subprocess.run(["wget", url, "-O", zipfile_path])
        for filename in os.listdir("./zips"):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join("./zips/",filename)
                shutil.unpack_archive(zipfile_path, "./unzips", 'zip')
                for root, dirs, files in os.walk('./unzips'):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file.endswith(".index"):
                            os.mkdir(f'./logs/{model}')
                            shutil.copy2(file_path,f'./logs/{model}')
                        elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                            shutil.copy(file_path,f'./assets/weights/{model}.pth')
            elif filename.endswith(".pth"):
                shutil.copy2(os.path.join("./zips/",filename),f'./assets/weights/{model}.pth')
            elif filename.endswith(".index"):
                os.mkdir(f'./logs/{model}')
                shutil.copy2(os.path.join("./zips/",filename),f'./logs/{model}/')
            else:
                return "No zipfile found.", {"choices":show_available("assets/weights"),"__type__":"update"}
        shutil.rmtree("zips")
        shutil.rmtree("unzips")
        return "Success.", {"choices":show_available("assets/weights"),"__type__":"update"}
    except:
        return "There's been an error.", {"choices":show_available("assets/weights"),"__type__":"update"}

def import_from_name(model):
    try:
        url = models[f'{model}']
    except:
        return "", {"__type__":"update"}
    url=url.replace('/blob/main/','/resolve/main/')
    print(f"Model name: {model}")
    if url == '':
        return "", {"__type__":"update"}
    url = url.strip()
    zip_dirs = ["zips", "unzips"]
    for directory in zip_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    os.makedirs("zips", exist_ok=True)
    os.makedirs("unzips", exist_ok=True)
    zipfile = model + '.zip'
    zipfile_path = './zips/' + zipfile
    try:
        if url.endswith('.pth'):
            subprocess.run(["wget", url, "-O", f'./assets/weights/{model}.pth'])
            return f"", {"choices":show_available("assets/weights"),"__type__":"update","value":f"{model}.pth"}
        if "drive.google.com" in url:
            subprocess.run(["gdown", url, "--fuzzy", "-O", zipfile_path])
        elif "mega.nz" in url:
            m = Mega()
            m.download_url(url, './zips')
        else:
            subprocess.run(["wget", url, "-O", zipfile_path])
        for filename in os.listdir("./zips"):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join("./zips/",filename)
                shutil.unpack_archive(zipfile_path, "./unzips", 'zip')
                for root, dirs, files in os.walk('./unzips'):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file.endswith(".index"):
                            os.mkdir(f'./logs/{model}')
                            shutil.copy2(file_path,f'./logs/{model}')
                        elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                            shutil.copy(file_path,f'./assets/weights/{model}.pth')
            elif filename.endswith(".pth"):
                shutil.copy2(os.path.join("./zips/",filename),f'./assets/weights/{model}.pth')
            elif filename.endswith(".index"):
                os.mkdir(f'./logs/{model}')
                shutil.copy2(os.path.join("./zips/",filename),f'./logs/{model}/')
            else:
                return "", {"__type__":"update"}
        shutil.rmtree("zips")
        shutil.rmtree("unzips")
        return "", {"choices":show_available("assets/weights"),"__type__":"update","value":f"{model}.pth"}
    except:
        return "", {"__type__":"update"}

def show_available(filepath,format=None):
    if format:
        print(f"Format: {format}")
        files = []
        for file in os.listdir(filepath):
            if file.endswith(format):
                print(f"Matches format: {file}")
                files.append(file)
            else:
                print(f"Does not match format: {file}")
        print(f"Matches: {files}")
        if len(files) < 1:
            return ['']
        return files
    if len(os.listdir(filepath)) < 1:
        return ['']
    return os.listdir(filepath)
  
def upload_file(file):
    audio_formats = ['.wav', '.mp3', '.ogg', '.flac', '.aac']
    print(file)
    try:
        _, ext = os.path.splitext(file.name)
        filename = os.path.basename(file.name)
        file_path = file.name
    except AttributeError:
        _, ext = os.path.splitext(file)
        filename = os.path.basename(file)
        file_path = file
    if ext.lower() in audio_formats:
        if os.path.exists(f'audios/{filename}'): 
            os.remove(f'audios/{filename}')
        shutil.move(file_path,'audios')
    else:
        warn('File incompatible')
    return {"choices":show_available('audios'),"__type__": "update","value":filename}

def refresh():
    return {"choices":show_available("audios"),"__type__": "update"},{"choices":show_available("assets/weights",".pth"),"__type__": "update"},{"choices":show_available("logs"),"__type__": "update"}

def update_audio_player(choice):
    return os.path.join("audios",choice)

with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            gr.Markdown("# RVC PlayGround 🎮")
        with gr.Column():
            gr.HTML("<a href='https://ko-fi.com/rejekts' target='_blank'><img src='file/kofi_button.png' alt='🤝 Support Me'></a>")
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("1.Choose a voice model:"):
                    model_picker = gr.Dropdown(label="Model: ",choices=show_available('assets/weights','.pth'),value=show_available('assets/weights','.pth')[0],interactive=True,allow_custom_value=True)
                    index_picker = gr.Dropdown(label="Index:",interactive=True,choices=show_available('logs'),value=show_available('logs')[0],allow_custom_value=True)
                    model_picker.change(fn=load_model,inputs=[model_picker,index_picker],outputs=[index_picker])
                with gr.TabItem("(Or download a model here)"):
                    with gr.Row():
                        url = gr.Textbox(label="Paste the URL here:",value="",placeholder="(i.e. https://huggingface.co/repo/model/resolve/main/model.zip)")
                    with gr.Row():
                        with gr.Column():
                            model_rename = gr.Textbox(placeholder="My-Model", label="Name your model:",value="")
                        with gr.Column():
                            download_button = gr.Button("Download")
                            download_button.click(fn=download_from_url,inputs=[url,model_rename],outputs=[url,model_picker])
                    with gr.Row():
                        selected_import = gr.Dropdown(choices=list(models.keys()),label="OR Search Models (Quality UNKNOWN)",scale=5)
                        import_model = gr.Button("Download")
                        import_model.click(fn=import_from_name,inputs=[selected_import],outputs=[selected_import,model_picker])
                with gr.TabItem("Advanced"):
                    index_rate = gr.Slider(label='Index Rate: ',minimum=0,maximum=1,value=0.66,step=0.01)
                    pitch = gr.Slider(label='Pitch (-12 lowers it an octave, 0 keeps the original pitch, 12 lifts it an octave): ',minimum =-12, maximum=12, step=1, value=0, interactive=True)
                    method = gr.Dropdown(label="Method:",choices=["rmvpe","pm"],value="rmvpe")
        
    with gr.Row():
        with gr.Tabs():
            with gr.TabItem("2.Choose an audio file:"):
                recorder = gr.Microphone(label="Record audio here...",type='filepath')
                audio_picker = gr.Dropdown(label="",choices=show_available('audios'),value='',interactive=True)
                try:
                    recorder.stop_recording(upload_file, inputs=[recorder],outputs=[audio_picker])
                except:
                    recorder.upload(upload_file, inputs=[recorder],outputs=[audio_picker])
            with gr.TabItem("(Or upload a new file here)"):
                try:
                    dropbox = gr.File(label="Drop an audio here.",file_types=['.wav', '.mp3', '.ogg', '.flac', '.aac'], type="filepath")
                except:#Version Compatibiliy
                    dropbox = gr.File(label="Drop an audio here.",file_types=['.wav', '.mp3', '.ogg', '.flac', '.aac'], type="file")
                dropbox.upload(fn=upload_file, inputs=[dropbox],outputs=[audio_picker])
        audio_refresher = gr.Button("Refresh")
        audio_refresher.click(fn=refresh,inputs=[],outputs=[audio_picker,model_picker,index_picker])
        convert_button = gr.Button("Convert")
    with gr.Row():
        audio_player = gr.Audio()
        inputs = [audio_picker,model_picker,index_picker,index_rate,pitch,method]
        audio_picker.change(fn=update_audio_player, inputs=[audio_picker],outputs=[audio_player])
        convert_button.click(convert, inputs=inputs,outputs=[audio_picker,audio_player])

app.queue(max_size=20).launch(debug=True,share=True)
