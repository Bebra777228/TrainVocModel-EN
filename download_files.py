import subprocess, os
assets_folder = "./assets/"
if not os.path.exists(assets_folder):
    os.makedirs(assets_folder)
files = {
    "rmvpe/rmvpe.pt":"https://huggingface.co/Rejekts/project/resolve/main/rmvpe.pt",
    "hubert/hubert_base.pt":"https://huggingface.co/Rejekts/project/resolve/main/hubert_base.pt"
}
for file, link in files.items():
    file_path = os.path.join(assets_folder, file)
    if not os.path.exists(file_path):
        try:
            subprocess.run(['aria2c', '--console-log-level=error', '-c', '-x', '16', '-s', '16', '-k', '1M', link, '-d', pretrained_folder, '-o', file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {file}: {e}")
