import os
import subprocess

assets_folder = "./assets/"
os.makedirs(assets_folder, exist_ok=True)

hugg_link = "https://huggingface.co/Politrees/all_RVC-pretrained_and_other/resolve/main"
file_links = {
    "rmvpe/rmvpe.pt": f"{hugg_link}/other/rmvpe.pt",
    "hubert/hubert_base.pt": f"{hugg_link}/HuBERTs/contentvec_base.pt"
}

for file, link in file_links.items():
    file_path = os.path.join(assets_folder, file)
    if not os.path.exists(file_path):
        try:
            subprocess.run(['wget', '-O', file_path, link], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {file}: {e}")
