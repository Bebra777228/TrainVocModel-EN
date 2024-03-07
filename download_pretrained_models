import subprocess, os
pretrained_folder = "/content/pretrained_models/"
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
for file, link in files.items():
    file_path = os.path.join(pretrained_folder, file)
    if not os.path.exists(file_path):
        try:
            subprocess.run(['wget', link, '-O', file_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {file}: {e}")
