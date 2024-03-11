import subprocess, os
pretrained_folder = ".assets/pretrained_v2"
if not os.path.exists(pretrained_folder):
    os.makedirs(pretrained_folder)
files = {
    "default_D.pth":"https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth",
    "default_G.pth":"https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth",
    "Ov2Super_D.pth":"https://huggingface.co/ORVC/Ov2Super/resolve/main/f0Ov2Super40kD.pth",
    "Ov2Super_G.pth":"https://huggingface.co/ORVC/Ov2Super/resolve/main/f0Ov2Super40kG.pth",
    "SnowieV2_D.pth":"https://huggingface.co/MUSTAR/SnowyRuPretrain_EnP_40k/resolve/main/D_Snowie_RuPretrain_EnP.pth",
    "SnowieV2_G.pth":"https://huggingface.co/MUSTAR/SnowyRuPretrain_EnP_40k/resolve/main/G_Snowie_RuPretrain_EnP.pth",
    "Rin_E3_D.pth":"https://huggingface.co/ORVC/RIN_E3/resolve/main/RIN_E3_D.pth",
    "Rin_E3_G.pth":"https://huggingface.co/ORVC/RIN_E3/resolve/main/RIN_E3_G.pth",
    "SnowieV3_D.pth":"https://huggingface.co/MUSTAR/SnowieV3-40k_pretrain/resolve/main/D_SnowieV3_40k.pth",
    "SnowieV3_G.pth":"https://huggingface.co/MUSTAR/SnowieV3-40k_pretrain/resolve/main/G_SnowieV3_40k.pth",
    "SnowieV3_x_Rin_E3_D.pth":"https://huggingface.co/MUSTAR/SnowieV3-X-RINE3-40K/resolve/main/D_Snowie-X-Rin_40k.pth",
    "SnowieV3_x_Rin_E3_G.pth":"https://huggingface.co/MUSTAR/SnowieV3-X-RINE3-40K/resolve/main/G_Snowie-X-Rin_40k.pth"
}
for file, link in files.items():
    file_path = os.path.join(pretrained_folder, file)
    if not os.path.exists(file_path):
        try:
            subprocess.run(['aria2c', '--console-log-level=error', '-c', '-x', '16', '-s', '16', '-k', '1M', link, '-d', pretrained_folder, '-o', file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {file}: {e}")
