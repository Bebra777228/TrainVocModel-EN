import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from librosa.display import specshow
from matplotlib.colors import Normalize

def plot_spectrogram_to_numpy(spectrogram, figsize=(10, 4), cmap='viridis'):
    """Визуализация Mel-спектрограммы."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap=cmap, norm=Normalize(vmin=-10, vmax=0))
    plt.colorbar(im, ax=ax, format='%+2.0f dB')
    plt.xlabel('Кадры')
    plt.ylabel('Частотные каналы')
    plt.tight_layout()
    
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf, dtype=np.uint8)
    plt.close(fig)
    return data

def plot_pitch_to_numpy(pitch_values, figsize=(12, 3)):
    """Визуализация кривой pitch (F0)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(pitch_values, color='r', linewidth=1, label='Pitch')
    ax.set_xlabel('Кадры')
    ax.set_ylabel('Частота (Гц)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf, dtype=np.uint8)
    plt.close(fig)
    return data

def calculate_snr(real, gen, eps=1e-7):
    """Вычисление Signal-to-Noise Ratio (SNR)."""
    noise = real - gen
    signal_power = torch.mean(real ** 2)
    noise_power = torch.mean(noise ** 2)
    return 10 * torch.log10(signal_power / (noise_power + eps))

def calculate_mse(real_wave, generated_wave):
    """Вычисление Mean Squared Error (MSE)."""
    mse = F.mse_loss(real_wave, generated_wave)
    return mse.item()
