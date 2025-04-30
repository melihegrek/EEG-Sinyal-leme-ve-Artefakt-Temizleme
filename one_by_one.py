import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.decomposition import FastICA
import os

def load_data_from_folder(folder_path, set_label):
    file_names = sorted([f for f in os.listdir(folder_path) if f.startswith(set_label) and f.lower().endswith(('.txt', '.TXT'))])
    all_signals = []
    for file in file_names:
        file_path = os.path.join(folder_path, file)
        signal_data = np.loadtxt(file_path)
        all_signals.append(signal_data)
    return np.array(all_signals) if all_signals else np.array([])

def apply_notch_filter(signal_data, fs=256, notch_freq=50):
    if len(signal_data) < 9:
        return signal_data
    Q = 60.0
    b, a = signal.iirnotch(notch_freq, Q, fs)
    return signal.filtfilt(b, a, signal_data)

def apply_thresholding(signal_data, threshold=100):
    return np.where(np.abs(signal_data) > threshold, np.median(signal_data), signal_data)

def apply_ica(signal_data, n_components=None):  
    if signal_data.ndim == 1:
        signal_data = signal_data.reshape(-1, 1)
    
    ica = FastICA(n_components=n_components, random_state=42, whiten="unit-variance")
    transformed = ica.fit_transform(signal_data)

    # Çıkışı orijinal sinyalin ölçeğine geri getir
    return transformed[:, 0] * np.std(signal_data)

def apply_bandpass_filter(signal_data, fs=256, lowcut=1, highcut=50):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, signal_data)

def process_and_plot(folder_path, set_label, filter_func, filter_name):
    raw_signals = load_data_from_folder(folder_path, set_label)
    if raw_signals.size == 0:
        print(f"HATA: '{folder_path}' klasöründe '{set_label}' ile başlayan dosya bulunamadı!")
        return
    raw_signal = raw_signals.flatten()
    processed_signal = filter_func(raw_signal)

    plt.figure(figsize=(20, 6))
    plt.plot(raw_signal, label="Öncesi (Ham Sinyal)", alpha=0.8)  # Alpha değerini artırdık
    plt.plot(processed_signal, label=f"Sonrası ({filter_name})", color="red", alpha=0.7)
    plt.legend()
    plt.title(f"{set_label} İçin {filter_name} Filtresi Uygulama Öncesi ve Sonrası")
    plt.xlabel("Zaman (Örnek Noktaları)")
    plt.ylabel("Genlik (µV)")
    plt.show()

# 📌 Filtrelerin tek tek uygulanması
process_and_plot("N", "N", apply_notch_filter, "Notch Filtresi")
process_and_plot("N", "N", apply_thresholding, "Thresholding")
process_and_plot("N", "N", lambda x: apply_ica(x.reshape(-1, 1)), "ICA")
process_and_plot("N", "N", apply_bandpass_filter, "Band-Pass Filtresi")
