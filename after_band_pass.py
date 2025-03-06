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
        signal_data = np.loadtxt(file_path)  # .txt içindeki 4097 veriyi oku
        all_signals.append(signal_data)

    return np.array(all_signals) if all_signals else np.array([])

def apply_notch_filter(signal_data, fs=173.61, notch_freq=50):
    Q = 30.0
    b, a = signal.iirnotch(notch_freq, Q, fs)
    return signal.filtfilt(b, a, signal_data)

def apply_bandpass_filter(signal_data, fs=173.61, lowcut=0.5, highcut=50.0):
    """
    Band-pass filtre uygular: 0.5 Hz - 50 Hz arası.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, signal_data)

def apply_thresholding(signal_data, threshold=100):
    return np.where(np.abs(signal_data) > threshold, np.median(signal_data), signal_data)

def apply_ica(signal_data, n_components=1):
    if signal_data.shape[0] < 2:
        return signal_data.flatten()

    ica = FastICA(n_components=n_components, random_state=42)
    return ica.fit_transform(signal_data)

def process_and_plot_all(folder_path, set_label):
    raw_signals = load_data_from_folder(folder_path, set_label)

    if raw_signals.size == 0:
        print(f"HATA: '{folder_path}' klasöründe '{set_label}' ile başlayan dosya bulunamadı!")
        return

    raw_signal = raw_signals.flatten()

    # Notch filtresi uygula
    filtered_signal = apply_notch_filter(raw_signal)

    # Band-pass filtre uygula (0.5 Hz - 50 Hz)
    bandpass_filtered_signal = apply_bandpass_filter(filtered_signal)

    # Thresholding uygula
    thresholded_signal = apply_thresholding(bandpass_filtered_signal)

    # ICA uygula
    cleaned_signal = apply_ica(thresholded_signal.reshape(-1, 1)).flatten()

    # Grafik çiz
    plt.figure(figsize=(20, 6))
    plt.plot(raw_signal, label="Öncesi (Ham Sinyal)", alpha=0.5)
    plt.plot(cleaned_signal, label="Sonrası (Temizlenmiş Sinyal)", color="red", alpha=0.7)
    plt.legend()
    plt.title(f"{set_label} İçin Artefakt Temizleme Öncesi ve Sonrası (Tüm Dosyalar)")
    plt.xlabel("Zaman (Örnek Noktaları)")
    plt.ylabel("Genlik (µV)")
    plt.show()

# Çalıştırmak için:
process_and_plot_all("S", "S")