import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt
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

def apply_notch_filter(signal_data, fs=173.61, notch_freq=50):
    Q = 30.0
    b, a = signal.iirnotch(notch_freq, Q, fs)
    return signal.filtfilt(b, a, signal_data)

def apply_bandpass_filter(signal_data, fs=173.61, lowcut=0.5, highcut=50.0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
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

def extract_time_domain_features(signal_data):
    mean_val = np.mean(signal_data)
    variance_val = np.var(signal_data)
    energy_val = np.sum(signal_data ** 2)
    return mean_val, variance_val, energy_val

def extract_frequency_domain_features(signal_data, fs=173.61):
    freqs, psd = signal.welch(signal_data, fs, nperseg=256)
    alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 12)])
    beta_power = np.sum(psd[(freqs >= 12) & (freqs <= 30)])
    return alpha_power, beta_power, psd, freqs

def extract_time_frequency_features(signal_data, fs=173.61):
    f, t, Zxx = signal.stft(signal_data, fs, nperseg=256)
    wavelet_coeffs, _ = pywt.cwt(signal_data, scales=np.arange(1, 128), wavelet='morl')
    return f, t, Zxx, wavelet_coeffs

def process_and_plot_all(folder_path, set_label):
    #plt.ion()  # Etkileşimli modu aç
    raw_signals = load_data_from_folder(folder_path, set_label)
    if raw_signals.size == 0:
        print(f"HATA: '{folder_path}' klasöründe '{set_label}' ile başlayan dosya bulunamadı!")
        return
    raw_signal = raw_signals.flatten()
    
    # Sinyal işleme adımları
    filtered_signal = apply_notch_filter(raw_signal)
    bandpass_filtered_signal = apply_bandpass_filter(filtered_signal)
    thresholded_signal = apply_thresholding(bandpass_filtered_signal)
    cleaned_signal = apply_ica(thresholded_signal.reshape(-1, 1)).flatten()
    
    # Özellik çıkarımı
    mean_val, variance_val, energy_val = extract_time_domain_features(cleaned_signal)
    alpha_power, beta_power, psd, freqs = extract_frequency_domain_features(cleaned_signal)
    f, t, Zxx, wavelet_coeffs = extract_time_frequency_features(cleaned_signal)
    
    # Zaman Alanı Özellikleri
    print(f"Ortalama: {mean_val}")
    print(f"Varyans: {variance_val}")
    print(f"Enerji: {energy_val}")
    
    # Frekans Alanı Özellikleri
    print(f"Alfa Gücü (8-12 Hz): {alpha_power}")
    print(f"Beta Gücü (12-30 Hz): {beta_power}")
    
    # Grafikler
    plt.figure(figsize=(20, 6))
    plt.plot(raw_signal, label="Ham Sinyal", alpha=0.5)
    plt.plot(cleaned_signal, label="Temizlenmiş Sinyal", color="red", alpha=0.7)
    plt.legend()
    plt.title(f"{set_label} İçin Sinyal İşleme ve Özellik Çıkarma")
    plt.xlabel("Zaman (Örnek Noktaları)")
    plt.ylabel("Genlik (µV)")
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, psd)
    plt.title("Güç Spektral Yoğunluğu (PSD)")
    plt.xlabel("Frekans (Hz)")
    plt.ylabel("Güç Yoğunluğu")
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title("Zaman-Frekans Analizi (STFT)")
    plt.xlabel("Zaman (s)")
    plt.ylabel("Frekans (Hz)")
    plt.colorbar(label="Genlik")
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.imshow(np.abs(wavelet_coeffs), aspect='auto', cmap='jet', extent=[0, len(cleaned_signal), 1, 128])
    plt.title("Wavelet Dönüşümü")
    plt.xlabel("Zaman")
    plt.ylabel("Ölçek")
    plt.colorbar(label="Katsayı Büyüklüğü")
    plt.show()

# Çalıştırmak için:
process_and_plot_all("S", "S")
