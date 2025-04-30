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

def apply_bandpass_filter(signal_data, fs=173.61, lowcut=0.5, highcut=40.0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, signal_data)   #hem ılerı hem gerı

def apply_thresholding(signal_data, threshold=100):
    return np.where(np.abs(signal_data) > threshold, np.median(signal_data), signal_data)

def apply_ica(signal_data, n_components=None):  
    if signal_data.ndim == 1:
        signal_data = signal_data.reshape(-1, 1)
    
    ica = FastICA(n_components=n_components, random_state=42, whiten="unit-variance")
    transformed = ica.fit_transform(signal_data)
    
    return transformed[:, 0] * np.std(signal_data)

def apply_gaussian_dp(signal_data: np.ndarray, epsilon: float = 0.5, delta: float = 2.44e-6, sensitivity: float = 500.0) -> np.ndarray:#delta verı setı buyuklugune baglı 1 / N
    """
    Gaussian Differential Privacy (DP) uygular.

    Parametreler:
    - signal_data (np.ndarray): Gürültü eklenecek sinyal verisi.
    - epsilon (float): Gizlilik parametresi.
    - delta (float): DP parametresi (küçük değerler daha güçlü gizlilik sağlar).
    - sensitivity (float): Maksimum duyarlılık değişimi.

    Dönüş:
    - np.ndarray: Gürültü eklenmiş sinyal.
    """
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * (sensitivity / epsilon)  # Gaussian noise standard deviation
    noise = np.random.normal(0, sigma, size=signal_data.shape)
    noisy_signal = signal_data + noise

    print(f"Gaussian noise sigma: {sigma}")

    # Gürültünün analizi
    noise_variation = noisy_signal - signal_data
    print(f"Gaussian DP Gürültü Min: {noise_variation.min()}, Max: {noise_variation.max()}")

    # Grafiklerle Gaussian DP etkisini inceleyelim
    plt.figure(figsize=(10, 5))
    plt.plot(signal_data, label="Gaussian DP Öncesi", alpha=0.7)
    plt.plot(noisy_signal, label="Gaussian DP Sonrası", linestyle='dashed', alpha=0.7)
    plt.legend()
    plt.title("Gaussian DP Öncesi ve Sonrası Sinyal Karşılaştırması")
    plt.xlabel("Zaman (Örnek Noktaları)")
    plt.ylabel("Genlik (µV)")
    plt.show()

    # Gürültüyü ayrı grafikte gösterelim
    plt.figure(figsize=(10, 5))
    plt.plot(noise_variation, color='purple', alpha=0.7)
    plt.title("Gaussian DP Tarafından Eklenen Gürültü")
    plt.xlabel("Zaman (Örnek Noktaları)")
    plt.ylabel("Gürültü Seviyesi (µV)")
    plt.show()

    # Gürültünün histogramını çizelim
    plt.figure(figsize=(8, 4))
    plt.hist(noise_variation, bins=50, color='blue', alpha=0.7, density=True)
    plt.title("Gaussian DP Gürültü Dağılımı")
    plt.xlabel("Gürültü Seviyesi")
    plt.ylabel("Frekans")
    plt.show()

    return noisy_signal

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
    raw_signals = load_data_from_folder(folder_path, set_label)
    if raw_signals.size == 0:
        print(f"HATA: '{folder_path}' klasöründe '{set_label}' ile başlayan dosya bulunamadı!")
        return
    raw_signal = raw_signals.flatten()
    
    filtered_signal = apply_notch_filter(raw_signal)
    bandpass_filtered_signal = apply_bandpass_filter(filtered_signal)
    thresholded_signal = apply_thresholding(bandpass_filtered_signal)
    cleaned_signal = apply_ica(thresholded_signal.reshape(-1, 1)).flatten()
    
    ldp_signal = apply_gaussian_dp(cleaned_signal, epsilon=2.0, sensitivity=0.5)

    print(f"LDP Öncesi: Min={cleaned_signal.min()}, Max={cleaned_signal.max()}")
    print(f"LDP Sonrası: Min={ldp_signal.min()}, Max={ldp_signal.max()}")
    
    mean_val, variance_val, energy_val = extract_time_domain_features(cleaned_signal)
    alpha_power, beta_power, psd, freqs = extract_frequency_domain_features(cleaned_signal)
    f, t, Zxx, wavelet_coeffs = extract_time_frequency_features(cleaned_signal)
    
    plt.figure(figsize=(20, 6))
    plt.plot(raw_signal, label="Ham Sinyal", alpha=0.5)
    plt.plot(cleaned_signal, label="Temizlenmiş Sinyal", color="red", alpha=0.7)
    plt.legend()
    plt.title(f"{set_label} İçin Sinyal İşleme ve Özellik Çıkarma")
    plt.xlabel("Zaman (Örnek Noktaları)")
    plt.ylabel("Genlik (µV)")
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(cleaned_signal, label="LDP Öncesi", alpha=0.7)
    plt.plot(ldp_signal, label="LDP Sonrası", linestyle='dashed', alpha=0.7)
    plt.legend()
    plt.title("LDP Öncesi ve Sonrası Sinyal Karşılaştırması")
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

process_and_plot_all("O", "O")
