import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.decomposition import FastICA
import os

def load_data_from_folder(folder_path, set_label):
    """
    Belirtilen klasördeki tüm .txt dosyalarını okuyup bir NumPy dizisi olarak döndürür.
    """
    file_names = sorted([f for f in os.listdir(folder_path) if f.startswith(set_label) and f.lower().endswith(('.txt', '.TXT'))])
    all_signals = []

    for file in file_names:
        file_path = os.path.join(folder_path, file)
        signal_data = np.loadtxt(file_path)  # .txt içindeki 4097 veriyi oku
        all_signals.append(signal_data)

    return np.array(all_signals) if all_signals else np.array([])  # Boşsa boş bir array döndür

def apply_notch_filter(signal_data, fs=256, notch_freq=50):
    """
    50 Hz elektriksel gürültüyü ortadan kaldırmak için Notch filtresi uygular.
    """
    if len(signal_data) < 9:
        print("UYARI: Sinyal uzunluğu 9'dan küçük olduğu için Notch filtresi uygulanamıyor.")
        return signal_data  # Küçük sinyalleri değiştirmeden geri döndür
    
    Q = 30.0  # Kalite faktörü
    b, a = signal.iirnotch(notch_freq, Q, fs)
    return signal.filtfilt(b, a, signal_data)

def apply_thresholding(signal_data, threshold=100):
    """
    Belirtilen eşik değerinin üzerinde veya altında olan anormal noktaları ortadan kaldırır.
    """
    return np.where(np.abs(signal_data) > threshold, np.median(signal_data), signal_data)

def apply_ica(signal_data, n_components=1):
    """
    Bağımsız Bileşen Analizi (ICA) ile göz kırpma gibi artefaktları temizler.
    """
    if signal_data.shape[0] < 2:
        print("UYARI: ICA uygulanamadı çünkü sinyal yeterince uzun değil.")
        return signal_data.flatten()  # Orijinal sinyali değiştirmeden döndür

    ica = FastICA(n_components=n_components, random_state=42)
    return ica.fit_transform(signal_data)

def process_and_plot_all(folder_path, set_label):
    """
    Tüm dosyaları okuyup, ham ve temizlenmiş sinyalleri tek bir grafikte çizer.
    """
    raw_signals = load_data_from_folder(folder_path, set_label)

    if raw_signals.size == 0:
        print(f"HATA: '{folder_path}' klasöründe '{set_label}' ile başlayan dosya bulunamadı!")
        return

    # Tüm dosyaları birleştir (100 x 4097 → 409700 noktaya dönüştür)
    raw_signal = raw_signals.flatten()

    # Notch filtresi uygula
    filtered_signal = apply_notch_filter(raw_signal)

    # Thresholding uygula
    thresholded_signal = apply_thresholding(filtered_signal)

    # ICA uygula (Önce uygunluk kontrolü yapıldı)
    cleaned_signal = apply_ica(thresholded_signal.reshape(-1, 1)).flatten()

    # Grafik çiz
    plt.figure(figsize=(20, 6))  # Genişliği artırdık
    plt.plot(raw_signal, label="Öncesi (Ham Sinyal)", alpha=0.5)
    plt.plot(cleaned_signal, label="Sonrası (Temizlenmiş Sinyal)", color="red", alpha=0.7)
    plt.legend()
    plt.title(f"{set_label} İçin Artefakt Temizleme Öncesi ve Sonrası (Tüm Dosyalar)")
    plt.xlabel("Zaman (Örnek Noktaları)")
    plt.ylabel("Genlik (µV)")
    plt.show()

# 📌 Çalıştırmak için:
process_and_plot_all("N", "N")
