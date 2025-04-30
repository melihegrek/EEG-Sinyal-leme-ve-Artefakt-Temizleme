import numpy as np
import matplotlib.pyplot as plt
import os

# Verileri klasörden yükle
def load_data_from_folder(folder_path, set_label):
    file_names = sorted([f for f in os.listdir(folder_path) if f.startswith(set_label) and f.lower().endswith(('.txt', '.TXT'))])
    all_signals = []

    if not file_names:
        print(f"[Uyarı] '{set_label}' etiketi için dosya bulunamadı!")

    for file in file_names:
        file_path = os.path.join(folder_path, file)
        signal_data = np.loadtxt(file_path)
        if signal_data.size == 0:
            print(f"[Uyarı] '{file}' dosyası boş!")
        all_signals.append(signal_data)

    if not all_signals:
        print(f"[Hata] '{set_label}' veri seti boş!")
        return np.array([])

    return np.array(all_signals)

# Verinin aralığını hesapla (min-max)
def calculate_range(signal_data):
    if signal_data.size == 0:
        return 0
    return np.max(signal_data) - np.min(signal_data)

# Fuzzy TOPSIS algoritması
def fuzzy_topsis(criteria_matrix, weights):
    # Normalize et
    normalized_matrix = criteria_matrix / np.linalg.norm(criteria_matrix, axis=0)
    
    # Ağırlıklı matrisi hesapla
    weighted_matrix = normalized_matrix * weights
    
    # İdeal ve negatif ideal çözümleri hesapla
    ideal_solution = np.max(weighted_matrix, axis=0)
    negative_ideal_solution = np.min(weighted_matrix, axis=0)
    
    # Uzaklık hesaplamaları
    distance_to_ideal = np.abs(weighted_matrix - ideal_solution)
    distance_to_negative = np.abs(weighted_matrix - negative_ideal_solution)
    
    # Relative closeness hesapla
    relative_closeness = distance_to_negative / (distance_to_ideal + distance_to_negative)
    
    return weighted_matrix, relative_closeness

# Fuzzy skorlarına göre epsilon hesaplama
def calculate_epsilon_from_fuzzy(closeness_scores, epsilon_total):
    total_closeness = np.sum(closeness_scores)
    if total_closeness == 0:
        print("[Hata] Relative closeness değerleri 0 toplamına sahip!")
        return {}
    
    epsilon_values = {}
    for i, score in enumerate(closeness_scores):
        epsilon_values[i] = (score / total_closeness) * epsilon_total
    return epsilon_values

# LDP (Laplace Mekanizması) uygulama
def apply_ldp(signal_data: np.ndarray, epsilon: float = 0.5, sensitivity: float = 1.0) -> np.ndarray:
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size=signal_data.shape)
    noisy_signal = signal_data + noise
    print(f"Laplace ölçek parametresi (scale): {scale}")
    print(f"Bu veri seti için epsilon değeri: {epsilon}")
    print(f"Kullanıcı tarafından verilen sensitivity: {sensitivity}")
    return noisy_signal

# Verilerin işlenmesi
def process_data(root_folder, set_labels, epsilon_total, sensitivity):
    raw_signals = []

    for label in set_labels:
        folder_path = os.path.join(root_folder, label)
        print(f"\n--- '{label}' klasörü işleniyor ---")

        signals = load_data_from_folder(folder_path, label)
        if signals.size == 0:
            print(f"[Hata] '{label}' klasöründen veri yüklenemedi!")
            return

        # Tüm sinyalleri birleştir (tek boyutlu hale getir)
        merged_signals = signals.flatten()
        raw_signals.append(merged_signals)

    if any(d.size == 0 for d in raw_signals):
        print("[Hata] Bir veya daha fazla veri seti boş. İşlem yapılamaz!")
        return

    # Fuzzy TOPSIS kriter matrisini oluştur
    fuzzy_criteria_matrix = np.array([calculate_range(data) for data in raw_signals])
    weights = np.ones(len(raw_signals))  # Eşit ağırlık
    
    # Fuzzy TOPSIS hesaplamaları
    weighted_matrix, relative_closeness = fuzzy_topsis(fuzzy_criteria_matrix, weights)

    print("\nFuzzy TOPSIS Ağırlıklı Matris ve Relative Closeness Değerleri:")
    print("Ağırlıklı Matris:")
    print(weighted_matrix)
    for i, score in enumerate(relative_closeness):
        print(f"{set_labels[i]} -> {score:.4f}")

    # Closeness'a göre epsilon dağılımı
    epsilon_values = calculate_epsilon_from_fuzzy(relative_closeness, epsilon_total)

    if not epsilon_values:
        print("[Hata] Epsilon değerleri hesaplanamadı!")
        return

    # Her veri seti için LDP uygulayıp görselleştir
    for idx, label in enumerate(set_labels):
        print(f"\n--- '{label}' veri seti için epsilon: {epsilon_values[idx]} ---")
        signal_data = raw_signals[idx]

        max_value = np.max(signal_data)
        min_value = np.min(signal_data)
        print(f"'{label}' veri seti: Max Değer = {max_value}, Min Değer = {min_value}")

        ldp_signal = apply_ldp(signal_data, epsilon=epsilon_values[idx], sensitivity=sensitivity)

        # Orijinal ve LDP sinyalleri karşılaştırması
        plt.figure(figsize=(10, 5))
        plt.plot(signal_data, label="Orijinal Sinyal", alpha=0.7)
        plt.plot(ldp_signal, label="LDP Uygulanmış Sinyal", linestyle='dashed', alpha=0.7)
        plt.legend()
        plt.title(f"'{label}' Veri Seti: LDP Öncesi ve Sonrası Sinyal Karşılaştırması")
        plt.xlabel("Zaman (Örnek Noktaları)")
        plt.ylabel("Genlik (µV)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root_folder = "."  # Script ile aynı klasör
    set_labels = ["F", "N", "O", "S", "Z"]
    epsilon_total = 2.5

    try:
        sensitivity_input = float(input("Lütfen hassasiyet (sensitivity) değerini girin (örn. 10.0): "))
        process_data(root_folder, set_labels, epsilon_total, sensitivity_input)
    except ValueError:
        print("[Hata] Geçersiz bir sayı girdiniz!")
