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

# Gizlilik bütçesi için epsilon değerlerinin dağılımı
def calculate_epsilon_for_datasets(dataset, epsilon_total):
    if not dataset or any(d.size == 0 for d in dataset):
        print("[Hata] Veri setinde boş veri bulunuyor. Epsilon hesaplanamaz!")
        return {}

    ranges = [calculate_range(data) for data in dataset]
    total_range = np.sum(ranges)
    
    epsilon_values = {}
    for i, data in enumerate(dataset):
        epsilon_values[i] = (ranges[i] / total_range) * epsilon_total
        
    return epsilon_values

# LDP (Laplace Mekanizması) uygulama
def apply_ldp(signal_data: np.ndarray, epsilon: float = 0.5, sensitivity: float = None) -> np.ndarray:
    if sensitivity is None:
        sensitivity = np.max(signal_data) - np.min(signal_data)
        print(f"Dinamik hesaplanan sensitivity: {sensitivity}")

    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size=signal_data.shape)
    noisy_signal = signal_data + noise
    print(f"Laplace ölçek parametresi (scale): {scale}")
    print(f"Bu veri seti için epsilon değeri: {epsilon}")

    return noisy_signal

# Fuzzy TOPSIS algoritması
def fuzzy_topsis(criteria_matrix, weights):
    normalized_matrix = criteria_matrix / np.linalg.norm(criteria_matrix)
    weighted_matrix = normalized_matrix * weights
    ideal_solution = np.max(weighted_matrix)
    negative_ideal_solution = np.min(weighted_matrix)
    distance_to_ideal = np.abs(weighted_matrix - ideal_solution)
    distance_to_negative = np.abs(weighted_matrix - negative_ideal_solution)
    relative_closeness = distance_to_negative / (distance_to_ideal + distance_to_negative)
    return relative_closeness

# Verilerin işlenmesi
def process_data(root_folder, set_labels, epsilon_total):
    raw_signals = []
    
    for label in set_labels:
        folder_path = os.path.join(root_folder, label)
        print(f"\n--- '{label}' klasörü işleniyor ---")
        
        signals = load_data_from_folder(folder_path, label)
        if signals.size == 0:
            print(f"[Hata] '{label}' klasöründen veri yüklenemedi!")
            return
        
        raw_signals.append(signals)
    
    if any(d.size == 0 for d in raw_signals):
        print("[Hata] Bir veya daha fazla veri seti boş. İşlem yapılamaz!")
        return
    
    epsilon_values = calculate_epsilon_for_datasets(raw_signals, epsilon_total)
    
    if not epsilon_values:
        print("[Hata] Epsilon değerleri hesaplanamadı!")
        return
    
    fuzzy_criteria_matrix = np.array([calculate_range(data) for data in raw_signals])
    weights = np.ones(len(raw_signals))  # 🔧 Burayı düzelttik
    relative_closeness = fuzzy_topsis(fuzzy_criteria_matrix, weights)
    
    for idx, label in enumerate(set_labels):
        print(f"\n--- '{label}' veri seti için epsilon: {epsilon_values[idx]} ---")
        signal_data = raw_signals[idx].flatten()
        
        ldp_signal = apply_ldp(signal_data, epsilon=epsilon_values[idx])
        
        plt.figure(figsize=(10, 5))
        plt.plot(signal_data, label="Orijinal Sinyal", alpha=0.7)
        plt.plot(ldp_signal, label="LDP Uygulanmış Sinyal", linestyle='dashed', alpha=0.7)
        plt.legend()
        plt.title(f"'{label}' Veri Seti: LDP Öncesi ve Sonrası Sinyal Karşılaştırması")
        plt.xlabel("Zaman (Örnek Noktaları)")
        plt.ylabel("Genlik (µV)")
        plt.show()

if __name__ == "__main__":
    root_folder = "."  # Script ile aynı klasör
    set_labels = ["F", "N", "O", "S", "Z"]
    epsilon_total = 2.5
    process_data(root_folder, set_labels, epsilon_total)
