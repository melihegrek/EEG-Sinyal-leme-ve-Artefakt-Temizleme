## EEG Sinyal İşleme ve Artefakt Temizleme

Bu proje, EEG (Elektroensefalogram) verilerini işlemek ve temizlemek amacıyla bir dizi sinyal işleme tekniğini içerir. Proje, EEG verilerindeki gürültü ve artefaktları temizlemek için çeşitli filtreleme ve istatistiksel yöntemler kullanır. Ayrıca, verilerin görselleştirilmesi ve temizlenmiş sinyallerin analizine olanak tanır.

## Özellikler

- **Notch Filtresi**: 50 Hz frekansındaki elektriksel gürültüleri ortadan kaldırmak için Notch filtresi uygulanır.
- **Band-Pass Filtresi**: EEG sinyalleri, 0.5 Hz ile 50 Hz arasındaki frekansları geçiren bir band-pass filtresinden geçirilir.
- **Eşikleme**: Belirli bir genlik değerinin üzerinde olan anormal noktalar tespit edilip düzeltilir.
- **Bağımsız Bileşen Analizi (ICA)**: EEG sinyallerindeki göz kırpma ve benzeri artefaktlar temizlenir.
- **Sinyal Görselleştirmesi**: Ham ve temizlenmiş sinyallerin karşılaştırmalı olarak çizildiği grafikler.

## Kullanım

1. **Veri Yükleme**:
   - `load_data_from_folder` fonksiyonu, belirtilen klasördeki `.txt` dosyalarındaki EEG verilerini yükler.
   - Bu veriler, sinyalleri içerir ve her bir dosya bir zaman serisi olarak ele alınır.

2. **Filtreleme**:
   - **Notch filtresi**: 50 Hz gürültüsünü ortadan kaldırmak için kullanılır.
   - **Band-Pass filtresi**: 0.5 Hz ile 50 Hz arasındaki frekanslar geçiren band-pass filtresi uygulanır.

3. **Eşikleme**: Sinyalin genlik değeri belirli bir eşiği geçtiğinde, bu noktalar medyan değerle değiştirilir.

4. **ICA (Bağımsız Bileşen Analizi)**: ICA, EEG sinyallerindeki artefaktları temizler ve bağımsız bileşenleri ayırır.

5. **Görselleştirme**: Ham sinyal ile temizlenmiş sinyal karşılaştırmalı olarak çizilir.

## Kullanılan Veri Seti

Bu projede kullanılan EEG veri seti [Bonn Epilepsy Database](https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/) adresinden temin edilmiştir.

## Gereksinimler

- `numpy`: Matematiksel hesaplamalar ve veri işleme için.
- `matplotlib`: Verilerin görselleştirilmesi için.
- `scipy`: Filtreleme ve sinyal işleme için.
- `sklearn`: Bağımsız bileşen analizi (ICA) için.

## Kurulum

Projenin gereksinimlerini kurmak için aşağıdaki komutları kullanabilirsiniz:

```bash
pip install numpy matplotlib scipy scikit-learn
```

## Kullanım Örneği

```python
process_and_plot_all("S", "S")
```

Bu fonksiyon, belirtilen dosya seti etiketine (`"S"`) göre verileri yükler ve sinyal işleme adımlarını uygular. Sonuç olarak, ham ve temizlenmiş sinyallerin karşılaştırıldığı bir grafik oluşturur.

## Katkıda Bulunma

Eğer projeye katkı sağlamak isterseniz, aşağıdaki adımları takip edebilirsiniz:

1. Projeyi kendi bilgisayarınıza klonlayın.
2. Yeni bir dal oluşturun (`git checkout -b feature-xyz`).
3. Gerekli değişiklikleri yapın ve testleri oluşturun.
4. Değişikliklerinizi ana dal ile birleştirin ve pull request gönderin.
