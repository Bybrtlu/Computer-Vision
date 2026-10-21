# YOLO Multi-Task Eğitim Paneli

PyQt6 kullanılarak geliştirilmiş, Ultralytics YOLO modellerini arayüz üzerinden kolayca eğitmenizi sağlayan masaüstü uygulaması.

## 🚀 Özellikler
- **Çoklu Sürüm Desteği:** YOLOv8, YOLOv9, YOLOv10, YOLOv11 ve YOLOv26 modellerini destekler.
- **Çoklu Görev (Multi-Task):** Detection (Nesne Tespiti), Segmentation (Segmentasyon), Classification (Sınıflandırma) ve Pose (Poz Tahmini) modları.
- **Otomatik Veriseti Hazırlama:** Ham görsellerinizi ve etiketlerinizi belirlenen oranlarda (`Train / Val / Test`) otomatik olarak böler ve gerekli `yaml` yapılandırmasını oluşturur.
- **Arka Plan İş Parçacığı (Multi-threading):** Eğitim sürerken arayüzün donmasını engeller ve logları anlık olarak ekrana yazdırır.

## ⚙️ Gereksinimler
Projeyi çalıştırabilmek için aşağıdaki Python kütüphanelerinin yüklü olması gerekmektedir:

```bash
pip install ultralytics PyQt6 pyyaml

🖥️ Kullanım

Uygulamayı başlatın:
    

python dosya_adi.py

Arayüzden eğitmek istediğiniz Model Versiyonunu (örn: yolo26), Model Boyutunu (n, s, m, l, x) ve Görevi (Detection, Segmentation vb.) seçin.

Hiperparametreleri (Epochs, Görüntü Boyutu, Batch) ihtiyacınıza göre ayarlayın.

Veriseti bölümünden ham verilerinizin bulunduğu klasörü seçin.

"Veriseti Hazırla ve Eğitimi Başlat" butonuna basarak işlemi başlatın.

Not: Uygulama işlevsel olarak sorunsuz çalışmaktadır, ancak görsel arayüzü geliştirilmeye ve güncellenmeye devam edecektir.
