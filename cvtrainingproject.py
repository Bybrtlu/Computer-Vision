import sys
import os
import random
import shutil
import yaml
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QLineEdit,
                             QPushButton, QProgressBar, QTextEdit, QFileDialog, QSpinBox)
from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO


class TrainingThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            # ultralytics kütüphanesi model ismindeki -seg veya -cls ekinden
            # hangi görevde çalışacağını otomatik anlar.
            model = YOLO(self.params['model_path'])

            results = model.train(
                data=self.params['data_yaml'],
                epochs=self.params['epochs'],
                imgsz=self.params['imgsz'],
                batch=self.params['batch'],
                patience=self.params['patience'],
                device=self.params['device']
            )
            self.finished_signal.emit(results)
        except Exception as e:
            self.log_signal.emit(f"Eğitim Hatası: {str(e)}")


class YOLOInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Multi-Task Eğitim Paneli")
        self.setMinimumSize(800, 850)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # --- Model, Görev ve Cihaz Ayarları ---
        h_model = QHBoxLayout()

        self.combo_ver = QComboBox()
        # DÜZELTME: YOLO26 için doğru isimlendirme 'yolo26' şeklindedir.
        self.combo_ver.addItems(["yolov8", "yolov9", "yolov10", "yolo11", "yolo26"])

        self.combo_size = QComboBox()
        self.combo_size.addItems(["n", "s", "m", "l", "x"])

        # Görev Seçici (Detection, Segmentation vb.)
        self.combo_task = QComboBox()
        self.combo_task.addItems(["Detection", "Segmentation", "Classification", "Pose"])

        self.combo_device = QComboBox()
        self.combo_device.addItems(["0 (GPU)", "CPU"])

        h_model.addWidget(QLabel("Versiyon:"))
        h_model.addWidget(self.combo_ver)
        h_model.addWidget(self.combo_size)
        h_model.addWidget(QLabel("Görev:"))
        h_model.addWidget(self.combo_task)
        h_model.addWidget(QLabel("Cihaz:"))
        h_model.addWidget(self.combo_device)
        layout.addLayout(h_model)

        # --- Hiperparametreler ---
        h_params = QHBoxLayout()

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(100)

        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(32, 2048)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(-1, 256)
        self.spin_batch.setValue(16)

        h_params.addWidget(QLabel("Epochs:"))
        h_params.addWidget(self.spin_epochs)
        h_params.addWidget(QLabel("Görüntü Boyutu:"))
        h_params.addWidget(self.spin_imgsz)
        h_params.addWidget(QLabel("Batch:"))
        h_params.addWidget(self.spin_batch)
        layout.addLayout(h_params)

        # --- Veriseti ve Bölme ---
        h_path = QHBoxLayout()
        self.txt_path = QLineEdit()
        btn_browse = QPushButton("Klasör Seç")
        btn_browse.clicked.connect(self.select_folder)
        h_path.addWidget(QLabel("Veriseti:"))
        h_path.addWidget(self.txt_path)
        h_path.addWidget(btn_browse)
        layout.addLayout(h_path)

        h_split = QHBoxLayout()
        self.txt_train = QLineEdit("70")
        self.txt_val = QLineEdit("20")
        self.txt_test = QLineEdit("10")
        h_split.addWidget(QLabel("Train %:"))
        h_split.addWidget(self.txt_train)
        h_split.addWidget(QLabel("Val %:"))
        h_split.addWidget(self.txt_val)
        h_split.addWidget(QLabel("Test %:"))
        h_split.addWidget(self.txt_test)
        layout.addLayout(h_split)

        # --- İşlem Butonu ---
        self.btn_start = QPushButton("Veriseti Hazırla ve Eğitimi Başlat")
        self.btn_start.setFixedHeight(45)
        self.btn_start.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; border-radius: 5px;")
        self.btn_start.clicked.connect(self.process_and_train)
        layout.addWidget(self.btn_start)

        # --- Progress ve Log ---
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            "background-color: #1e1e1e; color: #00ff00; font-family: 'Consolas' or 'Courier';")
        layout.addWidget(self.log_output)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Veriseti Seç")
        if folder: self.txt_path.setText(folder)

    def prepare_dataset(self, source_path):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        all_files = [f for f in os.listdir(source_path) if f.lower().endswith(valid_extensions)]

        if not all_files:
            self.log_output.append("Hata: Klasörde görsel bulunamadı!")
            return None

        random.shuffle(all_files)

        try:
            t_ratio = int(self.txt_train.text()) / 100
            v_ratio = int(self.txt_val.text()) / 100
        except ValueError:
            self.log_output.append("Hata: Oranlar geçerli sayı olmalıdır!")
            return None

        t_idx = int(len(all_files) * t_ratio)
        v_idx = t_idx + int(len(all_files) * v_ratio)

        splits = {'train': all_files[:t_idx], 'val': all_files[t_idx:v_idx], 'test': all_files[v_idx:]}
        dataset_root = os.path.join(source_path, "yolo_ready_data")

        for s in splits:
            for d in ["images", "labels"]:
                os.makedirs(os.path.join(dataset_root, s, d), exist_ok=True)
            for f in splits[s]:
                shutil.copy(os.path.join(source_path, f), os.path.join(dataset_root, s, "images", f))
                txt_f = os.path.splitext(f)[0] + ".txt"
                source_txt = os.path.join(source_path, txt_f)
                if os.path.exists(source_txt):
                    shutil.copy(source_txt, os.path.join(dataset_root, s, "labels", txt_f))

        classes_path = os.path.join(source_path, "classes.txt")
        if os.path.exists(classes_path):
            with open(classes_path, "r", encoding="utf-8") as f:
                class_list = [line.strip() for line in f.readlines() if line.strip()]
            class_dict = {i: name for i, name in enumerate(class_list)}
        else:
            class_dict = {0: "object"}

        yaml_content = {
            'path': os.path.abspath(dataset_root),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': class_dict
        }

        yaml_path = os.path.join(dataset_root, "config.yaml")
        with open(yaml_path, 'w', encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True)

        return yaml_path

    def process_and_train(self):
        source = self.txt_path.text()
        if not source or not os.path.exists(source):
            self.log_output.append("Hata: Geçersiz kaynak yolu!")
            return

        # --- MODEL İSMİ OLUŞTURMA MANTIĞI ---
        ver = self.combo_ver.currentText()  # yolov8, yolo26 vb.
        size = self.combo_size.currentText()  # n, s, m, l, x
        task = self.combo_task.currentText()  # Segmentation, Detection vb.

        # Görevlere göre ekleri belirle
        task_suffix = ""
        if task == "Segmentation":
            task_suffix = "-seg"
        elif task == "Classification":
            task_suffix = "-cls"
        elif task == "Pose":
            task_suffix = "-pose"

        # Örn: yolo26 + n + -seg + .pt -> yolo26n-seg.pt (Eğer yoksa otomatik indirilir)
        full_model_name = f"{ver}{size}{task_suffix}.pt"

        self.log_output.append(">>> Veriseti Hazırlanıyor...")
        y_path = self.prepare_dataset(source)
        if not y_path: return

        training_params = {
            'model_path': full_model_name,
            'data_yaml': y_path,
            'epochs': self.spin_epochs.value(),
            'imgsz': self.spin_imgsz.value(),
            'batch': self.spin_batch.value(),
            'patience': 50,
            'device': 0 if "GPU" in self.combo_device.currentText() else "cpu"
        }

        self.log_output.append(f">>> Görev: {task}")
        self.log_output.append(f">>> Eğitim Başlıyor: {full_model_name}")
        self.btn_start.setEnabled(False)

        self.thread = TrainingThread(training_params)
        self.thread.log_signal.connect(self.log_output.append)
        self.thread.finished_signal.connect(self.training_finished)
        self.thread.start()

    def training_finished(self, results):
        self.log_output.append(f"\n[BİTTİ] Sonuçlar şuraya kaydedildi: {results.save_dir}")
        self.btn_start.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOInterface()
    window.show()
    sys.exit(app.exec())
