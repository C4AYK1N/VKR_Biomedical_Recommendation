"""
Модуль для загрузки реальных датасетов ЭКГ.
"""
import os
import wget
import zipfile
import pandas as pd
import numpy as np
import wfdb
from sklearn.model_selection import train_test_split
from utils.config import Config
import logging

class RealECGDataLoader:
    """Класс для загрузки реальных данных ЭКГ."""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
    def download_mit_bih_dataset(self):
        """Загрузка MIT-BIH Arrhythmia Database."""
        dataset_url = "https://www.physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"
        local_zip_path = "data/mit-bih-arrhythmia-database-1.0.0.zip"
        extract_path = "data/mit-bih"
        
        # Создаем директории если нет
        os.makedirs("data", exist_ok=True)
        os.makedirs(extract_path, exist_ok=True)
        
        # Скачиваем если файла нет
        if not os.path.exists(local_zip_path):
            self.logger.info("Скачивание MIT-BIH датасета...")
            wget.download(dataset_url, local_zip_path)
            self.logger.info("Скачивание завершено!")
        
        # Распаковываем если нужно
        if not os.path.exists(os.path.join(extract_path, "RECORDS")):
            self.logger.info("Распаковка архива...")
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        
        return extract_path
    
    def load_mit_bih_records(self, record_names=None, max_records=10):
        """Загрузка записей из MIT-BIH."""
        extract_path = self.download_mit_bih_dataset()
        
        if record_names is None:
            # Берем первые записи
            with open(os.path.join(extract_path, "RECORDS"), 'r') as f:
                all_records = [line.strip() for line in f if line.strip()]
            record_names = all_records[:max_records]
        
        signals = []
        labels = []
        
        for record_name in record_names:
            try:
                # Чтение записи
                record_path = os.path.join(extract_path, record_name)
                record = wfdb.rdrecord(record_path)
                annotation = wfdb.rdann(record_path, 'atr')
                
                # Берем первый канал (обычно MLII)
                signal = record.p_signal[:, 0]
                
                # Сегментация сигнала
                segment_length = self.config.SEQUENCE_LENGTH
                n_segments = len(signal) // segment_length
                
                for i in range(n_segments):
                    start_idx = i * segment_length
                    end_idx = start_idx + segment_length
                    
                    segment = signal[start_idx:end_idx]
                    
                    # Определяем метку для сегмента
                    # (упрощенно - берем наиболее частую метку в сегменте)
                    segment_annotations = []
                    for ann_sample, ann_symbol in zip(annotation.sample, annotation.symbol):
                        if start_idx <= ann_sample < end_idx:
                            segment_annotations.append(ann_symbol)
                    
                    if segment_annotations:
                        # Преобразуем символы в числовые метки
                        label = self._symbol_to_label(segment_annotations[0])
                    else:
                        label = 0  # нормальный ритм
                    
                    signals.append(segment)
                    labels.append(label)
                    
            except Exception as e:
                self.logger.warning(f"Ошибка при загрузке записи {record_name}: {e}")
                continue
        
        return np.array(signals), np.array(labels)
    
    def _symbol_to_label(self, symbol):
        """Преобразование символов аннотаций в числовые метки."""
        label_mapping = {
            'N': 0,  # Normal beat
            'L': 0,  # Left bundle branch block beat
            'R': 0,  # Right bundle branch block beat
            'B': 0,  # Bundle branch block beat (unspecified)
            'A': 1,  # Atrial premature beat
            'a': 1,  # Aberrated atrial premature beat
            'J': 1,  # Nodal (junctional) premature beat
            'S': 1,  # Supraventricular premature beat
            'V': 2,  # Premature ventricular contraction
            'r': 3,  # R-on-T premature ventricular contraction
            'F': 4,  # Fusion of ventricular and normal beat
            'e': 5,  # Atrial escape beat
            'j': 5,  # Nodal (junctional) escape beat
            'n': 5,  # Supraventricular escape beat
            'E': 5,  # Ventricular escape beat
            '/': 6,  # Paced beat
            'f': 6,  # Fusion of paced and normal beat
            'Q': 7,  # Unclassifiable beat
            '?': 7   # Unknown beat
        }
        return label_mapping.get(symbol, 0)