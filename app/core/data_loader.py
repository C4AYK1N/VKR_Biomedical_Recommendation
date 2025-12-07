python:app/core/data_loader.py
"""
Модуль для загрузки и предварительной обработки биомедицинских данных.
"""
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
from utils.config import Config
from .dataset_loader import RealECGDataLoader

class BiomedicalDataLoader:
    def __init__(self):
        self.config = Config()
        self.real_data_loader = RealECGDataLoader()
        
    def download_dataset(self, use_real_data=True):
        """
        Загрузка датасета.
        
        Args:
            use_real_data (bool): Если True, загружает реальные данные MIT-BIH
        """
        print("Загрузка датасета...")
        
        if use_real_data:
            try:
                # Пробуем загрузить реальные данные
                X, y = self.real_data_loader.load_mit_bih_records(max_records=5)
                print(f"Загружено {len(X)} реальных записей ЭКГ")
                
                # Если данных мало, добавляем симулированные
                if len(X) < 1000:
                    print("Мало реальных данных, добавляем симулированные...")
                    X_sim, y_sim = self._create_simulated_data(2000)
                    X = np.vstack([X, X_sim])
                    y = np.concatenate([y, y_sim])
                
            except Exception as e:
                print(f"Ошибка при загрузке реальных данных: {e}")
                print("Используем симулированные данные...")
                X, y = self._create_simulated_data(5000)
        else:
            # Используем только симулированные данные
            X, y = self._create_simulated_data(5000)
        
        print(f"Всего образцов: {len(X)}")
        return X, y
    
    def _create_simulated_data(self, n_samples):
        """
        Создание симулированных данных ЭКГ для демонстрации.
        """
        np.random.seed(self.config.RANDOM_STATE)
        
        n_features = self.config.SEQUENCE_LENGTH
        
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            t = np.linspace(0, 1, n_features)
            base_ecg = 0.5 * np.sin(2 * np.pi * 1 * t)
            
            # Добавляем шум
            noise = 0.1 * np.random.normal(size=n_features)
            
            # Создаем разные типы аритмий
            arrhythmia_type = i % 5
            
            if arrhythmia_type == 0:  # Нормальный ритм
                ecg_signal = base_ecg + noise
            elif arrhythmia_type == 1:  # Апноэ
                ecg_signal = base_ecg * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)) + noise
            elif arrhythmia_type == 2:  # Фибрилляция предсердий
                ecg_signal = base_ecg + 0.3 * np.random.normal(size=n_features) + noise
            elif arrhythmia_type == 3:  # Шум
                ecg_signal = 0.8 * np.random.normal(size=n_features)
            else:  # Другая аритмия
                ecg_signal = base_ecg * (1 + 0.3 * np.sin(2 * np.pi * 2 * t)) + noise
            
            X[i] = ecg_signal
            y[i] = arrhythmia_type
        
        print(f"Создано {n_samples} симулированных образцов")
        return X, y
    
    def preprocess_ecg_signal(self, ecg_signal):
        """
        Предобработка сигнала ЭКГ.
        """
        # 1. Нормализация
        ecg_normalized = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
        
        # 2. Фильтрация (bandpass filter 0.5-40 Hz)
        nyquist = self.config.SAMPLING_RATE / 2
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        b, a = signal.butter(3, [low, high], btype='band')
        ecg_filtered = signal.filtfilt(b, a, ecg_normalized)
        
        return ecg_filtered
    
    def split_data(self, X, y):
        """
        Разделение данных на обучающую, валидационную и тестовую выборки.
        """
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.config.VALIDATION_SIZE/(1-self.config.TEST_SIZE),
            random_state=self.config.RANDOM_STATE,
            stratify=y_train_val
        )
        
        print(f"Разделение данных:")
        print(f"  Обучающая выборка: {len(X_train)}")
        print(f"  Валидационная выборка: {len(X_val)}")
        print(f"  Тестовая выборка: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test