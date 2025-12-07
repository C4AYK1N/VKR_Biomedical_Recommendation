"""
Модуль для извлечения признаков из биомедицинских сигналов.
Автор: Чайкин Виталий Федорович
Тема ВКР: Разработка рекомендательной системы на основе обработки биомедицинских данных
"""

import numpy as np
from scipy import stats, fft
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

class ECGFeatureEngineer:
    """Класс для инженерии признаков ЭКГ сигналов."""
    
    def __init__(self, sampling_rate=360):
        self.sampling_rate = sampling_rate
        
    def extract_temporal_features(self, ecg_signal):
        """
        Извлечение временных признаков из ЭКГ сигнала.
        
        Args:
            ecg_signal (array): Массив с отсчетами ЭКГ сигнала
            
        Returns:
            dict: Словарь с извлеченными признаками
        """
        features = {}
        
        # Базовые статистические признаки
        features['mean'] = np.mean(ecg_signal)
        features['std'] = np.std(ecg_signal)
        features['var'] = np.var(ecg_signal)
        features['skewness'] = stats.skew(ecg_signal)
        features['kurtosis'] = stats.kurtosis(ecg_signal)
        features['rms'] = np.sqrt(np.mean(ecg_signal**2))
        
        # Признаки на основе амплитуды
        features['max_amplitude'] = np.max(ecg_signal)
        features['min_amplitude'] = np.min(ecg_signal)
        features['peak_to_peak'] = features['max_amplitude'] - features['min_amplitude']
        features['mean_absolute'] = np.mean(np.abs(ecg_signal))
        
        # Признаки на основе производных
        derivative = np.diff(ecg_signal)
        features['max_derivative'] = np.max(derivative)
        features['min_derivative'] = np.min(derivative)
        
        return features
    
    def extract_frequency_features(self, ecg_signal):
        """
        Извлечение частотных признаков с помощью БПФ.
        
        Args:
            ecg_signal (array): Массив с отсчетами ЭКГ сигнала
            
        Returns:
            dict: Словарь с частотными признаками
        """
        # Вычисление БПФ
        fft_vals = fft.fft(ecg_signal)
        fft_freq = fft.fftfreq(len(ecg_signal), 1.0/self.sampling_rate)
        
        # Убираем отрицательные частоты
        positive_freq_idx = np.where(fft_freq > 0)
        fft_vals = fft_vals[positive_freq_idx]
        fft_freq = fft_freq[positive_freq_idx]
        
        # Вычисляем спектральную плотность мощности
        psd = np.abs(fft_vals)**2
        
        features = {}
        
        # Основные частотные признаки
        features['dominant_frequency'] = fft_freq[np.argmax(psd)]
        features['spectral_energy'] = np.sum(psd)
        features['spectral_entropy'] = stats.entropy(psd + 1e-8)  # Добавляем маленькое значение для стабильности
        
        # Спектральные моменты
        features['spectral_centroid'] = np.sum(fft_freq * psd) / np.sum(psd)
        features['spectral_bandwidth'] = np.sqrt(np.sum((fft_freq - features['spectral_centroid'])**2 * psd) / np.sum(psd))
        
        # Отношения в частотных bandах
        low_freq_mask = (fft_freq >= 0.5) & (fft_freq <= 5)
        mid_freq_mask = (fft_freq > 5) & (fft_freq <= 15)
        high_freq_mask = (fft_freq > 15) & (fft_freq <= 40)
        
        features['low_freq_power'] = np.sum(psd[low_freq_mask])
        features['mid_freq_power'] = np.sum(psd[mid_freq_mask])
        features['high_freq_power'] = np.sum(psd[high_freq_mask])
        features['lf_hf_ratio'] = features['low_freq_power'] / (features['high_freq_power'] + 1e-8)
        
        return features
    
    def extract_wavelet_features(self, ecg_signal):
        """
        Извлечение признаков на основе вейвлет-преобразования.
        Упрощенная версия для демонстрации.
        
        Args:
            ecg_signal (array): Массив с отсчетами ЭКГ сигнала
            
        Returns:
            dict: Словарь с вейвлет-признаками
        """
        # Простая имитация вейвлет-признаков
        # В реальном проекте использовать pywt
        
        # Разделяем сигнал на сегменты и вычисляем статистики
        segment_length = len(ecg_signal) // 4
        features = {}
        
        for i in range(4):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            segment = ecg_signal[start_idx:end_idx]
            
            features[f'segment_{i}_mean'] = np.mean(segment)
            features[f'segment_{i}_std'] = np.std(segment)
            features[f'segment_{i}_energy'] = np.sum(segment**2)
        
        return features
    
    def extract_all_features(self, ecg_signal):
        """
        Извлечение всех типов признаков из ЭКГ сигнала.
        
        Args:
            ecg_signal (array): Массив с отсчетами ЭКГ сигнала
            
        Returns:
            array: Вектор всех признаков
        """
        temporal_features = self.extract_temporal_features(ecg_signal)
        frequency_features = self.extract_frequency_features(ecg_signal)
        wavelet_features = self.extract_wavelet_features(ecg_signal)
        
        # Объединяем все признаки в один вектор
        all_features = {**temporal_features, **frequency_features, **wavelet_features}
        feature_vector = np.array(list(all_features.values()))
        
        return feature_vector
    
    def create_feature_dataset(self, X):
        """
        Создание датасета признаков из исходных сигналов.
        
        Args:
            X (array): Массив сигналов формы (n_samples, sequence_length)
            
        Returns:
            array: Массив признаков формы (n_samples, n_features)
        """
        n_samples = X.shape[0]
        feature_vectors = []
        
        for i in range(n_samples):
            ecg_signal = X[i]
            feature_vector = self.extract_all_features(ecg_signal)
            feature_vectors.append(feature_vector)
        
        return np.array(feature_vectors)