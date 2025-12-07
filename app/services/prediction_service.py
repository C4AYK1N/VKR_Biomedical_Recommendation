"""
Сервис для выполнения прогнозов с использованием обученных моделей.
Автор: Чайкин Виталий Федорович
Тема ВКР: Разработка рекомендательной системы на основе обработки биомедицинских данных
"""

import numpy as np
import pandas as pd
from utils.config import Config

class PredictionService:
    """Сервис для выполнения прогнозов."""
    
    def __init__(self, model_loader, feature_engineer, data_loader):
        self.model_loader = model_loader
        self.feature_engineer = feature_engineer
        self.data_loader = data_loader
        self.config = Config()
        self.loaded_models = {}
    
    def load_model(self, model_path, model_name):
        """
        Загрузка модели из файла.
        
        Args:
            model_path (str): Путь к файлу модели
            model_name (str): Имя модели
            
        Returns:
            bool: Успешность загрузки
        """
        try:
            model = self.model_loader.load_model(model_path)
            self.loaded_models[model_name] = {
                'model': model,
                'path': model_path,
                'type': 'keras' if model_path.endswith('.h5') else 'sklearn'
            }
            print(f"Модель {model_name} успешно загружена")
            return True
        except Exception as e:
            print(f"Ошибка при загрузке модели {model_name}: {str(e)}")
            return False
    
    def predict_single_ecg(self, ecg_signal, model_name, use_features=True):
        """
        Прогнозирование для одного сигнала ЭКГ.
        
        Args:
            ecg_signal (array): Сигнал ЭКГ
            model_name (str): Имя модели для использования
            use_features (bool): Использовать извлеченные признаки или сырой сигнал
            
        Returns:
            dict: Результаты прогнозирования
        """
        if model_name not in self.loaded_models:
            return {"error": f"Модель {model_name} не загружена"}
        
        model_info = self.loaded_models[model_name]
        model = model_info['model']
        model_type = model_info['type']
        
        try:
            # Предобработка сигнала
            processed_signal = self.data_loader.preprocess_ecg_signal(ecg_signal)
            
            if use_features and model_type == 'sklearn':
                # Извлечение признаков для sklearn моделей
                features = self.feature_engineer.extract_all_features(processed_signal)
                features = features.reshape(1, -1)
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
            else:
                # Использование сырого сигнала для CNN
                if model_type == 'keras':
                    signal_reshaped = processed_signal.reshape(1, len(processed_signal), 1)
                    probabilities = model.predict(signal_reshaped)[0]
                    prediction = np.argmax(probabilities)
                else:
                    signal_reshaped = processed_signal.reshape(1, -1)
                    prediction = model.predict(signal_reshaped)[0]
                    probabilities = model.predict_proba(signal_reshaped)[0]
            
            # Интерпретация результатов
            arrhythmia_type = self.config.ARRHYTHMIA_CLASSES.get(int(prediction), "Неизвестный тип")
            confidence = np.max(probabilities)
            
            # Формирование рекомендации
            recommendation = self._generate_recommendation(prediction, confidence)
            
            result = {
                'prediction': int(prediction),
                'arrhythmia_type': arrhythmia_type,
                'confidence': float(confidence),
                'probabilities': {
                    self.config.ARRHYTHMIA_CLASSES[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                },
                'recommendation': recommendation,
                'success': True
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Ошибка при прогнозировании: {str(e)}", "success": False}
    
    def predict_batch(self, ecg_signals, model_name, use_features=True):
        """
        Пакетное прогнозирование для нескольких сигналов.
        
        Args:
            ecg_signals (array): Массив сигналов ЭКГ
            model_name (str): Имя модели для использования
            use_features (bool): Использовать извлеченные признаки
            
        Returns:
            list: Список результатов прогнозирования
        """
        results = []
        
        for i, signal in enumerate(ecg_signals):
            result = self.predict_single_ecg(signal, model_name, use_features)
            result['signal_index'] = i
            results.append(result)
        
        return results
    
    def _generate_recommendation(self, prediction, confidence):
        """
        Генерация рекомендации на основе прогноза.
        
        Args:
            prediction (int): Предсказанный класс
            confidence (float): Уверенность модели
            
        Returns:
            str: Текст рекомендации
        """
        recommendations = {
            0: "Ритм в норме. Рекомендуется плановое наблюдение.",
            1: "Обнаружены признаки апноэ. Рекомендуется консультация сомнолога.",
            2: "Выявлена фибрилляция предсердий. Необходима срочная консультация кардиолога.",
            3: "Сигнал содержит значительный шум. Рекомендуется повторное измерение.",
            4: "Обнаружена другая аритмия. Требуется дополнительная диагностика."
        }
        
        base_recommendation = recommendations.get(prediction, "Требуется консультация специалиста.")
        
        if confidence < 0.7:
            base_recommendation += " Результат требует дополнительной проверки из-за низкой уверенности модели."
        
        return base_recommendation
    
    def compare_models(self, ecg_signal, model_names):
        """
        Сравнение прогнозов от разных моделей.
        
        Args:
            ecg_signal (array): Сигнал ЭКГ
            model_names (list): Список имен моделей для сравнения
            
        Returns:
            dict: Сравнительные результаты
        """
        results = {}
        
        for model_name in model_names:
            if model_name in self.loaded_models:
                result = self.predict_single_ecg(ecg_signal, model_name)
                results[model_name] = result
        
        return results
    
    def get_model_info(self, model_name):
        """
        Получение информации о загруженной модели.
        
        Args:
            model_name (str): Имя модели
            
        Returns:
            dict: Информация о модели
        """
        if model_name in self.loaded_models:
            model_info = self.loaded_models[model_name]
            return {
                'name': model_name,
                'type': model_info['type'],
                'path': model_info['path'],
                'loaded': True
            }
        else:
            return {'name': model_name, 'loaded': False}