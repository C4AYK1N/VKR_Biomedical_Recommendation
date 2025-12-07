"""
Модуль для загрузки и применения обученных моделей.
Автор: Чайкин Виталий Федорович
Тема ВКР: Разработка рекомендательной системы на основе обработки биомедицинских данных
"""

import pickle
import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class ModelLoader:
    """Класс для работы с моделями машинного обучения."""
    
    def __init__(self):
        self.models = {}
        
    def create_random_forest(self, n_estimators=100, max_depth=10):
        """
        Создание модели Random Forest.
        
        Args:
            n_estimators (int): Количество деревьев
            max_depth (int): Максимальная глубина деревьев
            
        Returns:
            RandomForestClassifier: Созданная модель
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        return model
    
    def create_cnn_model(self, input_shape, num_classes):
        """
        Создание CNN модели для обработки ЭКГ сигналов.
        
        Args:
            input_shape (tuple): Форма входных данных
            num_classes (int): Количество классов
            
        Returns:
            tf.keras.Model: Созданная модель
        """
        model = tf.keras.Sequential([
            # Первый сверточный блок
            tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', 
                                  input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Dropout(0.3),
            
            # Второй сверточный блок
            tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Dropout(0.3),
            
            # Третий сверточный блок
            tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.4),
            
            # Полносвязные слои
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            
            # Выходной слой
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Компиляция модели
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def save_model(self, model, filepath):
        """
        Сохранение модели в файл.
        
        Args:
            model: Модель для сохранения
            filepath (str): Путь для сохранения
        """
        if isinstance(model, tf.keras.Model):
            model.save(filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
    
    def load_model(self, filepath):
        """
        Загрузка модели из файла.
        
        Args:
            filepath (str): Путь к файлу модели
            
        Returns:
            Загруженная модель
        """
        if filepath.endswith('.h5'):
            return tf.keras.models.load_model(filepath)
        else:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    
    def evaluate_model(self, model, X_test, y_test, model_type='sklearn'):
        """
        Оценка производительности модели.
        
        Args:
            model: Обученная модель
            X_test: Тестовые данные
            y_test: Тестовые метки
            model_type (str): Тип модели ('sklearn' или 'keras')
            
        Returns:
            dict: Метрики оценки
        """
        if model_type == 'sklearn':
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        else:
            y_pred_proba = model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_test
        }
        
        return metrics