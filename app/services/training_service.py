"""
Сервис для обучения моделей машинного обучения.
Автор: Чайкин Виталий Федорович
Тема ВКР: Разработка рекомендательной системы на основе обработки биомедицинских данных
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from datetime import datetime
import json
import os

class TrainingService:
    """Сервис для обучения и настройки моделей."""
    
    def __init__(self, model_loader, feature_engineer):
        self.model_loader = model_loader
        self.feature_engineer = feature_engineer
        self.training_history = {}
        
    def train_random_forest(self, X_train, y_train, X_val=None, y_val=None, **params):
        """
        Обучение модели Random Forest.
        
        Args:
            X_train: Обучающие данные
            y_train: Обучающие метки
            X_val: Валидационные данные (опционально)
            y_val: Валидационные метки (опционально)
            **params: Параметры модели
            
        Returns:
            tuple: (обученная модель, история обучения)
        """
        print("Начало обучения Random Forest...")
        
        # Создание модели
        rf_model = self.model_loader.create_random_forest(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10)
        )
        
        # Обучение модели
        start_time = datetime.now()
        rf_model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Оценка на валидационной выборке
        val_accuracy = None
        if X_val is not None and y_val is not None:
            val_accuracy = rf_model.score(X_val, y_val)
            print(f"Accuracy на валидационной выборке: {val_accuracy:.4f}")
        
        # Сохранение истории обучения
        history = {
            'model_type': 'random_forest',
            'training_time': training_time,
            'train_accuracy': rf_model.score(X_train, y_train),
            'val_accuracy': val_accuracy,
            'parameters': params
        }
        
        print(f"Обучение Random Forest завершено за {training_time:.2f} секунд")
        
        return rf_model, history
    
    def train_cnn_model(self, X_train, y_train, X_val, y_val, **params):
        """
        Обучение CNN модели.
        
        Args:
            X_train: Обучающие данные
            y_train: Обучающие метки
            X_val: Валидационные данные
            y_val: Валидационные метки
            **params: Параметры модели
            
        Returns:
            tuple: (обученная модель, история обучения)
        """
        print("Начало обучения CNN модели...")
        
        # Подготовка данных для CNN
        if len(X_train.shape) == 2:
            X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        else:
            X_train_cnn = X_train
            X_val_cnn = X_val
        
        num_classes = len(np.unique(y_train))
        
        # Создание модели
        cnn_model = self.model_loader.create_cnn_model(
            input_shape=X_train_cnn.shape[1:],
            num_classes=num_classes
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Обучение модели
        start_time = datetime.now()
        history = cnn_model.fit(
            X_train_cnn, y_train,
            validation_data=(X_val_cnn, y_val),
            epochs=params.get('epochs', 50),
            batch_size=params.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Сохранение истории обучения
        training_history = {
            'model_type': 'cnn',
            'training_time': training_time,
            'final_epochs': len(history.history['loss']),
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'parameters': params,
            'full_history': history.history
        }
        
        print(f"Обучение CNN завершено за {training_time:.2f} секунд")
        print(f"Финальная точность: {training_history['final_val_accuracy']:.4f}")
        
        return cnn_model, training_history
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='random_forest'):
        """
        Подбор гиперпараметров для модели.
        
        Args:
            X_train: Обучающие данные
            y_train: Обучающие метки
            model_type (str): Тип модели
            
        Returns:
            tuple: (лучшая модель, лучшие параметры)
        """
        print(f"Подбор гиперпараметров для {model_type}...")
        
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
            
            model = self.model_loader.create_random_forest()
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Лучшие параметры: {grid_search.best_params_}")
            print(f"Лучшая точность: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_, grid_search.best_params_
        
        else:
            print("Подбор гиперпараметров для CNN пока не реализован")
            return None, None
    
    def evaluate_model_comprehensive(self, model, X_test, y_test, model_type='sklearn'):
        """
        Комплексная оценка модели.
        
        Args:
            model: Обученная модель
            X_test: Тестовые данные
            y_test: Тестовые метки
            model_type (str): Тип модели
            
        Returns:
            dict: Результаты оценки
        """
        print("Проведение комплексной оценки модели...")
        
        # Базовые метрики
        metrics = self.model_loader.evaluate_model(model, X_test, y_test, model_type)
        
        # Дополнительные метрики
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, fscore, support = precision_recall_fscore_support(
            y_test, metrics['predictions'], average='weighted'
        )
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1_score': fscore,
            'classification_report': classification_report(y_test, metrics['predictions']),
            'confusion_matrix': confusion_matrix(y_test, metrics['predictions'])
        })
        
        return metrics
    
    def plot_training_history(self, history, save_path=None):
        """
        Визуализация истории обучения.
        
        Args:
            history: История обучения
            save_path (str): Путь для сохранения графика
        """
        if history['model_type'] == 'cnn':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # График точности
            ax1.plot(history['full_history']['accuracy'], label='Обучающая')
            ax1.plot(history['full_history']['val_accuracy'], label='Валидационная')
            ax1.set_title('Точность модели')
            ax1.set_xlabel('Эпоха')
            ax1.set_ylabel('Точность')
            ax1.legend()
            ax1.grid(True)
            
            # График потерь
            ax2.plot(history['full_history']['loss'], label='Обучающая')
            ax2.plot(history['full_history']['val_loss'], label='Валидационная')
            ax2.set_title('Функция потерь')
            ax2.set_xlabel('Эпоха')
            ax2.set_ylabel('Потери')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()