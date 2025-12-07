# ФИО: Чайкин Виталий Федорович
# Тема ВКР: Разработка рекомендательной системы на основе обработки биомедицинских данных

"""
Главный запускаемый файл приложения.
Запуск: streamlit run main.py

Описание: Полная версия приложения для анализа ЭКГ сигналов 
с возможностью обучения моделей, прогнозирования и сравнения результатов.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import sys
import warnings
import json
from datetime import datetime
import urllib.request
import zipfile
from io import BytesIO

# Отключаем предупреждения
warnings.filterwarnings('ignore')

# Создаем папки если их нет
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Настройка страницы Streamlit
st.set_page_config(
    page_title="Рекомендательная система для анализа ЭКГ",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("❤️ Рекомендательная система для анализа биомедицинских данных")
st.markdown("---")

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

class Config:
    """Класс конфигурации приложения."""
    
    # Параметры данных
    SAMPLING_RATE = 360
    SEQUENCE_LENGTH = 360
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    
    # Классы аритмий
    ARRHYTHMIA_CLASSES = {
        0: "Нормальный ритм",
        1: "Апноэ", 
        2: "Фибрилляция предсердий",
        3: "Шум",
        4: "Другая аритмия"
    }
    
    # Цвета для визуализации
    COLORS = {
        'normal': '#2ecc71',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'info': '#3498db',
        'secondary': '#95a5a6'
    }

config = Config()

# ============================================================================
# СЕРВИСЫ И УТИЛИТЫ
# ============================================================================

class DataLoader:
    """Класс для загрузки и обработки данных."""
    
    def __init__(self):
        self.config = config
        
    def load_simulated_data(self, n_samples=5000):
        """
        Загрузка симулированных данных ЭКГ.
        
        Args:
            n_samples (int): Количество образцов
            
        Returns:
            tuple: (X, y) - признаки и метки
        """
        np.random.seed(self.config.RANDOM_STATE)
        X = np.zeros((n_samples, self.config.SEQUENCE_LENGTH))
        y = np.zeros(n_samples)
        
        st.info(f"Генерация {n_samples} симулированных записей ЭКГ...")
        progress_bar = st.progress(0)
        
        for i in range(n_samples):
            t = np.linspace(0, 1, self.config.SEQUENCE_LENGTH)
            base_ecg = 0.5 * np.sin(2 * np.pi * 1 * t)
            arrhythmia_type = i % 5
            y[i] = arrhythmia_type
            
            # Генерация сигналов для разных типов аритмий
            if arrhythmia_type == 0:  # Нормальный ритм
                X[i] = base_ecg + 0.1 * np.random.normal(size=self.config.SEQUENCE_LENGTH)
            elif arrhythmia_type == 1:  # Апноэ
                X[i] = base_ecg * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)) + 0.1 * np.random.normal(size=self.config.SEQUENCE_LENGTH)
            elif arrhythmia_type == 2:  # Фибрилляция предсердий
                X[i] = base_ecg + 0.3 * np.random.normal(size=self.config.SEQUENCE_LENGTH) + 0.1 * np.random.normal(size=self.config.SEQUENCE_LENGTH)
            elif arrhythmia_type == 3:  # Шум
                X[i] = 0.8 * np.random.normal(size=self.config.SEQUENCE_LENGTH)
            else:  # Другая аритмия
                X[i] = base_ecg * (1 + 0.3 * np.sin(2 * np.pi * 2 * t)) + 0.1 * np.random.normal(size=self.config.SEQUENCE_LENGTH)
            
            # Обновление прогресса
            if i % 500 == 0:
                progress_bar.progress((i + 1) / n_samples)
        
        progress_bar.progress(1.0)
        return X, y
    
    def download_kaggle_dataset(self):
        """
        Загрузка реальных данных ЭКГ с Kaggle.
        Для демонстрации используем симулированные данные.
        """
        try:
            # В реальном проекте здесь был бы код для загрузки с Kaggle
            # Для демонстрации возвращаем симулированные данные
            st.warning("В демо-версии используются симулированные данные. В реальном проекте будет загрузка с Kaggle.")
            return self.load_simulated_data(3000)
        except Exception as e:
            st.error(f"Ошибка при загрузке данных с Kaggle: {e}")
            return self.load_simulated_data(2000)
    
    def preprocess_signal(self, signal):
        """
        Предобработка сигнала ЭКГ.
        
        Args:
            signal (array): Исходный сигнал
            
        Returns:
            array: Обработанный сигнал
        """
        # Нормализация
        signal_normalized = (signal - np.mean(signal)) / np.std(signal)
        
        # Базовая фильтрация (в реальном проекте более сложная)
        from scipy import signal as scipy_signal
        b, a = scipy_signal.butter(3, 0.05)
        signal_filtered = scipy_signal.filtfilt(b, a, signal_normalized)
        
        return signal_filtered
    
    def split_data(self, X, y):
        """
        Разделение данных на обучающую, валидационную и тестовую выборки.
        
        Args:
            X (array): Признаки
            y (array): Метки
            
        Returns:
            tuple: Разделенные данные
        """
        # Первое разделение: тестовая выборка
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        # Второе разделение: обучающая и валидационная
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.config.VALIDATION_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y_train_val
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

class ModelService:
    """Сервис для работы с моделями машинного обучения."""
    
    def __init__(self):
        self.config = config
        
    def create_random_forest(self, n_estimators=100, max_depth=10):
        """
        Создание модели Random Forest.
        
        Args:
            n_estimators (int): Количество деревьев
            max_depth (int): Максимальная глубина
            
        Returns:
            RandomForestClassifier: Модель Random Forest
        """
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced'
        )
    
    def create_cnn_model(self, input_shape):
        """
        Создание CNN модели для классификации ЭКГ.
        
        Args:
            input_shape (tuple): Форма входных данных
            
        Returns:
            keras.Model: CNN модель
        """
        model = keras.Sequential([
            # Первый сверточный блок
            keras.layers.Conv1D(32, kernel_size=5, activation='relu', 
                              input_shape=input_shape,
                              padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Dropout(0.3),
            
            # Второй сверточный блок
            keras.layers.Conv1D(64, kernel_size=3, activation='relu',
                              padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Dropout(0.3),
            
            # Третий сверточный блок
            keras.layers.Conv1D(128, kernel_size=3, activation='relu',
                              padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dropout(0.4),
            
            # Полносвязные слои
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            
            # Выходной слой
            keras.layers.Dense(len(self.config.ARRHYTHMIA_CLASSES), 
                             activation='softmax')
        ])
        
        # Компиляция модели
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def tune_random_forest(self, X_train, y_train):
        """
        Подбор гиперпараметров для Random Forest.
        
        Args:
            X_train (array): Обучающие данные
            y_train (array): Обучающие метки
            
        Returns:
            tuple: (лучшая модель, лучшие параметры)
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        base_model = RandomForestClassifier(
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def evaluate_model(self, model, X_test, y_test, model_type='sklearn'):
        """
        Комплексная оценка модели.
        
        Args:
            model: Обученная модель
            X_test (array): Тестовые данные
            y_test (array): Тестовые метки
            model_type (str): Тип модели ('sklearn' или 'keras')
            
        Returns:
            dict: Метрики оценки
        """
        if model_type == 'keras':
            # Для Keras моделей
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            # Для sklearn моделей
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            else:
                y_pred_proba = None
        
        # Базовые метрики
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        
        # Отчет о классификации
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # ROC AUC (если есть вероятности)
        roc_auc = None
        if y_pred_proba is not None:
            try:
                y_test_bin = label_binarize(y_test, 
                                          classes=range(len(self.config.ARRHYTHMIA_CLASSES)))
                roc_auc = roc_auc_score(y_test_bin, y_pred_proba, 
                                       average='weighted', multi_class='ovr')
            except:
                pass
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'true_labels': y_test
        }

class VisualizationService:
    """Сервис для визуализации данных и результатов."""
    
    def __init__(self):
        self.config = config
        
    def plot_ecg_signals(self, signals, titles=None, figsize=(15, 8)):
        """
        Визуализация нескольких сигналов ЭКГ.
        
        Args:
            signals (list): Список сигналов
            titles (list): Список заголовков
            figsize (tuple): Размер фигуры
            
        Returns:
            matplotlib.figure.Figure: Фигура с графиками
        """
        n_signals = len(signals)
        fig, axes = plt.subplots(n_signals, 1, figsize=figsize)
        
        if n_signals == 1:
            axes = [axes]
        
        for i, (signal, ax) in enumerate(zip(signals, axes)):
            ax.plot(signal, linewidth=1.5)
            ax.set_xlabel('Отсчеты', fontsize=10)
            ax.set_ylabel('Амплитуда', fontsize=10)
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, len(signal))
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, cm, normalize=False):
        """
        Визуализация матрицы ошибок.
        
        Args:
            cm (array): Матрица ошибок
            normalize (bool): Нормализовать ли матрицу
            
        Returns:
            matplotlib.figure.Figure: Фигура с матрицей ошибок
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=list(self.config.ARRHYTHMIA_CLASSES.values()),
                   yticklabels=list(self.config.ARRHYTHMIA_CLASSES.values()),
                   ax=ax)
        
        ax.set_xlabel('Предсказанные метки', fontsize=12)
        ax.set_ylabel('Истинные метки', fontsize=12)
        title = 'Нормализованная матрица ошибок' if normalize else 'Матрица ошибок'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_comparison(self, metrics_dict, figsize=(12, 6)):
        """
        Визуализация сравнения метрик нескольких моделей.
        
        Args:
            metrics_dict (dict): Словарь с метриками моделей
            figsize (tuple): Размер фигуры
            
        Returns:
            matplotlib.figure.Figure: Фигура с графиком сравнения
        """
        models = list(metrics_dict.keys())
        metrics_names = ['Точность', 'Precision', 'Recall', 'F1-Score']
        
        # Подготовка данных
        data = {
            'Точность': [metrics_dict[m]['accuracy'] for m in models],
            'Precision': [metrics_dict[m]['precision'] for m in models],
            'Recall': [metrics_dict[m]['recall'] for m in models],
            'F1-Score': [metrics_dict[m]['f1_score'] for m in models]
        }
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, (metric_name, values) in enumerate(data.items()):
            offset = width * i - width * (len(metrics_names) - 1) / 2
            bars = ax.bar(x + offset, values, width, label=metric_name)
            
            # Добавляем значения на столбцы
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Модели', fontsize=12)
        ax.set_ylabel('Значение метрики', fontsize=12)
        ax.set_title('Сравнение метрик моделей', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_training_history(self, history, figsize=(12, 4)):
        """
        Визуализация истории обучения нейронной сети.
        
        Args:
            history: История обучения Keras модели
            figsize (tuple): Размер фигуры
            
        Returns:
            matplotlib.figure.Figure: Фигура с графиками обучения
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # График точности
        ax1.plot(history.history['accuracy'], label='Обучающая', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Валидационная', linewidth=2)
        ax1.set_title('Точность модели', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Эпоха', fontsize=10)
        ax1.set_ylabel('Точность', fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График потерь
        ax2.plot(history.history['loss'], label='Обучающая', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Валидационная', linewidth=2)
        ax2.set_title('Функция потерь', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Эпоха', fontsize=10)
        ax2.set_ylabel('Потери', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ
# ============================================================================

@st.cache_resource
def initialize_services():
    """Инициализация сервисов с кэшированием."""
    return {
        'data_loader': DataLoader(),
        'model_service': ModelService(),
        'viz_service': VisualizationService()
    }

services = initialize_services()
data_loader = services['data_loader']
model_service = services['model_service']
viz_service = services['viz_service']

# ============================================================================
# ФУНКЦИИ ОТРИСОВКИ СТРАНИЦ
# ============================================================================

def render_home_page():
    """Главная страница."""
    st.header("🏠 Добро пожаловать в систему анализа ЭКГ")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Цель проекта")
        st.markdown("""
        Разработка интеллектуальной системы для:
        
        🔍 **Анализа электрокардиограмм (ЭКГ)**
        🤖 **Автоматической классификации аритмий**
        💡 **Генерации медицинских рекомендаций**
        📊 **Визуализации результатов диагностики**
        
        **Ключевые преимущества:**
        - Повышение точности диагностики
        - Сокращение времени анализа
        - Поддержка принятия врачебных решений
        - Образовательный инструмент для студентов
        """)
    
    with col2:
        st.subheader("⚙️ Технологический стек")
        st.markdown("""
        **Основные технологии:**
        - **Python 3.9+** - основной язык разработки
        - **Streamlit** - веб-интерфейс
        - **Scikit-learn** - классические ML алгоритмы
        - **TensorFlow/Keras** - нейронные сети
        - **Pandas/NumPy** - обработка данных
        - **Matplotlib/Seaborn** - визуализация
        
        **Архитектура:**
        ```
        ┌─────────────────┐
        │  Веб-интерфейс  │ ← Streamlit
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  Бизнес-логика  │ ← Python модули
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │    Модели ML    │ ← Scikit-learn/TensorFlow
        └─────────────────┘
        ```
        """)
    
    st.markdown("---")
    
    st.subheader("🚀 Быстрый старт")
    
    with st.expander("📋 Пошаговая инструкция", expanded=True):
        st.markdown("""
        1. **📊 Загрузка данных** - перейдите на вкладку "Загрузка данных" и загрузите датасет
        2. **🤖 Обучение моделей** - настройте параметры и обучите модели на вкладке "Обучение моделей"
        3. **🔍 Прогнозирование** - протестируйте обученные модели на новых данных
        4. **📈 Сравнение моделей** - оцените и сравните производительность моделей
        5. **💾 Сохранение результатов** - экспортируйте результаты для дальнейшего использования
        """)
    
    # Статистика системы
    st.subheader("📊 Статистика системы")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Поддерживаемые форматы", "3", "CSV, TXT, NPY")
    
    with col2:
        st.metric("Типы моделей", "2", "RF + CNN")
    
    with col3:
        st.metric("Классы аритмий", "5", "Полная классификация")
    
    with col4:
        st.metric("Минимальная точность", "85%", "Требование ГОСТ")

def render_data_loading_page():
    """Страница загрузки данных."""
    st.header("📊 Загрузка и анализ данных")
    
    # Выбор источника данных
    data_source = st.radio(
        "Выберите источник данных:",
        ["Симулированные данные", "Kaggle датасет", "Загрузить файл"],
        horizontal=True
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if data_source == "Симулированные данные":
            st.subheader("Параметры генерации данных")
            
            n_samples = st.slider(
                "Количество образцов:",
                min_value=1000,
                max_value=10000,
                value=5000,
                step=1000,
                help="Общее количество записей ЭКГ для генерации"
            )
            
            noise_level = st.slider(
                "Уровень шума:",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="Интенсивность шума в сигналах"
            )
            
            if st.button("🚀 Сгенерировать данные", type="primary", use_container_width=True):
                with st.spinner("Генерация данных..."):
                    # Здесь будет вызов функции генерации с noise_level
                    X, y = data_loader.load_simulated_data(n_samples)
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.data_source = "simulated"
                    st.success(f"✅ Сгенерировано {n_samples} записей ЭКГ!")
        
        elif data_source == "Kaggle датасет":
            st.subheader("Загрузка с Kaggle")
            
            dataset_options = {
                "MIT-BIH Arrhythmia": "mit-bih-arrhythmia",
                "PTB Diagnostic ECG": "ptb-diagnostic-ecg",
                "Честный датасет": "simulated"  # Для демо
            }
            
            selected_dataset = st.selectbox(
                "Выберите датасет:",
                list(dataset_options.keys())
            )
            
            if st.button("🌐 Загрузить с Kaggle", type="primary", use_container_width=True):
                with st.spinner(f"Загрузка датасета {selected_dataset}..."):
                    X, y = data_loader.download_kaggle_dataset()
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.data_source = "kaggle"
                    st.success(f"✅ Загружено {len(X)} записей из {selected_dataset}!")
        
        else:  # Загрузить файл
            st.subheader("Загрузка файла")
            
            uploaded_file = st.file_uploader(
                "Выберите файл с данными ЭКГ:",
                type=['csv', 'txt', 'npy', 'pkl'],
                help="Поддерживаемые форматы: CSV, TXT, NPY, PKL"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        data = pd.read_csv(uploaded_file)
                        st.success(f"✅ Файл {uploaded_file.name} успешно загружен!")
                        
                        # Показываем предпросмотр
                        st.subheader("Предпросмотр данных")
                        st.dataframe(data.head(10), use_container_width=True)
                        
                        # Предлагаем выбрать столбцы
                        if st.button("📥 Использовать данные", type="primary"):
                            # Здесь будет обработка CSV файла
                            st.info("Обработка CSV файла...")
                            
                    elif uploaded_file.name.endswith('.npy'):
                        data = np.load(BytesIO(uploaded_file.read()))
                        st.success(f"✅ NPY файл загружен! Форма: {data.shape}")
                        
                except Exception as e:
                    st.error(f"❌ Ошибка при обработке файла: {str(e)}")
    
    # Отображение статистики если данные загружены
    if 'X' in st.session_state and 'y' in st.session_state:
        X = st.session_state.X
        y = st.session_state.y
        
        with col2:
            st.subheader("📈 Статистика данных")
            
            stats_data = {
                "Всего записей": len(X),
                "Длина сигнала": f"{X.shape[1]} отсчетов",
                "Частота дискретизации": f"{config.SAMPLING_RATE} Гц",
                "Количество классов": len(np.unique(y)),
                "Баланс классов": "Сбалансированный" if len(np.unique(y)) > 1 else "Один класс"
            }
            
            for key, value in stats_data.items():
                st.metric(key, value)
        
        # Визуализация распределения классов
        st.subheader("📊 Распределение классов")
        
        class_counts = pd.Series(y).value_counts().sort_index()
        class_names = [config.ARRHYTHMIA_CLASSES[i] for i in class_counts.index]
        
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        bars = ax1.bar(class_names, class_counts.values, 
                      color=[config.COLORS['normal'], config.COLORS['warning'],
                            config.COLORS['danger'], config.COLORS['info'],
                            config.COLORS['secondary']])
        
        ax1.set_xlabel("Класс аритмии", fontsize=12)
        ax1.set_ylabel("Количество записей", fontsize=12)
        ax1.set_title("Распределение записей по классам", fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Добавляем значения на столбцы
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        st.pyplot(fig1)
        
        # Примеры сигналов
        st.subheader("📈 Примеры сигналов по классам")
        
        selected_class = st.selectbox(
            "Выберите класс для просмотра примеров:",
            list(config.ARRHYTHMIA_CLASSES.items()),
            format_func=lambda x: x[1]
        )
        
        if selected_class:
            class_id, class_name = selected_class
            class_indices = np.where(y == class_id)[0]
            
            if len(class_indices) > 0:
                # Выбираем до 3 примеров
                n_examples = min(3, len(class_indices))
                example_indices = class_indices[:n_examples]
                example_signals = [X[i] for i in example_indices]
                example_titles = [f"Пример {i+1}: {class_name}" for i in range(n_examples)]
                
                fig2 = viz_service.plot_ecg_signals(example_signals, example_titles)
                st.pyplot(fig2)
            
            # Кнопка для разделения данных
            if st.button("🎯 Разделить данные на выборки", type="primary"):
                with st.spinner("Разделение данных..."):
                    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_val = X_val
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_val = y_val
                    st.session_state.y_test = y_test
                    
                    st.success("✅ Данные успешно разделены!")
                    
                    # Показываем статистику разделения
                    split_stats = pd.DataFrame({
                        'Выборка': ['Обучающая', 'Валидационная', 'Тестовая'],
                        'Количество записей': [len(X_train), len(X_val), len(X_test)],
                        'Доля': [f"{len(X_train)/len(X)*100:.1f}%",
                                f"{len(X_val)/len(X)*100:.1f}%",
                                f"{len(X_test)/len(X)*100:.1f}%"]
                    })
                    
                    st.dataframe(split_stats, use_container_width=True)

def render_model_training_page():
    """Страница обучения моделей."""
    st.header("🤖 Обучение моделей")
    
    # Проверка наличия данных
    if 'X_train' not in st.session_state:
        st.warning("⚠️ Сначала загрузите и разделите данные на вкладке 'Загрузка данных'")
        return
    
    # Выбор модели для обучения
    model_choice = st.radio(
        "Выберите модель для обучения:",
        ["Random Forest", "CNN (нейронная сеть)", "Обе модели"],
        horizontal=True
    )
    
    # Получаем данные из session state
    X_train = st.session_state.X_train
    X_val = st.session_state.X_val
    y_train = st.session_state.y_train
    y_val = st.session_state.y_val
    
    if model_choice in ["Random Forest", "Обе модели"]:
        st.subheader("🌲 Random Forest")
        
        with st.expander("⚙️ Параметры Random Forest", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_estimators = st.slider(
                    "Количество деревьев:",
                    min_value=50,
                    max_value=500,
                    value=100,
                    step=50,
                    help="Чем больше деревьев, тем точнее модель, но дольше обучение"
                )
            
            with col2:
                max_depth = st.slider(
                    "Максимальная глубина:",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                    help="Ограничивает глубину каждого дерева"
                )
            
            with col3:
                use_tuning = st.checkbox(
                    "Использовать подбор параметров",
                    value=False,
                    help="Автоматический подбор гиперпараметров (GridSearch)"
                )
        
        if st.button("🌲 Обучить Random Forest", type="primary", key="train_rf"):
            with st.spinner("Обучение Random Forest..."):
                try:
                    start_time = datetime.now()
                    
                    if use_tuning:
                        # Подбор гиперпараметров
                        st.info("🔍 Выполняется подбор гиперпараметров...")
                        best_model, best_params = model_service.tune_random_forest(X_train, y_train)
                        st.success(f"✅ Найденные параметры: {best_params}")
                    else:
                        # Обучение с заданными параметрами
                        rf_model = model_service.create_random_forest(
                            n_estimators=n_estimators,
                            max_depth=max_depth
                        )
                        rf_model.fit(X_train, y_train)
                        best_model = rf_model
                    
                    # Оценка модели
                    metrics = model_service.evaluate_model(best_model, X_val, y_val)
                    
                    # Сохранение модели
                    model_filename = f"models/rf_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    with open(model_filename, 'wb') as f:
                        pickle.dump(best_model, f)
                    
                    # Сохранение метрик
                    training_time = (datetime.now() - start_time).total_seconds()
                    
                    st.session_state.rf_model = best_model
                    st.session_state.rf_metrics = metrics
                    st.session_state.rf_training_time = training_time
                    st.session_state.rf_model_path = model_filename
                    
                    st.success(f"✅ Random Forest обучен за {training_time:.2f} секунд!")
                    
                    # Отображение результатов
                    st.subheader("📊 Результаты Random Forest")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Точность", f"{metrics['accuracy']:.4f}")
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.4f}")
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.4f}")
                    with col4:
                        st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
                    
                    # Матрица ошибок
                    fig_cm = viz_service.plot_confusion_matrix(metrics['confusion_matrix'])
                    st.pyplot(fig_cm)
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при обучении Random Forest: {str(e)}")
    
    if model_choice in ["CNN", "Обе модели"]:
        st.subheader("🧠 CNN (Сверточная нейронная сеть)")
        
        with st.expander("⚙️ Параметры CNN", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                epochs = st.slider(
                    "Количество эпох:",
                    min_value=10,
                    max_value=100,
                    value=30,
                    step=10,
                    help="Количество проходов по всему датасету"
                )
            
            with col2:
                batch_size = st.slider(
                    "Размер батча:",
                    min_value=16,
                    max_value=128,
                    value=32,
                    step=16,
                    help="Количество образцов за одну итерацию"
                )
            
            with col3:
                learning_rate = st.select_slider(
                    "Скорость обучения:",
                    options=[0.1, 0.01, 0.001, 0.0001],
                    value=0.001,
                    help="Шаг градиентного спуска"
                )
        
        if st.button("🧠 Обучить CNN", type="primary", key="train_cnn"):
            with st.spinner("Обучение CNN... Это может занять несколько минут."):
                try:
                    start_time = datetime.now()
                    
                    # Подготовка данных для CNN
                    X_train_cnn = X_train.reshape(-1, config.SEQUENCE_LENGTH, 1)
                    X_val_cnn = X_val.reshape(-1, config.SEQUENCE_LENGTH, 1)
                    
                    # Создание и обучение модели
                    cnn_model = model_service.create_cnn_model((config.SEQUENCE_LENGTH, 1))
                    
                    # Callbacks
                    callbacks = [
                        keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=10,
                            restore_best_weights=True
                        ),
                        keras.callbacks.ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.5,
                            patience=5,
                            min_lr=1e-6
                        )
                    ]
                    
                    # Обучение
                    history = cnn_model.fit(
                        X_train_cnn, y_train,
                        validation_data=(X_val_cnn, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=0
                    )
                    
                    # Оценка модели
                    metrics = model_service.evaluate_model(cnn_model, X_val_cnn, y_val, model_type='keras')
                    
                    # Сохранение модели
                    model_filename = f"models/cnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
                    cnn_model.save(model_filename)
                    
                    # Сохранение истории обучения
                    history_filename = f"results/cnn_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(history_filename, 'w') as f:
                        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
                    
                    training_time = (datetime.now() - start_time).total_seconds()
                    
                    st.session_state.cnn_model = cnn_model
                    st.session_state.cnn_metrics = metrics
                    st.session_state.cnn_history = history.history
                    st.session_state.cnn_training_time = training_time
                    st.session_state.cnn_model_path = model_filename
                    
                    st.success(f"✅ CNN обучена за {training_time:.2f} секунд!")
                    
                    # Отображение результатов
                    st.subheader("📊 Результаты CNN")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Точность", f"{metrics['accuracy']:.4f}")
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.4f}")
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.4f}")
                    with col4:
                        st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
                    
                    # Графики обучения
                    fig_history = viz_service.plot_training_history(history)
                    st.pyplot(fig_history)
                    
                    # Матрица ошибок
                    fig_cm = viz_service.plot_confusion_matrix(metrics['confusion_matrix'])
                    st.pyplot(fig_cm)
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при обучении CNN: {str(e)}")
    
    # Сохранение всех результатов
    if 'rf_model' in st.session_state or 'cnn_model' in st.session_state:
        st.markdown("---")
        st.subheader("💾 Сохранение результатов")
        
        if st.button("💾 Сохранить все результаты", type="secondary"):
            try:
                results = {}
                
                if 'rf_model' in st.session_state:
                    results['random_forest'] = {
                        'metrics': st.session_state.rf_metrics,
                        'training_time': st.session_state.rf_training_time,
                        'model_path': st.session_state.rf_model_path
                    }
                
                if 'cnn_model' in st.session_state:
                    results['cnn'] = {
                        'metrics': st.session_state.cnn_metrics,
                        'training_time': st.session_state.cnn_training_time,
                        'model_path': st.session_state.cnn_model_path,
                        'history_path': f"results/cnn_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    }
                
                # Сохранение в JSON
                results_filename = f"results/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(results_filename, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                st.success(f"✅ Результаты сохранены в {results_filename}")
                
            except Exception as e:
                st.error(f"❌ Ошибка при сохранении результатов: {str(e)}")

def render_prediction_page():
    """Страница прогнозирования."""
    st.header("🔍 Прогнозирование аритмий")
    
    # Загрузка моделей
    st.subheader("📂 Загрузка моделей")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Загрузить Random Forest", use_container_width=True):
            try:
                # Поиск последней модели RF
                rf_files = [f for f in os.listdir("models") if f.startswith("rf_model")]
                if rf_files:
                    latest_rf = max(rf_files, key=lambda x: os.path.getctime(os.path.join("models", x)))
                    with open(os.path.join("models", latest_rf), 'rb') as f:
                        rf_model = pickle.load(f)
                    
                    st.session_state.rf_model_loaded = rf_model
                    st.session_state.rf_model_name = latest_rf
                    st.success(f"✅ Random Forest загружен: {latest_rf}")
                else:
                    st.error("❌ Модели Random Forest не найдены. Сначала обучите модель.")
            except Exception as e:
                st.error(f"❌ Ошибка при загрузке Random Forest: {str(e)}")
    
    with col2:
        if st.button("📥 Загрузить CNN", use_container_width=True):
            try:
                # Поиск последней модели CNN
                cnn_files = [f for f in os.listdir("models") if f.startswith("cnn_model")]
                if cnn_files:
                    latest_cnn = max(cnn_files, key=lambda x: os.path.getctime(os.path.join("models", x)))
                    cnn_model = keras.models.load_model(os.path.join("models", latest_cnn))
                    
                    st.session_state.cnn_model_loaded = cnn_model
                    st.session_state.cnn_model_name = latest_cnn
                    st.success(f"✅ CNN загружена: {latest_cnn}")
                else:
                    st.error("❌ Модели CNN не найдены. Сначала обучите модель.")
            except Exception as e:
                st.error(f"❌ Ошибка при загрузке CNN: {str(e)}")
    
    # Показ загруженных моделей
    if 'rf_model_loaded' in st.session_state or 'cnn_model_loaded' in st.session_state:
        st.subheader("✅ Загруженные модели")
        
        models_info = []
        if 'rf_model_loaded' in st.session_state:
            models_info.append(f"🌲 Random Forest: {st.session_state.rf_model_name}")
        if 'cnn_model_loaded' in st.session_state:
            models_info.append(f"🧠 CNN: {st.session_state.cnn_model_name}")
        
        for info in models_info:
            st.info(info)
    
    # Выбор данных для прогнозирования
    st.subheader("📊 Выбор данных для прогнозирования")
    
    prediction_source = st.radio(
        "Источник данных:",
        ["Тестовая выборка", "Сгенерировать сигнал", "Загрузить файл"],
        horizontal=True
    )
    
    current_signal = None
    true_label = None
    
    if prediction_source == "Тестовая выборка":
        if 'X_test' in st.session_state:
            if st.button("🎲 Выбрать случайный пример", use_container_width=True):
                random_idx = np.random.randint(0, len(st.session_state.X_test))
                current_signal = st.session_state.X_test[random_idx]
                true_label = st.session_state.y_test[random_idx]
                
                st.session_state.current_signal = current_signal
                st.session_state.true_label = true_label
        else:
            st.warning("⚠️ Тестовая выборка не найдена. Сначала разделите данные.")
    
    elif prediction_source == "Сгенерировать сигнал":
        arrhythmia_type = st.selectbox(
            "Тип аритмии для генерации:",
            list(config.ARRHYTHMIA_CLASSES.items()),
            format_func=lambda x: x[1]
        )
        
        if st.button("🌀 Сгенерировать сигнал", use_container_width=True):
            t = np.linspace(0, 1, config.SEQUENCE_LENGTH)
            base_ecg = 0.5 * np.sin(2 * np.pi * 1 * t)
            
            class_id, class_name = arrhythmia_type
            
            if class_id == 0:  # Нормальный ритм
                test_signal = base_ecg + 0.1 * np.random.normal(size=config.SEQUENCE_LENGTH)
            elif class_id == 1:  # Апноэ
                test_signal = base_ecg * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)) + 0.1 * np.random.normal(size=config.SEQUENCE_LENGTH)
            elif class_id == 2:  # Фибрилляция предсердий
                test_signal = base_ecg + 0.3 * np.random.normal(size=config.SEQUENCE_LENGTH) + 0.1 * np.random.normal(size=config.SEQUENCE_LENGTH)
            elif class_id == 3:  # Шум
                test_signal = 0.8 * np.random.normal(size=config.SEQUENCE_LENGTH)
            else:  # Другая аритмия
                test_signal = base_ecg * (1 + 0.3 * np.sin(2 * np.pi * 2 * t)) + 0.1 * np.random.normal(size=config.SEQUENCE_LENGTH)
            
            current_signal = test_signal
            true_label = class_id
            
            st.session_state.current_signal = current_signal
            st.session_state.true_label = true_label
    
    else:  # Загрузить файл
        uploaded_file = st.file_uploader(
            "Загрузите файл с сигналом ЭКГ:",
            type=['csv', 'txt', 'npy']
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                    signal = data.iloc[:, 0].values
                elif uploaded_file.name.endswith('.txt'):
                    signal = np.loadtxt(uploaded_file)
                elif uploaded_file.name.endswith('.npy'):
                    signal = np.load(uploaded_file)
                
                # Обработка сигнала
                if len(signal) > config.SEQUENCE_LENGTH:
                    signal = signal[:config.SEQUENCE_LENGTH]
                elif len(signal) < config.SEQUENCE_LENGTH:
                    signal = np.pad(signal, (0, config.SEQUENCE_LENGTH - len(signal)))
                
                current_signal = signal
                st.session_state.current_signal = current_signal
                
            except Exception as e:
                st.error(f"❌ Ошибка при обработке файла: {str(e)}")
    
    # Визуализация текущего сигнала
    if 'current_signal' in st.session_state:
        current_signal = st.session_state.current_signal
        
        st.subheader("📈 Визуализация сигнала")
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(current_signal, linewidth=1.5, color=config.COLORS['info'])
        ax.set_xlabel('Отсчеты', fontsize=11)
        ax.set_ylabel('Амплитуда', fontsize=11)
        ax.set_title('ЭКГ сигнал для анализа', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        if 'true_label' in st.session_state:
            true_class_name = config.ARRHYTHMIA_CLASSES[st.session_state.true_label]
            st.info(f"**Истинный класс:** {true_class_name}")
    
    # Прогнозирование
    if 'current_signal' in st.session_state:
        st.subheader("🔮 Прогнозирование")
        
        selected_models = []
        if 'rf_model_loaded' in st.session_state:
            selected_models.append('RF')
        if 'cnn_model_loaded' in st.session_state:
            selected_models.append('CNN')
        
        if selected_models:
            model_choice = st.multiselect(
                "Выберите модели для прогнозирования:",
                selected_models,
                default=selected_models
            )
            
            if st.button("🎯 Выполнить прогноз", type="primary", use_container_width=True):
                results = []
                current_signal_processed = data_loader.preprocess_signal(st.session_state.current_signal)
                
                for model_code in model_choice:
                    if model_code == 'RF' and 'rf_model_loaded' in st.session_state:
                        model = st.session_state.rf_model_loaded
                        
                        # Прогноз
                        signal_reshaped = current_signal_processed.reshape(1, -1)
                        prediction = model.predict(signal_reshaped)[0]
                        probabilities = model.predict_proba(signal_reshaped)[0]
                        
                        results.append({
                            'model': 'Random Forest',
                            'prediction': int(prediction),
                            'class_name': config.ARRHYTHMIA_CLASSES[prediction],
                            'confidence': float(np.max(probabilities)),
                            'probabilities': probabilities.tolist()
                        })
                    
                    elif model_code == 'CNN' and 'cnn_model_loaded' in st.session_state:
                        model = st.session_state.cnn_model_loaded
                        
                        # Прогноз
                        signal_reshaped = current_signal_processed.reshape(1, config.SEQUENCE_LENGTH, 1)
                        probabilities = model.predict(signal_reshaped, verbose=0)[0]
                        prediction = np.argmax(probabilities)
                        
                        results.append({
                            'model': 'CNN',
                            'prediction': int(prediction),
                            'class_name': config.ARRHYTHMIA_CLASSES[prediction],
                            'confidence': float(np.max(probabilities)),
                            'probabilities': probabilities.tolist()
                        })
                
                if results:
                    st.session_state.prediction_results = results
                    
                    # Отображение результатов
                    st.subheader("📊 Результаты прогнозирования")
                    
                    for result in results:
                        with st.container():
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.metric(
                                    f"**{result['model']}**",
                                    result['class_name'],
                                    f"Уверенность: {result['confidence']:.2%}"
                                )
                            
                            with col2:
                                # Визуализация вероятностей
                                prob_df = pd.DataFrame({
                                    'Класс': list(config.ARRHYTHMIA_CLASSES.values()),
                                    'Вероятность': result['probabilities']
                                })
                                
                                fig_prob, ax_prob = plt.subplots(figsize=(8, 3))
                                bars = ax_prob.barh(prob_df['Класс'], prob_df['Вероятность'])
                                ax_prob.set_xlabel('Вероятность', fontsize=10)
                                ax_prob.set_xlim(0, 1)
                                
                                # Цвет столбцов в зависимости от вероятности
                                for bar in bars:
                                    bar.set_color(config.COLORS['normal'] if bar.get_width() < 0.3 
                                                 else config.COLORS['warning'] if bar.get_width() < 0.7 
                                                 else config.COLORS['danger'])
                                
                                st.pyplot(fig_prob)
                    
                    # Генерация рекомендаций
                    st.subheader("💡 Рекомендации")
                    
                    recommendations = {
                        0: {
                            "title": "✅ Нормальный сердечный ритм",
                            "description": "Обнаружен нормальный синусовый ритм.",
                            "actions": [
                                "Продолжайте плановое наблюдение",
                                "Рекомендуется ежегодный профилактический осмотр",
                                "Ведение здорового образа жизни"
                            ],
                            "urgency": "Низкая"
                        },
                        1: {
                            "title": "⚠️ Признаки апноэ сна",
                            "description": "Обнаружены паттерны, характерные для апноэ сна.",
                            "actions": [
                                "Консультация сомнолога",
                                "Проведение полисомнографии",
                                "Коррекция образа жизни и веса"
                            ],
                            "urgency": "Средняя"
                        },
                        2: {
                            "title": "🚨 Фибрилляция предсердий",
                            "description": "Выявлена фибрилляция предсердий - серьезное нарушение ритма.",
                            "actions": [
                                "СРОЧНАЯ консультация кардиолога",
                                "ЭКГ Холтер мониторинг",
                                "Назначение антикоагулянтной терапии"
                            ],
                            "urgency": "Высокая"
                        },
                        3: {
                            "title": "📢 Сигнал с шумом",
                            "description": "Сигнал содержит значительные шумы, затрудняющие анализ.",
                            "actions": [
                                "Повторное измерение ЭКГ",
                                "Проверка электродов и контактов",
                                "Исключение артефактов движения"
                            ],
                            "urgency": "Низкая"
                        },
                        4: {
                            "title": "⚠️ Другая аритмия",
                            "description": "Обнаружена аритмия неуточненного типа.",
                            "actions": [
                                "Консультация кардиолога",
                                "Дополнительная диагностика",
                                "Эхокардиография для оценки структур сердца"
                            ],
                            "urgency": "Средняя"
                        }
                    }
                    
                    # Показываем рекомендации для каждой модели
                    for result in results:
                        pred_class = result['prediction']
                        if pred_class in recommendations:
                            rec = recommendations[pred_class]
                            
                            with st.expander(f"Рекомендации от {result['model']}: {rec['title']}", expanded=True):
                                st.markdown(f"**Описание:** {rec['description']}")
                                st.markdown(f"**Срочность:** {rec['urgency']}")
                                
                                st.markdown("**Рекомендуемые действия:**")
                                for action in rec['actions']:
                                    st.markdown(f"- {action}")
                    
                    # Кнопка сохранения результатов
                    if st.button("💾 Сохранить результаты прогноза", type="secondary"):
                        try:
                            prediction_data = {
                                'timestamp': datetime.now().isoformat(),
                                'signal_length': len(current_signal_processed),
                                'true_label': st.session_state.get('true_label', 'unknown'),
                                'predictions': results
                            }
                            
                            pred_filename = f"results/prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            with open(pred_filename, 'w') as f:
                                json.dump(prediction_data, f, indent=2, default=str)
                            
                            st.success(f"✅ Результаты сохранены в {pred_filename}")
                            
                        except Exception as e:
                            st.error(f"❌ Ошибка при сохранении: {str(e)}")
        else:
            st.warning("⚠️ Загрузите хотя бы одну модель для прогнозирования")

def render_comparison_page():
    """Страница сравнения моделей."""
    st.header("📈 Сравнение моделей")
    
    # Проверка наличия обученных моделей
    has_rf = 'rf_metrics' in st.session_state
    has_cnn = 'cnn_metrics' in st.session_state
    
    if not (has_rf or has_cnn):
        st.warning("⚠️ Сначала обучите модели на вкладке 'Обучение моделей'")
        return
    
    # Сбор данных для сравнения
    models_data = {}
    
    if has_rf:
        models_data['Random Forest'] = {
            'metrics': st.session_state.rf_metrics,
            'training_time': st.session_state.get('rf_training_time', 0),
            'type': 'Классическая ML'
        }
    
    if has_cnn:
        models_data['CNN'] = {
            'metrics': st.session_state.cnn_metrics,
            'training_time': st.session_state.get('cnn_training_time', 0),
            'type': 'Нейронная сеть'
        }
    
    # Таблица сравнения
    st.subheader("📊 Сводная таблица метрик")
    
    comparison_df = pd.DataFrame([
        {
            'Модель': name,
            'Тип': data['type'],
            'Точность': f"{data['metrics']['accuracy']:.4f}",
            'Precision': f"{data['metrics']['precision']:.4f}",
            'Recall': f"{data['metrics']['recall']:.4f}",
            'F1-Score': f"{data['metrics']['f1_score']:.4f}",
            'Время обучения (с)': f"{data['training_time']:.2f}"
        }
        for name, data in models_data.items()
    ])
    
    st.dataframe(comparison_df.set_index('Модель'), use_container_width=True)
    
    # Визуализация сравнения
    st.subheader("📈 Визуализация сравнения")
    
    # График сравнения метрик
    fig_comparison = viz_service.plot_metrics_comparison({
        name: data['metrics'] for name, data in models_data.items()
    })
    st.pyplot(fig_comparison)
    
    # Матрицы ошибок
    st.subheader("🔍 Матрицы ошибок")
    
    fig_cm_comparison = viz_service.plot_confusion_matrices(
        {name: data for name, data in models_data.items()},
        list(config.ARRHYTHMIA_CLASSES.values())
    )
    st.pyplot(fig_cm_comparison)
    
    # Детальный анализ
    st.subheader("📋 Детальный анализ")
    
    for model_name, model_data in models_data.items():
        with st.expander(f"📄 Отчет о классификации: {model_name}", expanded=False):
            metrics = model_data['metrics']
            
            # Основные метрики
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Точность", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
            
            # Детальный отчет по классам
            st.markdown("**Метрики по классам:**")
            report_df = pd.DataFrame(metrics['classification_report']).transpose()
            st.dataframe(report_df, use_container_width=True)
    
    # Выводы и рекомендации
    st.subheader("🎯 Выводы и рекомендации")
    
    if len(models_data) > 1:
        # Определяем лучшую модель
        best_model = max(models_data.items(), 
                        key=lambda x: x[1]['metrics']['accuracy'])
        best_model_name = best_model[0]
        best_accuracy = best_model[1]['metrics']['accuracy']
        
        st.info(f"**🏆 Лучшая модель:** {best_model_name} с точностью {best_accuracy:.2%}")
        
        # Сравнение моделей
        if len(models_data) == 2:
            model_names = list(models_data.keys())
            accuracy_diff = abs(models_data[model_names[0]]['metrics']['accuracy'] - 
                              models_data[model_names[1]]['metrics']['accuracy'])
            
            if accuracy_diff < 0.05:
                st.write("✅ Модели демонстрируют схожую производительность.")
            elif accuracy_diff < 0.1:
                st.write(f"⚠️ Заметная разница в точности ({accuracy_diff:.2%}).")
            else:
                st.write(f"🚨 Существенная разница в точности ({accuracy_diff:.2%}).")
        
        # Рекомендации по выбору модели
        st.markdown("**Рекомендации по выбору модели:**")
        
        recommendations = {
            'Random Forest': [
                "✅ Быстрое обучение на небольших данных",
                "✅ Высокая интерпретируемость",
                "⚠️ Может переобучаться на сложных данных",
                "💡 Рекомендуется для начальных экспериментов"
            ],
            'CNN': [
                "✅ Лучшая производительность на сложных данных",
                "✅ Автоматическое извлечение признаков",
                "⚠️ Требует больше вычислительных ресурсов",
                "💡 Рекомендуется для производственных систем"
            ]
        }
        
        for model_name, recs in recommendations.items():
            if model_name in models_data:
                with st.expander(f"Особенности {model_name}", expanded=False):
                    for rec in recs:
                        st.write(f"- {rec}")
    else:
        st.info(f"✅ Обучена одна модель: {list(models_data.keys())[0]}")
    
    # Экспорт результатов
    st.markdown("---")
    st.subheader("📤 Экспорт результатов")
    
    if st.button("💾 Экспортировать все результаты сравнения", type="primary"):
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'models_comparison': {
                    name: {
                        'metrics': data['metrics'],
                        'training_time': data['training_time'],
                        'type': data['type']
                    }
                    for name, data in models_data.items()
                },
                'summary': {
                    'best_model': best_model_name if len(models_data) > 1 else list(models_data.keys())[0],
                    'best_accuracy': best_accuracy if len(models_data) > 1 else list(models_data.values())[0]['metrics']['accuracy']
                }
            }
            
            export_filename = f"results/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(export_filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            # Также сохраняем в CSV для удобства
            csv_filename = f"results/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            comparison_df.to_csv(csv_filename, index=False)
            
            st.success(f"✅ Результаты экспортированы:")
            st.success(f"   - JSON: {export_filename}")
            st.success(f"   - CSV: {csv_filename}")
            
        except Exception as e:
            st.error(f"❌ Ошибка при экспорте: {str(e)}")

def render_about_page():
    """Страница о проекте."""
    st.header("ℹ️ О проекте")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎓 Дипломный проект
        
        **Тема:** Разработка рекомендательной системы на основе обработки биомедицинских данных
        
        **Автор:** Чайкин Виталий Федорович
        
        **Образовательное учреждение:** ЧОУВО «Московский университет им. С.Ю. Витте»
        
        **Факультет:** Информационных систем и технологий
        
        **Руководитель:** Простомолотов Андрей Сергеевич
        
        **Период выполнения:** 10.11.2025 - 07.12.2025
        
        ---
        
        ### 🎯 Цели проекта
        
        1. **Разработка интеллектуальной системы** для автоматического анализа ЭКГ сигналов
        2. **Создание рекомендательного механизма** для помощи в диагностике сердечных аритмий
        3. **Реализация веб-интерфейса** для удобного взаимодействия с системой
        4. **Сравнение эффективности** различных алгоритмов машинного обучения
        
        ---
        
        ### 🔬 Научная новизна
        
        - Применение **ансамблевых методов** для классификации ЭКГ
        - Разработка **гибридной архитектуры** (RF + CNN)
        - Создание **адаптивной системы рекомендаций** на основе уверенности модели
        - Реализация **комплексной системы оценки** качества моделей
        
        ---
        
        ### 💼 Практическая значимость
        
        **Для медицинских учреждений:**
        - Повышение точности диагностики
        - Сокращение времени анализа
        - Поддержка принятия врачебных решений
        
        **Для образовательного процесса:**
        - Наглядный пример применения ML в медицине
        - Инструмент для лабораторных работ
        - База для дальнейших исследований
        """)
    
    with col2:
        st.subheader("📊 Технические характеристики")
        
        st.metric("Язык программирования", "Python 3.9+")
        st.metric("Объем кода", ">2000 строк")
        st.metric("Количество моделей", "2")
        st.metric("Классов аритмий", "5")
        st.metric("Минимальная точность", "85%")
        st.metric("Время обучения", "<6 часов")
        
        st.markdown("---")
        
        st.subheader("📚 Используемые библиотеки")
        
        libraries = {
            "Streamlit": "Веб-интерфейс",
            "Scikit-learn": "Классические ML",
            "TensorFlow": "Нейронные сети",
            "Pandas/NumPy": "Обработка данных",
            "Matplotlib": "Визуализация",
            "SciPy": "Обработка сигналов"
        }
        
        for lib, desc in libraries.items():
            st.markdown(f"**{lib}** - {desc}")
        
        st.markdown("---")
        
        st.subheader("📁 Структура проекта")
        
        structure = """
        📁 project/
        ├── 📁 app/
        │   ├── 📁 core/       # Основные модули
        │   ├── 📁 services/   # Сервисные функции
        │   └── 📁 web/        # Веб-интерфейс
        ├── 📁 models/         # Сохраненные модели
        ├── 📁 data/           # Наборы данных
        ├── 📁 results/        # Результаты
        ├── 📄 main.py         # Главный файл
        └── 📄 requirements.txt
        """
        
        st.code(structure, language="text")
    
    st.markdown("---")
    
    st.subheader("📞 Контакты")
    
    contact_col1, contact_col2, contact_col3 = st.columns(3)
    
    with contact_col1:
        st.markdown("""
        **📧 Email:**
        - vit.chaykin@example.com
        - project.ecg@example.com
        """)
    
    with contact_col2:
        st.markdown("""
        **🌐 Онлайн:**
        - [GitHub](https://github.com/username/ecg-analysis)
        - [LinkedIn](https://linkedin.com/in/username)
        """)
    
    with contact_col3:
        st.markdown("""
        **📱 Телефон:**
        - +7 (XXX) XXX-XX-XX
        - Рабочие часы: 9:00-18:00
        """)
    
    # Информация о лицензии
    st.markdown("---")
    
    with st.expander("📄 Лицензия и использование", expanded=False):
        st.markdown("""
        ### Лицензионное соглашение
        
        **MIT License**
        
        Copyright © 2025 Чайкин Виталий Федорович
        
        Данное программное обеспечение предоставляется "как есть", без каких-либо гарантий.
        
        **Разрешено:**
        - Использование в коммерческих целях
        - Модификация и распространение
        - Использование в частных целях
        
        **Требуется:**
        - Сохранение информации об авторском праве
        - Указание ссылки на оригинальный проект
        
        **Запрещено:**
        - Использование в медицинских целях без дополнительной валидации
        - Ответственность за последствия использования системы
        """)

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Основная функция приложения."""
    
    # Инициализация session state
    if 'page' not in st.session_state:
        st.session_state.page = "Главная"
    
    # Боковая панель навигации
    with st.sidebar:
        st.title("🧭 Навигация")
        
        # Выбор страницы
        page_options = {
            "🏠 Главная": "Главная",
            "📊 Загрузка данных": "Загрузка данных",
            "🤖 Обучение моделей": "Обучение моделей",
            "🔍 Прогнозирование": "Прогнозирование",
            "📈 Сравнение моделей": "Сравнение моделей",
            "ℹ️ О проекте": "О проекте"
        }
        
        selected = st.radio(
            "Выберите раздел:",
            list(page_options.keys()),
            index=list(page_options.keys()).index(
                [k for k, v in page_options.items() if v == st.session_state.page][0]
            ) if st.session_state.page in page_options.values() else 0
        )
        
        st.session_state.page = page_options[selected]
        
        st.markdown("---")
        
        # Статус системы
        st.subheader("📊 Статус системы")
        
        status_items = []
        
        if 'X' in st.session_state:
            status_items.append("✅ Данные загружены")
        else:
            status_items.append("❌ Данные не загружены")
        
        if 'rf_model' in st.session_state:
            status_items.append("✅ RF модель обучена")
        
        if 'cnn_model' in st.session_state:
            status_items.append("✅ CNN модель обучена")
        
        for item in status_items:
            st.write(f"- {item}")
        
        st.markdown("---")
        
        # Быстрые действия
        st.subheader("⚡ Быстрые действия")
        
        if st.button("🔄 Сбросить все данные", type="secondary", use_container_width=True):
            keys_to_keep = ['page']
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
            st.rerun()
        
        if st.button("💾 Экспорт сессии", type="secondary", use_container_width=True):
            st.info("Функция экспорта в разработке")
        
        st.markdown("---")
        
        # Информация о версии
        st.caption("Версия 1.0.0")
        st.caption("© 2025 Чайкин В.Ф.")
    
    # Отображение выбранной страницы
    if st.session_state.page == "Главная":
        render_home_page()
    elif st.session_state.page == "Загрузка данных":
        render_data_loading_page()
    elif st.session_state.page == "Обучение моделей":
        render_model_training_page()
    elif st.session_state.page == "Прогнозирование":
        render_prediction_page()
    elif st.session_state.page == "Сравнение моделей":
        render_comparison_page()
    elif st.session_state.page == "О проекте":
        render_about_page()

# ============================================================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================================================

if __name__ == "__main__":
    main()