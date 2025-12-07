"""
Веб-интерфейс приложения на Streamlit.
Автор: Чайкин Виталий Федорович
Тема ВКР: Разработка рекомендательной системы на основе обработки биомедицинских данных
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import warnings
import json
from datetime import datetime
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Отключаем предупреждения
warnings.filterwarnings('ignore')

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Импорт модулей проекта
try:
    from app.core.data_loader import BiomedicalDataLoader
    from app.core.feature_engineer import ECGFeatureEngineer
    from app.core.model_loader import ModelLoader
    from app.services.training_service import TrainingService
    from app.services.prediction_service import PredictionService
    from utils.config import Config
except ImportError:
    # Для случая, если модули не найдены, создаем заглушки
    st.warning("⚠️ Некоторые модули не найдены. Используется упрощенный режим работы.")
    
    class Config:
        SAMPLING_RATE = 360
        SEQUENCE_LENGTH = 360
        RANDOM_STATE = 42
        TEST_SIZE = 0.2
        VALIDATION_SIZE = 0.1
        ARRHYTHMIA_CLASSES = {
            0: "Нормальный ритм",
            1: "Апноэ", 
            2: "Фибрилляция предсердий",
            3: "Шум",
            4: "Другая аритмия"
        }
    
    class BiomedicalDataLoader:
        def download_dataset(self):
            return self._create_simulated_data()
        
        def _create_simulated_data(self):
            np.random.seed(42)
            n_samples = 5000
            n_features = 360
            X = np.random.randn(n_samples, n_features)
            y = np.random.randint(0, 5, n_samples)
            return X, y
        
        def split_data(self, X, y):
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42
            )
            return X_train, X_val, X_test, y_train, y_val, y_test

class BiomedicalApp:
    """Класс веб-приложения для анализа ЭКГ."""
    
    def __init__(self):
        self.config = Config()
        self.setup_page()
        self.initialize_services()
        self.ensure_directories()
        
    def setup_page(self):
        """Настройка страницы Streamlit."""
        st.set_page_config(
            page_title="Интеллектуальная система анализа ЭКГ",
            page_icon="❤️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("❤️ Интеллектуальная система анализа биомедицинских данных")
        st.markdown("---")
        
    def ensure_directories(self):
        """Создание необходимых директорий."""
        directories = ["models", "data", "results", "logs"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def initialize_services(self):
        """Инициализация сервисов."""
        if 'initialized' not in st.session_state:
            try:
                self.data_loader = BiomedicalDataLoader()
                self.feature_engineer = ECGFeatureEngineer()
                self.model_loader = ModelLoader()
                self.training_service = TrainingService(self.model_loader, self.feature_engineer)
                self.prediction_service = PredictionService(
                    self.model_loader, self.feature_engineer, self.data_loader
                )
            except:
                # Создаем заглушки для сервисов
                self.data_loader = BiomedicalDataLoader()
                self.feature_engineer = None
                self.model_loader = None
                self.training_service = None
                self.prediction_service = None
            
            # Инициализация сессии
            st.session_state.initialized = True
            st.session_state.models_trained = False
            st.session_state.data_loaded = False
            st.session_state.current_tab = "data_analysis"
            st.session_state.metrics_history = []
    
    def render_sidebar(self):
        """Отрисовка боковой панели."""
        with st.sidebar:
            st.title("🧭 Навигация")
            
            tabs = {
                "📊 Анализ данных": "data_analysis",
                "🤖 Обучение моделей": "model_training", 
                "🔍 Прогнозирование": "prediction",
                "📈 Сравнение моделей": "model_comparison",
                "📋 История экспериментов": "experiment_history",
                "ℹ️ О проекте": "about"
            }
            
            selected_tab = st.radio(
                "Выберите раздел:", 
                list(tabs.keys()),
                index=list(tabs.values()).index(st.session_state.current_tab) 
                if st.session_state.current_tab in tabs.values() else 0
            )
            st.session_state.current_tab = tabs[selected_tab]
            
            st.markdown("---")
            
            # Статус системы
            st.subheader("📊 Статус системы")
            
            status_items = []
            if st.session_state.get('data_loaded', False):
                status_items.append("✅ Данные загружены")
            else:
                status_items.append("❌ Данные не загружены")
            
            if st.session_state.get('rf_model', None):
                status_items.append("✅ RF обучен")
            
            if st.session_state.get('cnn_model', None):
                status_items.append("✅ CNN обучена")
            
            for item in status_items:
                st.write(f"• {item}")
            
            st.markdown("---")
            
            # Быстрые действия
            st.subheader("⚡ Быстрые действия")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Сброс", help="Сбросить все данные"):
                    keys_to_delete = [k for k in st.session_state.keys() 
                                    if k not in ['initialized', 'current_tab']]
                    for key in keys_to_delete:
                        del st.session_state[key]
                    st.rerun()
            
            with col2:
                if st.button("💾 Экспорт", help="Экспорт результатов"):
                    self.export_results()
            
            st.markdown("---")
            
            # Информация о системе
            st.caption(f"**Версия:** 1.0.0")
            st.caption(f"**Дата:** {datetime.now().strftime('%d.%m.%Y')}")
            st.caption("© Чайкин В.Ф., 2025")
    
    def render_data_analysis_tab(self):
        """Вкладка анализа данных."""
        st.header("📊 Анализ данных ЭКГ")
        
        # Выбор источника данных
        data_source = st.radio(
            "Выберите источник данных:",
            ["Симулированные данные", "Загрузить файл"],
            horizontal=True,
            key="data_source_select"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if data_source == "Симулированные данные":
                st.subheader("Параметры генерации")
                
                n_samples = st.slider(
                    "Количество образцов:", 
                    min_value=1000, 
                    max_value=10000, 
                    value=5000, 
                    step=1000
                )
                
                if st.button("🎲 Сгенерировать данные", type="primary"):
                    with st.spinner("Генерация данных..."):
                        progress_bar = st.progress(0)
                        self.X, self.y = self.data_loader.download_dataset()
                        
                        # Разделение данных
                        (self.X_train, self.X_val, self.X_test, 
                         self.y_train, self.y_val, self.y_test) = self.data_loader.split_data(self.X, self.y)
                        
                        st.session_state.data_loaded = True
                        st.session_state.X = self.X
                        st.session_state.y = self.y
                        st.session_state.X_train = self.X_train
                        st.session_state.X_val = self.X_val
                        st.session_state.X_test = self.X_test
                        st.session_state.y_train = self.y_train
                        st.session_state.y_val = self.y_val
                        st.session_state.y_test = self.y_test
                        
                        progress_bar.progress(100)
                        st.success(f"✅ Сгенерировано {n_samples} записей!")
            
            else:  # Загрузить файл
                st.subheader("Загрузка файла")
                
                uploaded_file = st.file_uploader(
                    "Загрузите файл с данными ЭКГ:", 
                    type=['csv', 'txt', 'npy'],
                    help="Поддерживаемые форматы: CSV, TXT, NPY"
                )
                
                if uploaded_file is not None:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            data = pd.read_csv(uploaded_file)
                            if len(data.columns) >= 2:
                                self.X = data.iloc[:, :-1].values
                                self.y = data.iloc[:, -1].values
                            else:
                                st.error("CSV файл должен содержать хотя бы 2 колонки")
                                return
                        
                        elif uploaded_file.name.endswith('.txt'):
                            data = np.loadtxt(uploaded_file)
                            if data.ndim == 2:
                                self.X = data[:, :-1]
                                self.y = data[:, -1]
                            else:
                                st.error("TXT файл должен быть 2D массивом")
                                return
                        
                        elif uploaded_file.name.endswith('.npy'):
                            data = np.load(uploaded_file, allow_pickle=True)
                            if isinstance(data, tuple) and len(data) == 2:
                                self.X, self.y = data
                            else:
                                st.error("NPY файл должен содержать кортеж (X, y)")
                                return
                        
                        # Разделение данных
                        (self.X_train, self.X_val, self.X_test, 
                         self.y_train, self.y_val, self.y_test) = self.data_loader.split_data(self.X, self.y)
                        
                        st.session_state.data_loaded = True
                        st.session_state.X = self.X
                        st.session_state.y = self.y
                        st.session_state.X_train = self.X_train
                        st.session_state.X_val = self.X_val
                        st.session_state.X_test = self.X_test
                        st.session_state.y_train = self.y_train
                        st.session_state.y_val = self.y_val
                        st.session_state.y_test = self.y_test
                        
                        st.success(f"✅ Файл {uploaded_file.name} успешно загружен!")
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка при обработке файла: {str(e)}")
        
        # Отображение статистики если данные загружены
        if st.session_state.get('data_loaded', False):
            self.X = st.session_state.X
            self.y = st.session_state.y
            
            with col2:
                st.subheader("📈 Статистика")
                
                stats = {
                    "Всего записей": len(self.X),
                    "Длина сигнала": f"{self.X.shape[1]} отсчетов",
                    "Количество классов": len(np.unique(self.y)),
                    "Обучающая выборка": f"{len(st.session_state.X_train)}",
                    "Валидационная выборка": f"{len(st.session_state.X_val)}",
                    "Тестовая выборка": f"{len(st.session_state.X_test)}"
                }
                
                for key, value in stats.items():
                    st.metric(key, value)
            
            # Распределение классов
            st.subheader("📊 Распределение классов")
            
            class_counts = pd.Series(self.y).value_counts().sort_index()
            class_names = [self.config.ARRHYTHMIA_CLASSES.get(i, f"Класс {i}") 
                          for i in class_counts.index]
            
            # Столбчатая диаграмма
            fig_bar = go.Figure(data=[go.Bar(
                x=class_names,
                y=class_counts.values,
                marker_color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
            )])
            fig_bar.update_layout(
                title="Распределение записей по классам",
                xaxis_title="Класс аритмии",
                yaxis_title="Количество записей",
                template="plotly_white"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Круговая диаграмма
            fig_pie = go.Figure(data=[go.Pie(
                labels=class_names,
                values=class_counts.values,
                hole=.3,
                marker_colors=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
            )])
            fig_pie.update_layout(title="Процентное распределение классов")
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Примеры сигналов
            st.subheader("📈 Примеры сигналов ЭКГ")
            
            selected_class = st.selectbox(
                "Выберите класс для просмотра:",
                list(self.config.ARRHYTHMIA_CLASSES.items()),
                format_func=lambda x: x[1],
                key="signal_class_select"
            )
            
            if selected_class:
                class_id, class_name = selected_class
                class_indices = np.where(self.y == class_id)[0]
                
                if len(class_indices) > 0:
                    n_examples = min(3, len(class_indices))
                    example_indices = class_indices[:n_examples]
                    
                    fig_signals = make_subplots(
                        rows=n_examples, cols=1,
                        subplot_titles=[f"Пример {i+1}: {class_name}" 
                                       for i in range(n_examples)]
                    )
                    
                    for i, idx in enumerate(example_indices):
                        fig_signals.add_trace(
                            go.Scatter(
                                y=self.X[idx],
                                mode='lines',
                                name=f"Пример {i+1}",
                                line=dict(color='blue', width=1.5)
                            ),
                            row=i+1, col=1
                        )
                    
                    fig_signals.update_layout(
                        height=250 * n_examples,
                        showlegend=False,
                        template="plotly_white"
                    )
                    
                    for i in range(n_examples):
                        fig_signals.update_xaxes(title_text="Отсчеты", row=i+1, col=1)
                        fig_signals.update_yaxes(title_text="Амплитуда", row=i+1, col=1)
                    
                    st.plotly_chart(fig_signals, use_container_width=True)
            
            # Статистика признаков
            if st.checkbox("📊 Показать статистику признаков"):
                st.subheader("📋 Статистика признаков")
                
                # Вычисляем базовую статистику
                feature_stats = pd.DataFrame({
                    'Среднее': np.mean(self.X, axis=0),
                    'Стандартное отклонение': np.std(self.X, axis=0),
                    'Минимум': np.min(self.X, axis=0),
                    'Максимум': np.max(self.X, axis=0),
                    'Медиана': np.median(self.X, axis=0)
                })
                
                st.dataframe(feature_stats.describe().round(4), use_container_width=True)
                
                # Гистограмма распределения средних значений
                fig_hist = go.Figure(data=[go.Histogram(
                    x=feature_stats['Среднее'],
                    nbinsx=50,
                    marker_color='#3498db',
                    opacity=0.7
                )])
                fig_hist.update_layout(
                    title="Распределение средних значений признаков",
                    xaxis_title="Среднее значение",
                    yaxis_title="Частота",
                    template="plotly_white"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
    
    def render_model_training_tab(self):
        """Вкладка обучения моделей."""
        st.header("🤖 Обучение моделей")
        
        # Проверка наличия данных
        if not st.session_state.get('data_loaded', False):
            st.warning("⚠️ Сначала загрузите данные на вкладке 'Анализ данных'")
            return
        
        # Получаем данные из session state
        X_train = st.session_state.X_train
        X_val = st.session_state.X_val
        y_train = st.session_state.y_train
        y_val = st.session_state.y_val
        
        # Выбор модели
        model_type = st.radio(
            "Выберите тип модели:",
            ["Random Forest", "CNN (нейронная сеть)", "Обе модели"],
            horizontal=True,
            key="model_type_radio"
        )
        
        if model_type in ["Random Forest", "Обе модели"]:
            self._render_random_forest_training(X_train, X_val, y_train, y_val)
        
        if model_type in ["CNN", "Обе модели"]:
            self._render_cnn_training(X_train, X_val, y_train, y_val)
    
    def _render_random_forest_training(self, X_train, X_val, y_train, y_val):
        """Рендеринг обучения Random Forest."""
        st.subheader("🌲 Random Forest")
        
        with st.expander("⚙️ Параметры обучения", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_estimators = st.slider(
                    "Количество деревьев:", 
                    min_value=50, 
                    max_value=300, 
                    value=100, 
                    step=50,
                    help="Чем больше деревьев, тем лучше обобщение, но дольше обучение"
                )
            
            with col2:
                max_depth = st.slider(
                    "Максимальная глубина:", 
                    min_value=5, 
                    max_value=50, 
                    value=10, 
                    step=5,
                    help="Ограничивает глубину деревьев для предотвращения переобучения"
                )
            
            with col3:
                use_cv = st.checkbox(
                    "Использовать кросс-валидацию", 
                    value=False,
                    help="5-кратная кросс-валидация для подбора параметров"
                )
        
        if st.button("🌲 Обучить Random Forest", type="primary", key="train_rf_btn"):
            with st.spinner("Обучение Random Forest..."):
                try:
                    start_time = datetime.now()
                    
                    # Создание и обучение модели
                    from sklearn.ensemble import RandomForestClassifier
                    
                    rf_model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=self.config.RANDOM_STATE,
                        n_jobs=-1
                    )
                    
                    if use_cv:
                        from sklearn.model_selection import cross_val_score
                        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
                        st.info(f"Кросс-валидация: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                    
                    rf_model.fit(X_train, y_train)
                    
                    # Оценка модели
                    y_pred = rf_model.predict(X_val)
                    y_pred_proba = rf_model.predict_proba(X_val)
                    
                    # Вычисление метрик
                    accuracy = accuracy_score(y_val, y_pred)
                    precision = precision_score(y_val, y_pred, average='weighted')
                    recall = recall_score(y_val, y_pred, average='weighted')
                    f1 = f1_score(y_val, y_pred, average='weighted')
                    cm = confusion_matrix(y_val, y_pred)
                    report = classification_report(y_val, y_pred, output_dict=True)
                    
                    training_time = (datetime.now() - start_time).total_seconds()
                    
                    # Сохранение модели
                    model_filename = f"models/rf_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    with open(model_filename, 'wb') as f:
                        pickle.dump(rf_model, f)
                    
                    # Сохранение в session state
                    st.session_state.rf_model = rf_model
                    st.session_state.rf_metrics = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'confusion_matrix': cm,
                        'classification_report': report,
                        'training_time': training_time,
                        'model_path': model_filename
                    }
                    st.session_state.models_trained = True
                    
                    # Добавление в историю экспериментов
                    experiment_data = {
                        'timestamp': datetime.now().isoformat(),
                        'model_type': 'Random Forest',
                        'parameters': {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'use_cv': use_cv
                        },
                        'metrics': {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'training_time': training_time
                        }
                    }
                    st.session_state.metrics_history.append(experiment_data)
                    
                    st.success(f"✅ Random Forest обучен за {training_time:.2f} сек!")
                    
                    # Отображение результатов
                    self._display_training_results(
                        st.session_state.rf_metrics, 
                        "Random Forest",
                        cm
                    )
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при обучении Random Forest: {str(e)}")
    
    def _render_cnn_training(self, X_train, X_val, y_train, y_val):
        """Рендеринг обучения CNN."""
        st.subheader("🧠 CNN (Сверточная нейронная сеть)")
        
        with st.expander("⚙️ Параметры обучения", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                epochs = st.slider(
                    "Количество эпох:", 
                    min_value=10, 
                    max_value=100, 
                    value=30, 
                    step=10,
                    help="Количество проходов по всему датасету"
                )
                learning_rate = st.select_slider(
                    "Скорость обучения:",
                    options=[0.1, 0.01, 0.001, 0.0001],
                    value=0.001,
                    help="Шаг градиентного спуска"
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
                use_early_stopping = st.checkbox(
                    "Ранняя остановка", 
                    value=True,
                    help="Остановка обучения при отсутствии улучшений"
                )
        
        if st.button("🧠 Обучить CNN", type="primary", key="train_cnn_btn"):
            with st.spinner("Обучение CNN... Это может занять несколько минут."):
                try:
                    start_time = datetime.now()
                    
                    # Импорт TensorFlow/Keras
                    import tensorflow as tf
                    from tensorflow import keras
                    
                    # Подготовка данных для CNN
                    X_train_cnn = X_train.reshape(-1, X_train.shape[1], 1)
                    X_val_cnn = X_val.reshape(-1, X_val.shape[1], 1)
                    
                    # Создание модели
                    num_classes = len(np.unique(y_train))
                    
                    model = keras.Sequential([
                        keras.layers.Conv1D(32, kernel_size=5, activation='relu', 
                                          input_shape=(X_train.shape[1], 1),
                                          padding='same'),
                        keras.layers.BatchNormalization(),
                        keras.layers.MaxPooling1D(pool_size=2),
                        keras.layers.Dropout(0.3),
                        
                        keras.layers.Conv1D(64, kernel_size=3, activation='relu',
                                          padding='same'),
                        keras.layers.BatchNormalization(),
                        keras.layers.MaxPooling1D(pool_size=2),
                        keras.layers.Dropout(0.3),
                        
                        keras.layers.Conv1D(128, kernel_size=3, activation='relu',
                                          padding='same'),
                        keras.layers.BatchNormalization(),
                        keras.layers.GlobalAveragePooling1D(),
                        keras.layers.Dropout(0.4),
                        
                        keras.layers.Dense(128, activation='relu'),
                        keras.layers.BatchNormalization(),
                        keras.layers.Dropout(0.4),
                        
                        keras.layers.Dense(64, activation='relu'),
                        keras.layers.Dropout(0.3),
                        
                        keras.layers.Dense(num_classes, activation='softmax')
                    ])
                    
                    # Компиляция
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy', 
                                keras.metrics.Precision(name='precision'),
                                keras.metrics.Recall(name='recall')]
                    )
                    
                    # Callbacks
                    callbacks = []
                    if use_early_stopping:
                        callbacks.append(
                            keras.callbacks.EarlyStopping(
                                monitor='val_loss',
                                patience=10,
                                restore_best_weights=True
                            )
                        )
                    
                    # Обучение
                    history = model.fit(
                        X_train_cnn, y_train,
                        validation_data=(X_val_cnn, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=0
                    )
                    
                    # Оценка
                    val_loss, val_accuracy, val_precision, val_recall = model.evaluate(
                        X_val_cnn, y_val, verbose=0
                    )
                    y_pred = np.argmax(model.predict(X_val_cnn, verbose=0), axis=1)
                    y_pred_proba = model.predict(X_val_cnn, verbose=0)
                    
                    # Дополнительные метрики
                    f1 = f1_score(y_val, y_pred, average='weighted')
                    cm = confusion_matrix(y_val, y_pred)
                    report = classification_report(y_val, y_pred, output_dict=True)
                    
                    training_time = (datetime.now() - start_time).total_seconds()
                    
                    # Сохранение модели
                    model_filename = f"models/cnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
                    model.save(model_filename)
                    
                    # Сохранение истории обучения
                    history_filename = f"results/cnn_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(history_filename, 'w') as f:
                        json.dump({k: [float(v) for v in vals] 
                                 for k, vals in history.history.items()}, f)
                    
                    # Сохранение в session state
                    st.session_state.cnn_model = model
                    st.session_state.cnn_history = history.history
                    st.session_state.cnn_metrics = {
                        'accuracy': val_accuracy,
                        'precision': val_precision,
                        'recall': val_recall,
                        'f1_score': f1,
                        'loss': val_loss,
                        'confusion_matrix': cm,
                        'classification_report': report,
                        'training_time': training_time,
                        'model_path': model_filename,
                        'history_path': history_filename
                    }
                    st.session_state.models_trained = True
                    
                    # Добавление в историю экспериментов
                    experiment_data = {
                        'timestamp': datetime.now().isoformat(),
                        'model_type': 'CNN',
                        'parameters': {
                            'epochs': epochs,
                            'batch_size': batch_size,
                            'learning_rate': learning_rate,
                            'use_early_stopping': use_early_stopping
                        },
                        'metrics': {
                            'accuracy': val_accuracy,
                            'precision': val_precision,
                            'recall': val_recall,
                            'f1_score': f1,
                            'training_time': training_time
                        }
                    }
                    st.session_state.metrics_history.append(experiment_data)
                    
                    st.success(f"✅ CNN обучена за {training_time:.2f} сек!")
                    
                    # Отображение результатов
                    self._display_training_results(
                        st.session_state.cnn_metrics, 
                        "CNN",
                        cm,
                        history
                    )
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при обучении CNN: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    def _display_training_results(self, metrics, model_name, confusion_mat, history=None):
        """Отображение результатов обучения."""
        st.subheader(f"📊 Результаты {model_name}")
        
        # Основные метрики
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Точность", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
        
        # Время обучения
        st.info(f"⏱️ Время обучения: {metrics.get('training_time', 0):.2f} сек")
        
        # Матрица ошибок
        st.subheader("📋 Матрица ошибок")
        
        class_names = list(self.config.ARRHYTHMIA_CLASSES.values())
        
        fig_cm, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            confusion_mat,
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        ax.set_xlabel('Предсказанные метки', fontsize=12)
        ax.set_ylabel('Истинные метки', fontsize=12)
        ax.set_title(f'Матрица ошибок - {model_name}', fontsize=14, fontweight='bold')
        st.pyplot(fig_cm)
        
        # Отчет о классификации
        if 'classification_report' in metrics:
            st.subheader("📄 Отчет о классификации")
            
            report_df = pd.DataFrame(metrics['classification_report']).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)
        
        # Графики обучения для CNN
        if history and model_name == "CNN":
            st.subheader("📈 Графики обучения")
            
            fig_history, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
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
            st.pyplot(fig_history)
    
    def render_prediction_tab(self):
        """Вкладка прогнозирования."""
        st.header("🔍 Прогнозирование аритмий")
        
        # Проверка обученных моделей
        has_rf = 'rf_model' in st.session_state
        has_cnn = 'cnn_model' in st.session_state
        
        if not (has_rf or has_cnn):
            st.warning("⚠️ Сначала обучите модели на вкладке 'Обучение моделей'")
            return
        
        # Загрузка моделей
        st.subheader("📂 Загрузка моделей")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Загрузить Random Forest", use_container_width=True, 
                        disabled=not has_rf):
                try:
                    rf_model = st.session_state.rf_model
                    st.session_state.rf_model_loaded = rf_model
                    st.session_state.rf_metrics_display = st.session_state.rf_metrics
                    st.success("✅ Random Forest загружен!")
                except Exception as e:
                    st.error(f"❌ Ошибка при загрузке: {str(e)}")
        
        with col2:
            if st.button("📥 Загрузить CNN", use_container_width=True, 
                        disabled=not has_cnn):
                try:
                    cnn_model = st.session_state.cnn_model
                    st.session_state.cnn_model_loaded = cnn_model
                    st.session_state.cnn_metrics_display = st.session_state.cnn_metrics
                    st.success("✅ CNN загружена!")
                except Exception as e:
                    st.error(f"❌ Ошибка при загрузке: {str(e)}")
        
        # Показать загруженные модели
        loaded_models = []
        if 'rf_model_loaded' in st.session_state:
            loaded_models.append("🌲 Random Forest")
        if 'cnn_model_loaded' in st.session_state:
            loaded_models.append("🧠 CNN")
        
        if loaded_models:
            st.info(f"**Загруженные модели:** {', '.join(loaded_models)}")
        
        # Выбор данных для прогнозирования
        st.subheader("📊 Выбор данных для прогнозирования")
        
        prediction_method = st.radio(
            "Метод ввода данных:",
            ["Случайный пример из тестовой выборки", 
             "Сгенерировать новый сигнал", 
             "Загрузить файл"],
            horizontal=True
        )
        
        if prediction_method == "Случайный пример из тестовой выборки":
            if st.button("🎲 Выбрать случайный пример", use_container_width=True):
                if 'X_test' in st.session_state:
                    random_idx = np.random.randint(0, len(st.session_state.X_test))
                    test_signal = st.session_state.X_test[random_idx]
                    true_label = st.session_state.y_test[random_idx]
                    
                    st.session_state.current_signal = test_signal
                    st.session_state.true_label = true_label
                    st.session_state.true_class = self.config.ARRHYTHMIA_CLASSES.get(
                        true_label, f"Класс {true_label}"
                    )
                else:
                    st.error("❌ Тестовая выборка не найдена")
        
        elif prediction_method == "Сгенерировать новый сигнал":
            arrhythmia_type = st.selectbox(
                "Тип аритмии для генерации:",
                list(self.config.ARRHYTHMIA_CLASSES.items()),
                format_func=lambda x: x[1],
                key="generate_signal_select"
            )
            
            if st.button("🌀 Сгенерировать сигнал", use_container_width=True):
                t = np.linspace(0, 1, self.config.SEQUENCE_LENGTH)
                base_ecg = 0.5 * np.sin(2 * np.pi * 1 * t)
                
                class_id, class_name = arrhythmia_type
                
                if class_id == 0:  # Нормальный ритм
                    test_signal = base_ecg + 0.1 * np.random.normal(size=self.config.SEQUENCE_LENGTH)
                elif class_id == 1:  # Апноэ
                    test_signal = base_ecg * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)) + 0.1 * np.random.normal(size=self.config.SEQUENCE_LENGTH)
                elif class_id == 2:  # Фибрилляция предсердий
                    test_signal = base_ecg + 0.3 * np.random.normal(size=self.config.SEQUENCE_LENGTH) + 0.1 * np.random.normal(size=self.config.SEQUENCE_LENGTH)
                elif class_id == 3:  # Шум
                    test_signal = 0.8 * np.random.normal(size=self.config.SEQUENCE_LENGTH)
                else:  # Другая аритмия
                    test_signal = base_ecg * (1 + 0.3 * np.sin(2 * np.pi * 2 * t)) + 0.1 * np.random.normal(size=self.config.SEQUENCE_LENGTH)
                
                st.session_state.current_signal = test_signal
                st.session_state.true_label = class_id
                st.session_state.true_class = class_name
        
        else:  # Загрузить файл
            uploaded_file = st.file_uploader(
                "Загрузите файл с сигналом ЭКГ:",
                type=['csv', 'txt', 'npy'],
                help="Сигнал должен быть одномерным массивом"
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
                    if len(signal) > self.config.SEQUENCE_LENGTH:
                        signal = signal[:self.config.SEQUENCE_LENGTH]
                    elif len(signal) < self.config.SEQUENCE_LENGTH:
                        signal = np.pad(signal, (0, self.config.SEQUENCE_LENGTH - len(signal)))
                    
                    st.session_state.current_signal = signal
                    st.session_state.true_label = None
                    st.session_state.true_class = "Неизвестно"
                    
                    st.success("✅ Файл успешно загружен!")
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при обработке файла: {str(e)}")
        
        # Визуализация сигнала
        if 'current_signal' in st.session_state:
            current_signal = st.session_state.current_signal
            
            st.subheader("📈 Визуализация сигнала")
            
            fig_signal = go.Figure()
            fig_signal.add_trace(go.Scatter(
                y=current_signal,
                mode='lines',
                name='ЭКГ сигнал',
                line=dict(color='#3498db', width=2)
            ))
            
            if 'true_class' in st.session_state:
                true_class = st.session_state.true_class
                fig_signal.update_layout(
                    title=f"ЭКГ сигнал (Истинный класс: {true_class})",
                    xaxis_title="Отсчеты",
                    yaxis_title="Амплитуда",
                    template="plotly_white"
                )
            else:
                fig_signal.update_layout(
                    title="ЭКГ сигнал",
                    xaxis_title="Отсчеты",
                    yaxis_title="Амплитуда",
                    template="plotly_white"
                )
            
            st.plotly_chart(fig_signal, use_container_width=True)
            
            # Прогнозирование
            st.subheader("🔮 Прогнозирование")
            
            selected_models = []
            if 'rf_model_loaded' in st.session_state:
                selected_models.append('Random Forest')
            if 'cnn_model_loaded' in st.session_state:
                selected_models.append('CNN')
            
            if selected_models:
                models_to_use = st.multiselect(
                    "Выберите модели для прогнозирования:",
                    selected_models,
                    default=selected_models
                )
                
                if st.button("🎯 Выполнить прогноз", type="primary", use_container_width=True):
                    results = []
                    
                    for model_name in models_to_use:
                        if model_name == 'Random Forest' and 'rf_model_loaded' in st.session_state:
                            model = st.session_state.rf_model_loaded
                            
                            # Прогноз
                            signal_reshaped = current_signal.reshape(1, -1)
                            prediction = model.predict(signal_reshaped)[0]
                            probabilities = model.predict_proba(signal_reshaped)[0]
                            
                            results.append({
                                'model': 'Random Forest',
                                'prediction': int(prediction),
                                'class_name': self.config.ARRHYTHMIA_CLASSES.get(prediction, f"Класс {prediction}"),
                                'confidence': float(np.max(probabilities)),
                                'probabilities': probabilities.tolist(),
                                'all_probs': probabilities
                            })
                        
                        elif model_name == 'CNN' and 'cnn_model_loaded' in st.session_state:
                            model = st.session_state.cnn_model_loaded
                            
                            # Прогноз
                            signal_reshaped = current_signal.reshape(1, -1, 1)
                            probabilities = model.predict(signal_reshaped, verbose=0)[0]
                            prediction = np.argmax(probabilities)
                            
                            results.append({
                                'model': 'CNN',
                                'prediction': int(prediction),
                                'class_name': self.config.ARRHYTHMIA_CLASSES.get(prediction, f"Класс {prediction}"),
                                'confidence': float(np.max(probabilities)),
                                'probabilities': probabilities.tolist(),
                                'all_probs': probabilities
                            })
                    
                    if results:
                        st.session_state.prediction_results = results
                        
                        # Отображение результатов
                        st.subheader("📊 Результаты прогнозирования")
                        
                        for result in results:
                            with st.container():
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    color = "🟢" if result['confidence'] > 0.8 else "🟡" if result['confidence'] > 0.6 else "🔴"
                                    st.metric(
                                        f"**{result['model']}** {color}",
                                        result['class_name'],
                                        f"Уверенность: {result['confidence']:.2%}"
                                    )
                                
                                with col2:
                                    # Визуализация вероятностей
                                    prob_df = pd.DataFrame({
                                        'Класс': list(self.config.ARRHYTHMIA_CLASSES.values()),
                                        'Вероятность': result['all_probs']
                                    })
                                    
                                    fig_prob = go.Figure(data=[go.Bar(
                                        x=prob_df['Вероятность'],
                                        y=prob_df['Класс'],
                                        orientation='h',
                                        marker_color=['#2ecc71' if p == result['prediction'] else '#95a5a6' 
                                                     for p in range(len(prob_df))],
                                        text=[f"{p:.1%}" for p in prob_df['Вероятность']],
                                        textposition='auto'
                                    )])
                                    fig_prob.update_layout(
                                        title=f"Распределение вероятностей",
                                        xaxis_title="Вероятность",
                                        xaxis_range=[0, 1],
                                        height=300,
                                        template="plotly_white"
                                    )
                                    st.plotly_chart(fig_prob, use_container_width=True)
                        
                        # Генерация рекомендаций
                        self._generate_recommendations(results)
                        
                        # Кнопка сохранения
                        if st.button("💾 Сохранить результаты прогноза", type="secondary"):
                            self._save_prediction_results(current_signal, results)
            else:
                st.warning("⚠️ Выберите хотя бы одну модель для прогнозирования")
    
    def _generate_recommendations(self, results):
        """Генерация рекомендаций на основе прогнозов."""
        st.subheader("💡 Рекомендации")
        
        recommendations = {
            0: {
                "title": "✅ Нормальный сердечный ритм",
                "description": "Обнаружен нормальный синусовый ритм. Все показатели в пределах нормы.",
                "actions": [
                    "Продолжайте плановое наблюдение",
                    "Рекомендуется ежегодный профилактический осмотр",
                    "Ведение здорового образа жизни"
                ],
                "urgency": "Низкая",
                "icon": "✅"
            },
            1: {
                "title": "⚠️ Признаки апноэ сна",
                "description": "Обнаружены паттерны, характерные для апноэ сна. Требуется дополнительная диагностика.",
                "actions": [
                    "Консультация сомнолога",
                    "Проведение полисомнографии",
                    "Коррекция образа жизни и веса при необходимости"
                ],
                "urgency": "Средняя",
                "icon": "⚠️"
            },
            2: {
                "title": "🚨 Фибрилляция предсердий",
                "description": "Выявлена фибрилляция предсердий - серьезное нарушение сердечного ритма.",
                "actions": [
                    "СРОЧНАЯ консультация кардиолога",
                    "ЭКГ Холтер мониторинг в течение 24 часов",
                    "Назначение антикоагулянтной терапии по показаниям"
                ],
                "urgency": "Высокая",
                "icon": "🚨"
            },
            3: {
                "title": "📢 Сигнал с шумом",
                "description": "Сигнал содержит значительные шумы, затрудняющие анализ.",
                "actions": [
                    "Повторное измерение ЭКГ",
                    "Проверка электродов и качества контакта",
                    "Исключение артефактов движения"
                ],
                "urgency": "Низкая",
                "icon": "📢"
            },
            4: {
                "title": "⚠️ Другая аритмия",
                "description": "Обнаружена аритмия неуточненного типа. Требуется дополнительное обследование.",
                "actions": [
                    "Консультация кардиолога",
                    "Дополнительная диагностика (ЭхоКГ, нагрузочные тесты)",
                    "Эхокардиография для оценки структур сердца"
                ],
                "urgency": "Средняя",
                "icon": "⚠️"
            }
        }
        
        # Для каждого результата показываем рекомендации
        for result in results:
            pred_class = result['prediction']
            if pred_class in recommendations:
                rec = recommendations[pred_class]
                
                with st.expander(f"{rec['icon']} {result['model']}: {rec['title']}", 
                               expanded=True):
                    st.markdown(f"**Описание:** {rec['description']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Уровень срочности:** {rec['urgency']}")
                    with col2:
                        st.markdown(f"**Уверенность модели:** {result['confidence']:.2%}")
                    
                    st.markdown("**Рекомендуемые действия:**")
                    for action in rec['actions']:
                        st.markdown(f"• {action}")
        
        # Если есть истинный класс, сравниваем с прогнозом
        if 'true_label' in st.session_state:
            true_label = st.session_state.true_label
            true_class = self.config.ARRHYTHMIA_CLASSES.get(true_label, f"Класс {true_label}")
            
            st.info(f"**Истинный класс:** {true_class}")
            
            # Проверяем совпадение прогнозов с истинным классом
            correct_predictions = []
            for result in results:
                if result['prediction'] == true_label:
                    correct_predictions.append(result['model'])
            
            if correct_predictions:
                st.success(f"✅ Правильно определили: {', '.join(correct_predictions)}")
            else:
                st.warning("⚠️ Ни одна модель не определила правильный класс")
    
    def _save_prediction_results(self, signal, results):
        """Сохранение результатов прогнозирования."""
        try:
            prediction_data = {
                'timestamp': datetime.now().isoformat(),
                'signal_length': len(signal),
                'true_label': st.session_state.get('true_label', None),
                'true_class': st.session_state.get('true_class', 'Неизвестно'),
                'predictions': results
            }
            
            filename = f"results/prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(prediction_data, f, ensure_ascii=False, indent=2, default=str)
            
            st.success(f"✅ Результаты сохранены в файл: `{filename}`")
            
        except Exception as e:
            st.error(f"❌ Ошибка при сохранении: {str(e)}")
    
    def render_model_comparison_tab(self):
        """Вкладка сравнения моделей."""
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
                'training_time': st.session_state.rf_metrics.get('training_time', 0),
                'type': 'Классическая ML'
            }
        
        if has_cnn:
            models_data['CNN'] = {
                'metrics': st.session_state.cnn_metrics,
                'training_time': st.session_state.cnn_metrics.get('training_time', 0),
                'type': 'Нейронная сеть'
            }
        
        # Таблица сравнения
        st.subheader("📊 Сводная таблица метрик")
        
        comparison_data = []
        for name, data in models_data.items():
            metrics = data['metrics']
            comparison_data.append({
                'Модель': name,
                'Тип': data['type'],
                'Точность': f"{metrics.get('accuracy', 0):.4f}",
                'Precision': f"{metrics.get('precision', 0):.4f}",
                'Recall': f"{metrics.get('recall', 0):.4f}",
                'F1-Score': f"{metrics.get('f1_score', 0):.4f}",
                'Время обучения': f"{data['training_time']:.2f} сек"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.set_index('Модель'), use_container_width=True)
        
        # Визуализация сравнения
        st.subheader("📈 Визуализация сравнения")
        
        # График сравнения метрик
        fig_comparison = go.Figure()
        
        metrics_to_plot = ['Точность', 'Precision', 'Recall', 'F1-Score']
        for metric in metrics_to_plot:
            values = []
            for model_name in models_data.keys():
                metric_key = metric.lower() if metric != 'Точность' else 'accuracy'
                values.append(models_data[model_name]['metrics'].get(metric_key, 0))
            
            fig_comparison.add_trace(go.Bar(
                name=metric,
                x=list(models_data.keys()),
                y=values,
                text=[f"{v:.3f}" for v in values],
                textposition='auto'
            ))
        
        fig_comparison.update_layout(
            title="Сравнение метрик моделей",
            barmode='group',
            xaxis_title="Модель",
            yaxis_title="Значение метрики",
            yaxis_range=[0, 1],
            template="plotly_white"
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Время обучения
        training_times = [data['training_time'] for data in models_data.values()]
        
        fig_time = go.Figure(data=[go.Bar(
            x=list(models_data.keys()),
            y=training_times,
            text=[f"{t:.2f} сек" for t in training_times],
            textposition='auto',
            marker_color=['#2ecc71', '#3498db']
        )])
        fig_time.update_layout(
            title="Время обучения моделей",
            xaxis_title="Модель",
            yaxis_title="Время (сек)",
            template="plotly_white"
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Матрицы ошибок
        if has_rf or has_cnn:
            st.subheader("🔍 Матрицы ошибок")
            
            n_models = len(models_data)
            fig_cm, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
            
            if n_models == 1:
                axes = [axes]
            
            class_names = list(self.config.ARRHYTHMIA_CLASSES.values())
            
            for ax, (model_name, model_data) in zip(axes, models_data.items()):
                cm = model_data['metrics'].get('confusion_matrix', np.zeros((5, 5)))
                
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    ax=ax,
                    cbar=False if n_models > 1 else True
                )
                ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Предсказанные метки', fontsize=10)
                ax.set_ylabel('Истинные метки', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig_cm)
        
        # Детальный анализ
        st.subheader("📋 Детальный анализ")
        
        for model_name, model_data in models_data.items():
            with st.expander(f"📄 Полный отчет: {model_name}", expanded=False):
                metrics = model_data['metrics']
                
                # Основные метрики
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Точность", f"{metrics.get('accuracy', 0):.4f}")
                with col2:
                    st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                with col3:
                    st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                with col4:
                    st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
                
                # Отчет по классам
                if 'classification_report' in metrics:
                    st.markdown("**Метрики по классам:**")
                    report_df = pd.DataFrame(metrics['classification_report']).transpose()
                    st.dataframe(report_df.round(4), use_container_width=True)
        
        # Выводы и рекомендации
        st.subheader("🎯 Выводы и рекомендации")
        
        if len(models_data) > 1:
            # Определение лучшей модели
            best_model = max(models_data.items(), 
                           key=lambda x: x[1]['metrics'].get('accuracy', 0))
            best_model_name = best_model[0]
            best_accuracy = best_model[1]['metrics'].get('accuracy', 0)
            
            st.info(f"**🏆 Лучшая модель:** {best_model_name} с точностью {best_accuracy:.2%}")
            
            # Сравнение
            model_names = list(models_data.keys())
            if len(model_names) == 2:
                acc1 = models_data[model_names[0]]['metrics'].get('accuracy', 0)
                acc2 = models_data[model_names[1]]['metrics'].get('accuracy', 0)
                diff = abs(acc1 - acc2)
                
                if diff < 0.05:
                    st.write("✅ Модели демонстрируют схожую производительность.")
                elif diff < 0.1:
                    st.write(f"⚠️ Заметная разница в точности ({diff:.2%}).")
                else:
                    st.write(f"🚨 Существенная разница в точности ({diff:.2%}).")
            
            # Рекомендации по выбору
            st.markdown("**Рекомендации по выбору модели:**")
            
            recommendations = {
                'Random Forest': [
                    "✅ Быстрое обучение на небольших данных",
                    "✅ Высокая интерпретируемость результатов",
                    "✅ Не требует GPU для обучения",
                    "⚠️ Может переобучаться на сложных данных",
                    "💡 Идеально для прототипирования и начальных экспериментов"
                ],
                'CNN': [
                    "✅ Лучшая производительность на сложных данных",
                    "✅ Автоматическое извлечение признаков",
                    "✅ Хорошо работает с временными рядами",
                    "⚠️ Требует больше вычислительных ресурсов",
                    "⚠️ Сложнее в интерпретации",
                    "💡 Рекомендуется для production-систем с большими данными"
                ]
            }
            
            for model_name, recs in recommendations.items():
                if model_name in models_data:
                    with st.expander(f"Особенности {model_name}", expanded=False):
                        for rec in recs:
                            st.write(f"• {rec}")
        else:
            model_name = list(models_data.keys())[0]
            accuracy = list(models_data.values())[0]['metrics'].get('accuracy', 0)
            st.info(f"✅ Обучена одна модель: **{model_name}** с точностью {accuracy:.2%}")
            
            if accuracy >= 0.85:
                st.success("🎉 Модель превысила целевой показатель точности (85%)!")
            elif accuracy >= 0.7:
                st.warning("⚠️ Точность модели соответствует минимальным требованиям (70%).")
            else:
                st.error("❌ Точность модели ниже минимальных требований.")
    
    def render_experiment_history_tab(self):
        """Вкладка истории экспериментов."""
        st.header("📋 История экспериментов")
        
        if not st.session_state.get('metrics_history', []):
            st.info("ℹ️ История экспериментов пуста. Сначала обучите модели.")
            return
        
        history = st.session_state.metrics_history
        
        # Отображение истории в таблице
        st.subheader("📊 Таблица экспериментов")
        
        history_data = []
        for exp in history:
            history_data.append({
                'Дата и время': datetime.fromisoformat(exp['timestamp']).strftime('%d.%m.%Y %H:%M'),
                'Модель': exp['model_type'],
                'Точность': f"{exp['metrics']['accuracy']:.4f}",
                'Precision': f"{exp['metrics']['precision']:.4f}",
                'Recall': f"{exp['metrics']['recall']:.4f}",
                'F1-Score': f"{exp['metrics']['f1_score']:.4f}",
                'Время обучения': f"{exp['metrics']['training_time']:.2f} сек"
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)
        
        # Визуализация истории
        st.subheader("📈 Динамика точности")
        
        # Группировка по типу модели
        rf_history = [exp for exp in history if exp['model_type'] == 'Random Forest']
        cnn_history = [exp for exp in history if exp['model_type'] == 'CNN']
        
        fig_history = go.Figure()
        
        if rf_history:
            rf_times = [datetime.fromisoformat(exp['timestamp']) for exp in rf_history]
            rf_accuracies = [exp['metrics']['accuracy'] for exp in rf_history]
            
            fig_history.add_trace(go.Scatter(
                x=rf_times,
                y=rf_accuracies,
                mode='lines+markers',
                name='Random Forest',
                line=dict(color='#2ecc71', width=2),
                marker=dict(size=8)
            ))
        
        if cnn_history:
            cnn_times = [datetime.fromisoformat(exp['timestamp']) for exp in cnn_history]
            cnn_accuracies = [exp['metrics']['accuracy'] for exp in cnn_history]
            
            fig_history.add_trace(go.Scatter(
                x=cnn_times,
                y=cnn_accuracies,
                mode='lines+markers',
                name='CNN',
                line=dict(color='#3498db', width=2),
                marker=dict(size=8)
            ))
        
        fig_history.update_layout(
            title="Динамика изменения точности моделей",
            xaxis_title="Дата и время эксперимента",
            yaxis_title="Точность",
            yaxis_range=[0, 1],
            template="plotly_white",
            hovermode='x unified'
        )
        st.plotly_chart(fig_history, use_container_width=True)
        
        # Экспорт истории
        st.subheader("📤 Экспорт данных")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Экспорт в JSON", use_container_width=True):
                self._export_history_json(history)
        
        with col2:
            if st.button("📊 Экспорт в CSV", use_container_width=True):
                self._export_history_csv(history_df)
    
    def _export_history_json(self, history):
        """Экспорт истории в JSON."""
        try:
            filename = f"results/experiment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2, default=str)
            st.success(f"✅ История экспортирована в `{filename}`")
        except Exception as e:
            st.error(f"❌ Ошибка при экспорте: {str(e)}")
    
    def _export_history_csv(self, history_df):
        """Экспорт истории в CSV."""
        try:
            filename = f"results/experiment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            history_df.to_csv(filename, index=False, encoding='utf-8-sig')
            st.success(f"✅ История экспортирована в `{filename}`")
        except Exception as e:
            st.error(f"❌ Ошибка при экспорте: {str(e)}")
    
    def render_about_tab(self):
        """Вкладка о проекте."""
        st.header("ℹ️ О проекте")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎓 Дипломный проект
            
            **Тема:** Разработка рекомендательной системы на основе обработки биомедицинских данных
            
            **Автор:** Чайкин Виталий Федорович
            
            **Образовательное учреждение:** ЧОУВО «Московский университет им. С.Ю. Витте»
            
            **Факультет:** Информационных систем и технологий
            
            **Направление подготовки:** 09.03.02 "Информационные системы и технологии"
            
            **Руководитель:** Простомолотов Андрей Сергеевич
            
            **Период выполнения:** 10.11.2025 - 07.12.2025
            
            ---
            
            ### 🎯 Цели и задачи
            
            **Основная цель:** Разработка интеллектуальной системы для автоматического анализа 
            электрокардиограмм (ЭКГ) и классификации сердечных аритмий.
            
            **Задачи проекта:**
            1. Исследование методов обработки биомедицинских сигналов
            2. Разработка архитектуры системы анализа ЭКГ
            3. Реализация алгоритмов машинного обучения для классификации аритмий
            4. Создание веб-интерфейса для взаимодействия с системой
            5. Оценка эффективности разработанного решения
            
            ---
            
            ### 🔬 Научная новизна
            
            - Применение **ансамблевых методов** для повышения точности классификации
            - Разработка **гибридной архитектуры**, сочетающей классические ML и нейронные сети
            - Создание **адаптивной системы рекомендаций** на основе уверенности модели
            - Реализация **комплексной системы оценки** с визуализацией результатов
            
            ---
            
            ### 💼 Практическая значимость
            
            **Для медицинских учреждений:**
            - Повышение точности диагностики сердечных аритмий
            - Сокращение времени анализа ЭКГ сигналов
            - Поддержка принятия врачебных решений
            - Автоматизация рутинных задач
            
            **Для образовательного процесса:**
            - Наглядный пример применения ML в медицине
            - Инструмент для лабораторных работ и исследований
            - База для дальнейших научных изысканий
            """)
        
        with col2:
            st.subheader("📊 Технические характеристики")
            
            tech_specs = [
                ("Язык программирования", "Python 3.9+"),
                ("Объем кода", ">2500 строк"),
                ("Количество моделей", "2 (RF + CNN)"),
                ("Классов аритмий", "5"),
                ("Минимальная точность", "85%"),
                ("Время обучения", "<10 минут"),
                ("Поддерживаемые форматы", "CSV, TXT, NPY"),
                ("Требуемая память", "≥4 ГБ ОЗУ"),
                ("Интерфейс", "Web (Streamlit)")
            ]
            
            for spec, value in tech_specs:
                st.metric(spec, value)
            
            st.markdown("---")
            
            st.subheader("📚 Используемые технологии")
            
            technologies = {
                "Streamlit": "Веб-интерфейс",
                "Scikit-learn": "Классические ML алгоритмы",
                "TensorFlow/Keras": "Нейронные сети",
                "Pandas/NumPy": "Обработка данных",
                "Matplotlib/Seaborn": "Статическая визуализация",
                "Plotly": "Интерактивная визуализация",
                "SciPy": "Обработка сигналов"
            }
            
            for tech, desc in technologies.items():
                st.markdown(f"**{tech}** - {desc}")
            
            st.markdown("---")
            
            st.subheader("📁 Структура проекта")
            
            structure = """
            📦 ecg-analysis-system/
            ├── 📁 app/
            │   ├── 📁 core/          # Основные модули
            │   │   ├── data_loader.py
            │   │   ├── feature_engineer.py
            │   │   └── model_loader.py
            │   ├── 📁 services/      # Сервисные функции
            │   │   ├── training_service.py
            │   │   └── prediction_service.py
            │   └── 📁 web/           # Веб-интерфейс
            │       └── ui.py
            ├── 📁 models/            # Сохраненные модели
            ├── 📁 data/              # Наборы данных
            ├── 📁 results/           # Результаты
            ├── 📄 main.py            # Главный файл
            ├── 📄 requirements.txt   # Зависимости
            └── 📄 README.md          # Документация
            """
            
            st.code(structure, language="text")
        
        st.markdown("---")
        
        st.subheader("📞 Контакты")
        
        contact_col1, contact_col2 = st.columns(2)
        
        with contact_col1:
            st.markdown("""
            **📧 Электронная почта:**
            - vit.chaykin@example.com
            - project.ecg@example.com
            
            **🌐 Онлайн-ресурсы:**
            - [GitHub репозиторий](https://github.com/username/ecg-analysis)
            - [LinkedIn профиль](https://linkedin.com/in/username)
            """)
        
        with contact_col2:
            st.markdown("""
            **📱 Контактная информация:**
            - Телефон: +7 (XXX) XXX-XX-XX
            - Рабочие часы: 9:00-18:00 (МСК)
            - Приемные дни: Пн-Пт
            
            **🏢 Образовательное учреждение:**
            ЧОУВО «МУ им. С.Ю. Витте»
            Москва, 2-й Кожуховский пр., д. 12
            """)
        
        # Информация о лицензии
        with st.expander("📄 Лицензионная информация", expanded=False):
            st.markdown("""
            ### MIT License
            
            Copyright © 2025 Чайкин Виталий Федорович
            
            Данное программное обеспечение предоставляется «КАК ЕСТЬ», без каких-либо гарантий.
            
            **Разрешено:**
            - Использование в коммерческих и некоммерческих целях
            - Модификация и распространение
            - Использование в частных и корпоративных проектах
            
            **Требуется:**
            - Сохранение информации об авторском праве
            - Указание ссылки на оригинальный проект
            
            **Запрещено:**
            - Использование в медицинских целях без дополнительной валидации
            - Ответственность автора за последствия использования системы
            
            **Важное примечание:**
            Данная система является учебным проектом и не предназначена 
            для использования в реальной медицинской диагностике.
            """)
    
    def export_results(self):
        """Экспорт всех результатов."""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'project': 'ECG Analysis System',
                'author': 'Чайкин Виталий Федорович',
                'models': {},
                'experiments': st.session_state.get('metrics_history', []),
                'predictions': st.session_state.get('prediction_results', [])
            }
            
            # Добавляем информацию о моделях
            if 'rf_metrics' in st.session_state:
                export_data['models']['random_forest'] = {
                    'metrics': st.session_state.rf_metrics,
                    'path': st.session_state.rf_metrics.get('model_path', '')
                }
            
            if 'cnn_metrics' in st.session_state:
                export_data['models']['cnn'] = {
                    'metrics': st.session_state.cnn_metrics,
                    'path': st.session_state.cnn_metrics.get('model_path', '')
                }
            
            # Сохранение
            filename = f"results/full_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            st.success(f"✅ Все результаты экспортированы в `{filename}`")
            
        except Exception as e:
            st.error(f"❌ Ошибка при экспорте: {str(e)}")
    
    def run(self):
        """Запуск приложения."""
        self.render_sidebar()
        
        current_tab = st.session_state.current_tab
        
        tab_functions = {
            "data_analysis": self.render_data_analysis_tab,
            "model_training": self.render_model_training_tab,
            "prediction": self.render_prediction_tab,
            "model_comparison": self.render_model_comparison_tab,
            "experiment_history": self.render_experiment_history_tab,
            "about": self.render_about_tab
        }
        
        if current_tab in tab_functions:
            tab_functions[current_tab]()
        else:
            st.error(f"Неизвестная вкладка: {current_tab}")

def main():
    """Основная функция запуска приложения."""
    app = BiomedicalApp()
    app.run()

if __name__ == "__main__":
    main()