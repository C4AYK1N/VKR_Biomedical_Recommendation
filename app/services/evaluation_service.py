"""
Сервис для оценки и сравнения моделей.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import json
import os

class EvaluationService:
    """Сервис для оценки моделей."""
    
    def __init__(self, config):
        self.config = config
        
    def evaluate_model(self, model, X_test, y_test, model_type='sklearn'):
        """
        Полная оценка модели.
        """
        # Получаем предсказания
        if model_type == 'keras':
            y_pred_proba = model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            else:
                y_pred_proba = None
        
        # Рассчитываем метрики
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'predictions': y_pred.tolist(),
            'true_labels': y_test.tolist()
        }
        
        # ROC AUC если есть вероятности
        if y_pred_proba is not None and len(np.unique(y_test)) > 1:
            try:
                y_test_bin = label_binarize(y_test, classes=range(len(self.config.ARRHYTHMIA_CLASSES)))
                metrics['roc_auc_macro'] = roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')
                metrics['roc_auc_weighted'] = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')
            except:
                metrics['roc_auc_macro'] = None
                metrics['roc_auc_weighted'] = None
        
        return metrics
    
    def compare_models(self, models_info, X_test, y_test):
        """
        Сравнение нескольких моделей.
        
        Args:
            models_info: Список словарей с информацией о моделях
                        [{'name': 'Random Forest', 'model': model, 'type': 'sklearn'}, ...]
        """
        results = {}
        
        for model_info in models_info:
            metrics = self.evaluate_model(
                model_info['model'], 
                X_test, 
                y_test, 
                model_type=model_info.get('type', 'sklearn')
            )
            
            results[model_info['name']] = {
                'metrics': metrics,
                'type': model_info.get('type', 'sklearn'),
                'accuracy': metrics['accuracy']
            }
        
        # Создаем сравнительную таблицу
        comparison_df = pd.DataFrame([
            {
                'Модель': name,
                'Точность': results[name]['metrics']['accuracy'],
                'Precision (weighted)': results[name]['metrics']['precision_weighted'],
                'Recall (weighted)': results[name]['metrics']['recall_weighted'],
                'F1-Score (weighted)': results[name]['metrics']['f1_weighted'],
                'Тип модели': results[name]['type']
            }
            for name in results.keys()
        ])
        
        return results, comparison_df
    
    def plot_comparison(self, comparison_df, save_path=None):
        """Визуализация сравнения моделей."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # График точности
        axes[0, 0].bar(comparison_df['Модель'], comparison_df['Точность'])
        axes[0, 0].set_title('Сравнение точности моделей')
        axes[0, 0].set_ylabel('Точность')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # График F1-Score
        axes[0, 1].bar(comparison_df['Модель'], comparison_df['F1-Score (weighted)'])
        axes[0, 1].set_title('Сравнение F1-Score моделей')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Heatmap метрик
        metrics_to_plot = ['Точность', 'Precision (weighted)', 'Recall (weighted)', 'F1-Score (weighted)']
        heatmap_data = comparison_df[metrics_to_plot].values.T
        
        im = axes[1, 0].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        axes[1, 0].set_title('Heatmap метрик')
        axes[1, 0].set_xticks(range(len(comparison_df['Модель'])))
        axes[1, 0].set_xticklabels(comparison_df['Модель'], rotation=45)
        axes[1, 0].set_yticks(range(len(metrics_to_plot)))
        axes[1, 0].set_yticklabels(metrics_to_plot)
        plt.colorbar(im, ax=axes[1, 0])
        
        # Радарная диаграмма
        axes[1, 1].axis('off')  # Для радиарной диаграммы нужно больше места
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrices(self, results, class_names, save_path=None):
        """Визуализация матриц ошибок для всех моделей."""
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, model_results) in zip(axes, results.items()):
            cm = np.array(model_results['metrics']['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=class_names,
                       yticklabels=class_names)
            ax.set_title(f'Матрица ошибок: {model_name}')
            ax.set_xlabel('Предсказанные метки')
            ax.set_ylabel('Истинные метки')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig