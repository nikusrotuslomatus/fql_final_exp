#!/usr/bin/env python3
"""
Анализ результатов экспериментов на Tesla T4
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def parse_csv_logs(results_dir):
    """Парсит CSV логи из экспериментов"""
    results_dir = Path(results_dir)
    
    all_data = []
    
    # Поиск всех CSV файлов
    for exp_dir in results_dir.glob("exp/fql/*/"):
        if exp_dir.is_dir():
            eval_csv = exp_dir / "eval.csv"
            train_csv = exp_dir / "train.csv"
            
            if eval_csv.exists():
                try:
                    df = pd.read_csv(eval_csv)
                    
                    # Извлекаем информацию из пути
                    parts = exp_dir.name.split('_')
                    if len(parts) >= 3:
                        config_type = parts[-2]  # baseline или kl_awfm
                        seed = parts[-1]
                        
                        # Определяем среду по логам
                        if "antmaze" in str(exp_dir):
                            env_name = "antmaze-large-play-v2"
                        elif "puzzle" in str(exp_dir):
                            env_name = "puzzle-3x3-play-singletask-v0"
                        else:
                            env_name = "unknown"
                        
                        df['config'] = config_type
                        df['seed'] = seed
                        df['environment'] = env_name
                        
                        all_data.append(df)
                        
                except Exception as e:
                    print(f"Ошибка чтения {eval_csv}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return None

def analyze_performance(df):
    """Анализирует производительность по конфигурациям"""
    if df is None or df.empty:
        print("❌ Нет данных для анализа")
        return
    
    print("📊 АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 50)
    
    # Группировка по среде и конфигурации
    for env in df['environment'].unique():
        env_data = df[df['environment'] == env]
        print(f"\n🎯 Среда: {env}")
        print("-" * 30)
        
        # Финальная производительность
        final_scores = env_data.groupby(['config', 'seed']).tail(1)
        
        if 'eval/return' in final_scores.columns:
            score_col = 'eval/return'
        elif 'return' in final_scores.columns:
            score_col = 'return'
        else:
            print("  ❌ Колонка с результатами не найдена")
            continue
        
        summary = final_scores.groupby('config')[score_col].agg(['mean', 'std', 'count'])
        
        for config in summary.index:
            mean_score = summary.loc[config, 'mean']
            std_score = summary.loc[config, 'std']
            count = summary.loc[config, 'count']
            
            print(f"  {config:12s}: {mean_score:6.1f} ± {std_score:4.1f} ({count} сидов)")
        
        # Сравнение конфигураций
        if len(summary) >= 2:
            configs = list(summary.index)
            if 'baseline' in configs and 'kl_awfm' in configs:
                baseline_mean = summary.loc['baseline', 'mean']
                kl_awfm_mean = summary.loc['kl_awfm', 'mean']
                improvement = ((kl_awfm_mean - baseline_mean) / baseline_mean) * 100
                
                if improvement > 0:
                    print(f"  🟢 Улучшение KL+AWFM: +{improvement:.1f}%")
                else:
                    print(f"  🔴 Ухудшение KL+AWFM: {improvement:.1f}%")

def create_performance_plots(df, output_dir):
    """Создает графики производительности"""
    if df is None or df.empty:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Настройка стиля
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Определяем колонку с результатами
    if 'eval/return' in df.columns:
        score_col = 'eval/return'
    elif 'return' in df.columns:
        score_col = 'return'
    else:
        print("❌ Не найдена колонка с результатами для графиков")
        return
    
    # График 1: Кривые обучения
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    environments = df['environment'].unique()
    
    for i, env in enumerate(environments):
        if i >= 2:  # Только 2 графика
            break
            
        env_data = df[df['environment'] == env]
        
        # Группировка по конфигурации
        for config in env_data['config'].unique():
            config_data = env_data[env_data['config'] == config]
            
            # Среднее по сидам
            if 'step' in config_data.columns:
                step_col = 'step'
            elif 'epoch' in config_data.columns:
                step_col = 'epoch'
            else:
                step_col = config_data.columns[0]
            
            mean_curve = config_data.groupby(step_col)[score_col].mean()
            std_curve = config_data.groupby(step_col)[score_col].std()
            
            axes[i].plot(mean_curve.index, mean_curve.values, 
                        label=f'{config}', linewidth=2)
            axes[i].fill_between(mean_curve.index, 
                               mean_curve.values - std_curve.values,
                               mean_curve.values + std_curve.values,
                               alpha=0.2)
        
        axes[i].set_title(f'{env.replace("-", " ").title()}')
        axes[i].set_xlabel('Шаги обучения')
        axes[i].set_ylabel('Средний возврат')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # График 2: Финальные результаты
    fig, ax = plt.subplots(figsize=(10, 6))
    
    final_scores = df.groupby(['environment', 'config', 'seed']).tail(1)
    
    # Box plot
    sns.boxplot(data=final_scores, x='environment', y=score_col, hue='config', ax=ax)
    ax.set_title('Финальная производительность по конфигурациям')
    ax.set_xlabel('Среда')
    ax.set_ylabel('Финальный результат')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'final_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Графики сохранены в: {output_dir}")

def generate_report(df, results_dir):
    """Генерирует отчет о результатах"""
    report_path = Path(results_dir) / "tesla_t4_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Отчет о результатах Tesla T4 экспериментов\n\n")
        
        f.write("## Конфигурация\n")
        f.write("- GPU: Tesla T4 16GB\n")
        f.write("- Batch size: 1024\n")
        f.write("- Шагов обучения: 200,000\n")
        f.write("- Архитектура: (512, 512, 512, 512)\n\n")
        
        if df is not None and not df.empty:
            f.write("## Результаты по средам\n\n")
            
            for env in df['environment'].unique():
                env_data = df[df['environment'] == env]
                f.write(f"### {env}\n\n")
                
                # Финальные результаты
                final_scores = env_data.groupby(['config', 'seed']).tail(1)
                
                if 'eval/return' in final_scores.columns:
                    score_col = 'eval/return'
                elif 'return' in final_scores.columns:
                    score_col = 'return'
                else:
                    continue
                
                summary = final_scores.groupby('config')[score_col].agg(['mean', 'std', 'count'])
                
                f.write("| Конфигурация | Среднее | Стд. откл. | Сиды |\n")
                f.write("|--------------|---------|------------|------|\n")
                
                for config in summary.index:
                    mean_score = summary.loc[config, 'mean']
                    std_score = summary.loc[config, 'std']
                    count = summary.loc[config, 'count']
                    f.write(f"| {config} | {mean_score:.1f} | {std_score:.1f} | {count} |\n")
                
                f.write("\n")
        
        f.write("## Графики\n\n")
        f.write("- `learning_curves.png` - Кривые обучения\n")
        f.write("- `final_performance.png` - Финальная производительность\n\n")
        
        f.write("## Рекомендации\n\n")
        f.write("1. Проверьте утилизацию GPU с помощью `python monitor_gpu.py analyze`\n")
        f.write("2. Если результаты неудовлетворительные, попробуйте увеличить batch_size до 1536\n")
        f.write("3. Для более длительных экспериментов увеличьте offline_steps до 300,000\n")
    
    print(f"📄 Отчет сохранен: {report_path}")

def main():
    if len(sys.argv) < 2:
        print("❌ Использование: python analyze_tesla_results.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.exists(results_dir):
        print(f"❌ Директория не найдена: {results_dir}")
        sys.exit(1)
    
    print(f"🔍 Анализ результатов в: {results_dir}")
    
    # Парсинг данных
    df = parse_csv_logs(results_dir)
    
    if df is None:
        print("❌ Не найдены CSV файлы с результатами")
        sys.exit(1)
    
    print(f"✅ Загружено {len(df)} записей")
    
    # Анализ производительности
    analyze_performance(df)
    
    # Создание графиков
    plots_dir = Path(results_dir) / "plots"
    create_performance_plots(df, plots_dir)
    
    # Генерация отчета
    generate_report(df, results_dir)
    
    print(f"\n🎉 Анализ завершен!")
    print(f"📊 Результаты в: {results_dir}")

if __name__ == "__main__":
    main() 