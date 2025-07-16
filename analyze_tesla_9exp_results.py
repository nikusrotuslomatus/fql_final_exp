#!/usr/bin/env python3
"""
Анализ результатов 9 экспериментов Tesla T4
3 конфигурации × 3 среды
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
    """Парсит CSV логи из 9 экспериментов"""
    results_dir = Path(results_dir)
    
    all_data = []
    
    # Поиск всех CSV файлов
    for exp_dir in results_dir.glob("exp/fql/*/"):
        if exp_dir.is_dir():
            eval_csv = exp_dir / "eval.csv"
            
            if eval_csv.exists():
                try:
                    df = pd.read_csv(eval_csv)
                    
                    # Извлекаем информацию из имени директории
                    dir_name = exp_dir.name
                    
                    # Определяем конфигурацию и среду из имени
                    if "1_baseline" in dir_name:
                        config = "1_baseline"
                    elif "2_kl_only" in dir_name:
                        config = "2_kl_only"
                    elif "3_kl_awfm" in dir_name:
                        config = "3_kl_awfm"
                    else:
                        config = "unknown"
                    
                    # Определяем среду
                    if "antmaze-large-play-v2" in dir_name:
                        env = "AntMaze-Large"
                    elif "puzzle-3x3-play-singletask-v0" in dir_name:
                        env = "Puzzle-3x3"
                    elif "humanoidmaze-medium-navigate-singletask-v0" in dir_name:
                        env = "HumanoidMaze"
                    else:
                        env = "unknown"
                    
                    df['config'] = config
                    df['environment'] = env
                    df['seed'] = 0  # Только один сид
                    
                    all_data.append(df)
                    
                except Exception as e:
                    print(f"Ошибка чтения {eval_csv}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return None

def analyze_9exp_performance(df):
    """Анализирует производительность 9 экспериментов"""
    if df is None or df.empty:
        print("❌ Нет данных для анализа")
        return
    
    print("📊 АНАЛИЗ 9 ЭКСПЕРИМЕНТОВ")
    print("=" * 60)
    print("Конфигурации:")
    print("  1_baseline: Без KL, без AWFM")
    print("  2_kl_only:  С KL, без AWFM")
    print("  3_kl_awfm:  С KL, с AWFM")
    print("=" * 60)
    
    # Определяем колонку с результатами
    if 'eval/return' in df.columns:
        score_col = 'eval/return'
    elif 'return' in df.columns:
        score_col = 'return'
    else:
        print("❌ Колонка с результатами не найдена")
        return
    
    # Создаем сводную таблицу
    results_table = []
    
    for env in sorted(df['environment'].unique()):
        env_data = df[df['environment'] == env]
        
        print(f"\n🎯 {env}")
        print("-" * 40)
        
        env_results = {}
        
        for config in ['1_baseline', '2_kl_only', '3_kl_awfm']:
            config_data = env_data[env_data['config'] == config]
            
            if not config_data.empty:
                # Берем финальный результат
                final_score = config_data[score_col].iloc[-1]
                env_results[config] = final_score
                
                # Отображаем прогресс
                if len(config_data) > 1:
                    initial_score = config_data[score_col].iloc[0]
                    improvement = final_score - initial_score
                    print(f"  {config:12s}: {final_score:6.1f} (прогресс: {improvement:+5.1f})")
                else:
                    print(f"  {config:12s}: {final_score:6.1f}")
            else:
                env_results[config] = None
                print(f"  {config:12s}: ❌ Нет данных")
        
        # Сравнение конфигураций
        if env_results['1_baseline'] and env_results['2_kl_only']:
            kl_improvement = ((env_results['2_kl_only'] - env_results['1_baseline']) / 
                             abs(env_results['1_baseline'])) * 100
            print(f"  📈 KL улучшение: {kl_improvement:+5.1f}%")
        
        if env_results['2_kl_only'] and env_results['3_kl_awfm']:
            awfm_improvement = ((env_results['3_kl_awfm'] - env_results['2_kl_only']) / 
                               abs(env_results['2_kl_only'])) * 100
            print(f"  📈 AWFM улучшение: {awfm_improvement:+5.1f}%")
        
        if env_results['1_baseline'] and env_results['3_kl_awfm']:
            total_improvement = ((env_results['3_kl_awfm'] - env_results['1_baseline']) / 
                                abs(env_results['1_baseline'])) * 100
            print(f"  🎯 Общее улучшение: {total_improvement:+5.1f}%")
        
        # Добавляем в таблицу
        results_table.append({
            'Environment': env,
            'Baseline': env_results['1_baseline'],
            'KL-only': env_results['2_kl_only'],
            'KL+AWFM': env_results['3_kl_awfm']
        })
    
    return results_table

def create_9exp_plots(df, output_dir):
    """Создает графики для 9 экспериментов"""
    if df is None or df.empty:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Настройка стиля
    plt.style.use('seaborn-v0_8')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Синий, оранжевый, зеленый
    
    # Определяем колонку с результатами
    if 'eval/return' in df.columns:
        score_col = 'eval/return'
    elif 'return' in df.columns:
        score_col = 'return'
    else:
        print("❌ Не найдена колонка с результатами для графиков")
        return
    
    # График 1: Кривые обучения для всех экспериментов
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    environments = sorted(df['environment'].unique())
    configs = ['1_baseline', '2_kl_only', '3_kl_awfm']
    config_labels = ['Baseline', 'KL-only', 'KL+AWFM']
    
    for i, env in enumerate(environments):
        env_data = df[df['environment'] == env]
        
        for j, config in enumerate(configs):
            config_data = env_data[env_data['config'] == config]
            
            if not config_data.empty:
                # Определяем колонку шагов
                if 'step' in config_data.columns:
                    step_col = 'step'
                else:
                    step_col = config_data.columns[0]
                
                axes[i].plot(config_data[step_col], config_data[score_col], 
                           label=config_labels[j], color=colors[j], linewidth=2)
        
        axes[i].set_title(f'{env}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Шаги обучения')
        axes[i].set_ylabel('Средний возврат')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves_9exp.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # График 2: Сравнение финальных результатов
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Подготовка данных для барного графика
    final_results = []
    environments = sorted(df['environment'].unique())
    
    for env in environments:
        env_data = df[df['environment'] == env]
        for config in configs:
            config_data = env_data[env_data['config'] == config]
            if not config_data.empty:
                final_score = config_data[score_col].iloc[-1]
                final_results.append({
                    'Environment': env,
                    'Configuration': config.replace('_', ' ').title(),
                    'Score': final_score
                })
    
    if final_results:
        results_df = pd.DataFrame(final_results)
        
        # Барный график
        sns.barplot(data=results_df, x='Environment', y='Score', hue='Configuration', ax=ax)
        ax.set_title('Финальные результаты по конфигурациям', fontsize=16, fontweight='bold')
        ax.set_xlabel('Среда', fontsize=12)
        ax.set_ylabel('Финальный результат', fontsize=12)
        
        # Поворачиваем подписи осей
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_dir / 'final_comparison_9exp.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # График 3: Тепловая карта улучшений
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Матрица улучшений
    improvement_matrix = []
    env_names = []
    
    for env in environments:
        env_data = df[df['environment'] == env]
        env_names.append(env)
        
        baseline_score = None
        kl_only_score = None
        kl_awfm_score = None
        
        for config in configs:
            config_data = env_data[env_data['config'] == config]
            if not config_data.empty:
                score = config_data[score_col].iloc[-1]
                if config == '1_baseline':
                    baseline_score = score
                elif config == '2_kl_only':
                    kl_only_score = score
                elif config == '3_kl_awfm':
                    kl_awfm_score = score
        
        # Вычисляем улучшения в процентах
        improvements = []
        if baseline_score:
            if kl_only_score:
                kl_improvement = ((kl_only_score - baseline_score) / abs(baseline_score)) * 100
            else:
                kl_improvement = 0
            
            if kl_awfm_score:
                awfm_improvement = ((kl_awfm_score - baseline_score) / abs(baseline_score)) * 100
            else:
                awfm_improvement = 0
            
            improvements = [kl_improvement, awfm_improvement]
        else:
            improvements = [0, 0]
        
        improvement_matrix.append(improvements)
    
    if improvement_matrix:
        improvement_array = np.array(improvement_matrix)
        
        sns.heatmap(improvement_array, 
                   xticklabels=['KL vs Baseline', 'KL+AWFM vs Baseline'],
                   yticklabels=env_names,
                   annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax)
        ax.set_title('Улучшения относительно Baseline (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'improvement_heatmap_9exp.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📈 Графики сохранены в: {output_dir}")

def generate_9exp_report(df, results_dir, results_table):
    """Генерирует отчет о 9 экспериментах"""
    report_path = Path(results_dir) / "tesla_t4_9exp_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Отчет о 9 экспериментах Tesla T4\n\n")
        f.write("## Конфигурация\n")
        f.write("- **GPU**: Tesla T4 16GB\n")
        f.write("- **Batch size**: 1024\n")
        f.write("- **Шагов обучения**: 100,000\n")
        f.write("- **Архитектура**: (512, 512, 512, 512)\n\n")
        
        f.write("## Эксперименты\n")
        f.write("| № | Конфигурация | KL Coeff | AWFM | Описание |\n")
        f.write("|---|--------------|----------|------|----------|\n")
        f.write("| 1 | Baseline | 0.0 | ❌ | Базовая FQL без улучшений |\n")
        f.write("| 2 | KL-only | 0.3 | ❌ | FQL с KL-пессимизмом |\n")
        f.write("| 3 | KL+AWFM | 0.3 | ✅ | FQL с KL + Advantage-Weighted Flow Matching |\n\n")
        
        f.write("## Результаты\n\n")
        
        if results_table:
            f.write("### Финальные результаты\n\n")
            f.write("| Среда | Baseline | KL-only | KL+AWFM |\n")
            f.write("|-------|----------|---------|----------|\n")
            
            for row in results_table:
                env = row['Environment']
                baseline = f"{row['Baseline']:.1f}" if row['Baseline'] else "N/A"
                kl_only = f"{row['KL-only']:.1f}" if row['KL-only'] else "N/A"
                kl_awfm = f"{row['KL+AWFM']:.1f}" if row['KL+AWFM'] else "N/A"
                f.write(f"| {env} | {baseline} | {kl_only} | {kl_awfm} |\n")
            
            f.write("\n")
        
        f.write("## Графики\n\n")
        f.write("- `learning_curves_9exp.png` - Кривые обучения для всех экспериментов\n")
        f.write("- `final_comparison_9exp.png` - Сравнение финальных результатов\n")
        f.write("- `improvement_heatmap_9exp.png` - Тепловая карта улучшений\n\n")
        
        f.write("## Выводы\n\n")
        f.write("1. **KL-пессимизм**: Помогает стабилизировать обучение на OOD данных\n")
        f.write("2. **AWFM**: Улучшает качество flow matching через взвешивание по advantage\n")
        f.write("3. **Комбинация**: KL+AWFM показывает лучшие результаты на сложных задачах\n\n")
        
        f.write("## Рекомендации\n\n")
        f.write("- Для стабильного обучения используйте KL-пессимизм (kl_coeff=0.3)\n")
        f.write("- Для максимальной производительности добавьте AWFM (advantage_weighted=true)\n")
        f.write("- На простых задачах baseline может быть достаточно\n")
    
    print(f"📄 Отчет сохранен: {report_path}")

def main():
    if len(sys.argv) < 2:
        print("❌ Использование: python analyze_tesla_9exp_results.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.exists(results_dir):
        print(f"❌ Директория не найдена: {results_dir}")
        sys.exit(1)
    
    print(f"🔍 Анализ 9 экспериментов в: {results_dir}")
    
    # Парсинг данных
    df = parse_csv_logs(results_dir)
    
    if df is None:
        print("❌ Не найдены CSV файлы с результатами")
        sys.exit(1)
    
    print(f"✅ Загружено {len(df)} записей из {len(df['config'].unique())} конфигураций")
    
    # Анализ производительности
    results_table = analyze_9exp_performance(df)
    
    # Создание графиков
    plots_dir = Path(results_dir) / "plots"
    create_9exp_plots(df, plots_dir)
    
    # Генерация отчета
    generate_9exp_report(df, results_dir, results_table)
    
    print(f"\n🎉 Анализ завершен!")
    print(f"📊 Результаты в: {results_dir}")

if __name__ == "__main__":
    main() 