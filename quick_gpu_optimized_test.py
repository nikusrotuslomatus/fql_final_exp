#!/usr/bin/env python3
"""
Оптимизированный тест для слабого сервера с улучшенной утилизацией GPU
- Увеличенный batch_size для лучшей загрузки GPU
- Сокращено до 100k итераций 
- Только самые важные эксперименты
- Эффективное использование памяти
"""

import subprocess
import os
import time
import json
from pathlib import Path

# Конфигурация для слабого сервера
SERVER_CONFIG = {
    'offline_steps': 100000,  # Сокращено с 500k до 100k
    'eval_episodes': 20,      # Сокращено с 50 до 20
    'eval_interval': 25000,   # Каждые 25k вместо 50k
    'log_interval': 5000,     # Каждые 5k вместо 10k
    'batch_size': 512,        # Увеличено с 256 до 512 для лучшей загрузки GPU
    'num_seeds': 2,           # Только 2 сида вместо 3
    'max_parallel': 1,        # Только 1 параллельный процесс
}

def get_gpu_optimized_config():
    """Конфигурация для оптимального использования GPU"""
    return {
        'batch_size': 512,           # Увеличенный batch для GPU
        'actor_hidden_dims': '(512, 512, 512)',  # Меньше слоев для экономии памяти
        'value_hidden_dims': '(512, 512, 512)',
        'flow_steps': 10,            # Меньше шагов flow для скорости
        'kl_num_samples': 6,         # Меньше сэмплов для KL
    }

def run_experiment(env_name, config_name, config_args, seed, exp_group):
    """Запуск одного эксперимента"""
    print(f"🧪 Запуск: {env_name} | {config_name} | seed {seed}")
    
    # Создание уникального имени
    run_name = f"{env_name}_{config_name}_seed{seed}"
    
    # Базовые аргументы
    base_args = [
        'python', 'main.py',
        f'--env_name={env_name}',
        f'--offline_steps={SERVER_CONFIG["offline_steps"]}',
        '--online_steps=0',
        f'--seed={seed}',
        f'--run_group={exp_group}',
        f'--eval_episodes={SERVER_CONFIG["eval_episodes"]}',
        f'--eval_interval={SERVER_CONFIG["eval_interval"]}',
        f'--log_interval={SERVER_CONFIG["log_interval"]}',
        '--use_wandb=False',  # Отключаем wandb для экономии ресурсов
        f'--agent.batch_size={SERVER_CONFIG["batch_size"]}',
    ]
    
    # Добавляем конфигурацию
    base_args.extend(config_args.split())
    
    # Создание директории для логов
    log_dir = Path(f"quick_test_results_{int(time.time())}")
    log_dir.mkdir(exist_ok=True)
    
    # Запуск с логированием
    with open(log_dir / f"{run_name}.log", 'w') as f:
        process = subprocess.Popen(
            base_args,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd='.'
        )
    
    return process

def main():
    """Основная функция для быстрого тестирования"""
    print("🚀 Быстрый тест FQL с оптимизацией GPU")
    print("=" * 50)
    print(f"Настройки сервера:")
    print(f"  - Итераций: {SERVER_CONFIG['offline_steps']:,}")
    print(f"  - Batch size: {SERVER_CONFIG['batch_size']}")
    print(f"  - Сидов: {SERVER_CONFIG['num_seeds']}")
    print(f"  - Параллельных процессов: {SERVER_CONFIG['max_parallel']}")
    print()
    
    # Оптимизированная конфигурация GPU
    gpu_config = get_gpu_optimized_config()
    gpu_args = ' '.join([f'--agent.{k}={v}' for k, v in gpu_config.items()])
    
    # Выбираем только самые важные эксперименты
    experiments = [
        {
            'env': 'antmaze-large-play-v2',
            'group': 'QuickTest_AntMaze',
            'alpha': '3',
            'extra_args': '--agent.q_agg=min'
        },
        {
            'env': 'puzzle-3x3-play-singletask-v0', 
            'group': 'QuickTest_Puzzle',
            'alpha': '100',
            'extra_args': ''
        }
    ]
    
    # Конфигурации для тестирования
    test_configs = {
        'baseline': f'--agent.kl_coeff=0.0 --agent.advantage_weighted=False',
        'kl_awfm': f'--agent.kl_coeff=0.3 --agent.advantage_weighted=True --agent.adv_weight_coeff=1.0',
    }
    
    processes = []
    
    for exp in experiments:
        print(f"🔬 Тестирование: {exp['env']}")
        print(f"   Alpha: {exp['alpha']}")
        
        for config_name, config_base in test_configs.items():
            # Полная конфигурация
            full_config = f"{config_base} --agent.alpha={exp['alpha']} {exp['extra_args']} {gpu_args}"
            
            for seed in range(SERVER_CONFIG['num_seeds']):
                # Ждем если слишком много процессов
                while len([p for p in processes if p.poll() is None]) >= SERVER_CONFIG['max_parallel']:
                    time.sleep(5)
                    # Очищаем завершенные процессы
                    processes = [p for p in processes if p.poll() is None]
                
                # Запускаем новый процесс
                process = run_experiment(
                    exp['env'],
                    config_name,
                    full_config,
                    seed,
                    exp['group']
                )
                processes.append(process)
                
                # Небольшая задержка между запусками
                time.sleep(2)
    
    # Ждем завершения всех процессов
    print("\n⏳ Ожидание завершения всех экспериментов...")
    for process in processes:
        process.wait()
    
    print("\n✅ Все эксперименты завершены!")
    print("\n📊 Для анализа результатов:")
    print("   - Проверьте файлы в exp/fql/QuickTest_*/")
    print("   - Используйте CSV логи для построения графиков")
    print("\n💡 Рекомендации по GPU:")
    print(f"   - Batch size увеличен до {SERVER_CONFIG['batch_size']} для лучшей загрузки")
    print("   - Сокращены размеры сетей для экономии памяти")
    print("   - Уменьшено количество flow steps для скорости")

if __name__ == "__main__":
    main() 