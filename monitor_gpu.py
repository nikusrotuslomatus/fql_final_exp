#!/usr/bin/env python3
"""
Простой мониторинг GPU во время обучения FQL
"""

import time
import subprocess
import json
import os
from datetime import datetime

def get_gpu_info():
    """Получает информацию о GPU"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 5:
                    gpu_info.append({
                        'gpu_id': i,
                        'utilization': int(parts[0]),
                        'memory_used': int(parts[1]),
                        'memory_total': int(parts[2]),
                        'temperature': int(parts[3]),
                        'power_draw': float(parts[4]) if parts[4] != '[Not Supported]' else 0
                    })
            return gpu_info
        else:
            return None
    except Exception as e:
        print(f"Ошибка получения информации о GPU: {e}")
        return None

def monitor_training(log_file="gpu_monitor.log", interval=30):
    """Мониторинг GPU во время обучения"""
    print("🔍 Запуск мониторинга GPU...")
    print(f"Логи сохраняются в: {log_file}")
    print(f"Интервал мониторинга: {interval} секунд")
    print("Нажмите Ctrl+C для остановки")
    print("-" * 50)
    
    with open(log_file, 'w') as f:
        # Заголовок CSV
        f.write("timestamp,gpu_id,utilization,memory_used,memory_total,memory_percent,temperature,power_draw\n")
        
        try:
            while True:
                gpu_info = get_gpu_info()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                if gpu_info:
                    for gpu in gpu_info:
                        memory_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
                        
                        # Запись в лог
                        f.write(f"{timestamp},{gpu['gpu_id']},{gpu['utilization']},"
                               f"{gpu['memory_used']},{gpu['memory_total']},{memory_percent:.1f},"
                               f"{gpu['temperature']},{gpu['power_draw']}\n")
                        f.flush()
                        
                        # Вывод на экран
                        print(f"GPU {gpu['gpu_id']}: "
                              f"Util={gpu['utilization']:2d}% "
                              f"Mem={memory_percent:4.1f}% "
                              f"Temp={gpu['temperature']:2d}°C "
                              f"Power={gpu['power_draw']:4.1f}W")
                else:
                    print("❌ Не удалось получить информацию о GPU")
                
                print(f"[{timestamp}] Следующая проверка через {interval}с...")
                print("-" * 50)
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Мониторинг остановлен")

def analyze_gpu_log(log_file="gpu_monitor.log"):
    """Анализирует лог GPU"""
    if not os.path.exists(log_file):
        print(f"❌ Файл лога не найден: {log_file}")
        return
    
    print(f"📊 Анализ GPU лога: {log_file}")
    print("=" * 50)
    
    with open(log_file, 'r') as f:
        lines = f.readlines()[1:]  # Пропускаем заголовок
    
    if not lines:
        print("❌ Лог пуст")
        return
    
    # Парсинг данных
    gpu_data = {}
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 8:
            gpu_id = int(parts[1])
            utilization = int(parts[2])
            memory_percent = float(parts[5])
            temperature = int(parts[6])
            power_draw = float(parts[7])
            
            if gpu_id not in gpu_data:
                gpu_data[gpu_id] = {
                    'utilization': [],
                    'memory_percent': [],
                    'temperature': [],
                    'power_draw': []
                }
            
            gpu_data[gpu_id]['utilization'].append(utilization)
            gpu_data[gpu_id]['memory_percent'].append(memory_percent)
            gpu_data[gpu_id]['temperature'].append(temperature)
            gpu_data[gpu_id]['power_draw'].append(power_draw)
    
    # Статистика
    for gpu_id, data in gpu_data.items():
        print(f"\n🎯 GPU {gpu_id} статистика:")
        
        util_avg = sum(data['utilization']) / len(data['utilization'])
        util_max = max(data['utilization'])
        util_min = min(data['utilization'])
        
        mem_avg = sum(data['memory_percent']) / len(data['memory_percent'])
        mem_max = max(data['memory_percent'])
        
        temp_avg = sum(data['temperature']) / len(data['temperature'])
        temp_max = max(data['temperature'])
        
        power_avg = sum(data['power_draw']) / len(data['power_draw'])
        power_max = max(data['power_draw'])
        
        print(f"  Утилизация: {util_avg:.1f}% (мин: {util_min}%, макс: {util_max}%)")
        print(f"  Память: {mem_avg:.1f}% (макс: {mem_max:.1f}%)")
        print(f"  Температура: {temp_avg:.1f}°C (макс: {temp_max}°C)")
        print(f"  Мощность: {power_avg:.1f}W (макс: {power_max:.1f}W)")
        
        # Рекомендации
        print(f"\n💡 Рекомендации для GPU {gpu_id}:")
        if util_avg < 50:
            print("  ⚠️  Низкая утилизация GPU! Увеличьте batch_size")
        elif util_avg > 90:
            print("  ✅ Отличная утилизация GPU!")
        else:
            print("  🟡 Умеренная утилизация GPU, можно увеличить batch_size")
        
        if mem_avg < 50:
            print("  💾 Много свободной памяти, можно увеличить размер модели")
        elif mem_avg > 90:
            print("  ⚠️  Высокое использование памяти, осторожно с batch_size")
        
        if temp_avg > 80:
            print("  🌡️  Высокая температура, проверьте охлаждение")

def get_optimal_batch_size():
    """Подбирает оптимальный batch_size для текущей GPU"""
    print("🔧 Подбор оптимального batch_size...")
    
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("❌ Не удалось получить информацию о GPU")
        return
    
    for gpu in gpu_info:
        memory_gb = gpu['memory_total'] / 1024  # Переводим в GB
        
        print(f"\nGPU {gpu['gpu_id']}: {memory_gb:.1f}GB памяти")
        
        # Эмпирические рекомендации
        if memory_gb < 6:
            recommended_batch = 256
            print("  📦 Рекомендуемый batch_size: 256 (малая GPU)")
        elif memory_gb < 12:
            recommended_batch = 512
            print("  📦 Рекомендуемый batch_size: 512 (средняя GPU)")
        elif memory_gb < 24:
            recommended_batch = 1024
            print("  📦 Рекомендуемый batch_size: 1024 (большая GPU)")
        else:
            recommended_batch = 2048
            print("  📦 Рекомендуемый batch_size: 2048 (очень большая GPU)")
        
        print(f"  🎯 Для вашей GPU попробуйте: --agent.batch_size={recommended_batch}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "monitor":
            monitor_training()
        elif command == "analyze":
            log_file = sys.argv[2] if len(sys.argv) > 2 else "gpu_monitor.log"
            analyze_gpu_log(log_file)
        elif command == "batch":
            get_optimal_batch_size()
        else:
            print("❌ Неизвестная команда")
    else:
        print("🖥️  GPU Monitor для FQL")
        print("=" * 30)
        print("Использование:")
        print("  python monitor_gpu.py monitor     - Запуск мониторинга")
        print("  python monitor_gpu.py analyze     - Анализ лога")
        print("  python monitor_gpu.py batch       - Подбор batch_size")
        print()
        
        # Показываем текущее состояние GPU
        gpu_info = get_gpu_info()
        if gpu_info:
            print("📊 Текущее состояние GPU:")
            for gpu in gpu_info:
                memory_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
                print(f"  GPU {gpu['gpu_id']}: {gpu['utilization']}% util, "
                      f"{memory_percent:.1f}% mem, {gpu['temperature']}°C")
        else:
            print("❌ NVIDIA GPU не найдена") 