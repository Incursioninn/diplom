"""
Упрощенный запуск ассистента
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def create_test_config():
    """Создает тестовую конфигурацию"""
    config = {
        "voice": {
            "speech_recognizer": "google",
            "language": "ru",
            "activation_mode": "always"
        },
        "execution": {
            "safe_mode": True
        }
    }
    
    config_path = Path("config/test_config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    import json
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    return config_path

def run_simple_assistant():
    """Запускает упрощенного ассистента"""
    print("=" * 60)
    print("ПРОСТОЙ АССИСТЕНТ - ТЕСТОВЫЙ РЕЖИМ")
    print("=" * 60)
    
    # Создаем конфигурацию
    config_path = create_test_config()
    
    # Импортируем компоненты
    try:
        from voice.voice_engine import NeuralVoiceEngine, VoiceConfig
        from neural_core.intent_recognition import NeuralIntentRecognizer
        from execution.command_executor import NeuralCommandDispatcher
        from memory.memory_assistent import AssistantMemory
        
        print("[OK] Все компоненты загружены")
        
    except ImportError as e:
        print(f"[ERROR] Ошибка импорта: {e}")
        print("Убедитесь что все файлы созданы:")
        print("  - voice/voice_engine.py")
        print("  - neural_core/intent_recognition.py") 
        print("  - execution/command_executor.py")
        print("  - memory/assistant_memory.py")
        return
    
    # Инициализируем компоненты
    print("\n[ИНИТ] Инициализация...")
    
    try:
        # 1. Голос
        voice_config = VoiceConfig(
            speech_recognizer="google",
            language="ru",
            activation_mode="always"
        )
        voice_engine = NeuralVoiceEngine(voice_config)
        print("[OK] Голосовой движок")
        
        # 2. Распознавание
        intent_recognizer = NeuralIntentRecognizer("models/intent_model.pt")
        print("[OK] Распознаватель намерений")
        
        # 3. Исполнитель
        command_executor = NeuralCommandDispatcher({"safe_mode": True})
        print("[OK] Исполнитель команд")
        
        # 4. Память
        memory = AssistantMemory("memory/test_memory.json")
        print("[OK] Память")
        
    except Exception as e:
        print(f"[ERROR] Ошибка инициализации: {e}")
        return
    
    # Тестовый цикл
    print("\n" + "=" * 60)
    print("ТЕСТОВЫЙ РЕЖИМ")
    print("Команды: 'привет', 'время', 'открой', 'стоп'")
    print("=" * 60)
    
    while True:
        try:
            print("\n[СЛУШАЮ] Говорите команду...")
            
            # Слушаем команду
            text = voice_engine.listen_once(timeout=5)
            
            if not text:
                print("[ТАЙМАУТ] Ничего не сказано")
                continue
            
            print(f"[СЛЫШУ] Вы сказали: {text}")
            
            if "стоп" in text.lower():
                print("[ВЫХОД] Завершение работы...")
                break
            
            # Распознаем намерение
            intent_result = intent_recognizer.predict(text)
            print(f"[НАМЕРЕНИЕ] {intent_result['intent']} ({intent_result['confidence']:.0%})")
            
            # Выполняем команду
            execution_result = command_executor.execute(
                intent=intent_result['intent'],
                entities=intent_result.get('entities', []),
                original_text=text,
                context={}
            )
            
            print(f"[ВЫПОЛНЕНИЕ] {execution_result.message}")
            
            # Сохраняем в память
            memory.add_command(
                text=text,
                intent=intent_result['intent'],
                entities=intent_result.get('entities', []),
                success=execution_result.success,
                result=execution_result.data
            )
            
            # Озвучиваем ответ
            voice_engine.speak(execution_result.message)
            
        except KeyboardInterrupt:
            print("\n[ПРЕРВАНО] Ctrl+C")
            break
        except Exception as e:
            print(f"[ОШИБКА] {e}")
    
    # Завершение
    print("\n[ЗАВЕРШЕНИЕ] Сохранение памяти...")
    memory.save()
    voice_engine.cleanup()
    print("[ГОТОВО] Работа завершена")

if __name__ == "__main__":
    run_simple_assistant()