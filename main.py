"""
Основной скрипт запуска нейросетевого ассистента
"""
import os
import sys
import json
import time
import signal
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Добавляем пути для импорта
sys.path.insert(0, str(Path(__file__).parent))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AI_Assistant")

def check_dependencies():
    """Проверяет зависимости"""
    required_packages = [
        'torch', 'torchaudio', 'speech_recognition', 'pyttsx3',
        'pyaudio', 'pyautogui'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"[ERROR] Отсутствуют зависимости: {', '.join(missing)}")
        print("Установите: pip install " + " ".join(missing))
        return False
    
    print("[SUCCESS] Все зависимости установлены")
    return True

def setup_directories():
    """Создает необходимые директории"""
    directories = [
        "models",
        "models/actions",
        "data",
        "memory", 
        "logs",
        "cache/audio",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"[SUCCESS]Создана директория: {directory}")

def create_default_config():
    """Создает конфигурацию по умолчанию"""
    config_path = Path("config/config.json")
    
    if config_path.exists():
        print(f"[INFO] Конфигурация уже существует: {config_path}")
        return
    
    default_config = {
        "assistant": {
            "name": "Нейро-Ассистент",
            "version": "1.0.0",
            "auto_start": True,
            "log_level": "INFO"
        },
        "voice": {
            "speech_recognizer": "google",  # google, vosk, whisper
            "tts_engine": "pyttsx3",  # pyttsx3, silero
            "language": "ru",
            "activation_keyword": "ассистент",
            "activation_mode": "keyword",  # keyword, hotkey, always
            "sample_rate": 16000,
            "voice_gender": "female"
        },
        "neural_models": {
            "intent_model": "models/intent_model.pt",
            "learning_model": "models/learning_model.pt",
            "code_generator": "models/code_generator.pt",
            "use_gpu": False,
            "model_size": "small"
        },
        "execution": {
            "safe_mode": True,
            "confirm_destructive": True,
            "max_code_length": 1000,
            "timeout_seconds": 30
        },
        "memory": {
            "memory_file": "memory/assistant_memory.json",
            "max_items": 2000,
            "auto_cleanup_days": 7
        },
        "learning": {
            "auto_learn": True,
            "min_confidence_for_learning": 0.3,
            "save_examples": True,
            "examples_file": "data/learned_commands.json"
        },
        "paths": {
            "browser": "chrome.exe",
            "editor": "notepad.exe",
            "calculator": "calc.exe",
            "file_manager": "explorer.exe"
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, ensure_ascii=False, indent=2)
    
    print(f"[SUCCESS] Создана конфигурация: {config_path}")

class NeuralAIAssistant:
    """
    Главный класс интегрирующий все компоненты ассистента
    """
    
    def __init__(self, config_path: str = "config/config.json"):
        logger.info("=" * 60)
        logger.info("[LOADING] ИНИЦИАЛИЗАЦИЯ НЕЙРОСЕТЕВОГО АССИСТЕНТА")
        logger.info("=" * 60)
        
        # Загрузка конфигурации
        self.config = self._load_config(config_path)
        
        # Флаги состояния
        self.is_running = False
        self.is_learning = False
        self.last_command = None
        
        # Компоненты ассистента
        self.assistant_core = None
        self.voice_engine = None
        self.memory = None
        self.intent_recognizer = None
        self.learning_engine = None
        self.code_generator = None
        self.command_executor = None
        
        # История выполнения
        self.command_history = []
        self.max_history = 100
        
        # Потоки
        self.main_thread = None
        
        # Загрузка компонентов
        self._initialize_components()
        
        logger.info("✅ Ассистент инициализирован")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загружает конфигурацию"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Конфигурация не найдена: {config_path}")
            return self._get_default_config()
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logger.info(f"[INFO] Конфигурация загружена из {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию по умолчанию"""
        return {
            "assistant": {"name": "Ассистент", "auto_start": True},
            "voice": {"language": "ru", "activation_keyword": "ассистент"},
            "execution": {"safe_mode": True}
        }
        
    def _ask_user(self, prompt_tts: str, text_prompt: str, timeout: int = 10) -> str:
        """Универсальный запрос: сначала пытаемся получить голосовой ввод, если не вышло — консоль."""
        if self.voice_engine:
            try:
                self.voice_engine.speak(prompt_tts)
                answer = self.voice_engine.listen_once(timeout=timeout)
                if answer:
                    return answer.strip()
            except Exception:
                pass
        # fallback
        return input(text_prompt).strip()
    
    def _initialize_components(self):
        """Инициализирует все компоненты ассистента"""
        logger.info("[LOADING] Инициализация компонентов...")
        
        try:
            # 1. Память ассистента
            from memory.memory_assistent import AssistantMemory
            memory_file = self.config.get("memory", {}).get("memory_file", "memory/assistant_memory.json")
            self.memory = AssistantMemory(memory_file)
            logger.info("[SUCCESS] Память инициализирована")
            
            # 2. Распознаватель намерений
            from neural_core.intent_recognition import NeuralIntentRecognizer
            intent_model = self.config.get("neural_models", {}).get("intent_model", "models/intent_model.pt")
            self.intent_recognizer = NeuralIntentRecognizer(intent_model)
            logger.info("[SUCCESS] Распознаватель намерений инициализирован")
            
            # 3. Движок обучения
            from neural_core.learning_engine import NeuralLearningEngine
            learning_model = self.config.get("neural_models", {}).get("learning_model", "models/learning_model.pt")
            learning_data = "data/learned_commands.json"
            self.learning_engine = NeuralLearningEngine(learning_model, learning_data)
            logger.info("[SUCCESS] Движок обучения инициализирован")
            
            # 4. Генератор кода
            from neural_core.code_generator import NeuralCodeGenerator
            code_model = self.config.get("neural_models", {}).get("code_generator", "models/code_generator.pt")
            code_data = "data/code_examples.json"
            self.code_generator = NeuralCodeGenerator(code_model, code_data)
            logger.info("[SUCCESS] Генератор кода инициализирован")
            
            # 5. Исполнитель команд
            from execution.command_executor import NeuralCommandDispatcher
            self.command_executor = NeuralCommandDispatcher(self.config.get("execution", {}))
            logger.info("[SUCCESS] Исполнитель команд инициализирован")
            
            # 6. Голосовой движок
            from voice.voice_engine import NeuralVoiceEngine, VoiceConfig
            
            voice_config_dict = self.config.get("voice", {})
            voice_config = VoiceConfig(
                speech_recognizer=voice_config_dict.get("speech_recognizer", "google"),
                language=voice_config_dict.get("language", "ru"),
                activation_keyword=voice_config_dict.get("activation_keyword", "ассистент"),
                activation_mode=voice_config_dict.get("activation_mode", "keyword"),
                tts_engine=voice_config_dict.get("tts_engine", "pyttsx3")
            )
            
            self.voice_engine = NeuralVoiceEngine(voice_config)
            self.voice_engine.set_command_callback(self._process_voice_command)
            self.voice_engine.set_activation_callback(self._on_voice_activation)
            logger.info("[SUCCESS] Голосовой движок инициализирован")
            
            # 7. Основной класс ассистента
            from assistant import NeuralAssistant
            self.assistant_core = NeuralAssistant(config=self.config)
            
            # Подключаем компоненты к ядру
            self.assistant_core._intent_recognizer = self.intent_recognizer
            self.assistant_core._command_executor = self.command_executor
            self.assistant_core._voice_engine = self.voice_engine
            self.assistant_core._memory = self.memory
            self.assistant_core._learning_engine = self.learning_engine
            self.assistant_core._code_generator = self.code_generator
            
            logger.info("[SUCCESS] Все компоненты успешно инициализированы")
            
        except ImportError as e:
            logger.error(f"[ERROR] Ошибка импорта компонента: {e}")
            logger.error("Проверьте что все файлы на месте и зависимости установлены")
            raise
        except Exception as e:
            logger.error(f"[ERROR] Ошибка инициализации компонентов: {e}")
            raise
    
    def _process_voice_command(self, text: str):
        """Обрабатывает голосовую команду"""
        logger.info(f"[INPUT] Голосовая команда: {text}")
        
        if not text or not text.strip():
            return
        
        self.last_command = text
        
        # Обрабатываем команду
        result = self.process_command(text)
        
        # Озвучиваем ответ если есть
        if result and result.get("message"):
            self.voice_engine.speak(result["message"])
    
    def _on_voice_activation(self, text: str):
        """Обрабатывает активацию по ключевому слову"""
        logger.info(f"[INFO] Активация по ключевому слову: {text}")
        self.voice_engine.speak("Да, слушаю вас!")
    
    def process_command(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Основной метод обработки команды
        
        Returns:
            Результат выполнения команды
        """
        if not text or not text.strip():
            return None
        
        logger.info(f"[INFO] Обработка команды: {text}")
        
        try:
            # 1. Распознаем намерение и сущности
            intent_result = self.intent_recognizer.predict(text)
            intent = intent_result.get("intent", "unknown")
            confidence = intent_result.get("confidence", 0.0)
            entities = intent_result.get("entities", [])
            
            logger.info(f"   Намерение: {intent} (уверенность: {confidence:.2%})")
            if entities:
                logger.info(f"   Сущности: {entities}")
            
            # 2. Ищем в памяти похожие команды
            similar_commands = self.memory.find_similar(text, intent)
            if similar_commands:
                logger.info(f"   Найдено похожих команд: {len(similar_commands)}")
            
            # 3. Если уверенность низкая, предлагаем обучение
            learning_cfg = self.config.get("learning", {})
            min_conf = learning_cfg.get("min_confidence_for_learning", 0.3)

            if (intent == "unknown" or confidence < min_conf) and learning_cfg.get("auto_learn", True):
                logger.info(" [WARNING] Неизвестное или неуверенное намерение, предлагаю обучение")
                return self._handle_unknown_command(text)
            
            # 4. Выполняем команду
            execution_result = self.command_executor.execute(
                intent=intent,
                entities=entities,
                original_text=text,
                context={"last_command": self.last_command}
            )
            
            # 5. Сохраняем в память
            if confidence > 0.5:  # Сохраняем только если уверены
                self.memory.add_command(
                    text=text,
                    intent=intent,
                    entities=entities,
                    success=execution_result.success,
                    result=execution_result.data
                )
                
                # Если это выученная команда, обновляем статистику
                if intent.startswith("learned_"):
                    self.memory.update_learned_command_stats(text, execution_result.success)
            
            # 6. Добавляем в историю
            self.command_history.append({
                "text": text,
                "intent": intent,
                "success": execution_result.success,
                "timestamp": time.time()
            })
            
            # Ограничиваем размер истории
            if len(self.command_history) > self.max_history:
                self.command_history = self.command_history[-self.max_history:]
            
            logger.info(f"   [SUCCESS] Команда выполнена: {execution_result.success}")
            
            return {
                "success": execution_result.success,
                "message": execution_result.message,
                "data": execution_result.data,
                "requires_confirmation": execution_result.requires_confirmation
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Ошибка обработки команды: {e}")
            
            return {
                "success": False,
                "message": f"Ошибка обработки команды: {str(e)}",
                "error": str(e)
            }
    
    def _handle_unknown_command(self, text: str) -> Dict[str, Any]:
        logger.info(f"[WARNING] Неизвестная команда: {text}")

        answer = self._ask_user(
            prompt_tts=f"Я не знаю команду '{text}'. Хотите научить меня? Скажите да или нет.",
            text_prompt="[INPUT] Хотите научить команду? (да/нет): "
        ).lower()

        if answer not in ["да", "yes", "ага"]:
            return {
                "success": False,
                "message": f"Команда '{text}' не обучена",
                "needs_learning": True,
                "command_text": text
            }

        explanation = self._ask_user(
            prompt_tts="Опишите, что должна делать эта команда.",
            text_prompt="[INPUT] Объяснение команды: "
        )

        examples_input = self._ask_user(
            prompt_tts="Приведите примеры похожих команд через запятую.",
            text_prompt="[INPUT] Примеры похожих команд: "
        )
        examples = [ex.strip() for ex in examples_input.split(",") if ex.strip()]

        result = self.assistant_core.train_on_unknown(text, explanation, examples)

        # 🔥 ОБЯЗАТЕЛЬНО: дообучаем классификатор интентов
        if result.get("success") and self.intent_recognizer:
            try:
                self.intent_recognizer.train_on_example(
                    text=text,
                    intent=f"learned_{hash(text) % 1000}",
                    entities=[]
                )
            except Exception as e:
                logger.error(f"Ошибка обучения распознавателя интентов: {e}")

        if self.voice_engine:
            self.voice_engine.speak(result.get("message", "Обучение завершено"))

        return result
    
    def start_learning_mode(self):
        """Запускает режим обучения"""
        logger.info("[LOADING] Запуск режима обучения")
        self.is_learning = True
        
        self.voice_engine.speak("Включен режим обучения. Говорите команды для обучения.")
        
        while self.is_learning and self.is_running:
            try:
                # Слушаем команду для обучения
                command = self.voice_engine.listen_once(timeout=10)
                
                if not command:
                    continue
                
                if "стоп" in command.lower() or "выход" in command.lower():
                    self.voice_engine.speak("Режим обучения завершен.")
                    break
                
                # Спрашиваем объяснение
                self.voice_engine.speak(f"Что должна делать команда '{command}'?")
                explanation = self.voice_engine.listen_once(timeout=10)
                
                if not explanation:
                    self.voice_engine.speak("Не расслышала объяснение. Попробуем еще раз?")
                    continue
                
                # Спрашиваем примеры
                self.voice_engine.speak("Приведите 2-3 примера похожих команд.")
                examples = []
                
                for i in range(3):
                    self.voice_engine.speak(f"Пример {i+1}:")
                    example = self.voice_engine.listen_once(timeout=5)
                    if example:
                        examples.append(example)
                
                # Генерируем код
                self.voice_engine.speak("Генерирую код для выполнения команды...")
                
                generated_code = self.code_generator.generate(
                    description=explanation,
                    intent_type="program",
                    safe_mode=self.config.get("execution", {}).get("safe_mode", True)
                )
                
                # Обучаем модель
                success = self.learning_engine.train_on_example(
                    text=command,
                    explanation=explanation,
                    examples=examples,
                    generated_code=generated_code
                )
                
                if success:
                    self.voice_engine.speak(f"Отлично! Я выучила команду '{command}'.")
                    
                    # Сохраняем в память
                    self.memory.add_learned_command(
                        command_text=command,
                        explanation=explanation,
                        generated_code=generated_code,
                        examples=examples
                    )
                    
                    # Переобучаем распознаватель
                    self.intent_recognizer.train_on_example(
                        text=command,
                        intent=f"learned_{hash(command) % 1000}",
                        entities=[]
                    )
                else:
                    self.voice_engine.speak(f"Не удалось выучить команду '{command}'. Попробуем другую?")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Ошибка в режиме обучения: {e}")
                self.voice_engine.speak("Произошла ошибка. Попробуем еще раз?")
        
        self.is_learning = False
    
    def start(self):
        """Запускает ассистента"""
        if self.is_running:
            logger.warning("Ассистент уже запущен")
            return
        
        logger.info("[INFO] Запуск ассистента...")
        self.is_running = True
        
        # Запускаем голосовой движок
        self.voice_engine.start_listening()
        
        # Приветствие
        assistant_name = self.config.get("assistant", {}).get("name", "Ассистент")
        welcome_message = f"Привет! Я {assistant_name}. Готова к работе."
        self.voice_engine.speak(welcome_message)
        
        # Запускаем главный цикл в отдельном потоке
        self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.main_thread.start()
        
        logger.info("[SUCCESS] Ассистент запущен")
        
        # Ожидание завершения
        try:
            while self.is_running and self.main_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n[WARNING] Получен сигнал Ctrl+C")
            self.stop()
    
    def _main_loop(self):
        """Главный цикл работы ассистента"""
        logger.info("🔄 Главный цикл запущен")
        
        try:
            while self.is_running:
                # Проверяем команды из других источников (например, текстовый ввод)
                # В этой версии основная обработка через голосовой движок
                
                # Можно добавить обработку горячих клавиш и т.д.
                
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Ошибка в главном цикле: {e}")
        finally:
            logger.info("Главный цикл завершен")
    
    def stop(self):
        """Останавливает ассистента"""
        if not self.is_running:
            return
        
        logger.info("[WARNING] Остановка ассистента...")
        self.is_running = False
        
        # Останавливаем голосовой движок
        if self.voice_engine:
            self.voice_engine.cleanup()
        
        # Сохраняем память
        if self.memory:
            self.memory.save()
        
        # Ждем завершения потоков
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5)
        
        logger.info("[SUCCESS] Ассистент остановлен")
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус ассистента"""
        status = {
            "running": self.is_running,
            "learning": self.is_learning,
            "last_command": self.last_command,
            "command_history_count": len(self.command_history),
            "components": {}
        }
        
        # Статистика компонентов
        if self.memory:
            status["components"]["memory"] = self.memory.get_statistics()
        
        if self.intent_recognizer:
            status["components"]["intent_recognizer"] = self.intent_recognizer.get_statistics()
        
        if self.voice_engine:
            status["components"]["voice_engine"] = self.voice_engine.get_stats()
        
        if self.command_executor:
            status["components"]["command_executor"] = self.command_executor.get_stats()
        
        return status

def signal_handler(signum, frame):
    """Обработчик сигналов для корректного завершения"""
    logger.info(f"Получен сигнал {signum}, завершаю работу...")
    sys.exit(0)

def main():
    """Основная функция запуска"""
    print("\n" + "=" * 60)
    print("[RUN] НЕЙРОСЕТЕВОЙ ГОЛОСОВОЙ АССИСТЕНТ")
    print("=" * 60)
    
    # Регистрация обработчиков сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Проверка зависимостей
    if not check_dependencies():
        sys.exit(1)
    
    # Создание директорий
    setup_directories()
    
    # Создание конфигурации
    create_default_config()
    
    # Запуск ассистента
    try:
        assistant = NeuralAIAssistant()
        
        print("\n[INFO] Статус компонентов:")
        status = assistant.get_status()
        for component, data in status.get("components", {}).items():
            if isinstance(data, dict):
                print(f"  • {component}: ✓")
        
        print("\n[INFO] Команды управления:")
        print("  • Скажите 'ассистент' для активации")
        print("  • Скажите 'научи меня' для обучения новым командам")
        print("  • Скажите 'статус' для получения информации")
        print("  • Скажите 'стоп' или нажмите Ctrl+C для выхода")
        print("\n" + "=" * 60)
        
        # Запуск
        assistant.start()
        
    except KeyboardInterrupt:
        print("\n[WARNING] Завершение по Ctrl+C")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        print(f"\n[ERROR] Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def test_mode():
    """Режим тестирования (без голоса)"""
    print("\n[WARNING] РЕЖИМ ТЕСТИРОВАНИЯ")
    
    # Создаем ассистента
    assistant = NeuralAIAssistant()
    
    # Тестовые команды
    test_commands = [
        "привет",
        "открой браузер",
        "сколько времени",
        "создай файл test.txt",
        "пока"
    ]
    
    print("\n[INFO] Тестирую команды:")
    for cmd in test_commands:
        print(f"\n[USER] Команда: {cmd}")
        result = assistant.process_command(cmd)
        
        if result:
            success = "[SUCCESS]" if result.get("success") else "❌"
            print(f"{success} Результат: {result.get('message', '')[:100]}...")
        else:
            print("❌ Нет результата")
        
        time.sleep(1)
    
    # Статус
    print("\n[INFO] Статус ассистента:")
    status = assistant.get_status()
    print(json.dumps(status, indent=2, ensure_ascii=False, default=str))

def interactive_mode():
    """Интерактивный режим (текстовый ввод)"""
    print("\n[INFO] ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("Вводите команды или 'стоп' для выхода\n")
    
    assistant = NeuralAIAssistant()
    
    while True:
        try:
            cmd = input("[INPUT] Команда: ").strip()
            
            if not cmd:
                continue
            
            if cmd.lower() in ['стоп', 'выход', 'exit', 'quit']:
                print("Завершение работы...")
                break
            
            if cmd.lower() == 'статус':
                status = assistant.get_status()
                print(json.dumps(status, indent=2, ensure_ascii=False, default=str))
                continue
            
            if cmd.lower() == 'память':
                if assistant.memory:
                    stats = assistant.memory.get_statistics()
                    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
                continue
            
            # Обработка команды
            result = assistant.process_command(cmd)
            
            if result:
                success = "[SUCCESS]" if result.get("success") else "❌"
                print(f"{success} {result.get('message', '')}")
            else:
                print("[ERROR] Не удалось обработать команду")
        
        except KeyboardInterrupt:
            print("\nЗавершение...")
            break
        except Exception as e:
            print(f"[ERROR] Ошибка: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Нейросетевой голосовой ассистент")
    parser.add_argument("--mode", choices=["full", "test", "interactive"], 
                       default="full", help="Режим работы")
    parser.add_argument("--config", default="config/config.json", 
                       help="Путь к конфигурации")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        test_mode()
    elif args.mode == "interactive":
        interactive_mode()
    else:
        main()