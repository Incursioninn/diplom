"""
Исполнитель команд с нейросетевой диспетчеризацией и безопасным выполнением
"""
import subprocess
import os
import sys
import time
import json
import shutil
import webbrowser
import pyautogui
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import threading
import inspect
from dataclasses import dataclass
from enum import Enum
import importlib.util

class CommandSafetyLevel(Enum):
    """Уровни безопасности команд"""
    SAFE = 1        # Чтение, открытие программ
    MODERATE = 2    # Создание/удаление файлов
    RISKY = 3       # Системные команды
    DANGEROUS = 4   # Потенциально опасные

@dataclass
class ExecutionResult:
    """Результат выполнения команды"""
    success: bool
    message: str
    data: Dict[str, Any]
    execution_time: float
    safety_level: CommandSafetyLevel
    requires_confirmation: bool = False

class NeuralCommandDispatcher:
    """Нейросетевой диспетчер команд"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_actions = self._load_base_actions()
        self.learned_actions = {}
        self.safety_rules = self._load_safety_rules()
        
        # Кэш загруженных модулей
        self.loaded_modules = {}
        
        # Статистика выполнения
        self.execution_stats = {
            'total_commands': 0,
            'successful': 0,
            'failed': 0,
            'blocked': 0
        }
        
        # Очередь команд для выполнения
        self.command_queue = []
        self.queue_lock = threading.Lock()
        
        # Инициализация безопасного выполнения
        self._init_safe_execution()
    
    def _load_base_actions(self) -> Dict[str, Callable]:
        """Загружает базовые действия"""
        actions = {
            'open_program': self._execute_open_program,
            'type_text': self._execute_type_text,
            'search_web': self._execute_search_web,
            'create_file': self._execute_create_file,
            'delete_file': self._execute_delete_file,
            'copy_text': self._execute_copy_text,
            'paste_text': self._execute_paste_text,
            'save_file': self._execute_save_file,
            'get_time': self._execute_get_time,
            'list_files': self._execute_list_files,
            'create_folder': self._execute_create_folder,
            'take_screenshot': self._execute_take_screenshot,
            'system_info': self._execute_system_info,
            'greeting': self._execute_greeting,
            'goodbye': self._execute_goodbye,
            'help': self._execute_help
        }
        
        return actions
    
    def _load_safety_rules(self) -> Dict[str, Any]:
        """Загружает правила безопасности"""
        return {
            'allowed_directories': [
                str(Path.home() / "Desktop"),
                str(Path.home() / "Documents"),
                str(Path.home() / "Downloads"),
                os.getcwd()
            ],
            'blocked_commands': [
                'rm -rf', 'format', 'del /f', 'shutdown', 'taskkill',
                'reg delete', 'chmod 777', 'wmic', 'diskpart'
            ],
            'max_file_size_mb': 100,
            'require_confirmation_for': [
                'delete', 'remove', 'uninstall', 'format', 'shutdown',
                'kill', 'terminate', 'override', 'overwrite'
            ]
        }
    
    def _init_safe_execution(self):
        """Инициализирует безопасное выполнение"""
        # Создаем песочницу для выполнения кода
        self.safe_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'isinstance': isinstance,
                'type': type,
                'Exception': Exception
            },
            'os': self._get_safe_os_module(),
            'sys': sys,
            'time': time,
            'json': json,
            'pathlib': Path,
            'subprocess': self._get_safe_subprocess(),
            'webbrowser': webbrowser,
            'pyautogui': self._get_safe_pyautogui()
        }
    
    def _get_safe_os_module(self):
        """Возвращает безопасную версию модуля os"""
        safe_os = type('SafeOS', (), {})()
        
        # Разрешенные методы os
        safe_methods = [
            'getcwd', 'listdir', 'mkdir', 'makedirs', 'remove',
            'rmdir', 'rename', 'path.exists', 'path.isdir', 'path.isfile',
            'path.join', 'path.basename', 'path.dirname', 'path.splitext'
        ]
        
        # Динамически добавляем методы
        for method in safe_methods:
            if '.' in method:
                # Вложенные атрибуты (например, os.path.exists)
                parts = method.split('.')
                obj = os
                for part in parts:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        break
                else:
                    # Создаем безопасную обертку
                    setattr(safe_os, parts[-1], obj)
            else:
                if hasattr(os, method):
                    setattr(safe_os, method, getattr(os, method))
        
        return safe_os
    
    def _get_safe_subprocess(self):
        """Возвращает безопасную версию subprocess"""
        safe_subprocess = type('SafeSubprocess', (), {})()
        
        # Только безопасные методы
        def safe_run(cmd, **kwargs):
            """Безопасный запуск команды"""
            # Проверяем команду на безопасность
            if isinstance(cmd, str):
                cmd_lower = cmd.lower()
                for blocked in self.safety_rules['blocked_commands']:
                    if blocked in cmd_lower:
                        raise PermissionError(f"Команда заблокирована: {blocked}")
            
            # Ограничиваем время выполнения
            kwargs['timeout'] = kwargs.get('timeout', 30)
            
            # Запрещаем shell=True для строковых команд
            if isinstance(cmd, str):
                kwargs['shell'] = False
            
            return subprocess.run(cmd, **kwargs)
        
        safe_subprocess.run = safe_run
        safe_subprocess.Popen = lambda *args, **kwargs: None  # Заглушка
        
        return safe_subprocess
    
    def _get_safe_pyautogui(self):
        """Возвращает безопасную версию pyautogui"""
        safe_pyautogui = type('SafePyAutoGUI', (), {})()
        
        # Разрешенные методы
        allowed_methods = ['write', 'hotkey', 'click', 'moveTo', 'size']
        
        for method in allowed_methods:
            if hasattr(pyautogui, method):
                # Создаем обертку с задержкой для безопасности
                def make_safe_wrapper(original_method):
                    def wrapper(*args, **kwargs):
                        # Добавляем небольшую задержку для предотвращения ошибок
                        time.sleep(0.1)
                        return original_method(*args, **kwargs)
                    return wrapper
                
                setattr(safe_pyautogui, method, 
                       make_safe_wrapper(getattr(pyautogui, method)))
        
        return safe_pyautogui
    
    def execute(self, intent: str, entities: List[Dict], original_text: str, 
                context: Dict[str, Any]) -> ExecutionResult:
        """
        Выполняет команду на основе намерения и сущностей
        
        Args:
            intent: Намерение (intent label)
            entities: Извлеченные сущности
            original_text: Оригинальный текст команды
            context: Контекст выполнения
            
        Returns:
            Результат выполнения
        """
        start_time = time.time()
        self.execution_stats['total_commands'] += 1
        
        print(f"🚀 Выполняю команду: {original_text}")
        print(f"   Намерение: {intent}")
        print(f"   Сущности: {entities}")
        
        # Проверяем безопасность
        safety_level = self._assess_safety(intent, entities, original_text)
        
        if safety_level == CommandSafetyLevel.DANGEROUS:
            self.execution_stats['blocked'] += 1
            return ExecutionResult(
                success=False,
                message="Команда заблокирована из соображений безопасности",
                data={'blocked': True, 'safety_level': 'DANGEROUS'},
                execution_time=time.time() - start_time,
                safety_level=safety_level,
                requires_confirmation=False
            )
        
        # Проверяем нужна ли подтверждение
        requires_confirmation = self._requires_confirmation(intent, entities, original_text)
        
        # Определяем действие
        action_func = self._find_action(intent, entities)
        
        if not action_func:
            # Пытаемся найти загруженное действие
            action_func = self._load_learned_action(intent)
            
            if not action_func:
                self.execution_stats['failed'] += 1
                return ExecutionResult(
                    success=False,
                    message=f"Неизвестное намерение: {intent}",
                    data={'intent': intent, 'entities': entities},
                    execution_time=time.time() - start_time,
                    safety_level=safety_level,
                    requires_confirmation=False
                )
        
        try:
            # Выполняем действие
            if requires_confirmation and self.config.get('confirm_destructive', True):
                # В реальном ассистенте здесь было бы ожидание подтверждения
                print(f"⚠️ Требуется подтверждение для: {original_text}")
            
            result_data = action_func(entities, original_text, context)
            
            # Форматируем результат
            if isinstance(result_data, dict):
                message = result_data.get('message', 'Команда выполнена')
                data = result_data.get('data', {})
            else:
                message = str(result_data)
                data = {'raw_result': result_data}
            
            self.execution_stats['successful'] += 1
            
            return ExecutionResult(
                success=True,
                message=message,
                data=data,
                execution_time=time.time() - start_time,
                safety_level=safety_level,
                requires_confirmation=requires_confirmation
            )
            
        except Exception as e:
            self.execution_stats['failed'] += 1
            print(f"❌ Ошибка выполнения: {e}")
            
            return ExecutionResult(
                success=False,
                message=f"Ошибка выполнения: {str(e)}",
                data={'error': str(e), 'traceback': self._get_traceback()},
                execution_time=time.time() - start_time,
                safety_level=safety_level,
                requires_confirmation=False
            )
    
    def _assess_safety(self, intent: str, entities: List[Dict], text: str) -> CommandSafetyLevel:
        """Оценивает уровень безопасности команды"""
        text_lower = text.lower()
        
        # Опасные команды
        dangerous_keywords = [
            'удали все', 'форматируй', 'отключи компьютер', 'убей процесс',
            'стереть все', 'уничтожь', 'взломай', 'взлом'
        ]
        
        if any(keyword in text_lower for keyword in dangerous_keywords):
            return CommandSafetyLevel.DANGEROUS
        
        # Рискованные команды
        risky_keywords = [
            'удали', 'стереть', 'отключи', 'перезагрузи', 'выключи',
            'измени реестр', 'настрой систему', 'установи'
        ]
        
        if any(keyword in text_lower for keyword in risky_keywords):
            return CommandSafetyLevel.RISKY
        
        # Команды средней опасности
        moderate_keywords = [
            'создай', 'измени', 'переименуй', 'перемести', 'скопируй',
            'запиши', 'сохрани', 'отредактируй'
        ]
        
        if any(keyword in text_lower for keyword in moderate_keywords):
            return CommandSafetyLevel.MODERATE
        
        # Безопасные команды
        return CommandSafetyLevel.SAFE
    
    def _requires_confirmation(self, intent: str, entities: List[Dict], text: str) -> bool:
        """Определяет, требуется ли подтверждение"""
        text_lower = text.lower()
        
        confirmation_keywords = self.safety_rules['require_confirmation_for']
        
        if any(keyword in text_lower for keyword in confirmation_keywords):
            return True
        
        # Проверяем по сущностям
        for entity in entities:
            if entity.get('label') in ['FILE', 'DIRECTORY', 'PROGRAM']:
                if 'delete' in intent or 'remove' in intent:
                    return True
        
        return False
    
    def _find_action(self, intent: str, entities: List[Dict]) -> Optional[Callable]:
        """Находит функцию для выполнения"""
        # Прямое сопоставление
        if intent in self.base_actions:
            return self.base_actions[intent]
        
        # Ищем по паттернам
        intent_lower = intent.lower()
        
        if 'open' in intent_lower or 'запусти' in intent_lower:
            return self.base_actions.get('open_program')
        elif 'type' in intent_lower or 'напечатай' in intent_lower:
            return self.base_actions.get('type_text')
        elif 'search' in intent_lower or 'найди' in intent_lower:
            return self.base_actions.get('search_web')
        elif 'create' in intent_lower or 'создай' in intent_lower:
            if 'file' in intent_lower or 'файл' in intent_lower:
                return self.base_actions.get('create_file')
            elif 'folder' in intent_lower or 'папк' in intent_lower:
                return self.base_actions.get('create_folder')
        
        return None
    
    def _load_learned_action(self, intent_label: str) -> Optional[Callable]:
        """Загружает выученное действие"""
        if intent_label in self.learned_actions:
            return self.learned_actions[intent_label]
        
        # Пытаемся найти файл с кодом
        actions_dir = Path(__file__).parent.parent / "models" / "actions"
        action_file = actions_dir / f"{intent_label}.py"
        
        if action_file.exists():
            try:
                # Динамически загружаем модуль
                spec = importlib.util.spec_from_file_location(intent_label, action_file)
                module = importlib.util.module_from_spec(spec)
                
                # Загружаем в безопасное окружение
                with open(action_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # Выполняем код в безопасном окружении
                exec(code, {**self.safe_globals, '__name__': '__main__'})
                
                # Ищем функцию execute
                if 'execute' in locals():
                    action_func = locals()['execute']
                    
                    # Создаем обертку
                    def wrapped_action(entities, original_text, context):
                        try:
                            # Подготавливаем параметры
                            params = self._prepare_parameters(entities, original_text)
                            
                            # Вызываем функцию
                            result = action_func(**params)
                            return {'message': str(result), 'data': {'result': result}}
                        except Exception as e:
                            raise Exception(f"Ошибка в выученной команде: {e}")
                    
                    self.learned_actions[intent_label] = wrapped_action
                    return wrapped_action
                    
            except Exception as e:
                print(f"❌ Ошибка загрузки выученного действия: {e}")
        
        return None
    
    def _prepare_parameters(self, entities: List[Dict], original_text: str) -> Dict[str, Any]:
        """Подготавливает параметры для функции"""
        params = {}
        
        # Извлекаем сущности
        for entity in entities:
            label = entity.get('label', '').lower()
            text = entity.get('text', '')
            
            if label in ['file', 'file_name']:
                params['file_path'] = text
            elif label in ['directory', 'folder']:
                params['directory_path'] = text
            elif label in ['program', 'app', 'application']:
                params['program_name'] = text
            elif label in ['text', 'content']:
                params['text'] = text
            elif label in ['query', 'search']:
                params['query'] = text
            elif label in ['url', 'website']:
                params['url'] = text
        
        # Если не нашли сущностей, пытаемся извлечь из текста
        if not params and original_text:
            # Простой парсинг
            if 'напечатай' in original_text.lower():
                # Извлекаем текст после "напечатай"
                parts = original_text.lower().split('напечатай', 1)
                if len(parts) > 1:
                    params['text'] = parts[1].strip()
            
            elif 'создай файл' in original_text.lower():
                # Извлекаем имя файла
                import re
                match = re.search(r'создай файл\s+([^\s]+)', original_text.lower())
                if match:
                    params['file_path'] = match.group(1)
        
        return params
    
    def _get_traceback(self) -> str:
        """Получает traceback ошибки"""
        import traceback
        return traceback.format_exc()
    
    # ========== БАЗОВЫЕ ДЕЙСТВИЯ ==========
    
    def _execute_open_program(self, entities: List[Dict], original_text: str, context: Dict) -> Dict[str, Any]:
        """Умное открытие программ с использованием SmartProgramOpener"""
        try:
            # Импортируем умный открыватель
            from execution.program_opener import SmartProgramOpener
        
            # Создаем или используем существующий открыватель
            if not hasattr(self, '_program_opener'):
                self._program_opener = SmartProgramOpener()
        
            # Открываем программу
            result = self._program_opener.open_program(original_text)
        
            if result['success']:
                return {
                    'message': result['message'],
                    'data': {
                        'program': result.get('program', {}),
                        'method': result.get('method', 'unknown'),
                        'original_text': original_text
                    }
                }
            else:
                # Если программа не найдена
                if result.get('needs_learning'):
                    return {
                        'message': result['message'],
                        'data': {
                            'needs_learning': True,
                            'program_name': result.get('program_name'),
                            'original_text': original_text
                        },
                        'requires_confirmation': True
                    }
                else:
                    raise Exception(result['message'])
                
        except ImportError:
            # Фолбэк на старую логику если модуль не найден
            return self._execute_open_program_fallback(entities, original_text, context)
    
    def _execute_type_text(self, entities: List[Dict], original_text: str, context: Dict) -> Dict[str, Any]:
        """Печатает текст"""
        text_to_type = None
        
        # Ищем текст в сущностях
        for entity in entities:
            if entity.get('label') in ['TEXT', 'CONTENT']:
                text_to_type = entity.get('text')
                break
        
        # Если не нашли, извлекаем из команды
        if not text_to_type:
            import re
            match = re.search(r'(?:напечатай|напиши|введи)\s+(.+)', original_text, re.IGNORECASE)
            if match:
                text_to_type = match.group(1).strip()
        
        if not text_to_type:
            raise Exception("Не указан текст для печати")
        
        # Ждем перед печатью
        time.sleep(1)
        
        try:
            pyautogui.write(text_to_type, interval=0.05)
            return {
                'message': f"Напечатано: {text_to_type}",
                'data': {'text': text_to_type, 'length': len(text_to_type)}
            }
        except Exception as e:
            raise Exception(f"Ошибка печати: {e}")
    
    def _execute_search_web(self, entities: List[Dict], original_text: str, context: Dict) -> Dict[str, Any]:
        """Ищет в интернете"""
        query = None
        
        # Ищем запрос в сущностях
        for entity in entities:
            if entity.get('label') in ['QUERY', 'SEARCH']:
                query = entity.get('text')
                break
        
        # Если не нашли, извлекаем из команды
        if not query:
            import re
            match = re.search(r'(?:найди|поищи|ищи)\s+(.+)', original_text, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
        
        if not query:
            raise Exception("Не указан запрос для поиска")
        
        # Кодируем запрос
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        
        # Открываем в браузере
        search_url = f"https://www.google.com/search?q={encoded_query}"
        webbrowser.open(search_url)
        
        return {
            'message': f"Ищу в интернете: {query}",
            'data': {'query': query, 'url': search_url}
        }
    
    def _execute_create_file(self, entities: List[Dict], original_text: str, context: Dict) -> Dict[str, Any]:
        """Создает файл"""
        file_path = None
        content = ""
        
        # Ищем путь к файлу в сущностях
        for entity in entities:
            if entity.get('label') == 'FILE':
                file_path = entity.get('text')
                break
        
        # Ищем содержимое
        for entity in entities:
            if entity.get('label') == 'CONTENT':
                content = entity.get('text', '')
                break
        
        # Если не нашли путь, создаем стандартный
        if not file_path:
            timestamp = int(time.time())
            file_path = f"новый_файл_{timestamp}.txt"
        
        # Проверяем безопасность пути
        self._validate_path_safety(file_path)
        
        # Создаем директорию если нужно
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Создаем файл
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            'message': f"Файл создан: {file_path}",
            'data': {'file_path': file_path, 'content_length': len(content)}
        }
    
    def _execute_delete_file(self, entities: List[Dict], original_text: str, context: Dict) -> Dict[str, Any]:
        """Удаляет файл"""
        file_path = None
        
        # Ищем путь к файлу в сущностях
        for entity in entities:
            if entity.get('label') == 'FILE':
                file_path = entity.get('text')
                break
        
        if not file_path:
            raise Exception("Не указан файл для удаления")
        
        # Проверяем безопасность пути
        self._validate_path_safety(file_path)
        
        # Проверяем что файл существует
        if not os.path.exists(file_path):
            raise Exception(f"Файл не существует: {file_path}")
        
        # Удаляем файл
        os.remove(file_path)
        
        return {
            'message': f"Файл удален: {file_path}",
            'data': {'file_path': file_path}
        }
    
    def _execute_copy_text(self, entities: List[Dict], original_text: str, context: Dict) -> Dict[str, Any]:
        """Копирует текст в буфер обмена"""
        try:
            pyautogui.hotkey('ctrl', 'c')
            time.sleep(0.2)
            return {
                'message': "Текст скопирован в буфер обмена",
                'data': {'action': 'copy'}
            }
        except Exception as e:
            raise Exception(f"Ошибка копирования: {e}")
    
    def _execute_paste_text(self, entities: List[Dict], original_text: str, context: Dict) -> Dict[str, Any]:
        """Вставляет текст из буфера обмена"""
        try:
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(0.2)
            return {
                'message': "Текст вставлен из буфера обмена",
                'data': {'action': 'paste'}
            }
        except Exception as e:
            raise Exception(f"Ошибка вставки: {e}")
    
    def _execute_save_file(self, entities: List[Dict], original_text: str, context: Dict) -> Dict[str, Any]:
        """Сохраняет файл"""
        try:
            pyautogui.hotkey('ctrl', 's')
            time.sleep(0.2)
            return {
                'message': "Файл сохранен",
                'data': {'action': 'save'}
            }
        except Exception as e:
            raise Exception(f"Ошибка сохранения: {e}")
    
    def _execute_get_time(self, entities: List[Dict], original_text: str, context: Dict) -> Dict[str, Any]:
        """Возвращает текущее время"""
        from datetime import datetime
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%d.%m.%Y")
        
        return {
            'message': f"Сейчас {time_str}, {date_str}",
            'data': {'time': time_str, 'date': date_str}
        }
    
    def _execute_list_files(self, entities: List[Dict], original_text: str, context: Dict) -> Dict[str, Any]:
        """Список файлов в директории"""
        directory = os.getcwd()
        
        # Ищем директорию в сущностях
        for entity in entities:
            if entity.get('label') == 'DIRECTORY':
                directory = entity.get('text')
                break
        
        # Проверяем безопасность пути
        self._validate_path_safety(directory)
        
        # Получаем список файлов
        try:
            files = os.listdir(directory)
            files_str = "\n".join(files[:20])  # Ограничиваем вывод
            
            if len(files) > 20:
                files_str += f"\n... и еще {len(files) - 20} файлов"
            
            return {
                'message': f"Файлы в {directory}:\n{files_str}",
                'data': {'directory': directory, 'file_count': len(files)}
            }
        except Exception as e:
            raise Exception(f"Не удалось получить список файлов: {e}")
    
    def _execute_create_folder(self, entities: List[Dict], original_text: str, context: Dict) -> Dict[str, Any]:
        """Создает папку"""
        folder_path = None
        
        # Ищем путь в сущностях
        for entity in entities:
            if entity.get('label') == 'DIRECTORY':
                folder_path = entity.get('text')
                break
        
        # Если не нашли, создаем стандартную папку
        if not folder_path:
            timestamp = int(time.time())
            folder_path = f"новая_папка_{timestamp}"
        
        # Проверяем безопасность пути
        self._validate_path_safety(folder_path)
        
        # Создаем папку
        os.makedirs(folder_path, exist_ok=True)
        
        return {
            'message': f"Папка создана: {folder_path}",
            'data': {'folder_path': folder_path}
        }
    
    def _execute_take_screenshot(self, entities: List[Dict], original_text: str, context: Dict) -> Dict[str, Any]:
        """Делает скриншот"""
        try:
            screenshot = pyautogui.screenshot()
            timestamp = int(time.time())
            filename = f"screenshot_{timestamp}.png"
            screenshot.save(filename)
            
            return {
                'message': f"Скриншот сохранен: {filename}",
                'data': {'filename': filename, 'size': screenshot.size}
            }
        except Exception as e:
            raise Exception(f"Ошибка создания скриншота: {e}")
    
    def _execute_system_info(self, entities: List[Dict], original_text: str, context: Dict) -> Dict[str, Any]:
        """Возвращает информацию о системе"""
        import platform
        
        system_info = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
        
        info_str = "\n".join([f"{k}: {v}" for k, v in system_info.items()])
        
        return {
            'message': f"Информация о системе:\n{info_str}",
            'data': system_info
        }
    
    def _execute_greeting(self, entities: List[Dict], original_text: str, context: Dict) -> Dict[str, Any]:
        """Приветствует пользователя"""
        import random
        
        greetings = [
            "Привет! Рад вас видеть!",
            "Здравствуйте! Чем могу помочь?",
            "Приветствую! Готов к работе!",
            "Здорово! Что нужно сделать?"
        ]
        
        greeting = random.choice(greetings)
        
        return {
            'message': greeting,
            'data': {'greeting_type': 'welcome'}
        }
    
    def _execute_goodbye(self, entities: List[Dict], original_text: str, context: Dict) -> Dict[str, Any]:
        """Прощается с пользователем"""
        import random
        
        goodbyes = [
            "До свидания! Была рада помочь!",
            "Пока! Возвращайтесь снова!",
            "Всего хорошего!",
            "До новых встреч!"
        ]
        
        goodbye = random.choice(goodbyes)
        
        return {
            'message': goodbye,
            'data': {'action': 'exit'}
        }
    
    def _execute_help(self, entities: List[Dict], original_text: str, context: Dict) -> Dict[str, Any]:
        """Показывает помощь"""
        help_text = """
Доступные команды:
• Открой [программу] - открыть программу
• Напечатай [текст] - напечатать текст
• Найди [запрос] - поиск в интернете
• Создай файл [имя] - создать файл
• Создай папку [имя] - создать папку
• Удали файл [имя] - удалить файл
• Сколько времени - узнать время
• Список файлов - показать файлы в папке
• Скриншот - сделать скриншот
• Информация о системе - показать системную информацию
• Привет/Пока - приветствие/прощание

Также я могу учиться новым командам!
        """
        
        return {
            'message': help_text,
            'data': {'help_type': 'general'}
        }
    
    def _validate_path_safety(self, path: str):
        """Проверяет безопасность пути"""
        # Получаем абсолютный путь
        abs_path = os.path.abspath(path)
        
        # Проверяем что путь в разрешенной директории
        allowed = False
        for allowed_dir in self.safety_rules['allowed_directories']:
            if abs_path.startswith(allowed_dir):
                allowed = True
                break
        
        if not allowed:
            raise PermissionError(f"Доступ к пути запрещен: {abs_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику выполнения"""
        return self.execution_stats.copy()

# Утилита для тестирования
def test_command_executor():
    """Тестирует исполнитель команд"""
    print("🧪 Тестирование исполнителя команд...")
    
    config = {
        'confirm_destructive': True,
        'paths': {
            'browser': 'chrome.exe'
        }
    }
    
    executor = NeuralCommandDispatcher(config)
    
    # Тестовые команды
    test_cases = [
        {
            'intent': 'greeting',
            'entities': [],
            'text': 'привет',
            'context': {}
        },
        {
            'intent': 'open_program',
            'entities': [{'label': 'PROGRAM', 'text': 'блокнот'}],
            'text': 'открой блокнот',
            'context': {}
        },
        {
            'intent': 'get_time',
            'entities': [],
            'text': 'сколько времени',
            'context': {}
        },
        {
            'intent': 'system_info',
            'entities': [],
            'text': 'информация о системе',
            'context': {}
        }
    ]
    
    for test in test_cases:
        print(f"\n📋 Тест: {test['text']}")
        
        result = executor.execute(
            intent=test['intent'],
            entities=test['entities'],
            original_text=test['text'],
            context=test['context']
        )
        
        print(f"✅ Результат: {result.success}")
        print(f"📝 Сообщение: {result.message[:100]}...")
        print(f"🛡️ Уровень безопасности: {result.safety_level.name}")
    
    # Проверяем статистику
    stats = executor.get_stats()
    print(f"\n📊 Статистика выполнения:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_command_executor()