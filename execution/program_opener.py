"""
Умный открыватель программ с поиском в системе и обучением
"""
import os
import subprocess
import json
import glob
import shutil
import winreg
import getpass
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple
import sys

class SmartProgramOpener:
    """
    Умный открыватель программ:
    1. Ищет в известных программах
    2. Ищет в системе (реестр, PATH, стандартные пути)
    3. Учится новым программам
    4. Запоминает выбор пользователя
    """
    
    def __init__(self, config_path: str = "data/programs.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(exist_ok=True)
        
        self.username = getpass.getuser()
        self.known_programs = self._load_base_programs()
        self.learned_programs = self._load_learned_programs()
        self.user_preferences = self._load_preferences()
        
        # Кэш найденных программ
        self.program_cache = {}
        
        print(f"🤖 Умный открыватель программ инициализирован")
        print(f"   Известно {len(self.known_programs)} базовых программ")
        print(f"   Выучено {len(self.learned_programs)} пользовательских программ")
    
    def _load_base_programs(self) -> Dict[str, Dict[str, Any]]:
        """Базовые программы (встроенные)"""
        return {
            'calculator': {
                'id': 'calculator',
                'names': ['калькулятор', 'calc', 'считалка', 'calculator', 'кальк'],
                'command': 'calc.exe',
                'type': 'system',
                'category': 'system_tools',
                'weight': 10
            },
            'notepad': {
                'id': 'notepad',
                'names': ['блокнот', 'notepad', 'текстовый', 'заметки', 'notes'],
                'command': 'notepad.exe',
                'type': 'system',
                'category': 'editors',
                'weight': 10
            },
            'chrome': {
                'id': 'chrome',
                'names': ['браузер', 'хром', 'chrome', 'browser', 'интернет', 'гугл'],
                'command': self._find_chrome_path(),
                'type': 'browser',
                'category': 'internet',
                'weight': 9
            },
            'explorer': {
                'id': 'explorer',
                'names': ['проводник', 'explorer', 'файлы', 'папки', 'диск'],
                'command': 'explorer.exe',
                'type': 'system',
                'category': 'file_management',
                'weight': 8
            },
            'cmd': {
                'id': 'cmd',
                'names': ['терминал', 'cmd', 'командная строка', 'консоль', 'powershell'],
                'command': 'cmd.exe',
                'type': 'system',
                'category': 'development',
                'weight': 7
            },
            'control': {
                'id': 'control',
                'names': ['панель управления', 'control', 'настройки', 'settings'],
                'command': 'control.exe',
                'type': 'system',
                'category': 'system_tools',
                'weight': 6
            },
            'mspaint': {
                'id': 'mspaint',
                'names': ['паинт', 'paint', 'рисование', 'краска'],
                'command': 'mspaint.exe',
                'type': 'system',
                'category': 'graphics',
                'weight': 5
            },
            'wordpad': {
                'id': 'wordpad',
                'names': ['вордпад', 'wordpad', 'текстовый редактор'],
                'command': 'write.exe',  # write.exe = WordPad в Windows
                'type': 'system',
                'category': 'editors',
                'weight': 4
            }
        }
    
    def _find_chrome_path(self) -> str:
        """Находит путь к Chrome"""
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            fr"C:\Users\{self.username}\AppData\Local\Google\Chrome\Application\chrome.exe",
            "chrome.exe"
        ]
        
        for path in chrome_paths:
            if os.path.exists(path):
                return path
        
        return "chrome.exe"
    
    def _load_learned_programs(self) -> Dict[str, Dict[str, Any]]:
        """Загружает выученные программы"""
        if not self.config_path.exists():
            return {}
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('learned_programs', {})
        except:
            return {}
    
    def _load_preferences(self) -> Dict[str, Any]:
        """Загружает предпочтения пользователя"""
        if not self.config_path.exists():
            return {}
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('preferences', {})
        except:
            return {}
    
    def _save_data(self):
        """Сохраняет все данные"""
        data = {
            'learned_programs': self.learned_programs,
            'preferences': self.user_preferences,
            'metadata': {
                'saved_at': 'now',
                'total_programs': len(self.known_programs) + len(self.learned_programs)
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def find_program(self, request: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Находит программу по запросу
        
        Returns:
            (программа, метод_поиска)
        """
        request_lower = request.lower()
        
        # 1. Кэш (быстрый поиск)
        if request_lower in self.program_cache:
            cached = self.program_cache[request_lower]
            return cached['program'], cached['method']
        
        # 2. Предпочтения пользователя
        if request_lower in self.user_preferences:
            program_id = self.user_preferences[request_lower]
            if program_id in self.learned_programs:
                program = self.learned_programs[program_id]
                self._update_cache(request_lower, program, 'preference')
                return program, 'preference'
        
        # 3. Известные программы (прямое совпадение)
        all_programs = {**self.known_programs, **self.learned_programs}
        
        for program_id, program in all_programs.items():
            for name in program.get('names', []):
                if name.lower() == request_lower or name.lower() in request_lower:
                    self._update_cache(request_lower, program, 'exact_match')
                    return program, 'exact_match'
        
        # 4. Похожие названия (fuzzy match)
        best_match = None
        best_score = 0
        
        for program_id, program in all_programs.items():
            for name in program.get('names', []):
                # Простой алгоритм схожести
                score = self._similarity_score(request_lower, name.lower())
                if score > best_score and score > 0.3:
                    best_score = score
                    best_match = program
        
        if best_match:
            self._update_cache(request_lower, best_match, 'similarity')
            return best_match, 'similarity'
        
        # 5. Поиск в системе
        system_program = self._search_in_system(request_lower)
        if system_program:
            self._update_cache(request_lower, system_program, 'system_search')
            return system_program, 'system_search'
        
        # 6. Поиск по категориям
        category_program = self._search_by_category(request_lower)
        if category_program:
            self._update_cache(request_lower, category_program, 'category')
            return category_program, 'category'
        
        return None, 'not_found'
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """Вычисляет схожесть двух строк"""
        if not text1 or not text2:
            return 0.0
        
        # Простой алгоритм
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        common = words1.intersection(words2)
        return len(common) / max(len(words1), len(words2))
    
    def _search_in_system(self, request: str) -> Optional[Dict[str, Any]]:
        """Ищет программу в системе Windows"""
        
        # Популярные программы (шаблоны путей)
        popular_templates = {
            'telegram': [
                fr"C:\Users\{self.username}\AppData\Roaming\Telegram Desktop\Telegram.exe",
                r"C:\Program Files\Telegram Desktop\Telegram.exe"
            ],
            'discord': fr"C:\Users\{self.username}\AppData\Local\Discord\app-*\Discord.exe",
            'whatsapp': fr"C:\Users\{self.username}\AppData\Local\WhatsApp\WhatsApp.exe",
            'vscode': [
                fr"C:\Users\{self.username}\AppData\Local\Programs\Microsoft VS Code\Code.exe",
                r"C:\Program Files\Microsoft VS Code\Code.exe"
            ],
            'pycharm': r"C:\Program Files\JetBrains\PyCharm *\bin\pycharm64.exe",
            'intellij': r"C:\Program Files\JetBrains\IntelliJ IDEA *\bin\idea64.exe",
            'photoshop': r"C:\Program Files\Adobe\Adobe Photoshop *\Photoshop.exe",
            'illustrator': r"C:\Program Files\Adobe\Adobe Illustrator *\Support Files\Contents\Windows\Illustrator.exe",
            'word': r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
            'excel': r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
            'powerpoint': r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE",
            'steam': r"C:\Program Files (x86)\Steam\steam.exe",
            'spotify': fr"C:\Users\{self.username}\AppData\Roaming\Spotify\Spotify.exe",
            'obs': r"C:\Program Files\obs-studio\bin\64bit\obs64.exe",
            'vlc': r"C:\Program Files\VideoLAN\VLC\vlc.exe",
            'firefox': [
                r"C:\Program Files\Mozilla Firefox\firefox.exe",
                r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe"
            ],
            'edge': r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            'opera': r"C:\Program Files\Opera\launcher.exe"
        }
        
        # Проверяем каждую популярную программу
        for prog_key, templates in popular_templates.items():
            if prog_key in request or any(word in request for word in prog_key.split('_')):
                if isinstance(templates, str):
                    templates = [templates]
                
                for template in templates:
                    # Заменяем звездочки
                    if '*' in template:
                        matches = glob.glob(template)
                        if matches:
                            path = matches[0]
                            if os.path.exists(path):
                                return self._create_program_info(prog_key, path, 'system')
                    else:
                        if os.path.exists(template):
                            return self._create_program_info(prog_key, template, 'system')
        
        # Поиск в реестре
        registry_program = self._search_in_registry(request)
        if registry_program:
            return registry_program
        
        # Поиск в PATH
        path_program = self._search_in_path(request)
        if path_program:
            return path_program
        
        return None
    
    def _search_in_registry(self, request: str) -> Optional[Dict[str, Any]]:
        """Ищет программу в реестре Windows"""
        try:
            # App Paths
            key_paths = [
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths",
                r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths"
            ]
            
            for key_path in key_paths:
                try:
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                    
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            if request in subkey_name.lower():
                                subkey = winreg.OpenKey(key, subkey_name)
                                path = winreg.QueryValue(subkey, "")
                                winreg.CloseKey(subkey)
                                
                                if os.path.exists(path):
                                    program_name = os.path.splitext(subkey_name)[0]
                                    return self._create_program_info(program_name, path, 'registry')
                        except:
                            continue
                    
                    winreg.CloseKey(key)
                except:
                    continue
                    
        except Exception as e:
            print(f"⚠️ Ошибка поиска в реестре: {e}")
        
        return None
    
    def _search_in_path(self, request: str) -> Optional[Dict[str, Any]]:
        """Ищет программу в PATH"""
        # Пробуем разные расширения
        extensions = ['.exe', '.bat', '.cmd', '.msi']
        
        for ext in extensions:
            program_name = request + ext
            path = shutil.which(program_name)
            if path:
                return self._create_program_info(request, path, 'path')
        
        return None
    
    def _search_by_category(self, request: str) -> Optional[Dict[str, Any]]:
        """Ищет программу по категории"""
        category_keywords = {
            'браузер': ['chrome', 'firefox', 'edge', 'opera'],
            'редактор': ['notepad', 'vscode', 'sublime', 'word'],
            'игра': ['steam', 'game', 'игра'],
            'музыка': ['spotify', 'music', 'плеер'],
            'видео': ['vlc', 'player', 'медиа'],
            'графика': ['paint', 'photoshop', 'рисование'],
            'офис': ['word', 'excel', 'powerpoint']
        }
        
        for category, programs in category_keywords.items():
            if category in request:
                # Возвращаем самую популярную программу из категории
                for program_id in programs:
                    if program_id in self.known_programs:
                        return self.known_programs[program_id]
        
        return None
    
    def _create_program_info(self, name: str, path: str, source: str) -> Dict[str, Any]:
        """Создает информацию о программе"""
        program_id = name.lower().replace(' ', '_').replace('.exe', '')
        
        return {
            'id': program_id,
            'names': [name],
            'command': path,
            'type': 'detected',
            'source': source,
            'weight': 5
        }
    
    def _update_cache(self, request: str, program: Dict[str, Any], method: str):
        """Обновляет кэш"""
        self.program_cache[request] = {
            'program': program,
            'method': method,
            'timestamp': 'now'
        }
    
    def learn_new_program(self, request: str, program_path: str, aliases: List[str] = None):
        """Учит новую программу"""
        program_name = os.path.basename(program_path).replace('.exe', '')
        program_id = program_name.lower().replace(' ', '_')
        
        self.learned_programs[program_id] = {
            'id': program_id,
            'names': aliases or [program_name, request],
            'command': program_path,
            'type': 'learned',
            'weight': 8,
            'learned_at': 'now'
        }
        
        # Сохраняем предпочтение
        self.user_preferences[request.lower()] = program_id
        
        self._save_data()
        
        print(f"🎓 Выучена новая программа: {program_name}")
        return True
    
    def open_program(self, request: str) -> Dict[str, Any]:
        """
        Открывает программу
        
        Returns:
            Словарь с результатом
        """
        print(f"🔍 Ищу программу для запроса: '{request}'")
        
        program, method = self.find_program(request)
        
        if not program:
            return {
                'success': False,
                'message': f"Не знаю как открыть '{request}'. Не найдено в системе.",
                'program_name': request,
                'needs_learning': True
            }
        
        print(f"   Найдено: {program.get('id')} (метод: {method})")
        print(f"   Команда: {program.get('command')}")
        
        try:
            # Запускаем программу
            command = program['command']
            
            if os.path.exists(command):
                subprocess.Popen(command, shell=True)
            else:
                # Пробуем без полного пути
                exe_name = os.path.basename(command)
                subprocess.Popen(exe_name, shell=True)
            
            # Получаем человекочитаемое имя
            display_name = program.get('names', [program.get('id', 'программу')])[0]
            
            return {
                'success': True,
                'message': f"Открываю {display_name}",
                'program': program,
                'method': method
            }
            
        except Exception as e:
            print(f"❌ Ошибка открытия: {e}")
            return {
                'success': False,
                'message': f"Не удалось открыть программу: {str(e)}",
                'error': str(e),
                'program': program
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Статистика"""
        return {
            'total_base_programs': len(self.known_programs),
            'total_learned_programs': len(self.learned_programs),
            'cache_size': len(self.program_cache),
            'user_preferences': len(self.user_preferences)
        }

# Тестирование
def test_program_opener():
    """Тестирование открывателя программ"""
    print("🧪 Тестирование умного открывателя программ")
    print("=" * 60)
    
    opener = SmartProgramOpener("test_programs.json")
    
    test_cases = [
        "калькулятор",
        "блокнот",
        "браузер",
        "проводник",
        "паинт",
        "терминал",
        "telegram",  # Должен найти если установлен
        "дискорд",   # Должен найти если установлен
        "вскод",     # VS Code
        "стим",      # Steam
        "неизвестная программа 123"
    ]
    
    for test in test_cases:
        print(f"\n🔧 Запрос: '{test}'")
        result = opener.open_program(test)
        
        if result['success']:
            print(f"   ✅ {result['message']}")
            if 'method' in result:
                print(f"   📊 Метод поиска: {result['method']}")
        else:
            print(f"   ❌ {result['message']}")
            if result.get('needs_learning'):
                print(f"   🎓 Нужно обучение")
    
    # Статистика
    stats = opener.get_statistics()
    print(f"\n📊 Статистика: {stats}")

if __name__ == "__main__":
    test_program_opener()