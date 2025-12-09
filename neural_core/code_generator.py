"""
Нейросеть для генерации Python-кода по описанию команды
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class CodeExample:
    """Пример кода для обучения"""
    description: str
    code: str
    intent_type: str
    complexity: int  # 1-простой, 2-средний, 3-сложный

class CodeDataset(Dataset):
    """Датасет пар (описание → код) для обучения"""
    
    def __init__(self, examples: List[CodeExample], vocab_size: int = 5000, max_len: int = 200):
        self.examples = examples
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Словари для токенизации
        self.char_to_idx = {}
        self.idx_to_char = {}
        self._build_char_vocab()
        
        # Токенизатор для Python кода
        self.code_tokens = set()
        self._build_code_vocab()
    
    def _build_char_vocab(self):
        """Строит словарь символов"""
        chars = set()
        
        for example in self.examples:
            chars.update(example.description)
            chars.update(example.code)
        
        # Специальные токены
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        for token in special_tokens:
            chars.add(token)
        
        # Создаем маппинг
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # Если символов больше vocab_size, ограничиваем
        if len(self.char_to_idx) > self.vocab_size:
            # Оставляем самые частые символы
            char_counts = {}
            for example in self.examples:
                for char in example.description + example.code:
                    char_counts[char] = char_counts.get(char, 0) + 1
            
            sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
            top_chars = [char for char, _ in sorted_chars[:self.vocab_size - len(special_tokens)]]
            
            self.char_to_idx = {char: idx for idx, char in enumerate(special_tokens + top_chars)}
            self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
    
    def _build_code_vocab(self):
        """Строит словарь ключевых слов Python"""
        python_keywords = [
            'def', 'return', 'import', 'from', 'as', 'if', 'else', 'elif',
            'for', 'while', 'try', 'except', 'with', 'class', 'self'
        ]
        
        python_stdlib = [
            'os', 'sys', 'subprocess', 'time', 'datetime', 'json',
            'shutil', 'pathlib', 're', 'typing', 'webbrowser'
        ]
        
        self.code_tokens = set(python_keywords + python_stdlib)
    
    def encode_text(self, text: str, add_special: bool = True) -> List[int]:
        """Кодирует текст в индексы"""
        indices = []
        
        if add_special:
            indices.append(self.char_to_idx.get('<SOS>', 1))
        
        for char in text[:self.max_len - 2]:
            indices.append(self.char_to_idx.get(char, self.char_to_idx.get('<UNK>', 3)))
        
        if add_special:
            indices.append(self.char_to_idx.get('<EOS>', 2))
        
        # Паддинг
        if len(indices) < self.max_len:
            indices += [self.char_to_idx['<PAD>']] * (self.max_len - len(indices))
        
        return indices[:self.max_len]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Кодируем описание и код
        desc_encoded = self.encode_text(example.description, add_special=True)
        code_encoded = self.encode_text(example.code, add_special=True)
        
        return {
            'description': torch.tensor(desc_encoded, dtype=torch.long),
            'code': torch.tensor(code_encoded, dtype=torch.long),
            'intent_type': example.intent_type,
            'complexity': example.complexity
        }

class CodeGeneratorModel(nn.Module):
    """Модель генерации кода (seq2seq с вниманием)"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        # Эмбеддинги
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Энкодер (описание → скрытое состояние)
        self.encoder_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Декодер (скрытое состояние → код)
        self.decoder_lstm = nn.LSTM(
            input_size=embedding_dim + hidden_dim * 2,  # эмбеддинг + контекст
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Механизм внимания
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=0.3,
            batch_first=True
        )
        
        # Линейные слои
        self.encoder_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Выходной слой
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        
        # Слой для типа интента
        self.intent_embedding = nn.Embedding(10, 32)  # 10 типов интентов
        
        # Слой для сложности
        self.complexity_embedding = nn.Embedding(3, 16)  # 3 уровня сложности
    
    def forward(self, description, code_input=None, intent_type=None, complexity=None, teacher_forcing_ratio=0.5):
        """
        Прямой проход
        
        Args:
            description: Описание команды [batch, seq_len]
            code_input: Входной код для обучения [batch, seq_len]
            intent_type: Тип намерения
            complexity: Сложность команды
            teacher_forcing_ratio: Вероятность учительского форсинга
        """
        batch_size = description.size(0)
        seq_len = description.size(1)
        
        # Эмбеддинг описания
        desc_embedded = self.embedding(description)
        
        # Пропускаем через энкодер
        encoder_outputs, (hidden, cell) = self.encoder_lstm(desc_embedded)
        encoder_outputs = self.encoder_proj(encoder_outputs)
        
        # Подготавливаем декодер
        decoder_input = torch.full((batch_size, 1), 1, dtype=torch.long, device=description.device)  # <SOS>
        decoder_hidden = hidden
        decoder_cell = cell
        
        # Эмбеддинги дополнительных признаков
        if intent_type is not None:
            intent_emb = self.intent_embedding(intent_type).unsqueeze(1)
        else:
            intent_emb = torch.zeros(batch_size, 1, 32, device=description.device)
        
        if complexity is not None:
            complexity_emb = self.complexity_embedding(complexity).unsqueeze(1)
        else:
            complexity_emb = torch.zeros(batch_size, 1, 16, device=description.device)
        
        # Список выходов
        outputs = []
        
        for t in range(seq_len - 1):
            # Эмбеддинг текущего токена декодера
            decoder_emb = self.embedding(decoder_input)
            
            # Конкатенация с эмбеддингами признаков
            decoder_emb = torch.cat([
                decoder_emb,
                intent_emb.expand(-1, decoder_emb.size(1), -1),
                complexity_emb.expand(-1, decoder_emb.size(1), -1)
            ], dim=-1)
            
            # Механизм внимания
            attn_output, _ = self.attention(
                query=decoder_emb,
                key=encoder_outputs,
                value=encoder_outputs
            )
            
            # Конкатенация с выходом внимания
            decoder_input_full = torch.cat([decoder_emb, attn_output], dim=-1)
            
            # Декодер
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                decoder_input_full, (decoder_hidden, decoder_cell)
            )
            
            decoder_output = self.decoder_proj(decoder_output)
            
            # Выходной слой
            output = self.output_layer(decoder_output[:, -1, :])
            outputs.append(output.unsqueeze(1))
            
            # Следующий вход декодера
            if code_input is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Учительский форсинг
                decoder_input = code_input[:, t:t+1]
            else:
                # Используем предсказание
                _, top_idx = output.topk(1)
                decoder_input = top_idx.detach()
        
        # Собираем все выходы
        outputs = torch.cat(outputs, dim=1)
        
        return outputs

class NeuralCodeGenerator:
    """Нейросеть для генерации кода"""
    
    def __init__(self, model_path: str, data_path: str):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        
        # Параметры
        self.vocab_size = 5000
        self.embedding_dim = 128
        self.hidden_dim = 256
        self.max_len = 200
        
        # Модель и данные
        self.model = None
        self.dataset = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # игнорируем <PAD>
        
        # Типы интентов
        self.intent_types = {
            'open_program': 0,
            'type_text': 1,
            'create_file': 2,
            'search_web': 3,
            'delete_file': 4,
            'copy_text': 5,
            'paste_text': 6,
            'save_file': 7,
            'system_command': 8,
            'custom': 9
        }
        
        # Загружаем или инициализируем
        self._load_or_initialize()
        
        # База шаблонов для простых команд
        self.code_templates = self._load_templates()
    
    def _load_or_initialize(self):
        """Загружает или инициализирует модель"""
        if self.model_path.exists():
            print(f"📂 Загружаю модель генератора кода из {self.model_path}")
            self._load_model()
        else:
            print("🆕 Инициализирую новую модель генератора кода")
            self._initialize_model()
        
        if self.data_path.exists():
            self._load_training_data()
    
    def _initialize_model(self):
        """Инициализирует новую модель"""
        self.model = CodeGeneratorModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
    
    def _load_model(self):
        """Загружает сохраненную модель"""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            self.model = CodeGeneratorModel(
                vocab_size=checkpoint['vocab_size'],
                embedding_dim=checkpoint['embedding_dim'],
                hidden_dim=checkpoint['hidden_dim']
            )
            
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=0.001
            )
            
            if 'intent_types' in checkpoint:
                self.intent_types = checkpoint['intent_types']
            
            print(f"✅ Модель загружена")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            self._initialize_model()
    
    def _load_training_data(self):
        """Загружает данные для обучения"""
        if not self.data_path.exists():
            self.dataset = None
            return
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            examples = []
            for item in data:
                example = CodeExample(
                    description=item['description'],
                    code=item['code'],
                    intent_type=item.get('intent_type', 'custom'),
                    complexity=item.get('complexity', 1)
                )
                examples.append(example)
            
            self.dataset = CodeDataset(examples, self.vocab_size, self.max_len)
            print(f"📊 Загружено {len(examples)} примеров для обучения")
            
        except Exception as e:
            print(f"⚠️ Ошибка загрузки данных: {e}")
            self.dataset = None
    
    def _load_templates(self) -> Dict[str, str]:
        """Загружает шаблоны кода для разных типов команд"""
        templates = {
            'open_program': '''
def execute(program_name=None):
    """Открывает программу на компьютере"""
    import subprocess
    import os
    
    # Словарь программ
    programs = {{
        'браузер': 'chrome.exe',
        'блокнот': 'notepad.exe',
        'калькулятор': 'calc.exe',
        'проводник': 'explorer.exe',
        'терминал': 'cmd.exe',
        'панель управления': 'control.exe'
    }}
    
    if program_name and program_name in programs:
        target = programs[program_name]
    else:
        # Пытаемся найти программу
        for name, exe in programs.items():
            if name in '{program_keyword}':
                target = exe
                break
        else:
            target = 'notepad.exe'
    
    try:
        subprocess.Popen(target)
        return f"Открываю {{target}}"
    except Exception as e:
        return f"Ошибка: {{str(e)}}"
''',
            
            'type_text': '''
def execute(text_to_type=None):
    """Печатает текст в активном окне"""
    import pyautogui
    import time
    
    if not text_to_type:
        return "Не указан текст для печати"
    
    # Ждем перед печатью
    time.sleep(1)
    
    try:
        pyautogui.write(text_to_type, interval=0.05)
        return f"Напечатано: {{text_to_type}}"
    except Exception as e:
        return f"Ошибка печати: {{str(e)}}"
''',
            
            'create_file': '''
def execute(file_path=None, content=None):
    """Создает файл с указанным содержимым"""
    import os
    from pathlib import Path
    
    if not file_path:
        return "Не указан путь к файлу"
    
    try:
        # Создаем директории если нужно
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content if content else '')
        
        return f"Файл создан: {{file_path}}"
    except Exception as e:
        return f"Ошибка: {{str(e)}}"
''',
            
            'search_web': '''
def execute(query=None):
    """Ищет информацию в интернете"""
    import webbrowser
    import urllib.parse
    
    if not query:
        return "Не указан запрос для поиска"
    
    # Кодируем запрос
    encoded_query = urllib.parse.quote(query)
    
    # Открываем в браузере
    search_url = f"https://www.google.com/search?q={{encoded_query}}"
    webbrowser.open(search_url)
    
    return f"Ищу в интернете: {{query}}"
'''
        }
        
        return templates
    
    def save_model(self):
        """Сохраняет модель"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'intent_types': self.intent_types
        }
        
        torch.save(checkpoint, self.model_path)
        print(f"💾 Модель генератора кода сохранена")
    
    def generate(self, description: str, intent_type: str = 'custom', safe_mode: bool = True) -> str:
        """
        Генерирует код по описанию команды
        
        Args:
            description: Описание команды
            intent_type: Тип намерения
            safe_mode: Безопасный режим (использовать шаблоны если нейросеть не уверена)
            
        Returns:
            Сгенерированный Python код
        """
        print(f"⚙️ Генерация кода для: '{description}'")
        
        # Если модель не обучена, используем шаблоны
        if self.model is None or self.dataset is None:
            print("⚠️ Модель не обучена, использую шаблоны")
            return self._generate_from_template(description, intent_type)
        
        try:
            # Определяем сложность
            complexity = self._estimate_complexity(description)
            
            # Получаем индекс типа интента
            intent_idx = self.intent_types.get(intent_type, 9)  # 9 = custom
            
            # Кодируем описание
            desc_encoded = self.dataset.encode_text(description, add_special=True)
            desc_tensor = torch.tensor([desc_encoded], dtype=torch.long)
            
            # Генерация
            self.model.eval()
            with torch.no_grad():
                # Прямой проход
                outputs = self.model(
                    desc_tensor,
                    intent_type=torch.tensor([intent_idx], dtype=torch.long),
                    complexity=torch.tensor([complexity], dtype=torch.long),
                    teacher_forcing_ratio=0.0
                )
                
                # Декодируем выход
                _, predicted = torch.max(outputs, dim=2)
                predicted = predicted.squeeze(0).tolist()
                
                # Декодируем в текст
                generated_code = self._decode_indices(predicted)
                
                # Очищаем код
                generated_code = self._clean_generated_code(generated_code)
                
                # Проверяем качество
                is_valid, error = self._validate_generated_code(generated_code)
                
                if not is_valid and safe_mode:
                    print(f"⚠️ Сгенерированный код невалиден: {error}")
                    print("🔄 Использую шаблонный подход")
                    return self._generate_from_template(description, intent_type)
                
                return generated_code
                
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            return self._generate_from_template(description, intent_type)
    
    def _decode_indices(self, indices: List[int]) -> str:
        """Декодирует индексы обратно в текст"""
        if not hasattr(self, 'dataset') or self.dataset is None:
            return ""
        
        # Убираем специальные токены
        decoded = []
        for idx in indices:
            if idx == 0 or idx == 1 or idx == 2:  # <PAD>, <SOS>, <EOS>
                continue
            
            if idx in self.dataset.idx_to_char:
                decoded.append(self.dataset.idx_to_char[idx])
            else:
                decoded.append('?')
        
        # Объединяем в строку
        return ''.join(decoded).strip()
    
    def _clean_generated_code(self, code: str) -> str:
        """Очищает сгенерированный код"""
        # Убираем лишние пробелы
        code = re.sub(r'\n\s*\n', '\n\n', code)
        
        # Убираем неполные строки в конце
        lines = code.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.rstrip()
            if line:  # Не пустая строка
                cleaned_lines.append(line)
        
        # Проверяем что есть def
        if not any('def ' in line for line in cleaned_lines):
            # Добавляем базовую функцию
            cleaned_lines.insert(0, 'def execute():')
            cleaned_lines.insert(1, '    """Выполняет команду"""')
            cleaned_lines.insert(2, '    return "Команда выполнена"')
        
        return '\n'.join(cleaned_lines)
    
    def _validate_generated_code(self, code: str) -> Tuple[bool, str]:
        """Проверяет валидность сгенерированного кода"""
        try:
            # Проверяем синтаксис
            ast.parse(code)
            
            # Проверяем наличие опасных конструкций
            dangerous_patterns = [
                (r'__import__\s*\(', '__import__'),
                (r'eval\s*\(', 'eval'),
                (r'exec\s*\(', 'exec'),
                (r'os\.system\s*\(', 'os.system'),
                (r'subprocess\.(?:call|run|Popen)\(.*shell\s*=\s*True', 'shell=True')
            ]
            
            for pattern, name in dangerous_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return False, f"Обнаружена опасная конструкция: {name}"
            
            # Проверяем наличие функции execute
            if not re.search(r'def\s+\w+\(', code):
                return False, "Нет функции для выполнения"
            
            return True, ""
            
        except SyntaxError as e:
            return False, f"Синтаксическая ошибка: {e}"
        except Exception as e:
            return False, f"Ошибка валидации: {e}"
    
    def _generate_from_template(self, description: str, intent_type: str) -> str:
        """Генерирует код из шаблона"""
        template = self.code_templates.get(intent_type, self.code_templates['custom'])
        
        # Извлекаем ключевые слова из описания
        program_keywords = ['браузер', 'блокнот', 'калькулятор', 'проводник', 'терминал', 'панель']
        found_keywords = [kw for kw in program_keywords if kw in description.lower()]
        
        if found_keywords:
            program_keyword = found_keywords[0]
        else:
            program_keyword = 'блокнот'
        
        # Заменяем плейсхолдеры
        code = template.format(program_keyword=program_keyword)
        
        # Добавляем импорты если их нет
        if 'import' not in code:
            code = "import os\nimport subprocess\n" + code
        
        return code
    
    def _estimate_complexity(self, description: str) -> int:
        """Оценивает сложность команды"""
        desc_lower = description.lower()
        
        # Простые команды
        simple_keywords = ['открой', 'закрой', 'привет', 'пока', 'время']
        if any(kw in desc_lower for kw in simple_keywords):
            return 1
        
        # Средние команды
        medium_keywords = ['создай', 'удали', 'найди', 'поищи', 'скопируй', 'вставь']
        if any(kw in desc_lower for kw in medium_keywords):
            return 2
        
        # Сложные команды
        complex_keywords = ['запусти процесс', 'настрой', 'автоматизируй', 'сделай резервную копию']
        if any(kw in desc_lower for kw in complex_keywords):
            return 3
        
        return 2  # по умолчанию средняя сложность
    
    def train_on_example(self, description: str, code: str, intent_type: str = 'custom'):
        """
        Обучает модель на новом примере
        
        Args:
            description: Описание команды
            code: Python код
            intent_type: Тип намерения
        """
        if self.dataset is None:
            # Создаем первый пример
            example = CodeExample(
                description=description,
                code=code,
                intent_type=intent_type,
                complexity=self._estimate_complexity(description)
            )
            self.dataset = CodeDataset([example], self.vocab_size, self.max_len)
        else:
            # Добавляем к существующим
            examples = self.dataset.examples.copy()
            examples.append(CodeExample(
                description=description,
                code=code,
                intent_type=intent_type,
                complexity=self._estimate_complexity(description)
            ))
            self.dataset = CodeDataset(examples, self.vocab_size, self.max_len)
        
        # Обучаем на одном примере
        self._train_one_epoch()
        
        # Сохраняем данные
        self._save_training_example(description, code, intent_type)
        
        print(f"✅ Модель обучена на примере: '{description[:50]}...'")
    
    def _train_one_epoch(self):
        """Обучает модель на одной эпохе"""
        if self.dataset is None or len(self.dataset) == 0:
            return
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=2,
            shuffle=True
        )
        
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            description = batch['description']
            code = batch['code']
            intent_type = batch.get('intent_type', None)
            complexity = batch.get('complexity', None)
            
            # Прямой проход
            outputs = self.model(
                description,
                code_input=code[:, :-1],  # Вход для учительского форсинга
                intent_type=intent_type,
                complexity=complexity,
                teacher_forcing_ratio=0.7
            )
            
            # Вычисляем потерю
            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                code[:, 1:].reshape(-1)  # Сдвинутая цель
            )
            
            # Обратный проход
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"  Обучение генератора: Loss = {avg_loss:.4f}")
    
    def _save_training_example(self, description: str, code: str, intent_type: str):
        """Сохраняет пример обучения"""
        data = []
        if self.data_path.exists():
            try:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                pass
        
        data.append({
            'description': description,
            'code': code,
            'intent_type': intent_type,
            'complexity': self._estimate_complexity(description),
            'timestamp': time.time() if 'time' in globals() else 0
        })
        
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Пример сохранен в {self.data_path}")

# Утилита для тестирования
def test_code_generator():
    """Тестирует генератор кода"""
    print("🧪 Тестирование генератора кода...")
    
    import tempfile
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "code_generator.pt")
    data_path = os.path.join(temp_dir, "code_data.json")
    
    # Создаем генератор
    generator = NeuralCodeGenerator(model_path, data_path)
    
    # Тестовые примеры
    test_cases = [
        {
            "description": "открой браузер",
            "intent_type": "open_program"
        },
        {
            "description": "напечатай привет мир",
            "intent_type": "type_text"
        },
        {
            "description": "создай файл report.txt с текстом отчет",
            "intent_type": "create_file"
        }
    ]
    
    for test in test_cases:
        print(f"\n📝 Тест: '{test['description']}'")
        
        # Генерируем код
        generated_code = generator.generate(
            description=test['description'],
            intent_type=test['intent_type']
        )
        
        print(f"📄 Сгенерированный код:\n{generated_code[:200]}...")
        
        # Проверяем валидность
        is_valid, error = generator._validate_generated_code(generated_code)
        if is_valid:
            print("✅ Код валиден")
        else:
            print(f"❌ Код невалиден: {error}")
    
    # Тестируем обучение
    print("\n🎓 Тестируем обучение на новом примере...")
    
    training_description = "закрой все окна"
    training_code = '''
def execute():
    """Закрывает все окна"""
    import pyautogui
    import time
    
    # Закрываем активное окно
    pyautogui.hotkey('alt', 'f4')
    time.sleep(0.5)
    
    return "Окна закрыты"
'''
    
    generator.train_on_example(training_description, training_code, "custom")
    
    # Сохраняем модель
    generator.save_model()
    
    # Очистка
    import shutil
    shutil.rmtree(temp_dir)
    print("\n🧹 Тестовые файлы удалены")

if __name__ == "__main__":
    import time
    test_code_generator()