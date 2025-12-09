"""
Память ассистента с нейросетевым поиском и контекстным запоминанием
"""
import json
import os
import time
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from enum import Enum

class MemoryType(Enum):
    """Типы памяти"""
    COMMAND = "command"          # Команды пользователя
    CONTEXT = "context"          # Контекст диалога
    PREFERENCE = "preference"   # Предпочтения пользователя
    LEARNED = "learned"         # Выученные команды
    ERROR = "error"             # Ошибки для обучения
    SUCCESS = "success"         # Успешные выполнения

@dataclass
class MemoryItem:
    """Элемент памяти"""
    id: str
    type: MemoryType
    content: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    timestamp: float = None
    weight: float = 1.0  # Важность (увеличивается при использовании)
    access_count: int = 0
    last_accessed: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.last_accessed is None:
            self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь"""
        data = asdict(self)
        data['type'] = self.type.value
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Создает из словаря"""
        data = data.copy()
        data['type'] = MemoryType(data['type'])
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'])
        return cls(**data)

class NeuralMemorySearch:
    """Нейросетевой поиск в памяти"""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.embeddings = {}  # id -> embedding
        self.item_index = {}  # id -> item
        
        # Простая нейросеть для трансформации текста в эмбеддинг
        self._init_text_encoder()
    
    def _init_text_encoder(self):
        """Инициализирует кодировщик текста"""
        # В реальной системе здесь была бы нейросеть (BERT, SentenceTransformer)
        # Для простоты используем TF-IDF like подход
        
        # Собираем статистику по словам
        self.word_vectors = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Преобразует текст в эмбеддинг"""
        # Упрощенный подход: усреднение эмбеддингов слов
        words = text.lower().split()
        
        if not words:
            return np.zeros(self.embedding_dim)
        
        # Создаем эмбеддинги для слов (хэширование)
        word_embeddings = []
        for word in words:
            # Используем хэш для создания псевдо-эмбеддинга
            seed = int(hashlib.md5(word.encode()).hexdigest(), 16) % (10**8)
            np.random.seed(seed)
            embedding = np.random.randn(self.embedding_dim)
            word_embeddings.append(embedding)
        
        # Усредняем
        if word_embeddings:
            return np.mean(word_embeddings, axis=0)
        else:
            return np.zeros(self.embedding_dim)
    
    def add_item(self, item: MemoryItem, text: Optional[str] = None):
        """Добавляет элемент в поисковый индекс"""
        self.item_index[item.id] = item
        
        # Создаем эмбеддинг если нужно
        if item.embedding is None and text:
            item.embedding = self._text_to_embedding(text)
        
        if item.embedding is not None:
            self.embeddings[item.id] = item.embedding
    
    def remove_item(self, item_id: str):
        """Удаляет элемент из индекса"""
        if item_id in self.item_index:
            del self.item_index[item_id]
        if item_id in self.embeddings:
            del self.embeddings[item_id]
    
    def search_similar(self, query: str, item_type: Optional[MemoryType] = None, 
                      top_k: int = 5, threshold: float = 0.5) -> List[Tuple[MemoryItem, float]]:
        """
        Ищет похожие элементы в памяти
        
        Returns:
            Список пар (элемент, схожесть)
        """
        query_embedding = self._text_to_embedding(query)
        
        results = []
        
        for item_id, embedding in self.embeddings.items():
            item = self.item_index.get(item_id)
            
            if item is None:
                continue
            
            # Фильтр по типу если указан
            if item_type and item.type != item_type:
                continue
            
            # Вычисляем косинусное сходство
            similarity = self._cosine_similarity(query_embedding, embedding)
            
            if similarity >= threshold:
                results.append((item, similarity))
        
        # Сортируем по схожести
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Косинусное сходство"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)

class AssistantMemory:
    """
    Память ассистента с долговременной и рабочей памятью
    """
    
    def __init__(self, memory_file: str = "memory/assistant_memory.json"):
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Компоненты памяти
        self.long_term_memory = {}  # Постоянная память
        self.working_memory = deque(maxlen=10)  # Кратковременная память
        self.neural_search = NeuralMemorySearch()
        
        # Кэш быстрого доступа
        self.command_cache = {}
        self.user_preferences = {}
        
        # Статистика
        self.stats = {
            "total_items": 0,
            "memory_size_mb": 0,
            "search_count": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Загружаем память
        self.load()
        
        # Фоновое обслуживание
        self._start_maintenance_thread()
        
        print(f"🧠 Память инициализирована. Загружено {len(self.long_term_memory)} элементов")
    
    def load(self):
        """Загружает память из файла"""
        if not self.memory_file.exists():
            print("📭 Файл памяти не найден, создаю новую память")
            return
        
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Загружаем долговременную память
            self.long_term_memory = {}
            for item_id, item_data in data.get('long_term_memory', {}).items():
                try:
                    item = MemoryItem.from_dict(item_data)
                    self.long_term_memory[item_id] = item
                    
                    # Добавляем в нейросетевой поиск
                    if 'text' in item.content:
                        self.neural_search.add_item(item, item.content.get('text'))
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки элемента памяти {item_id}: {e}")
            
            # Загружаем предпочтения
            self.user_preferences = data.get('user_preferences', {})
            
            # Обновляем статистику
            self.stats['total_items'] = len(self.long_term_memory)
            self.stats['memory_size_mb'] = os.path.getsize(self.memory_file) / 1024 / 1024
            
            print(f"📂 Память загружена: {len(self.long_term_memory)} элементов")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки памяти: {e}")
            self.long_term_memory = {}
    
    def save(self):
        """Сохраняет память в файл"""
        try:
            # Подготавливаем данные для сохранения
            save_data = {
                'long_term_memory': {},
                'user_preferences': self.user_preferences,
                'metadata': {
                    'save_time': time.time(),
                    'version': '1.0',
                    'total_items': len(self.long_term_memory)
                }
            }
            
            # Сохраняем долговременную память
            for item_id, item in self.long_term_memory.items():
                # Не сохраняем эмбеддинги в JSON (слишком большой)
                item_copy = MemoryItem.from_dict(item.to_dict())
                item_copy.embedding = None
                save_data['long_term_memory'][item_id] = item_copy.to_dict()
            
            # Сохраняем в файл
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
            
            # Обновляем размер
            self.stats['memory_size_mb'] = os.path.getsize(self.memory_file) / 1024 / 1024
            
            print(f"💾 Память сохранена: {len(self.long_term_memory)} элементов")
            
        except Exception as e:
            print(f"❌ Ошибка сохранения памяти: {e}")
    
    def add_command(self, text: str, intent: str, entities: List[Dict], 
                   success: bool, result: Optional[Dict] = None) -> str:
        """
        Добавляет команду в память
        
        Returns:
            ID добавленного элемента
        """
        item_id = self._generate_id("cmd", text)
        
        content = {
            'text': text,
            'intent': intent,
            'entities': entities,
            'success': success,
            'result': result if result else {},
            'timestamp': time.time()
        }
        
        item = MemoryItem(
            id=item_id,
            type=MemoryType.COMMAND,
            content=content,
            weight=1.0 if success else 0.5  # Неудачные команды менее важны
        )
        
        # Добавляем в долговременную память
        self.long_term_memory[item_id] = item
        
        # Добавляем в рабочую память
        self.working_memory.append(item)
        
        # Добавляем в нейросетевой поиск
        self.neural_search.add_item(item, text)
        
        # Кэшируем для быстрого доступа
        self.command_cache[text.lower()] = item
        
        # Обновляем статистику
        self.stats['total_items'] = len(self.long_term_memory)
        
        # Автосохранение при добавлении важных команд
        if success:
            self.save()
        
        return item_id
    
    def add_context(self, user_id: str, context_data: Dict[str, Any]) -> str:
        """Добавляет контекст пользователя"""
        item_id = self._generate_id("ctx", user_id)
        
        content = {
            'user_id': user_id,
            'context': context_data,
            'timestamp': time.time()
        }
        
        item = MemoryItem(
            id=item_id,
            type=MemoryType.CONTEXT,
            content=content,
            weight=0.8  # Контекст важен, но может устаревать
        )
        
        self.long_term_memory[item_id] = item
        return item_id
    
    def add_preference(self, user_id: str, key: str, value: Any) -> str:
        """Добавляет предпочтение пользователя"""
        item_id = self._generate_id("pref", f"{user_id}:{key}")
        
        content = {
            'user_id': user_id,
            'key': key,
            'value': value,
            'timestamp': time.time()
        }
        
        item = MemoryItem(
            id=item_id,
            type=MemoryType.PREFERENCE,
            content=content,
            weight=0.9  # Предпочтения очень важны
        )
        
        self.long_term_memory[item_id] = item
        self.user_preferences[f"{user_id}:{key}"] = value
        
        return item_id
    
    def add_learned_command(self, command_text: str, explanation: str, 
                          generated_code: str, examples: List[str]) -> str:
        """Добавляет выученную команду"""
        item_id = self._generate_id("learned", command_text)
        
        content = {
            'command': command_text,
            'explanation': explanation,
            'generated_code': generated_code,
            'examples': examples,
            'learned_at': time.time(),
            'execution_count': 0,
            'success_count': 0
        }
        
        item = MemoryItem(
            id=item_id,
            type=MemoryType.LEARNED,
            content=content,
            weight=1.0  # Выученные команды очень важны
        )
        
        self.long_term_memory[item_id] = item
        self.save()  # Сохраняем выученные команды сразу
        
        return item_id
    
    def find_similar(self, text: str, intent: Optional[str] = None, 
                    limit: int = 3, min_similarity: float = 0.6) -> List[Dict[str, Any]]:
        """
        Ищет похожие команды в памяти
        
        Returns:
            Список найденных команд с информацией
        """
        self.stats['search_count'] += 1
        
        # Сначала проверяем кэш
        cache_key = text.lower()
        if cache_key in self.command_cache:
            self.stats['cache_hits'] += 1
            item = self.command_cache[cache_key]
            return [{
                'command': item.content.get('text', ''),
                'intent': item.content.get('intent', ''),
                'similarity': 1.0,
                'success_rate': item.content.get('success', False),
                'last_used': item.last_accessed
            }]
        
        self.stats['cache_misses'] += 1
        
        # Ищем через нейросеть
        results = []
        
        # Ищем похожие команды
        similar_items = self.neural_search.search_similar(
            text, 
            MemoryType.COMMAND,
            top_k=limit * 2,  # Ищем больше, потом фильтруем
            threshold=min_similarity
        )
        
        for item, similarity in similar_items:
            # Фильтруем по intent если указан
            if intent and item.content.get('intent') != intent:
                continue
            
            # Вычисляем успешность
            success = item.content.get('success', False)
            access_count = item.access_count
            
            # Обновляем статистику доступа
            item.access_count += 1
            item.last_accessed = time.time()
            
            results.append({
                'command': item.content.get('text', ''),
                'intent': item.content.get('intent', ''),
                'similarity': float(similarity),
                'success_rate': success,
                'access_count': access_count,
                'last_used': item.last_accessed,
                'item_id': item.id
            })
            
            if len(results) >= limit:
                break
        
        # Сортируем по схожести и успешности
        results.sort(key=lambda x: (x['similarity'], x['success_rate']), reverse=True)
        
        return results[:limit]
    
    def get_user_context(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Получает контекст пользователя"""
        contexts = []
        
        for item_id, item in self.long_term_memory.items():
            if item.type == MemoryType.CONTEXT and item.content.get('user_id') == user_id:
                contexts.append(item.content)
                
                if len(contexts) >= limit:
                    break
        
        # Сортируем по времени (новые первые)
        contexts.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        return contexts[:limit]
    
    def get_preference(self, user_id: str, key: str) -> Any:
        """Получает предпочтение пользователя"""
        return self.user_preferences.get(f"{user_id}:{key}")
    
    def get_learned_command(self, command_text: str) -> Optional[Dict[str, Any]]:
        """Получает выученную команду"""
        for item_id, item in self.long_term_memory.items():
            if (item.type == MemoryType.LEARNED and 
                item.content.get('command', '').lower() == command_text.lower()):
                return item.content
        
        return None
    
    def update_learned_command_stats(self, command_text: str, success: bool):
        """Обновляет статистику выученной команды"""
        for item_id, item in self.long_term_memory.items():
            if (item.type == MemoryType.LEARNED and 
                item.content.get('command', '').lower() == command_text.lower()):
                
                item.content['execution_count'] = item.content.get('execution_count', 0) + 1
                if success:
                    item.content['success_count'] = item.content.get('success_count', 0) + 1
                
                # Обновляем вес на основе успешности
                success_rate = item.content['success_count'] / max(item.content['execution_count'], 1)
                item.weight = success_rate
                
                self.save()
                break
    
    def get_working_memory(self) -> List[Dict[str, Any]]:
        """Получает рабочую память (последние команды)"""
        return [item.content for item in self.working_memory]
    
    def clear_old_memory(self, max_age_days: int = 30):
        """Очищает старую память"""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        items_to_remove = []
        
        for item_id, item in self.long_term_memory.items():
            # Не удаляем важные элементы
            if item.weight > 0.8:
                continue
            
            item_age = current_time - item.timestamp
            
            if item_age > max_age_seconds:
                items_to_remove.append(item_id)
        
        # Удаляем старые элементы
        for item_id in items_to_remove:
            del self.long_term_memory[item_id]
            self.neural_search.remove_item(item_id)
        
        if items_to_remove:
            print(f"🧹 Удалено {len(items_to_remove)} старых элементов памяти")
            self.save()
    
    def optimize_memory(self, max_items: int = 1000):
        """Оптимизирует память, удаляя наименее важные элементы"""
        if len(self.long_term_memory) <= max_items:
            return
        
        # Сортируем элементы по весу и времени доступа
        items_sorted = sorted(
            self.long_term_memory.items(),
            key=lambda x: (x[1].weight, x[1].last_accessed)
        )
        
        # Удаляем наименее важные
        items_to_remove = len(self.long_term_memory) - max_items
        removed_ids = []
        
        for i in range(items_to_remove):
            item_id, item = items_sorted[i]
            
            # Не удаляем выученные команды и предпочтения
            if item.type in [MemoryType.LEARNED, MemoryType.PREFERENCE]:
                continue
            
            del self.long_term_memory[item_id]
            self.neural_search.remove_item(item_id)
            removed_ids.append(item_id)
        
        if removed_ids:
            print(f"🧹 Оптимизирована память, удалено {len(removed_ids)} элементов")
            self.save()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику памяти"""
        # Анализируем типы памяти
        type_counts = defaultdict(int)
        success_rate = 0
        total_commands = 0
        
        for item in self.long_term_memory.values():
            type_counts[item.type.value] += 1
            
            if item.type == MemoryType.COMMAND:
                total_commands += 1
                if item.content.get('success', False):
                    success_rate += 1
        
        success_rate = success_rate / max(total_commands, 1)
        
        return {
            **self.stats,
            'type_distribution': dict(type_counts),
            'command_success_rate': success_rate,
            'working_memory_size': len(self.working_memory),
            'cache_size': len(self.command_cache),
            'user_preferences_count': len(self.user_preferences)
        }
    
    def _generate_id(self, prefix: str, seed: str) -> str:
        """Генерирует уникальный ID"""
        hash_input = f"{prefix}:{seed}:{time.time()}:{np.random.rand()}"
        hash_obj = hashlib.md5(hash_input.encode())
        return f"{prefix}_{hash_obj.hexdigest()[:12]}"
    
    def _start_maintenance_thread(self):
        """Запускает фоновое обслуживание памяти"""
        import threading
        
        def maintenance_loop():
            while True:
                try:
                    time.sleep(3600)  # Каждый час
                    
                    # Оптимизация памяти
                    self.clear_old_memory(max_age_days=7)  # Удаляем старше недели
                    self.optimize_memory(max_items=2000)  # Максимум 2000 элементов
                    
                    # Автосохранение
                    self.save()
                    
                except Exception as e:
                    print(f"⚠️ Ошибка в maintenance loop: {e}")
        
        thread = threading.Thread(target=maintenance_loop, daemon=True)
        thread.start()
    
    def __len__(self):
        return len(self.long_term_memory)
    
    def __contains__(self, item_id: str):
        return item_id in self.long_term_memory

# Утилита для тестирования
def test_memory():
    """Тестирует память ассистента"""
    print("🧪 Тестирование памяти ассистента...")
    
    import tempfile
    import shutil
    
    # Создаем временную директорию
    temp_dir = tempfile.mkdtemp()
    memory_file = Path(temp_dir) / "test_memory.json"
    
    try:
        # Создаем память
        memory = AssistantMemory(str(memory_file))
        
        # Тест 1: Добавление команд
        print("\n1. Тестирую добавление команд...")
        
        test_commands = [
            ("открой браузер", "open_browser", [], True),
            ("напечатай привет мир", "type_text", [{"label": "TEXT", "text": "привет мир"}], True),
            ("создай файл отчет.txt", "create_file", [{"label": "FILE", "text": "отчет.txt"}], False),
            ("сколько времени", "get_time", [], True),
        ]
        
        for text, intent, entities, success in test_commands:
            item_id = memory.add_command(text, intent, entities, success)
            print(f"  ✅ Добавлена команда: '{text}' (ID: {item_id})")
        
        # Тест 2: Поиск похожих команд
        print("\n2. Тестирую поиск похожих команд...")
        
        test_queries = [
            "открой хром",
            "напиши текст",
            "создай документ",
            "скажи время"
        ]
        
        for query in test_queries:
            similar = memory.find_similar(query, limit=2)
            print(f"  🔍 Поиск: '{query}' → найдено: {len(similar)}")
            for result in similar:
                print(f"    - '{result['command']}' (схожесть: {result['similarity']:.2f})")
        
        # Тест 3: Предпочтения пользователя
        print("\n3. Тестирую предпочтения пользователя...")
        
        memory.add_preference("user1", "favorite_browser", "chrome")
        memory.add_preference("user1", "default_editor", "notepad")
        
        browser = memory.get_preference("user1", "favorite_browser")
        editor = memory.get_preference("user1", "default_editor")
        
        print(f"  ⚙️ Предпочтения: браузер={browser}, редактор={editor}")
        
        # Тест 4: Выученные команды
        print("\n4. Тестирую выученные команды...")
        
        learned_id = memory.add_learned_command(
            command_text="сделай скриншот",
            explanation="Создает скриншот экрана",
            generated_code="def execute(): return 'Скриншот создан'",
            examples=["сними скриншот", "заскринь экран", "сделай фото экрана"]
        )
        
        learned_cmd = memory.get_learned_command("сделай скриншот")
        print(f"  🎓 Выученная команда: {learned_cmd['command'] if learned_cmd else 'нет'}")
        
        # Тест 5: Контекст пользователя
        print("\n5. Тестирую контекст пользователя...")
        
        memory.add_context("user1", {
            "last_command": "открой браузер",
            "last_intent": "open_browser",
            "working_directory": "/home/user",
            "active_program": "browser"
        })
        
        context = memory.get_user_context("user1")
        print(f"  📝 Контекст пользователя: {len(context)} записей")
        
        # Тест 6: Статистика
        print("\n6. Статистика памяти...")
        
        stats = memory.get_statistics()
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  📊 {key}: {value}")
        
        # Сохраняем память
        memory.save()
        
        # Проверяем что файл создан
        if memory_file.exists():
            file_size = memory_file.stat().st_size / 1024
            print(f"\n💾 Файл памяти сохранен: {file_size:.1f} KB")
        
    finally:
        # Очистка
        shutil.rmtree(temp_dir)
        print("\n🧹 Временные файлы удалены")

if __name__ == "__main__":
    test_memory()