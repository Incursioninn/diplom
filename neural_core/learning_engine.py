"""
Нейросеть для обучения новым командам
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TrainingExample:
    """Пример для обучения нейросети"""
    text: str
    intent_label: str
    explanation: str
    examples: List[str]
    embedding: Optional[np.ndarray] = None

class CommandDataset(Dataset):
    """Датасет команд для обучения нейросети"""
    
    def __init__(self, examples: List[TrainingExample], vocab_size: int = 10000):
        self.examples = examples
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """Строит словарь из всех текстов"""
        word_counts = defaultdict(int)
        
        for example in self.examples:
            for word in example.text.lower().split():
                word_counts[word] += 1
        
        # Берем самые частые слова
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocab_words = [word for word, _ in sorted_words[:self.vocab_size]]
        
        # Добавляем специальные токены
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for idx, word in enumerate(vocab_words, start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def text_to_indices(self, text: str, max_len: int = 50) -> List[int]:
        """Преобразует текст в индексы"""
        words = text.lower().split()[:max_len]
        indices = [self.word_to_idx.get(word, 1) for word in words]
        
        # Паддинг
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        
        return indices
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text_indices = self.text_to_indices(example.text)
        
        # Получаем метку интента как индекс
        intent_idx = self._get_intent_index(example.intent_label)
        
        return {
            'text': torch.tensor(text_indices, dtype=torch.long),
            'intent': torch.tensor(intent_idx, dtype=torch.long),
            'text_str': example.text,
            'intent_label': example.intent_label
        }
    
    def _get_intent_index(self, intent_label: str) -> int:
        """Получает индекс намерения"""
        # Простая хэш-функция для интентов
        return hash(intent_label) % 1000

class IntentClassifier(nn.Module):
    """Нейросеть для классификации намерений"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, num_intents: int = 50):
        super().__init__()
        
        # Слой эмбеддингов
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM для последовательностей
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Внимание
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_intents)
        )
        
        # Слой для новых интентов
        self.new_intent_projection = nn.Linear(hidden_dim * 2, 64)
        
    def forward(self, text_ids, attention_mask=None):
        # Эмбеддинги
        embedded = self.embedding(text_ids)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Внимание
        attention_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        
        # Взвешенная сумма
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        # Классификация
        logits = self.classifier(context_vector)
        
        # Эмбеддинг для новых интентов
        new_intent_embedding = self.new_intent_projection(context_vector)
        
        return logits, new_intent_embedding

class SimilarityFinder(nn.Module):
    """Нейросеть для поиска похожих команд"""
    
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Проекция в пространство эмбеддингов
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, x1, x2):
        """Вычисляет схожесть между двумя эмбеддингами"""
        # Косинусное сходство
        cos_sim = F.cosine_similarity(x1, x2, dim=-1)
        return cos_sim

class NeuralLearningEngine:
    """Движок обучения нейросети новым командам"""
    
    def __init__(self, model_path: str, data_path: str):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        
        # Создаем директории
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Параметры модели
        self.vocab_size = 10000
        self.embedding_dim = 128
        self.hidden_dim = 256
        self.max_seq_len = 50
        
        # Инициализация моделей
        self.intent_classifier = None
        self.similarity_finder = None
        self.dataset = None
        
        # Оптимизатор
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Данные
        self.known_intents = set()
        self.intent_examples = defaultdict(list)
        self.intent_embeddings = {}
        
        # Загружаем если существует
        self._load_or_initialize()
    
    def _load_or_initialize(self):
        """Загружает существующую модель или инициализирует новую"""
        if self.model_path.exists():
            print(f"📂 Загружаю модель из {self.model_path}")
            self._load_model()
        else:
            print("🆕 Инициализирую новую модель")
            self._initialize_model()
        
        if self.data_path.exists():
            self._load_data()
    
    def _initialize_model(self):
        """Инициализирует новые модели"""
        self.intent_classifier = IntentClassifier(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_intents=50  # Начальное количество интентов
        )
        
        self.similarity_finder = SimilarityFinder(embedding_dim=64)
        
        # Инициализируем оптимизатор
        self.optimizer = optim.AdamW(
            list(self.intent_classifier.parameters()) + 
            list(self.similarity_finder.parameters()),
            lr=0.001
        )
    
    def _load_model(self):
        """Загружает сохраненную модель"""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            self.intent_classifier = IntentClassifier(
                vocab_size=checkpoint['vocab_size'],
                embedding_dim=checkpoint['embedding_dim'],
                hidden_dim=checkpoint['hidden_dim'],
                num_intents=checkpoint['num_intents']
            )
            
            self.intent_classifier.load_state_dict(checkpoint['intent_classifier_state'])
            
            self.similarity_finder = SimilarityFinder(
                embedding_dim=checkpoint['similarity_embedding_dim']
            )
            
            if 'similarity_finder_state' in checkpoint:
                self.similarity_finder.load_state_dict(checkpoint['similarity_finder_state'])
            
            self.optimizer = optim.AdamW(
                list(self.intent_classifier.parameters()) + 
                list(self.similarity_finder.parameters()),
                lr=0.001
            )
            
            self.known_intents = set(checkpoint['known_intents'])
            self.vocab_size = checkpoint['vocab_size']
            
            print(f"✅ Модель загружена. Знаю {len(self.known_intents)} намерений")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            self._initialize_model()
    
    def _load_data(self):
        """Загружает данные из файла"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Создаем примеры для обучения
            examples = []
            for item in data:
                example = TrainingExample(
                    text=item['text'],
                    intent_label=item['intent_label'],
                    explanation=item['explanation'],
                    examples=item['examples']
                )
                examples.append(example)
                self.known_intents.add(item['intent_label'])
            
            # Создаем датасет
            self.dataset = CommandDataset(examples, self.vocab_size)
            
            print(f"📊 Загружено {len(examples)} примеров для {len(self.known_intents)} намерений")
            
        except Exception as e:
            print(f"⚠️ Ошибка загрузки данных: {e}")
            self.dataset = None
    
    def save_model(self):
        """Сохраняет модель"""
        checkpoint = {
            'intent_classifier_state': self.intent_classifier.state_dict(),
            'similarity_finder_state': self.similarity_finder.state_dict() if self.similarity_finder else None,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'similarity_embedding_dim': 64,
            'num_intents': len(self.known_intents) + 10,  # +10 для запаса
            'known_intents': list(self.known_intents)
        }
        
        torch.save(checkpoint, self.model_path)
        print(f"💾 Модель сохранена в {self.model_path}")
    
    def train_on_example(self, text: str, explanation: str, examples: List[str], generated_code: str) -> bool:
        """
        Обучает нейросеть на новом примере команды
        
        Args:
            text: Текст команды
            explanation: Объяснение команды
            examples: Примеры похожих команд
            generated_code: Сгенерированный код для выполнения
            
        Returns:
            True если обучение успешно
        """
        try:
            print(f"🎓 Обучаю нейросеть на команде: '{text}'")
            
            # Генерируем метку для нового интента
            intent_label = self._generate_intent_label(text, explanation)
            
            # Создаем пример для обучения
            training_example = TrainingExample(
                text=text,
                intent_label=intent_label,
                explanation=explanation,
                examples=examples
            )
            
            # Добавляем в известные интенты
            self.known_intents.add(intent_label)
            
            # Обновляем датасет
            if self.dataset is None:
                # Создаем новый датасет
                self.dataset = CommandDataset([training_example], self.vocab_size)
            else:
                # Добавляем в существующий датасет
                # Для простоты пересоздаем с новым примером
                current_examples = self.dataset.examples.copy()
                current_examples.append(training_example)
                self.dataset = CommandDataset(current_examples, self.vocab_size)
            
            # Подготавливаем данные для обучения
            dataloader = DataLoader(
                self.dataset,
                batch_size=4,
                shuffle=True
            )
            
            # Обучение
            self._train_epoch(dataloader, num_epochs=5)
            
            # Сохраняем модель
            self.save_model()
            
            # Сохраняем данные
            self._save_training_example(training_example, generated_code)
            
            # Создаем эмбеддинг для новой команды
            self._create_command_embedding(training_example)
            
            print(f"✅ Нейросеть обучена на команде '{text}'")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка обучения: {e}")
            return False
    
    def _generate_intent_label(self, text: str, explanation: str) -> str:
        """Генерирует уникальную метку для намерения"""
        # Используем хэш от текста и объяснения
        import hashlib
        combined = text + "||" + explanation[:100]
        hash_obj = hashlib.sha256(combined.encode())
        
        # Определяем тип по ключевым словам
        if any(word in explanation.lower() for word in ['открыть', 'запустить']):
            prefix = "open_"
        elif any(word in explanation.lower() for word in ['создать', 'сделать']):
            prefix = "create_"
        elif any(word in explanation.lower() for word in ['напечатать', 'написать']):
            prefix = "type_"
        else:
            prefix = "cmd_"
        
        return prefix + hash_obj.hexdigest()[:8]
    
    def _train_epoch(self, dataloader: DataLoader, num_epochs: int = 5):
        """Обучает модель на одной эпохе"""
        self.intent_classifier.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                self.optimizer.zero_grad()
                
                # Получаем данные
                text_ids = batch['text']
                intent_labels = batch['intent']
                
                # Прямой проход
                logits, embeddings = self.intent_classifier(text_ids)
                
                # Вычисляем потерю
                loss = self.criterion(logits, intent_labels)
                
                # Обратный проход
                loss.backward()
                self.optimizer.step()
                
                # Статистика
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += intent_labels.size(0)
                correct += (predicted == intent_labels).sum().item()
            
            accuracy = 100 * correct / total if total > 0 else 0
            avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
            
            print(f"  Эпоха {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.1f}%")
    
    def _save_training_example(self, example: TrainingExample, generated_code: str):
        """Сохраняет пример обучения"""
        # Загружаем существующие данные
        data = []
        if self.data_path.exists():
            try:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                pass
        
        # Добавляем новый пример
        data.append({
            'text': example.text,
            'intent_label': example.intent_label,
            'explanation': example.explanation,
            'examples': example.examples,
            'generated_code': generated_code,
            'timestamp': time.time() if 'time' in globals() else 0
        })
        
        # Сохраняем
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _create_command_embedding(self, example: TrainingExample):
        """Создает эмбеддинг для команды"""
        try:
            # Преобразуем текст в индексы
            if self.dataset:
                indices = self.dataset.text_to_indices(example.text)
                text_tensor = torch.tensor([indices], dtype=torch.long)
                
                # Получаем эмбеддинг
                with torch.no_grad():
                    _, embedding = self.intent_classifier(text_tensor)
                
                # Сохраняем эмбеддинг
                self.intent_embeddings[example.intent_label] = embedding.numpy()
                
        except Exception as e:
            print(f"⚠️ Ошибка создания эмбеддинга: {e}")
    
    def find_similar_command(self, text: str, threshold: float = 0.7) -> Optional[Dict[str, Any]]:
        """
        Ищет похожие команды в памяти
        
        Args:
            text: Текст команды
            threshold: Порог схожести
            
        Returns:
            Информация о похожей команде или None
        """
        if not self.intent_embeddings:
            return None
        
        try:
            # Создаем эмбеддинг для входной команды
            if self.dataset:
                indices = self.dataset.text_to_indices(text)
                text_tensor = torch.tensor([indices], dtype=torch.long)
                
                with torch.no_grad():
                    _, query_embedding = self.intent_classifier(text_tensor)
                
                query_embedding = query_embedding.numpy()
                
                # Ищем похожие эмбеддинги
                best_match = None
                best_similarity = 0
                
                for intent_label, stored_embedding in self.intent_embeddings.items():
                    # Вычисляем косинусное сходство
                    cos_sim = np.dot(query_embedding.flatten(), stored_embedding.flatten())
                    cos_sim /= (np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding) + 1e-8)
                    
                    if cos_sim > best_similarity:
                        best_similarity = cos_sim
                        best_match = intent_label
                
                if best_match and best_similarity > threshold:
                    # Загружаем информацию о команде
                    if self.data_path.exists():
                        with open(self.data_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        for item in data:
                            if item['intent_label'] == best_match:
                                return {
                                    'command': item['text'],
                                    'intent_label': best_match,
                                    'similarity': best_similarity,
                                    'explanation': item['explanation'],
                                    'generated_code': item.get('generated_code', '')
                                }
            
            return None
            
        except Exception as e:
            print(f"⚠️ Ошибка поиска похожих команд: {e}")
            return None
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """
        Предсказывает намерение для текста
        
        Returns:
            Метка намерения и уверенность
        """
        try:
            if not self.dataset:
                return "unknown", 0.0
            
            # Преобразуем текст
            indices = self.dataset.text_to_indices(text)
            text_tensor = torch.tensor([indices], dtype=torch.long)
            
            # Предсказание
            with torch.no_grad():
                self.intent_classifier.eval()
                logits, _ = self.intent_classifier(text_tensor)
                probabilities = F.softmax(logits, dim=1)
                
                # Получаем самое вероятное намерение
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                # Преобразуем индекс в метку (нужно сохранять маппинг)
                # Пока возвращаем индекс
                intent_label = f"intent_{predicted_idx.item()}"
                
                return intent_label, confidence.item()
                
        except Exception as e:
            print(f"⚠️ Ошибка предсказания: {e}")
            return "error", 0.0
    
    def retrain_on_all_data(self):
        """Переобучает модель на всех данных"""
        if not self.data_path.exists():
            print("📭 Нет данных для переобучения")
            return False
        
        try:
            print("🔄 Переобучение на всех данных...")
            
            # Загружаем все данные
            with open(self.data_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            
            # Создаем примеры
            examples = []
            for item in all_data:
                example = TrainingExample(
                    text=item['text'],
                    intent_label=item['intent_label'],
                    explanation=item['explanation'],
                    examples=item['examples']
                )
                examples.append(example)
                self.known_intents.add(item['intent_label'])
            
            # Создаем датасет
            self.dataset = CommandDataset(examples, self.vocab_size)
            
            # Обучаем
            dataloader = DataLoader(
                self.dataset,
                batch_size=8,
                shuffle=True
            )
            
            self._train_epoch(dataloader, num_epochs=10)
            
            # Пересоздаем эмбеддинги
            self.intent_embeddings = {}
            for example in examples:
                self._create_command_embedding(example)
            
            # Сохраняем модель
            self.save_model()
            
            print(f"✅ Переобучение завершено. Знаю {len(self.known_intents)} намерений")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка переобучения: {e}")
            return False

# Утилита для тестирования
def test_learning_engine():
    """Тестирует движок обучения"""
    print("🧪 Тестирование движока обучения...")
    
    # Создаем временные пути
    import tempfile
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "test_model.pt")
    data_path = os.path.join(temp_dir, "test_data.json")
    
    # Создаем движок
    engine = NeuralLearningEngine(model_path, data_path)
    
    # Тестовая команда для обучения
    test_command = "открой калькулятор"
    explanation = "Открыть программу калькулятор на компьютере"
    examples = ["запусти калькулятор", "включи калькулятор", "открой приложение калькулятор"]
    
    # Генерируем простой код
    generated_code = '''
def execute():
    """Открывает калькулятор"""
    import subprocess
    try:
        subprocess.Popen("calc.exe")
        return "Калькулятор открыт"
    except Exception as e:
        return f"Ошибка: {str(e)}"
'''
    
    # Обучаем
    success = engine.train_on_example(
        text=test_command,
        explanation=explanation,
        examples=examples,
        generated_code=generated_code
    )
    
    if success:
        print("✅ Обучение успешно")
        
        # Проверяем поиск похожих команд
        similar = engine.find_similar_command("запусти калькулятор")
        if similar:
            print(f"✅ Найдена похожая команда: {similar['command']}")
        
        # Предсказываем намерение
        intent, confidence = engine.predict_intent(test_command)
        print(f"✅ Предсказано намерение: {intent} (уверенность: {confidence:.2%})")
    
    # Очистка
    import shutil
    shutil.rmtree(temp_dir)
    print("🧹 Тестовые файлы удалены")

if __name__ == "__main__":
    import time
    test_learning_engine()