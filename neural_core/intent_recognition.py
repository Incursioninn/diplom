"""
Нейросеть для распознавания намерений и извлечения сущностей
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, Counter
import pickle

@dataclass
class IntentExample:
    """Пример для обучения распознаванию намерений"""
    text: str
    intent: str
    entities: List[Dict[str, str]]
    tokens: List[str] = None
    
    def __post_init__(self):
        if self.tokens is None:
            self.tokens = self.text.lower().split()

@dataclass 
class Entity:
    """Извлеченная сущность"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0

class IntentNERDataset(Dataset):
    """Датасет для распознавания намерений и NER"""
    
    def __init__(self, examples: List[IntentExample], vocab_size: int = 5000, max_len: int = 50):
        self.examples = examples
        self.max_len = max_len
        
        # Строим словарь
        self.word2idx, self.idx2word = self._build_vocab(vocab_size)
        
        # Словарь интентов
        self.intent2idx = {}
        self.idx2intent = {}
        self._build_intent_vocab()
        
        # Словарь сущностей
        self.entity_labels = ['O', 'B-PROGRAM', 'I-PROGRAM', 'B-FILE', 'I-FILE',
                            'B-DIRECTORY', 'I-DIRECTORY', 'B-TEXT', 'I-TEXT',
                            'B-QUERY', 'I-QUERY', 'B-URL', 'I-URL', 'B-NUMBER',
                            'I-NUMBER', 'B-DATETIME', 'I-DATETIME']
        self.entity2idx = {label: idx for idx, label in enumerate(self.entity_labels)}
        self.idx2entity = {idx: label for label, idx in self.entity2idx.items()}
    
    def _build_vocab(self, vocab_size: int) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Строит словарь слов"""
        word_counts = Counter()
        
        for example in self.examples:
            word_counts.update(example.tokens)
        
        # Берем самые частые слова
        most_common = word_counts.most_common(vocab_size - 2)  # -2 для специальных токенов
        
        word2idx = {'<PAD>': 0, '<UNK>': 1}
        idx2word = {0: '<PAD>', 1: '<UNK>'}
        
        for idx, (word, _) in enumerate(most_common, start=2):
            word2idx[word] = idx
            idx2word[idx] = word
        
        return word2idx, idx2word
    
    def _build_intent_vocab(self):
        """Строит словарь интентов"""
        intents = set(example.intent for example in self.examples)
        self.intent2idx = {intent: idx for idx, intent in enumerate(intents)}
        self.idx2intent = {idx: intent for intent, idx in self.intent2idx.items()}
    
    def text_to_indices(self, text: str) -> List[int]:
        """Преобразует текст в индексы"""
        tokens = text.lower().split()[:self.max_len]
        indices = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        # Паддинг
        if len(indices) < self.max_len:
            indices += [self.word2idx['<PAD>']] * (self.max_len - len(indices))
        
        return indices[:self.max_len]
    
    def create_entity_labels(self, text: str, entities: List[Dict]) -> List[int]:
        """Создает метки сущностей для текста"""
        tokens = text.lower().split()[:self.max_len]
        labels = [0] * len(tokens)  # 0 = 'O' (не сущность)
        
        # Сопоставляем сущности с токенами
        for entity in entities:
            entity_text = entity.get('text', '').lower()
            entity_label = entity.get('label', 'O')
            
            # Ищем сущность в тексте
            if entity_text in text.lower():
                # Находим позиции токенов сущности
                entity_tokens = entity_text.split()
                text_tokens = text.lower().split()
                
                for i in range(len(text_tokens) - len(entity_tokens) + 1):
                    if text_tokens[i:i+len(entity_tokens)] == entity_tokens:
                        # Помечаем начало сущности
                        if entity_label != 'O':
                            labels[i] = self.entity2idx.get(f'B-{entity_label}', 0)
                            # Помечаем продолжение сущности
                            for j in range(1, len(entity_tokens)):
                                if i + j < len(labels):
                                    labels[i+j] = self.entity2idx.get(f'I-{entity_label}', 0)
                        break
        
        # Паддинг
        if len(labels) < self.max_len:
            labels += [0] * (self.max_len - len(labels))
        
        return labels[:self.max_len]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Токенизируем текст
        text_indices = self.text_to_indices(example.text)
        
        # Получаем метку интента
        intent_idx = self.intent2idx.get(example.intent, 0)
        
        # Получаем метки сущностей
        entity_labels = self.create_entity_labels(example.text, example.entities)
        
        return {
            'text': torch.tensor(text_indices, dtype=torch.long),
            'intent': torch.tensor(intent_idx, dtype=torch.long),
            'entities': torch.tensor(entity_labels, dtype=torch.long),
            'text_str': example.text,
            'intent_str': example.intent
        }

class JointIntentNERModel(nn.Module):
    """Совместная модель для распознавания намерений и извлечения сущностей"""
    
    def __init__(self, vocab_size: int, num_intents: int, num_entities: int,
                 embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        # Слой эмбеддингов
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM для последовательностей
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Внимание для интентов
        self.intent_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.3,
            batch_first=True
        )
        
        # Классификатор интентов
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_intents)
        )
        
        # Классификатор сущностей (CRF или линейный)
        self.entity_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_entities)
        )
        
        # Слой для новых интентов
        self.new_intent_projection = nn.Linear(hidden_dim, 64)
        
    def forward(self, text_ids, return_attentions=False):
        # Эмбеддинги
        embedded = self.embedding(text_ids)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(embedded)
        
        # Внимание для интентов
        intent_attn_out, intent_attn_weights = self.intent_attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Пулинг для интента (взвешенное среднее)
        attention_weights = F.softmax(intent_attn_weights.mean(dim=1), dim=-1)
        intent_context = torch.bmm(attention_weights.unsqueeze(1), intent_attn_out).squeeze(1)
        
        # Классификация интента
        intent_logits = self.intent_classifier(intent_context)
        
        # Классификация сущностей
        entity_logits = self.entity_classifier(lstm_out)
        
        # Эмбеддинг для новых интентов
        new_intent_embedding = self.new_intent_projection(intent_context)
        
        outputs = {
            'intent_logits': intent_logits,
            'entity_logits': entity_logits,
            'intent_embedding': new_intent_embedding
        }
        
        if return_attentions:
            outputs['attention_weights'] = intent_attn_weights
        
        return outputs

class NeuralIntentRecognizer:
    """
    Нейросетевое распознавание намерений и извлечение сущностей
    """
    
    def __init__(self, model_path: str, data_path: Optional[str] = None):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path) if data_path else self.model_path.parent / "intent_data.json"
        
        # Параметры модели
        self.vocab_size = 5000
        self.embedding_dim = 128
        self.hidden_dim = 256
        self.max_len = 50
        
        # Модель и данные
        self.model = None
        self.dataset = None
        self.intent_labels = []
        self.entity_labels = []
        
        # Статистика
        self.stats = {
            'total_predictions': 0,
            'high_confidence': 0,
            'low_confidence': 0,
            'unknown_intents': 0
        }
        
        # Загружаем или инициализируем
        self._load_or_initialize()
        
        # Правила для базовых интентов (если модель не обучена)
        self.basic_intent_rules = self._load_basic_rules()
        
        # Паттерны для сущностей
        self.entity_patterns = self._load_entity_patterns()
    
    def _load_or_initialize(self):
        """Загружает или инициализирует модель"""
        if self.model_path.exists():
            print(f"📂 Загружаю модель распознавания намерений из {self.model_path}")
            self._load_model()
        else:
            print("🆕 Инициализирую новую модель распознавания намерений")
            self._initialize_model()
        
        if self.data_path and self.data_path.exists():
            self._load_training_data()
    
    def _initialize_model(self):
        """Инициализирует новую модель"""
        # Базовые интенты
        base_intents = [
            'open_program', 'type_text', 'search_web', 'create_file',
            'delete_file', 'copy_text', 'paste_text', 'save_file',
            'get_time', 'list_files', 'create_folder', 'take_screenshot',
            'system_info', 'greeting', 'goodbye', 'help',
            'unknown', 'learn_command'
        ]
        
        # Базовые сущности
        entity_labels = ['O', 'B-PROGRAM', 'I-PROGRAM', 'B-FILE', 'I-FILE',
                        'B-DIRECTORY', 'I-DIRECTORY', 'B-TEXT', 'I-TEXT',
                        'B-QUERY', 'I-QUERY', 'B-URL', 'I-URL']
        
        self.intent_labels = base_intents
        self.entity_labels = entity_labels
        
        # Создаем модель
        self.model = JointIntentNERModel(
            vocab_size=self.vocab_size,
            num_intents=len(self.intent_labels),
            num_entities=len(self.entity_labels),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
        print(f"✅ Модель инициализирована: {len(self.intent_labels)} интентов, {len(self.entity_labels)} сущностей")
    
    def _load_model(self):
        """Загружает сохраненную модель"""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            self.intent_labels = checkpoint['intent_labels']
            self.entity_labels = checkpoint['entity_labels']
            
            self.model = JointIntentNERModel(
                vocab_size=checkpoint['vocab_size'],
                num_intents=len(self.intent_labels),
                num_entities=len(self.entity_labels),
                embedding_dim=checkpoint['embedding_dim'],
                hidden_dim=checkpoint['hidden_dim']
            )
            
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()
            
            # Загружаем словарь если есть
            if 'word2idx' in checkpoint:
                self.word2idx = checkpoint['word2idx']
                self.idx2word = checkpoint['idx2word']
            
            print(f"✅ Модель загружена: {len(self.intent_labels)} интентов")
            
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
                example = IntentExample(
                    text=item['text'],
                    intent=item['intent'],
                    entities=item.get('entities', [])
                )
                examples.append(example)
            
            if examples:
                self.dataset = IntentNERDataset(examples, self.vocab_size, self.max_len)
                print(f"📊 Загружено {len(examples)} примеров для обучения")
                
        except Exception as e:
            print(f"⚠️ Ошибка загрузки данных: {e}")
            self.dataset = None
    
    def _load_basic_rules(self) -> Dict[str, List[str]]:
        """Загружает правила для базовых интентов"""
        return {
            'open_program': ['открой', 'запусти', 'открыть', 'запустить', 'включи'],
            'type_text': ['напечатай', 'напиши', 'введи', 'печатай', 'ввод'],
            'search_web': ['найди', 'поищи', 'ищи', 'поиск', 'найти'],
            'create_file': ['создай', 'сделай', 'создать', 'новый файл'],
            'delete_file': ['удали', 'стереть', 'удалить', 'уничтожь'],
            'get_time': ['время', 'который час', 'сколько времени', 'времени'],
            'greeting': ['привет', 'здравствуй', 'добрый', 'hello', 'хай'],
            'goodbye': ['пока', 'до свидания', 'прощай', 'выход', 'стоп'],
            'help': ['помощь', 'помоги', 'что ты умеешь', 'возможности'],
            'learn_command': ['научи', 'запомни', 'выучи', 'обучи']
        }
    
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Загружает паттерны для извлечения сущностей"""
        return {
            'PROGRAM': ['браузер', 'хром', 'блокнот', 'калькулятор', 'проводник',
                       'терминал', 'word', 'excel', 'панель', 'notepad'],
            'FILE': ['.txt', '.doc', '.docx', '.pdf', '.jpg', '.png', 'файл'],
            'DIRECTORY': ['папка', 'директория', 'каталог', 'folder'],
            'TEXT': ['текст', 'сообщение', 'запись', 'note'],
            'QUERY': ['запрос', 'вопрос', 'информация', 'что такое'],
            'URL': ['http://', 'https://', 'www.', '.ru', '.com'],
            'NUMBER': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            'DATETIME': ['сегодня', 'завтра', 'вчера', 'час', 'минута', 'секунда']
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Предсказывает намерение и сущности для текста
        
        Returns:
            Словарь с результатами распознавания
        """
        self.stats['total_predictions'] += 1
        
        # Если модель не обучена, используем правила
        if self.model is None or self.dataset is None:
            return self._predict_with_rules(text)
        
        try:
            # Токенизируем текст
            text_indices = self._text_to_indices(text)
            text_tensor = torch.tensor([text_indices], dtype=torch.long)
            
            # Предсказание
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(text_tensor)
                
                # Обрабатываем интент
                intent_logits = outputs['intent_logits']
                intent_probs = F.softmax(intent_logits, dim=1)
                confidence, intent_idx = torch.max(intent_probs, dim=1)
                
                confidence = confidence.item()
                intent_idx = intent_idx.item()
                
                # Получаем метку интента
                if intent_idx < len(self.intent_labels):
                    intent = self.intent_labels[intent_idx]
                else:
                    intent = 'unknown'
                
                # Обрабатываем сущности
                entity_logits = outputs['entity_logits']
                entity_probs = F.softmax(entity_logits, dim=2)
                _, entity_idxs = torch.max(entity_probs, dim=2)
                
                # Извлекаем сущности
                entities = self._extract_entities(text, entity_idxs[0].tolist())
                
                # Статистика
                if confidence > 0.7:
                    self.stats['high_confidence'] += 1
                else:
                    self.stats['low_confidence'] += 1
                
                if intent == 'unknown':
                    self.stats['unknown_intents'] += 1
                
                return {
                    'intent': intent,
                    'confidence': confidence,
                    'entities': entities,
                    'intent_probs': intent_probs.tolist()[0],
                    'method': 'neural'
                }
                
        except Exception as e:
            print(f"❌ Ошибка нейросетевого предсказания: {e}")
            return self._predict_with_rules(text)
    
    def _predict_with_rules(self, text: str) -> Dict[str, Any]:
        """Предсказывает с помощью правил (fallback)"""
        text_lower = text.lower()
        
        # Определяем интент по ключевым словам
        intent = 'unknown'
        confidence = 0.5
        max_matches = 0
        
        for intent_name, keywords in self.basic_intent_rules.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            
            if matches > max_matches:
                max_matches = matches
                intent = intent_name
                confidence = min(0.3 + matches * 0.2, 0.9)  # Динамическая уверенность
        
        # Извлекаем сущности по паттернам
        entities = self._extract_entities_with_patterns(text)
        
        return {
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'method': 'rules'
        }
    
    def _text_to_indices(self, text: str) -> List[int]:
        """Преобразует текст в индексы"""
        if not hasattr(self, 'dataset') or self.dataset is None:
            # Простая токенизация если нет датасета
            tokens = text.lower().split()[:self.max_len]
            
            if hasattr(self, 'word2idx'):
                indices = [self.word2idx.get(token, 1) for token in tokens]
            else:
                # Используем простой хэш
                indices = []
                for token in tokens:
                    token_hash = hash(token) % (self.vocab_size - 2) + 2
                    indices.append(token_hash)
            
            # Паддинг
            if len(indices) < self.max_len:
                indices += [0] * (self.max_len - len(indices))
            
            return indices[:self.max_len]
        else:
            # Используем датасет
            return self.dataset.text_to_indices(text)
    
    def _extract_entities(self, text: str, entity_labels: List[int]) -> List[Dict[str, Any]]:
        """Извлекает сущности из предсказанных меток"""
        if not hasattr(self, 'dataset') or self.dataset is None:
            return self._extract_entities_with_patterns(text)
        
        tokens = text.lower().split()
        entities = []
        current_entity = None
        
        for i, label_idx in enumerate(entity_labels[:len(tokens)]):
            if label_idx >= len(self.entity_labels):
                continue
            
            label = self.entity_labels[label_idx]
            
            if label == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            elif label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]  # Убираем B-
                current_entity = {
                    'text': tokens[i],
                    'label': entity_type,
                    'start': i,
                    'end': i + 1,
                    'confidence': 0.8
                }
            elif label.startswith('I-') and current_entity:
                # Продолжение сущности
                current_entity['text'] += ' ' + tokens[i]
                current_entity['end'] = i + 1
        
        if current_entity:
            entities.append(current_entity)
        
        # Восстанавливаем оригинальный текст сущности
        original_tokens = text.split()
        for entity in entities:
            entity_text = ' '.join(original_tokens[entity['start']:entity['end']])
            entity['text'] = entity_text
        
        return entities
    
    def _extract_entities_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Извлекает сущности с помощью паттернов"""
        text_lower = text.lower()
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # Находим позицию
                    idx = text_lower.find(pattern)
                    
                    entities.append({
                        'text': text[idx:idx+len(pattern)],
                        'label': entity_type,
                        'start': idx,
                        'end': idx + len(pattern),
                        'confidence': 0.7,
                        'method': 'pattern'
                    })
        
        # Убираем пересекающиеся сущности
        if entities:
            entities.sort(key=lambda x: x['start'])
            filtered = [entities[0]]
            
            for entity in entities[1:]:
                if entity['start'] >= filtered[-1]['end']:
                    filtered.append(entity)
            
            entities = filtered
        
        return entities
    
    def train_on_example(self, text: str, intent: str, entities: List[Dict[str, str]]):
        """
        Обучает модель на новом примере
        
        Args:
            text: Текст команды
            intent: Намерение
            entities: Список сущностей
        """
        print(f"🎓 Обучаю модель на примере: '{text}' -> {intent}")
        
        # Создаем пример
        example = IntentExample(
            text=text,
            intent=intent,
            entities=entities
        )
        
        # Обновляем датасет
        if self.dataset is None:
            examples = [example]
            self.dataset = IntentNERDataset(examples, self.vocab_size, self.max_len)
            
            # Обновляем интенты
            if intent not in self.intent_labels:
                self.intent_labels.append(intent)
        else:
            # Добавляем к существующим примерам
            examples = self.dataset.examples.copy()
            examples.append(example)
            self.dataset = IntentNERDataset(examples, self.vocab_size, self.max_len)
            
            # Обновляем интенты
            if intent not in self.intent_labels:
                self.intent_labels.append(intent)
        
        # Если модель не создана, создаем
        if self.model is None:
            self._initialize_model()
        
        # Пересоздаем модель с новым количеством интентов
        self.model = JointIntentNERModel(
            vocab_size=self.vocab_size,
            num_intents=len(self.intent_labels),
            num_entities=len(self.dataset.entity_labels),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Обучаем на одном примере
        self._train_one_epoch()
        
        # Сохраняем модель
        self.save_model()
        
        # Сохраняем данные
        self._save_training_example(example)
        
        print(f"✅ Модель обучена на примере")
    
    def _train_one_epoch(self):
        """Обучает модель на одной эпохе"""
        if self.dataset is None or len(self.dataset) == 0:
            return
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=2,
            shuffle=True
        )
        
        criterion_intent = nn.CrossEntropyLoss()
        criterion_entity = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            text_ids = batch['text']
            intent_labels = batch['intent']
            entity_labels = batch['entities']
            
            # Прямой проход
            outputs = self.model(text_ids)
            
            # Вычисляем потери
            loss_intent = criterion_intent(outputs['intent_logits'], intent_labels)
            loss_entity = criterion_entity(
                outputs['entity_logits'].view(-1, outputs['entity_logits'].size(-1)),
                entity_labels.view(-1)
            )
            
            loss = loss_intent + loss_entity
            
            # Обратный проход
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"  Обучение модели: Loss = {avg_loss:.4f}")
    
    def save_model(self):
        """Сохраняет модель"""
        checkpoint = {
            'model_state': self.model.state_dict() if self.model else None,
            'intent_labels': self.intent_labels,
            'entity_labels': self.entity_labels,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim
        }
        
        # Сохраняем словарь если есть
        if hasattr(self, 'dataset') and self.dataset:
            checkpoint['word2idx'] = self.dataset.word2idx
            checkpoint['idx2word'] = self.dataset.idx2word
        
        torch.save(checkpoint, self.model_path)
        print(f"💾 Модель распознавания сохранена")
    
    def _save_training_example(self, example: IntentExample):
        """Сохраняет пример обучения"""
        if not self.data_path:
            return
        
        data = []
        if self.data_path.exists():
            try:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                pass
        
        data.append({
            'text': example.text,
            'intent': example.intent,
            'entities': example.entities,
            'timestamp': time.time() if 'time' in globals() else 0
        })
        
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику"""
        return {
            **self.stats,
            'total_intents': len(self.intent_labels),
            'total_entities': len(self.entity_labels),
            'has_model': self.model is not None,
            'has_dataset': self.dataset is not None and len(self.dataset) > 0
        }

# Утилита для тестирования
def test_intent_recognizer():
    """Тестирует распознаватель намерений"""
    print("🧪 Тестирование распознавателя намерений...")
    
    import tempfile
    import shutil
    
    # Создаем временную директорию
    temp_dir = tempfile.mkdtemp()
    model_path = Path(temp_dir) / "intent_model.pt"
    data_path = Path(temp_dir) / "intent_data.json"
    
    try:
        # Создаем распознаватель
        recognizer = NeuralIntentRecognizer(str(model_path), str(data_path))
        
        # Тест 1: Предсказание с правилами (модель не обучена)
        print("\n1. Тестирую предсказание с правилами...")
        
        test_texts = [
            "открой браузер пожалуйста",
            "напечатай текст документа",
            "сколько сейчас времени",
            "привет как дела",
            "неизвестная команда для теста"
        ]
        
        for text in test_texts:
            result = recognizer.predict(text)
            print(f"  📝 '{text[:30]}...' → {result['intent']} ({result['confidence']:.2%})")
            if result['entities']:
                print(f"    Сущности: {result['entities']}")
        
        # Тест 2: Обучение модели
        print("\n2. Тестирую обучение модели...")
        
        training_examples = [
            ("открой калькулятор", "open_program", [{"text": "калькулятор", "label": "PROGRAM"}]),
            ("создай файл отчет.txt", "create_file", [{"text": "отчет.txt", "label": "FILE"}]),
            ("найди в интернете python", "search_web", [{"text": "python", "label": "QUERY"}])
        ]
        
        for text, intent, entities in training_examples:
            recognizer.train_on_example(text, intent, entities)
        
        # Тест 3: Предсказание с обученной моделью
        print("\n3. Тестирую предсказание с обученной моделью...")
        
        for text in test_texts:
            result = recognizer.predict(text)
            print(f"  🧠 '{text[:30]}...' → {result['intent']} ({result['confidence']:.2%}, метод: {result['method']})")
        
        # Тест 4: Статистика
        print("\n4. Статистика...")
        
        stats = recognizer.get_statistics()
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  📊 {key}: {value}")
        
        # Сохраняем модель
        recognizer.save_model()
        
    finally:
        # Очистка
        shutil.rmtree(temp_dir)
        print("\n🧹 Временные файлы удалены")

if __name__ == "__main__":
    import time
    test_intent_recognizer()