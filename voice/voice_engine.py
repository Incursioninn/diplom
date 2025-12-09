"""
Голосовой движок на основе готовых open-source решений
"""
import os
import sys
import json
import time
import queue
import threading
import wave
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# Проверяем и импортируем зависимости
try:
    import pyaudio
    import torch
    #import torchaudio
    import speech_recognition as sr
    import pyttsx3
    HAS_AUDIO_DEPS = True
except ImportError as e:
    print(f"⚠️ Не установлены аудио зависимости: {e}")
    print("Установите: pip install pyaudio torch torchaudio SpeechRecognition pyttsx3")
    HAS_AUDIO_DEPS = False

@dataclass
class VoiceConfig:
    """Конфигурация голосового движка"""
    # Распознавание речи
    speech_recognizer: str = "vosk"  # vosk, whisper, google
    language: str = "ru"
    sample_rate: int = 16000
    chunk_size: int = 4000
    
    # Синтез речи
    tts_engine: str = "pyttsx3"  # pyttsx3, silero
    voice_gender: str = "female"  # male, female
    speech_rate: int = 180
    
    # VAD (Voice Activity Detection)
    use_vad: bool = True
    vad_threshold: float = 0.5
    silence_duration: float = 1.0
    
    # Активация
    activation_mode: str = "keyword"  # keyword, hotkey, always
    activation_keyword: str = "ассистент"
    hotkey: str = "ctrl+alt+a"
    
    # Пути к моделям
    model_path: str = "models/voice"
    cache_path: str = "cache/audio"

class NeuralVoiceEngine:
    """
    Голосовой движок на основе готовых open-source решений
    """
    
    def __init__(self, config: Optional[VoiceConfig] = None):
        if not HAS_AUDIO_DEPS:
            raise ImportError("Не установлены аудио зависимости")
        
        self.config = config or VoiceConfig()
        
        # Создаем директории
        self._create_directories()
        
        # Инициализация компонентов
        self._init_speech_recognizer()
        self._init_tts_engine()
        self._init_vad()
        
        # Состояние
        self.is_listening = False
        self.is_speaking = False
        self.last_speech_time = 0
        self.activation_detected = False
        
        # Очереди и потоки
        self.audio_queue = queue.Queue(maxsize=100)
        self.command_queue = queue.Queue()
        self.processing_thread = None
        
        # Коллбэки
        self.on_command_callback = None
        self.on_activation_callback = None
        
        # Статистика
        self.stats = {
            "total_audio_chunks": 0,
            "speech_detected": 0,
            "commands_recognized": 0,
            "recognition_errors": 0,
            "total_listening_time": 0.0
        }
        self._init_audio_stream()
        print("🎤 Голосовой движок инициализирован")
    
    def _create_directories(self):
        """Создает необходимые директории"""
        directories = [
            self.config.model_path,
            self.config.cache_path,
            "logs/audio"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _init_audio_stream(self):
        """Инициализирует аудио поток"""
        try:
            self.audio = pyaudio.PyAudio()
            
            # Находим микрофон
            self.input_device_index = self._find_microphone()
            
            if self.input_device_index is None:
                print("⚠️ Микрофон не найден, использую устройство по умолчанию")
                self.input_device_index = self.audio.get_default_input_device_info()["index"]
            
            # Параметры потока
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._audio_callback
            )
            
            print(f"🎤 Аудио поток создан: {self.config.sample_rate}Hz, chunk={self.config.chunk_size}")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации аудио потока: {e}")
            self.audio = None
            self.stream = None
    
    def _find_microphone(self) -> Optional[int]:
        """Находит микрофон по названию"""
        try:
            device_count = self.audio.get_device_count()
            
            for i in range(device_count):
                device_info = self.audio.get_device_info_by_index(i)
                
                # Проверяем что устройство имеет вход
                if device_info["maxInputChannels"] > 0:
                    device_name = device_info["name"].lower()
                    
                    # Ищем микрофон в названии
                    keywords = ["microphone", "mic", "микрофон", "вход"]
                    if any(keyword in device_name for keyword in keywords):
                        print(f"✅ Найден микрофон: {device_info['name']}")
                        return i
            
            return None
            
        except Exception as e:
            print(f"⚠️ Ошибка поиска микрофона: {e}")
            return None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback для аудио потока"""
        try:
            if hasattr(self, 'is_listening') and self.is_listening:
                # Конвертируем в numpy массив
                audio_data = np.frombuffer(in_data, dtype=np.int16)
            
                # Добавляем в очередь для обработки
                try:
                    self.audio_queue.put_nowait({
                        'data': audio_data,
                        'timestamp': time.time()
                    })
                    self.stats["total_audio_chunks"] += 1
                except queue.Full:
                    pass
        except Exception as e:
            print(f"⚠️ Ошибка в audio callback: {e}")
    
        return (in_data, pyaudio.paContinue)
    
    def _init_speech_recognizer(self):
        """Инициализирует распознаватель речи"""
        print(f"🔄 Инициализация распознавателя речи: {self.config.speech_recognizer}")
        
        if self.config.speech_recognizer == "vosk":
            self._init_vosk_recognizer()
        elif self.config.speech_recognizer == "whisper":
            self._init_whisper_recognizer()
        elif self.config.speech_recognizer == "google":
            self._init_google_recognizer()
        else:
            print(f"⚠️ Неизвестный распознаватель: {self.config.speech_recognizer}")
            print("🔄 Использую Google Speech Recognition")
            self._init_google_recognizer()
    
    def _init_vosk_recognizer(self):
        """Инициализирует Vosk распознаватель (оффлайн)"""
        try:
            import vosk
            
            # Загружаем модель
            model_path = Path(self.config.model_path) / "vosk-model-small-ru"
            
            if not model_path.exists():
                print("📥 Скачиваю модель Vosk для русского языка...")
                self._download_vosk_model()
            
            if model_path.exists():
                self.vosk_model = vosk.Model(str(model_path))
                self.vosk_recognizer = vosk.KaldiRecognizer(
                    self.vosk_model, 
                    self.config.sample_rate
                )
                print("✅ Vosk распознаватель инициализирован")
            else:
                print("⚠️ Модель Vosk не найдена, использую Google")
                self._init_google_recognizer()
                
        except ImportError:
            print("⚠️ Vosk не установлен, использую Google")
            print("Установите: pip install vosk")
            self._init_google_recognizer()
        except Exception as e:
            print(f"❌ Ошибка инициализации Vosk: {e}")
            self._init_google_recognizer()
    
    def _download_vosk_model(self):
        """Скачивает модель Vosk"""
        try:
            import urllib.request
            import zipfile
            
            model_url = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
            zip_path = Path(self.config.model_path) / "vosk-model-small-ru.zip"
            
            print(f"📥 Скачиваю модель с {model_url}")
            
            # Скачиваем
            urllib.request.urlretrieve(model_url, zip_path)
            
            # Распаковываем
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.config.model_path)
            
            # Удаляем архив
            zip_path.unlink()
            
            print("✅ Модель Vosk загружена")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели Vosk: {e}")
    
    def _init_whisper_recognizer(self):
        """Инициализирует Whisper распознаватель"""
        try:
            import whisper
            
            # Проверяем наличие модели
            model_size = "small"  # tiny, base, small, medium, large
            
            print(f"🔄 Загружаю модель Whisper ({model_size})...")
            self.whisper_model = whisper.load_model(model_size)
            
            print("✅ Whisper распознаватель инициализирован")
            
        except ImportError:
            print("⚠️ Whisper не установлен, использую Google")
            print("Установите: pip install openai-whisper")
            self._init_google_recognizer()
        except Exception as e:
            print(f"❌ Ошибка инициализации Whisper: {e}")
            self._init_google_recognizer()
    
    def _init_google_recognizer(self):
        """Инициализирует Google Speech Recognition"""
        try:
            self.google_recognizer = sr.Recognizer()
        
            # Пробуем найти микрофон
            try:
                self.google_microphone = sr.Microphone()
            
                # Настраиваем для шумной среды
                with self.google_microphone as source:
                    self.google_recognizer.adjust_for_ambient_noise(source, duration=1)
            
                print("✅ Google Speech Recognition инициализирован")
            
            except Exception as mic_error:
                print(f"⚠️ Не удалось инициализировать микрофон: {mic_error}")
                self.google_microphone = None
            
        except Exception as e:
            print(f"❌ Ошибка инициализации Google Speech Recognition: {e}")
            print("⚠️ Голосовой ввод будет недоступен")
            self.google_recognizer = None
            self.google_microphone = None
    
    def _init_tts_engine(self):
        """Инициализирует синтезатор речи"""
        print(f"🔄 Инициализация TTS: {self.config.tts_engine}")
        
        if self.config.tts_engine == "pyttsx3":
            self._init_pyttsx3()
        elif self.config.tts_engine == "silero":
            self._init_silero()
        else:
            print(f"⚠️ Неизвестный TTS: {self.config.tts_engine}")
            print("🔄 Использую pyttsx3")
            self._init_pyttsx3()
    
    def _init_pyttsx3(self):
        """Инициализирует pyttsx3 TTS"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Настройки голоса
            self.tts_engine.setProperty('rate', self.config.speech_rate)
            
            # Выбираем голос
            voices = self.tts_engine.getProperty('voices')
            
            if self.config.voice_gender == "female" and len(voices) > 1:
                # Пытаемся найти женский голос
                for voice in voices:
                    if "female" in voice.name.lower() or "женск" in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    # Если не нашли, используем второй голос
                    self.tts_engine.setProperty('voice', voices[1].id)
            else:
                # Мужской голос
                self.tts_engine.setProperty('voice', voices[0].id)
            
            print(f"✅ pyttsx3 TTS инициализирован")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации pyttsx3: {e}")
            self.tts_engine = None
    
    def _init_silero(self):
        """Инициализирует Silero TTS (качественный оффлайн)"""
        try:
            import torch
            language = 'ru'
            model_id = 'v3_1_ru'
            device = torch.device('cpu')
            
            # Загружаем модель
            torch.hub.download_url_to_file(
                f'https://models.silero.ai/models/tts/{language}/{model_id}.pt',
                f'{self.config.model_path}/silero_{model_id}.pt'
            )
            
            # Инициализируем
            self.silero_model = torch.package.PackageImporter(
                f'{self.config.model_path}/silero_{model_id}.pt'
            ).load_pickle("tts_models", "model")
            
            self.silero_model.to(device)
            
            # Выбираем голос
            self.silero_speaker = 'baya'  # baya, kseniya, aidar, eugene, random
            
            print("✅ Silero TTS инициализирован")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации Silero: {e}")
            print("🔄 Использую pyttsx3")
            self._init_pyttsx3()
    
    def _init_vad(self):
        """Инициализирует Voice Activity Detection"""
        if self.config.use_vad:
            try:
                import webrtcvad
                self.vad = webrtcvad.Vad(2)  # 0-3, где 3 самый агрессивный
                print("✅ VAD инициализирован")
            except ImportError:
                print("⚠️ webrtcvad не установлен, VAD отключен")
                print("Установите: pip install webrtcvad")
                self.config.use_vad = False
            except Exception as e:
                print(f"❌ Ошибка инициализации VAD: {e}")
                self.config.use_vad = False
    
    def start_listening(self):
        """Начинает прослушивание"""
        if not self.is_listening:
            print("👂 Начинаю прослушивание...")
            self.is_listening = True
            
            # Запускаем поток обработки
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            
            # Запускаем поток активации
            if self.config.activation_mode == "keyword":
                self.activation_thread = threading.Thread(
                    target=self._activation_detection_loop,
                    daemon=True
                )
                self.activation_thread.start()
    
    def stop_listening(self):
        """Останавливает прослушивание"""
        if self.is_listening:
            print("🛑 Останавливаю прослушивание...")
            self.is_listening = False
            
            if self.processing_thread:
                self.processing_thread.join(timeout=2)
            
            # Очищаем очереди
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
    
    def _processing_loop(self):
        """Основной цикл обработки аудио"""
        audio_buffer = []
        buffer_duration = 0
        last_audio_time = time.time()
        
        while self.is_listening:
            try:
                # Получаем аудио чанк
                chunk = self.audio_queue.get(timeout=0.1)
                audio_data = chunk['data']
                timestamp = chunk['timestamp']
                
                # Проверяем VAD
                is_speech = self._detect_speech(audio_data)
                
                if is_speech:
                    self.stats["speech_detected"] += 1
                    
                    # Добавляем в буфер
                    audio_buffer.append(audio_data)
                    buffer_duration += len(audio_data) / self.config.sample_rate
                    last_audio_time = timestamp
                    
                    # Если накопили достаточно аудио, распознаем
                    if buffer_duration >= 1.0:  # 1 секунда
                        self._process_audio_buffer(audio_buffer)
                        audio_buffer = []
                        buffer_duration = 0
                
                else:
                    # Если была речь и теперь тишина
                    if audio_buffer and (timestamp - last_audio_time) > self.config.silence_duration:
                        self._process_audio_buffer(audio_buffer)
                        audio_buffer = []
                        buffer_duration = 0
                
                # Обновляем статистику времени
                self.stats["total_listening_time"] += 0.1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Ошибка в processing loop: {e}")
                self.stats["recognition_errors"] += 1
    
    def _detect_speech(self, audio_data: np.ndarray) -> bool:
        """Определяет есть ли речь в аудио"""
        if not self.config.use_vad:
            return True  # Если VAD отключен, считаем что всегда есть речь
        
        try:
            # Конвертируем в нужный формат для VAD
            audio_int16 = audio_data.astype(np.int16)
            
            # Проверяем каждый фрейм (30ms)
            frame_duration = 30  # ms
            frame_size = int(self.config.sample_rate * frame_duration / 1000)
            
            is_speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_int16), frame_size):
                frame = audio_int16[i:i+frame_size]
                if len(frame) < frame_size:
                    continue
                
                try:
                    if self.vad.is_speech(frame.tobytes(), self.config.sample_rate):
                        is_speech_frames += 1
                    total_frames += 1
                except:
                    pass
            
            if total_frames > 0:
                speech_ratio = is_speech_frames / total_frames
                return speech_ratio > self.config.vad_threshold
            
            return False
            
        except Exception as e:
            print(f"⚠️ Ошибка VAD: {e}")
            return True  # В случае ошибки считаем что есть речь
    
    def _process_audio_buffer(self, audio_buffer: List[np.ndarray]):
        """Обрабатывает накопленный аудио буфер"""
        if not audio_buffer:
            return
        
        try:
            # Объединяем все чанки
            combined_audio = np.concatenate(audio_buffer)
            
            # Распознаем речь
            text = self._recognize_speech(combined_audio)
            
            if text and text.strip():
                print(f"🎤 Распознано: {text}")
                self.stats["commands_recognized"] += 1
                
                # Проверяем активацию
                if self._check_activation(text):
                    self.activation_detected = True
                    if self.on_activation_callback:
                        self.on_activation_callback(text)
                
                # Если всегда слушаем или активация обнаружена
                if self.config.activation_mode == "always" or self.activation_detected:
                    if self.on_command_callback:
                        self.on_command_callback(text)
                    
                    # Сбрасываем активацию после команды
                    if self.config.activation_mode == "keyword":
                        self.activation_detected = False
        
        except Exception as e:
            print(f"❌ Ошибка обработки аудио: {e}")
            self.stats["recognition_errors"] += 1
    
    def _recognize_speech(self, audio_data: np.ndarray) -> Optional[str]:
        """Распознает речь из аудио данных"""
        try:
            if self.config.speech_recognizer == "vosk":
                return self._recognize_vosk(audio_data)
            elif self.config.speech_recognizer == "whisper":
                return self._recognize_whisper(audio_data)
            elif self.config.speech_recognizer == "google":
                return self._recognize_google(audio_data)
            else:
                return self._recognize_google(audio_data)
                
        except Exception as e:
            print(f"❌ Ошибка распознавания речи: {e}")
            return None
    
    def _recognize_vosk(self, audio_data: np.ndarray) -> Optional[str]:
        """Распознает с помощью Vosk"""
        if not hasattr(self, 'vosk_recognizer'):
            return None
        
        try:
            # Конвертируем в байты
            audio_bytes = audio_data.astype(np.int16).tobytes()
            
            # Распознаем
            if self.vosk_recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(self.vosk_recognizer.Result())
                return result.get("text", "")
            else:
                result = json.loads(self.vosk_recognizer.PartialResult())
                return result.get("partial", "")
                
        except Exception as e:
            print(f"❌ Ошибка Vosk распознавания: {e}")
            return None
    
    def _recognize_whisper(self, audio_data: np.ndarray) -> Optional[str]:
        """Распознает с помощью Whisper"""
        if not hasattr(self, 'whisper_model'):
            return None
        
        try:
            # Конвертируем в float32
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Распознаем
            result = self.whisper_model.transcribe(
                audio_float,
                language=self.config.language,
                fp16=False  # Используем float32 для CPU
            )
            
            return result.get("text", "")
            
        except Exception as e:
            print(f"❌ Ошибка Whisper распознавания: {e}")
            return None
    
    def _recognize_google(self, audio_data: np.ndarray) -> Optional[str]:
        """Распознает с помощью Google Speech Recognition"""
        if not hasattr(self, 'google_recognizer'):
            return None
        
        try:
            # Создаем AudioData объект
            audio_sr = sr.AudioData(
                audio_data.tobytes(),
                self.config.sample_rate,
                2  # sample width in bytes
            )
            
            # Распознаем
            text = self.google_recognizer.recognize_google(
                audio_sr,
                language=f"{self.config.language}-{self.config.language.upper()}"
            )
            
            return text
            
        except sr.UnknownValueError:
            # Речь не распознана
            return None
        except sr.RequestError as e:
            print(f"❌ Ошибка Google Speech Recognition: {e}")
            return None
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {e}")
            return None
    
    def _activation_detection_loop(self):
        """Цикл обнаружения ключевого слова"""
        print("🔍 Запускаю обнаружение ключевого слова...")
        
        activation_buffer = []
        
        while self.is_listening and self.config.activation_mode == "keyword":
            try:
                # Получаем аудио чанк
                chunk = self.audio_queue.get(timeout=0.5)
                audio_data = chunk['data']
                
                # Добавляем в буфер
                activation_buffer.append(audio_data)
                
                # Ограничиваем размер буфера (3 секунды)
                if len(activation_buffer) > 3 * self.config.sample_rate / self.config.chunk_size:
                    activation_buffer.pop(0)
                
                # Периодически проверяем буфер
                if len(activation_buffer) >= 5:  # Примерно 0.5 секунды
                    combined = np.concatenate(activation_buffer)
                    text = self._recognize_speech(combined)
                    
                    if text and self._check_activation(text):
                        print(f"✅ Ключевое слово обнаружено: {text}")
                        self.activation_detected = True
                        activation_buffer = []  # Очищаем буфер
                    
                    # Очищаем старый буфер
                    activation_buffer = activation_buffer[-5:]
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Ошибка в активационном цикле: {e}")
    
    def _check_activation(self, text: str) -> bool:
        """Проверяет содержит ли текст ключевое слово"""
        if not text:
            return False
        
        text_lower = text.lower()
        keyword = self.config.activation_keyword.lower()
        
        # Проверяем разные формы ключевого слова
        variations = [
            keyword,
            keyword + " ",
            " " + keyword,
            keyword + ",",
            keyword + "."
        ]
        
        for variation in variations:
            if variation in text_lower:
                return True
        
        return False
    
    def speak(self, text: str, wait: bool = True):
        """
        Озвучивает текст
        
        Args:
            text: Текст для озвучивания
            wait: Ждать завершения воспроизведения
        """
        if not text or self.is_speaking:
            return
        
        print(f"🗣️ Озвучиваю: {text[:50]}...")
        self.is_speaking = True
        
        try:
            if self.config.tts_engine == "silero" and hasattr(self, 'silero_model'):
                self._speak_silero(text, wait)
            elif hasattr(self, 'tts_engine') and self.tts_engine:
                self._speak_pyttsx3(text, wait)
            else:
                print(f"⚠️ TTS движок не доступен")
        
        except Exception as e:
            print(f"❌ Ошибка синтеза речи: {e}")
        
        finally:
            self.is_speaking = False
    
    def _speak_pyttsx3(self, text: str, wait: bool):
        """Озвучивает с помощью pyttsx3"""
        self.tts_engine.say(text)
        
        if wait:
            self.tts_engine.runAndWait()
        else:
            # Запускаем в отдельном потоке
            def speak_thread():
                self.tts_engine.runAndWait()
            
            threading.Thread(target=speak_thread, daemon=True).start()
    
    def _speak_silero(self, text: str, wait: bool):
        """Озвучивает с помощью Silero"""
        try:
            import torchaudio
            
            # Генерируем речь
            audio = self.silero_model.apply_tts(
                text=text,
                speaker=self.silero_speaker,
                sample_rate=24000
            )
            
            # Сохраняем во временный файл
            temp_file = Path(self.config.cache_path) / f"tts_{int(time.time())}.wav"
            torchaudio.save(str(temp_file), audio.unsqueeze(0), 24000)
            
            # Воспроизводим
            self._play_audio_file(temp_file)
            
            # Удаляем временный файл
            if wait:
                while self.is_playing_audio:
                    time.sleep(0.1)
                temp_file.unlink()
            else:
                threading.Thread(
                    target=self._wait_and_delete,
                    args=(temp_file,),
                    daemon=True
                ).start()
                
        except Exception as e:
            print(f"❌ Ошибка Silero TTS: {e}")
            # Fallback на pyttsx3
            if hasattr(self, 'tts_engine'):
                self._speak_pyttsx3(text, wait)
    
    def _play_audio_file(self, filepath: Path):
        """Воспроизводит аудио файл"""
        try:
            import pyaudio
            import wave
            
            wf = wave.open(str(filepath), 'rb')
            
            p = pyaudio.PyAudio()
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
            
        except Exception as e:
            print(f"❌ Ошибка воспроизведения аудио: {e}")
    
    def _wait_and_delete(self, filepath: Path):
        """Ждет и удаляет файл"""
        time.sleep(5)  # Ждем завершения воспроизведения
        if filepath.exists():
            filepath.unlink()
    
    def listen_once(self, timeout: float = 5.0) -> Optional[str]:
        """
        Слушает одну команду с таймаутом
        """
        print(f"👂 Слушаю команду (таймаут: {timeout}с)...")
    
        if not hasattr(self, 'google_recognizer') or not self.google_recognizer:
            print("❌ Распознаватель не инициализирован")
            return None
    
        if not hasattr(self, 'google_microphone') or not self.google_microphone:
            print("❌ Микрофон не найден")
            return None
    
        try:
            with self.google_microphone as source:
                self.google_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("🎤 Говорите сейчас...")
                audio = self.google_recognizer.listen(source, timeout=timeout, phrase_time_limit=3)
            
                text = self.google_recognizer.recognize_google(
                    audio,
                    language=f"{self.config.language}-{self.config.language.upper()}"
                )
            
                print(f"✅ Распознано: {text}")
                return text
            
        except sr.WaitTimeoutError:
            print("⏰ Таймаут ожидания команды")
            return None
        except sr.UnknownValueError:
            print("🤷 Речь не распознана")
            return None
        except Exception as e:
            print(f"❌ Ошибка прослушивания: {e}")
            return None
    
    def set_command_callback(self, callback: Callable[[str], None]):
        """Устанавливает коллбэк для распознанных команд"""
        self.on_command_callback = callback
    
    def set_activation_callback(self, callback: Callable[[str], None]):
        """Устанавливает коллбэк для активации"""
        self.on_activation_callback = callback
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику"""
        return self.stats.copy()
    
    def save_audio_debug(self, audio_data: np.ndarray, filename: str):
        """Сохраняет аудио для отладки"""
        try:
            import soundfile as sf
            
            debug_dir = Path("logs/audio")
            debug_dir.mkdir(exist_ok=True)
            
            filepath = debug_dir / f"{filename}_{int(time.time())}.wav"
            sf.write(str(filepath), audio_data, self.config.sample_rate)
            
            print(f"💾 Аудио сохранено: {filepath}")
            
        except Exception as e:
            print(f"⚠️ Ошибка сохранения аудио: {e}")
    
    def cleanup(self):
        """Очищает ресурсы"""
        print("🧹 Очистка голосового движка...")
        
        self.stop_listening()
        
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()

# Утилита для тестирования
def test_voice_engine():
    """Тестирует голосовой движок"""
    print("🧪 Тестирование голосового движка...")
    
    if not HAS_AUDIO_DEPS:
        print("❌ Аудио зависимости не установлены")
        return
    
    # Конфигурация
    config = VoiceConfig(
        speech_recognizer="google",  # Начнем с Google для теста
        activation_mode="always",    # Всегда слушаем для теста
        use_vad=False                # Отключаем VAD для теста
    )
    
    try:
        engine = NeuralVoiceEngine(config)
        
        print("\n🔊 Тест синтеза речи...")
        engine.speak("Привет! Я голосовой движок. Готов к тестированию.")
        time.sleep(2)
        
        print("\n👂 Тест прослушивания одной команды...")
        print("Скажите что-нибудь в течение 5 секунд...")
        
        text = engine.listen_once(timeout=5)
        
        if text:
            print(f"✅ Распознано: {text}")
            engine.speak(f"Вы сказали: {text}")
        else:
            print("❌ Ничего не распознано")
            engine.speak("К сожалению, я ничего не расслышала")
        
        # Статистика
        stats = engine.get_stats()
        print(f"\n📊 Статистика: {stats}")
        
        # Очистка
        engine.cleanup()
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()

# Простой пример использования
class SimpleVoiceAssistant:
    """Простой голосовой ассистент для демонстрации"""
    
    def __init__(self):
        self.config = VoiceConfig(
            speech_recognizer="google",
            activation_mode="keyword",
            activation_keyword="ассистент"
        )
        
        self.engine = NeuralVoiceEngine(self.config)
        self.engine.set_command_callback(self.on_command)
        self.engine.set_activation_callback(self.on_activation)
        
        self.is_running = False
    
    def on_command(self, text: str):
        """Обработка команды"""
        print(f"🎯 Команда: {text}")
        
        # Простая логика ответа
        if "привет" in text.lower():
            self.engine.speak("Привет! Рада вас слышать!")
        elif "как дела" in text.lower():
            self.engine.speak("Всё отлично! Готова помогать!")
        elif "пока" in text.lower() or "до свидания" in text.lower():
            self.engine.speak("До свидания! Была рада помочь!")
            self.stop()
        else:
            self.engine.speak(f"Вы сказали: {text}")
    
    def on_activation(self, text: str):
        """Обработка активации"""
        print(f"🔔 Активация по ключевому слову: {text}")
        self.engine.speak("Да, я слушаю!")
    
    def start(self):
        """Запускает ассистента"""
        print("🚀 Запуск голосового ассистента...")
        print(f"💡 Скажите '{self.config.activation_keyword}' для активации")
        
        self.is_running = True
        self.engine.start_listening()
        
        # Главный цикл
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Остановка по Ctrl+C")
        finally:
            self.stop()
    
    def stop(self):
        """Останавливает ассистента"""
        if self.is_running:
            print("🛑 Остановка ассистента...")
            self.is_running = False
            self.engine.cleanup()

if __name__ == "__main__":
    # Тестируем
    test_voice_engine()
    
    # Запускаем простого ассистента
    # assistant = SimpleVoiceAssistant()
    # assistant.start()