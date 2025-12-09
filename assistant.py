"""
Упрощенный основной класс ассистента
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class NeuralAssistant:
    """
    Основной класс ассистента (упрощенная версия)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        print("[ИНИТ] Инициализация ядра ассистента...")
        
        # Конфигурация
        self.config = config or {}
        
        # Компоненты будут установлены из main.py
        self.intent_recognizer = None
        self.command_executor = None
        self.voice_engine = None
        self.memory = None
        self.learning_engine = None
        self.code_generator = None
        
        # Состояние
        self.is_active = False
        
        print("[OK] Ядро ассистента инициализировано")
    
    def process_command(self, text: str) -> Dict[str, Any]:
        """Обрабатывает команду (упрощенная версия)"""
        if not self.intent_recognizer or not self.command_executor:
            return {"success": False, "message": "Компоненты не инициализированы"}
        
        try:
            # Распознаем намерение
            intent_result = self.intent_recognizer.predict(text)
            
            # Выполняем команду
            execution_result = self.command_executor.execute(
                intent=intent_result['intent'],
                entities=intent_result.get('entities', []),
                original_text=text,
                context={}
            )
            
            # Сохраняем в память если есть
            if self.memory:
                self.memory.add_command(
                    text=text,
                    intent=intent_result['intent'],
                    entities=intent_result.get('entities', []),
                    success=execution_result.success,
                    result=execution_result.data
                )
            
            return {
                "success": execution_result.success,
                "message": execution_result.message,
                "data": execution_result.data
            }
            
        except Exception as e:
            print(f"[ERROR] Ошибка обработки команды: {e}")
            return {"success": False, "message": f"Ошибка: {str(e)}"}
    
    
    def teach_program(self, program_name: str, program_path: str):
        """Обучает ассистента новой программе"""
        if hasattr(self, '_program_opener'):
            success = self._program_opener.learn_new_program(
                program_name, 
                program_path,
                aliases=[program_name]
            )
        
            if success:
                return {
                    'success': True,
                    'message': f'Выучил программу "{program_name}"'
                }
    
        return {
            'success': False,
            'message': 'Не удалось обучить программе'
        }
        
    
    def start_learning(self, command: str, explanation: str):
        """Запускает обучение новой команде"""
        if not self.learning_engine or not self.code_generator:
            return {"success": False, "message": "Компоненты обучения не инициализированы"}
        
        try:
            # Генерируем код
            generated_code = self.code_generator.generate(
                description=explanation,
                intent_type="custom"
            )
            
            # Обучаем нейросеть
            success = self.learning_engine.train_on_example(
                text=command,
                explanation=explanation,
                examples=[command],
                generated_code=generated_code
            )
            
            if success and self.memory:
                self.memory.add_learned_command(
                    command_text=command,
                    explanation=explanation,
                    generated_code=generated_code,
                    examples=[command]
                )
            
            return {"success": success, "message": "Обучение завершено"}
            
        except Exception as e:
            print(f"[ERROR] Ошибка обучения: {e}")
            return {"success": False, "message": f"Ошибка обучения: {str(e)}"}
        
    def teach_command_interactive(self, command_text: str, explanation: str, examples: list = None) -> dict:
        """
        Обучение ассистента новой команде.
        command_text: текст команды
        explanation: описание что делает команда
        examples: список примеров похожих команд
        """
        if not self.learning_engine or not self.code_generator:
            return {"success": False, "message": "Компоненты обучения не инициализированы"}

        examples = examples or [command_text]

        try:
            # Генерируем код для команды
            generated_code = self.code_generator.generate(
                description=explanation,
                intent_type="custom"
            )

            # Обучаем движок
            success = self.learning_engine.train_on_example(
                text=command_text,
                explanation=explanation,
                examples=examples,
                generated_code=generated_code
            )

            # Сохраняем в память
            if success and self.memory:
                self.memory.add_learned_command(
                    command_text=command_text,
                    explanation=explanation,
                    generated_code=generated_code,
                    examples=examples
                )

            return {"success": success, "message": "Команда успешно выучена" if success else "Ошибка обучения"}

        except Exception as e:
            print(f"[ERROR] Ошибка обучения команды: {e}")
            return {"success": False, "message": f"Ошибка обучения: {str(e)}"}
        
