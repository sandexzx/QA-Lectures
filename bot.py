import asyncio
import logging
import os
import json
import random
from typing import List, Dict, Any
from pathlib import Path
import re

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

from openai import OpenAI
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
BOT_TOKEN = os.getenv("BOT_TOKEN")
NEUROAPI_API_KEY = os.getenv("NEUROAPI_API_KEY")
LECTURES_DIR = Path("lectures")

# Инициализация клиента OpenAI
client = OpenAI(
    base_url="https://neuroapi.host/v1",
    api_key=NEUROAPI_API_KEY,
)

# Состояния FSM
class UserStates(StatesGroup):
    waiting_for_answer = State()
    selecting_lecture = State()

# Класс для работы с данными пользователей
class UserDataManager:
    def __init__(self):
        self.data_file = Path("user_data.json")
        self.load_data()
    
    def load_data(self):
        if self.data_file.exists():
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = {}
    
    def save_data(self):
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def get_user_data(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.data:
            self.data[user_id] = {
                "max_lecture": None,
                "interview_active": False,
                "asked_questions": [],
                "ai_model": "gemini-2.5-flash"
            }
        return self.data[user_id]
    
    def update_user_data(self, user_id: str, **kwargs):
        user_data = self.get_user_data(user_id)
        user_data.update(kwargs)
        self.save_data()

# Класс для работы с лекциями
class LectureManager:
    def __init__(self, lectures_dir: Path):
        self.lectures_dir = lectures_dir
        self.lectures = self.load_lectures()
    
    def load_lectures(self) -> List[Dict[str, Any]]:
        lectures = []
        if not self.lectures_dir.exists():
            logger.error(f"Директория {self.lectures_dir} не найдена")
            return lectures
        
        for file_path in self.lectures_dir.glob("*.md"):
            match = re.match(r'(\d+(?:\.\d+)*)\.\s*(.+)\.md', file_path.name)
            if match:
                number = match.group(1)
                title = match.group(2).replace('_', ' ').replace('-', ' ')
                lectures.append({
                    "number": number,
                    "title": title,
                    "file_path": file_path
                })
        
        # Сортировка по номерам
        lectures.sort(key=lambda x: [int(n) for n in x["number"].split('.')])
        return lectures
    
    def get_lectures_up_to(self, max_lecture: str) -> List[Dict[str, Any]]:
        if not max_lecture:
            return []
        
        max_parts = [int(n) for n in max_lecture.split('.')]
        filtered_lectures = []
        
        for lecture in self.lectures:
            lecture_parts = [int(n) for n in lecture["number"].split('.')]
            if self.compare_lecture_numbers(lecture_parts, max_parts) <= 0:
                filtered_lectures.append(lecture)
        
        return filtered_lectures
    
    def compare_lecture_numbers(self, parts1: List[int], parts2: List[int]) -> int:
        min_len = min(len(parts1), len(parts2))
        for i in range(min_len):
            if parts1[i] < parts2[i]:
                return -1
            elif parts1[i] > parts2[i]:
                return 1
        return len(parts1) - len(parts2)
    
    def get_lecture_content(self, lecture: Dict[str, Any]) -> str:
        try:
            with open(lecture["file_path"], 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Ошибка чтения файла {lecture['file_path']}: {e}")
            return ""

# Менеджеры
user_manager = UserDataManager()
lecture_manager = LectureManager(LECTURES_DIR)

# Инициализация бота
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

# Главное меню
def get_main_menu():
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⚙️ Настройки", callback_data="settings")],
        [InlineKeyboardButton(text="🎯 Режим собеседования", callback_data="interview_mode")]
    ])
    return keyboard

# Меню настроек
def get_settings_menu():
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📚 Выбрать последнюю лекцию", callback_data="select_lecture")],
        [InlineKeyboardButton(text="🤖 Выбрать AI модель", callback_data="select_model")],
        [InlineKeyboardButton(text="🔙 Назад", callback_data="back_to_main")]
    ])
    return keyboard

# Генерация клавиатуры для выбора лекций
def get_lecture_selection_keyboard():
    keyboard = []
    for lecture in lecture_manager.lectures:
        button_text = f"{lecture['number']}. {lecture['title']}"
        callback_data = f"lecture_{lecture['number']}"
        keyboard.append([InlineKeyboardButton(text=button_text, callback_data=callback_data)])
    
    keyboard.append([InlineKeyboardButton(text="🔙 Назад", callback_data="settings")])
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

# Генерация клавиатуры для выбора AI модели
def get_model_selection_keyboard():
    models = [
        ("gpt-4.1", "GPT-4.1"),
        ("claude-sonnet-4-all", "Claude Sonnet 4 All"),
        ("o3-mini", "O3 Mini"),
        ("o4-mini", "O4 Mini"),
        ("gemini-2.5-flash-lite-preview-06-17", "Gemini 2.5 Flash Lite Preview"),
        ("gemini-2.5-flash", "Gemini 2.5 Flash")
    ]
    
    keyboard = []
    for model_code, model_name in models:
        callback_data = f"model_{model_code}"
        keyboard.append([InlineKeyboardButton(text=model_name, callback_data=callback_data)])
    
    keyboard.append([InlineKeyboardButton(text="🔙 Назад", callback_data="settings")])
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

# Обработчики команд
@dp.message(Command("start"))
async def start_command(message: Message):
    await message.answer(
        "Добро пожаловать в бота для подготовки к собеседованию! 🎯\n\n"
        "Выберите действие:",
        reply_markup=get_main_menu()
    )

@dp.callback_query(F.data == "settings")
async def settings_callback(callback: CallbackQuery):
    user_id = str(callback.from_user.id)
    user_data = user_manager.get_user_data(user_id)
    
    current_lecture = user_data.get("max_lecture", "Не выбрана")
    current_model = user_data.get("ai_model", "gemini-2.5-flash")
    text = f"⚙️ Настройки\n\nТекущая последняя лекция: {current_lecture}\nТекущая AI модель: {current_model}"
    
    await callback.message.edit_text(text, reply_markup=get_settings_menu())

@dp.callback_query(F.data == "select_lecture")
async def select_lecture_callback(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text(
        "📚 Выберите последнюю изученную лекцию:",
        reply_markup=get_lecture_selection_keyboard()
    )
    await state.set_state(UserStates.selecting_lecture)

@dp.callback_query(F.data.startswith("lecture_"))
async def lecture_selected_callback(callback: CallbackQuery, state: FSMContext):
    lecture_number = callback.data.replace("lecture_", "")
    user_id = str(callback.from_user.id)
    
    user_manager.update_user_data(user_id, max_lecture=lecture_number)
    
    await callback.message.edit_text(
        f"✅ Последняя лекция установлена: {lecture_number}",
        reply_markup=get_settings_menu()
    )
    await state.clear()

@dp.callback_query(F.data == "select_model")
async def select_model_callback(callback: CallbackQuery):
    await callback.message.edit_text(
        "🤖 Выберите AI модель:",
        reply_markup=get_model_selection_keyboard()
    )

@dp.callback_query(F.data.startswith("model_"))
async def model_selected_callback(callback: CallbackQuery):
    model_code = callback.data.replace("model_", "")
    user_id = str(callback.from_user.id)
    
    user_manager.update_user_data(user_id, ai_model=model_code)
    
    await callback.message.edit_text(
        f"✅ AI модель установлена: {model_code}",
        reply_markup=get_settings_menu()
    )

@dp.callback_query(F.data == "back_to_main")
async def back_to_main_callback(callback: CallbackQuery):
    await callback.message.edit_text(
        "Выберите действие:",
        reply_markup=get_main_menu()
    )

@dp.callback_query(F.data == "interview_mode")
async def interview_mode_callback(callback: CallbackQuery, state: FSMContext):
    user_id = str(callback.from_user.id)
    user_data = user_manager.get_user_data(user_id)
    
    if not user_data.get("max_lecture"):
        await callback.message.edit_text(
            "❌ Сначала выберите последнюю изученную лекцию в настройках!",
            reply_markup=get_main_menu()
        )
        return
    
    user_manager.update_user_data(user_id, interview_active=True)
    
    await callback.message.edit_text("🎯 Режим собеседования активирован! Генерирую вопрос...")
    await generate_and_send_question(callback.message, user_id, state)

async def generate_and_send_question(message: Message, user_id: str, state: FSMContext):
    user_data = user_manager.get_user_data(user_id)
    max_lecture = user_data.get("max_lecture")
    asked_questions = user_data.get("asked_questions", [])
    
    available_lectures = lecture_manager.get_lectures_up_to(max_lecture)
    
    if not available_lectures:
        await message.answer("❌ Нет доступных лекций для генерации вопросов.")
        return
    
    # Случайно выбираем 1-3 лекции для генерации вопроса
    num_lectures_to_use = random.randint(1, min(3, len(available_lectures)))
    selected_lectures = random.sample(available_lectures, num_lectures_to_use)
    
    # Получаем содержимое выбранных лекций
    lecture_contents = []
    for lecture in selected_lectures:
        content = lecture_manager.get_lecture_content(lecture)
        lecture_contents.append(f"Лекция {lecture['number']}: {lecture['title']}\n{content}")
    
    all_content = "\n\n".join(lecture_contents)
    
    logger.info(f"Selected {num_lectures_to_use} lectures out of {len(available_lectures)} available")
    logger.info(f"Selected lectures: {[l['number'] for l in selected_lectures]}")
    
    logger.info(f"Loaded {len(available_lectures)} lectures")
    logger.info(f"Total content length: {len(all_content)} characters")
    
    # Проверяем, что есть контент для генерации вопроса
    if len(all_content.strip()) == 0:
        await message.answer("❌ Не удалось загрузить содержимое лекций. Проверьте файлы.")
        return
    
    # Генерируем вопрос через API
    try:
        # Создаем историю заданных вопросов для контекста
        asked_questions_text = ""
        if asked_questions:
            asked_questions_text = f"\n\nРанее заданные вопросы (НЕ повторяй их):\n" + "\n".join(asked_questions[-5:])  # Последние 5 вопросов
        
        question_types = [
            "вопрос на понимание теоретических основ (только на знание того, что есть в лекции)",
            "вопрос про инструменты и их использование (только на знание того, что есть в лекции)"
        ]
        
        question_type = random.choice(question_types)
        
        selected_titles = [f"{l['number']}. {l['title']}" for l in selected_lectures]
        
        prompt = f"""Ты - опытный интервьюер по QA/тестированию. На основе предоставленного материала создай один конкретный вопрос для собеседования.

Материал из лекций ({', '.join(selected_titles)}):
{all_content}{asked_questions_text}

Требования к вопросу:
- Сгенерируй {question_type}
- Основывайся на конкретном материале из предоставленных лекций
- Вопрос должен быть практическим и реалистичным
- Задавай вопросы только о том, что ПОДРОБНО объяснено в лекции
- НЕ спрашивай о практическом применении инструментов, если в лекции нет детального описания их использования
- НЕ требуй знаний о процессах и методиках, которые только упомянуты, но не раскрыты
- Проверять понимание основных концепций из материала
- Быть сформулированным как на реальном собеседовании
- Не должен быть слишком простым или слишком сложным
- Один конкретный вопрос, не список вопросов
- ВАЖНО: Не выходи за пределы материала лекций
- ВАЖНО: Если в лекции что-то только упоминается без объяснения - не спрашивай об этом
- ВАЖНО: Не повторяй ранее заданные вопросы!
- Можешь указать контекст: "В рамках работы QA-инженера..." или "При тестировании..."

Сгенерируй только вопрос, без дополнительных пояснений."""

        user_data = user_manager.get_user_data(user_id)
        selected_model = user_data.get("ai_model", "gemini-2.5-flash")
        
        completion = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        logger.info(f"API Response: {completion}")
        
        question = completion.choices[0].message.content
        if question:
            question = question.strip()
        else:
            question = ""
        
        logger.info(f"Generated question: '{question}'")
        
        # Проверяем, что вопрос не пустой
        if not question or len(question.strip()) == 0:
            logger.warning("Получен пустой вопрос от API")
            question = "Расскажите о ключевых принципах тестирования программного обеспечения и их применении на практике."
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Следующий вопрос", callback_data="next_question")],
            [InlineKeyboardButton(text="🏁 Завершить собеседование", callback_data="end_interview")]
        ])
        
        # Сохраняем заданный вопрос
        asked_questions.append(question)
        user_manager.update_user_data(user_id, asked_questions=asked_questions)
        
        # Информация о выбранных лекциях
        lectures_info = f"📚 *Основано на:* {', '.join(selected_titles)}\n\n"
        
        await message.answer(f"❓ **Вопрос для собеседования:**\n\n{lectures_info}{question}", reply_markup=keyboard)
        await state.set_state(UserStates.waiting_for_answer)
        await state.update_data(last_question=question, lecture_content=all_content, conversation_history=[])
        
    except Exception as e:
        logger.error(f"Ошибка генерации вопроса: {e}")
        await message.answer("❌ Произошла ошибка при генерации вопроса. Попробуйте еще раз.")

@dp.callback_query(F.data == "next_question")
async def next_question_callback(callback: CallbackQuery, state: FSMContext):
    user_id = str(callback.from_user.id)
    await callback.message.edit_text("🔄 Генерирую следующий вопрос...")
    await generate_and_send_question(callback.message, user_id, state)

@dp.callback_query(F.data == "end_interview")
async def end_interview_callback(callback: CallbackQuery, state: FSMContext):
    user_id = str(callback.from_user.id)
    user_data = user_manager.get_user_data(user_id)
    questions_count = len(user_data.get("asked_questions", []))
    
    user_manager.update_user_data(user_id, interview_active=False, asked_questions=[])
    
    await callback.message.edit_text(
        f"🏁 Собеседование завершено!\n\nВы ответили на {questions_count} вопросов. Хорошая работа! 👏",
        reply_markup=get_main_menu()
    )
    await state.clear()

@dp.message(StateFilter(UserStates.waiting_for_answer))
async def handle_answer(message: Message, state: FSMContext):
    user_id = str(message.from_user.id)
    user_data = user_manager.get_user_data(user_id)
    
    if not user_data.get("interview_active"):
        return
    
    # Получаем данные о последнем вопросе из состояния
    state_data = await state.get_data()
    last_question = state_data.get("last_question", "")
    conversation_history = state_data.get("conversation_history", [])
    
    await message.answer("🤔 Анализирую ваш ответ...")
    
    # Добавляем новый ответ пользователя в историю
    conversation_history.append({"role": "user", "content": message.text})
    await state.update_data(conversation_history=conversation_history)
    
    # Анализируем ответ пользователя
    await analyze_user_answer(message, user_id, last_question, conversation_history, state)

async def analyze_user_answer(message: Message, user_id: str, question: str, conversation_history: list, state: FSMContext):
    state_data = await state.get_data()
    lecture_content = state_data.get("lecture_content", "")
    
    try:
        user_data = user_manager.get_user_data(user_id)
        selected_model = user_data.get("ai_model", "gemini-2.5-flash")
        
        # Если это первый ответ, создаем системный промпт
        if len(conversation_history) == 1:
            system_prompt = f"""Ты - опытный преподаватель и наставник по QA/тестированию. Ученик отвечает на вопрос по материалу лекций.

Материал лекций:
{lecture_content}

Вопрос: {question}

Твоя задача как учителя:
1. Проанализировать ответ ученика
2. Дать конструктивную обратную связь
3. Если ответ неполный или неточный - задать наводящие вопросы
4. Если ответ хороший - похвалить и дополнить полезной информацией
5. Быть доброжелательным и поддерживающим

Формат ответа:
- Начни с оценки ответа (по 5 балльной шкале)
- Дай конструктивную обратную связь
- Если нужно - задай наводящий вопрос или дай подсказку

Отвечай как опытный учитель, а не как строгий HR."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_history[0]["content"]}
            ]
        else:
            # Для продолжающегося диалога используем историю сообщений
            messages = conversation_history.copy()
        
        completion = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        feedback = completion.choices[0].message.content.strip()
        
        # Добавляем ответ AI в историю диалога
        conversation_history.append({"role": "assistant", "content": feedback})
        await state.update_data(conversation_history=conversation_history)
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="💬 Дополнить ответ", callback_data="continue_answer")],
            [InlineKeyboardButton(text="🔄 Следующий вопрос", callback_data="next_question")],
            [InlineKeyboardButton(text="🏁 Завершить собеседование", callback_data="end_interview")]
        ])
        
        await message.answer(f"👨‍🏫 **Обратная связь:**\n\n{feedback}", reply_markup=keyboard)
        
    except Exception as e:
        logger.error(f"Ошибка анализа ответа: {e}")
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Следующий вопрос", callback_data="next_question")],
            [InlineKeyboardButton(text="🏁 Завершить собеседование", callback_data="end_interview")]
        ])
        
        await message.answer(
            "✅ Ваш ответ принят! Что дальше?",
            reply_markup=keyboard
        )

@dp.callback_query(F.data == "continue_answer")
async def continue_answer_callback(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text(
        "💬 Можете дополнить или уточнить свой ответ:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Следующий вопрос", callback_data="next_question")],
            [InlineKeyboardButton(text="🏁 Завершить собеседование", callback_data="end_interview")]
        ])
    )
    await state.set_state(UserStates.waiting_for_answer)

async def main():
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN не установлен!")
        return
    
    if not NEUROAPI_API_KEY:
        logger.error("NEUROAPI_API_KEY не установлен!")
        return
    
    logger.info("Запуск бота...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())