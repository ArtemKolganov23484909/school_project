import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler, CallbackContext
from back import verified_appeal_to_ai
from PIL import Image

# Создаем папку для сохранения изображений
IMAGE_FOLDER = "received_images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Словарь ответов для анализа (пример)
ANSWER_DICTIONARY = {
    0: "Class A",
    1: "Class B",
    2: "Class C",
}

# Стартовая команда
def start(update: Update, context: CallbackContext):
    keyboard = [
        [InlineKeyboardButton("Анализировать фото", callback_data="analyze_photo")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text("Что вы хотите сделать?", reply_markup=reply_markup)

# Обработчик кнопок
def button_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()

    if query.data == "analyze_photo":
        query.edit_message_text(text="Пришлите фото для анализа.")

# Обработчик получения фото
def photo_handler(update: Update, context: CallbackContext):
    photo = update.message.photo[-1]  # Берем фото наивысшего качества
    file = context.bot.get_file(photo.file_id)
    
    # Сохраняем изображение
    image_path = os.path.join(IMAGE_FOLDER, f"{photo.file_id}.jpg")
    file.download(image_path)

    # Анализ изображения
    try:
        with Image.open(image_path) as img:
            result = verified_appeal_to_ai(img, ANSWER_DICTIONARY)
            if result[0]:
                update.message.reply_text(f"Анализ завершен. Результат: {result[1]}")
            else:
                update.message.reply_text(f"Анализ завершен. Результаты неоднозначны: {result[1]}")
    except Exception as e:
        update.message.reply_text(f"Произошла ошибка при анализе изображения: {e}")

# Главная функция
def main():
    updater = Updater("YOUR_TOKEN", use_context=True)
    dispatcher = updater.dispatcher

    # Обработчики команд
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CallbackQueryHandler(button_handler))

    # Обработчик фото
    dispatcher.add_handler(MessageHandler(Filters.photo, photo_handler))

    # Запуск бота
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
