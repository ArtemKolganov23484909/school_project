import os
from PIL import Image

def convert_and_resize_images(input_dir, output_dir, target_size=(512, 512), target_format='jpg'):
    """
    Конвертирует изображения в заданный формат, изменяет их размер и сохраняет в новую папку.

    :param input_dir: Папка с исходными изображениями.
    :param output_dir: Папка для сохранения обработанных изображений.
    :param target_size: Размер изображений (ширина, высота).
    :param target_format: Формат изображений ('jpg', 'png').
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, filename in enumerate(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, filename)

        try:
            # Открываем изображение
            with Image.open(file_path) as img:
                # Конвертируем в RGB (на случай, если исходное изображение в другом цветовом пространстве)
                img = img.convert('RGB')
                
                # Изменяем размер с использованием нового атрибута
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Сохраняем изображение в нужном формате
                output_path = os.path.join(output_dir, f'image_{idx + 1:04d}.{target_format}')
                img.save(output_path, target_format.upper())
                print(f"Обработано: {filename} -> {output_path}")

        except Exception as e:
            print(f"Ошибка при обработке {filename}: {e}")

# Пример использования
convert_and_resize_images(
    input_dir='E:/датасет/Lasius Niger1',  # Папка с сырыми изображениями
    output_dir='E:/датасет/Lasius Niger',  # Папка для сохранения обработанных изображений
    target_size=(512, 512),  # Размер для приведения изображений
    target_format='JPEG'  # Формат для сохранения
)
