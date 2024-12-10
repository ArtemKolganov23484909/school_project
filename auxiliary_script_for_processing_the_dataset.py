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


from PIL import Image, ImageOps
import os

def augment_images(folder_path):
    """
    Увеличивает датасет изображений, выполняя несколько преобразований для каждого изображения.
    Каждое изображение поворачивается на 30 градусов 12 раз, затем зеркально отражается, 
    а также применяются дополнительные трансформации.
    
    :param folder_path: Путь к папке с изображениями.
    """
    if not os.path.exists(folder_path):
        print(f"Папка {folder_path} не найдена.")
        return
    
    # Создаем папку для сохранения измененных изображений
    augmented_folder = os.path.join(folder_path, "augmented")
    os.makedirs(augmented_folder, exist_ok=True)

    # Список файлов в папке
    images = sorted([f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))], key=lambda x: int(x.split('.')[0]))

    for image_name in images:
        # Открываем изображение
        image_path = os.path.join(folder_path, image_name)
        try:
            img = Image.open(image_path)
        except Exception as e:
            print(f"Ошибка при открытии {image_name}: {e}")
            continue

        # Базовое имя файла
        base_name = os.path.splitext(image_name)[0]

        # 12 поворотов на 30 градусов
        for i in range(12):
            rotated = img.rotate(i * 30, expand=True)
            rotated.save(os.path.join(augmented_folder, f"{base_name}_{i+1}.png"))

        # Зеркальное отражение
        mirrored = ImageOps.mirror(img)
        mirrored.save(os.path.join(augmented_folder, f"{base_name}_mirrored.png"))

        # Дополнительные преобразования (например, изменение яркости)
        for factor in [0.8, 1.2]:  # Уменьшаем и увеличиваем яркость
            brightened = ImageEnhance.Brightness(img).enhance(factor)
            brightened.save(os.path.join(augmented_folder, f"{base_name}_brightness_{factor}.png"))

        # Сохранение в градациях серого
        grayscale = ImageOps.grayscale(img)
        grayscale.save(os.path.join(augmented_folder, f"{base_name}_grayscale.png"))

    print(f"Обработано {len(images)} изображений. Результаты сохранены в {augmented_folder}.")

