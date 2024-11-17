'''import'''
from PIL import Image
from collections import Counter
'''pip install pillow'''


def rotate_and_replace_image(image_path: str, angle: float = 30) -> None:
    """
    Поворачивает изображение на указанный угол и заменяет исходный файл.

    :param image_path: Путь к изображению.
    :param angle: Угол поворота в градусах (по умолчанию 30).
    """
    try:
        # Открываем изображение
        with Image.open(image_path) as img:
            # Поворачиваем изображение
            rotated_img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
            # Сохраняем изображение, заменяя оригинал
            rotated_img.save(image_path)
    except Exception as e:
        print(f"Ошибка при обработке изображения {image_path}: {e}")


def appeal_to_ai(image, answer_dictionary, *args, **kwargs): 
    '''
    обращаемся к ИИ и получаем коэфиценты на последнем слое
    coefficients = список из весов нейронов последнего слоя
    answer = список видов [индекс максимального веса]
    '''
    answer = None
    coefficients = None
    return (answer, coefficients)


def verified_appeal_to_ai(image, answer_dictionary, *args, **kwargs):
    '''
    Функция обращается к ИИ модели и возвращает её ответ с весами на последних нейронах.
    Для повышения точности:
    - Фото поворачивается 12 раз на 30 градусов.
    - Суммируются веса нейронов для каждой из этих попыток.
    '''
    answer_list = []
    coefficients_list = None  # Инициализация пустого списка для коэффициентов.
    for _ in range(12):
        # Обращение к ИИ
        answer, coefficients = appeal_to_ai(image, answer_dictionary, *args, **kwargs)
        answer_list.append(answer)
        # Суммируем коэффициенты
        if coefficients_list is None:
            coefficients_list = coefficients[:]  # Копируем первый список.
        else:
            coefficients_list = [x + y for x, y in zip(coefficients_list, coefficients)]
        # Поворот изображения на 30 градусов
        image = image.rotate(30, resample=Image.BICUBIC, expand=True)
    return answer_list, coefficients_list


def verified_appeal_to_ai(image, answer_dictionary, *args, **kwargs):
    """
    Обращается к ИИ модели, возвращает наиболее вероятный ответ.
    Повышает точность за счёт многократного поворота изображения 
    и анализа наилучшего совпадения.
    
    Если ответ не определён точно, повторы с меньшими углами. 
    При неудаче возвращает None и вызывает дополнительную функцию.
    
    :param image: Изображение для анализа.
    :param answer_dictionary: Словарь ответов {индекс нейрона: значение}.
    :param args, kwargs: Дополнительные параметры для appeal_to_ai.
    :return: True, exact_answer или False, [exact_answer, most_common_answer].
    """
    def get_most_common_answer(answers):
        """
        Возвращает наиболее частый элемент и его частоту.
        """
        if not answers:
            return None, 0
        counter = Counter(answers)
        most_common_answer, frequency = counter.most_common(1)[0]
        return most_common_answer, frequency


    def process_image(image, angle_step, rotations):
        """
        Обрабатывает изображение заданное число раз с указанным шагом угла.
        Суммирует веса нейронов и собирает ответы.
        """
        answer_list = []
        coefficients_list = None
        for _ in range(rotations):
            # Обращение к ИИ модели
            answer, coefficients = appeal_to_ai(image, answer_dictionary, *args, **kwargs)
            answer_list.append(answer)
            # Суммируем веса нейронов
            if coefficients_list is None:
                coefficients_list = coefficients[:]
            else:
                coefficients_list = [x + y for x, y in zip(coefficients_list, coefficients)]
            # Поворот изображения
            image = image.rotate(angle_step, resample=Image.BICUBIC, expand=True)
        return answer_list, coefficients_list

    # Первичный проход с шагом 30 градусов
    answer_list, coefficients_list = process_image(image, angle_step=30, rotations=12)
    # Находим индекс самого большого нейрона
    max_neuron_index = coefficients_list.index(max(coefficients_list))
    exact_answer = answer_dictionary[max_neuron_index]
    # Проверяем, совпадает ли с самым частым ответом
    most_common_answer, frequency = get_most_common_answer(answer_list)
    if exact_answer == most_common_answer:
        return exact_answer
    # Если не совпало, повторяем с шагом 15 градусов
    answer_list, coefficients_list = process_image(image, angle_step=15, rotations=24)
    # Повторная проверка
    max_neuron_index = coefficients_list.index(max(coefficients_list))
    exact_answer = answer_dictionary[max_neuron_index]
    most_common_answer, frequency = get_most_common_answer(answer_list)
    if exact_answer == most_common_answer:
        return True, exact_answer
    else:
        return False, [exact_answer, most_common_answer]
