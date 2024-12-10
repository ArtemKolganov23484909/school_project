from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Подготовка данных
def load_data():
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,          
        rotation_range=15,         
        width_shift_range=0.1,     
        height_shift_range=0.1,   
        shear_range=0.1,           
        zoom_range=0.1,            
        horizontal_flip=True,      
        validation_split=0.2        
    )

    train_data = datagen.flow_from_directory(
        directory="dataset/train",
        target_size=(512, 512),     
        batch_size=8,               
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        directory="dataset/train",
        target_size=(512, 512),
        batch_size=8,
        class_mode='categorical',
        subset='validation'
    )

    return train_data, val_data

train_data, val_data = load_data()
model = create_model(input_shape=(512, 512, 3), num_classes=train_data.num_classes)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20, 
    steps_per_epoch=len(train_data),
    validation_steps=len(val_data)
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.mixed_precision import set_global_policy
import tensorflow as tf

# Настройка смешанной точности для ускорения и уменьшения потребления памяти
set_global_policy('mixed_float16')

# Оптимизированная архитектура модели
def create_model(input_shape=(512, 512, 3), num_classes=10):
    """
    Углублённая модель с учётом больших ресурсов.
    """
    model = Sequential([
        # Сверточный блок 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Сверточный блок 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Сверточный блок 3
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Сверточный блок 4
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Глобальный пулинг вместо Flatten для уменьшения количества параметров
        GlobalAveragePooling2D(),
        
        # Полносвязные слои
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Количество классов зависит от задачи
    ])

    # Компиляция модели
    model.compile(
        optimizer='adam',  # Оптимизатор
        loss='categorical_crossentropy',  # Функция потерь для многоклассовой классификации
        metrics=['accuracy']  # Основная метрика
    )
    
    return model


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

def train_model(model, train_data, val_data, save_path='model.h5', epochs=20):
    """
    Запуск процесса обучения модели с сохранением лучшей версии.
    :param model: Скомпилированная модель.
    :param train_data: Данные для обучения.
    :param val_data: Данные для проверки.
    :param save_path: Путь для сохранения модели.
    :param epochs: Количество эпох.
    :return: История обучения.
    """
    # Настраиваем коллбеки
    callbacks = [
        ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss', verbose=1),  # Сохранение лучшей модели
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)  # Остановка при переобучении
    ]

    # Обучение
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        steps_per_epoch=len(train_data),
        validation_steps=len(val_data),
        callbacks=callbacks
    )
    
    return history

# Вызов функции
history = train_model(
    model=model, 
    train_data=train_data, 
    val_data=val_data,  
    save_path='ants_classifier.h5',  
    epochs=20  
)
def plot_training_history(history):
    """
    Отрисовка графиков обучения и проверки.
    
    :param history: История обучения модели.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # График точности
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Точность на обучении')
    plt.plot(epochs, val_acc, label='Точность на проверке')
    plt.title('Точность обучения')
    plt.legend()

    # График потерь
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Потери на обучении')
    plt.plot(epochs, val_loss, label='Потери на проверке')
    plt.title('Потери обучения')
    plt.legend()

    plt.show()

# Вызов функции
plot_training_history(history)
def train_model():
    """
    Загружает данные, создаёт модель и обучает её.
    """
    train_data, val_data = load_data()
    model = create_model()
    
    # Устанавливаем раннюю остановку и сохранение модели
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    ]
    
    # Обучение
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        callbacks=callbacks
    )
    
    # Сохранение модели
    model.save('final_model.h5')
    print("Модель сохранена в файлы: final_model.h5 и best_model.h5")

