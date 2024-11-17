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
