import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt  

IMG_SIZE = 24  
BATCH_SIZE = 32
EPOCHS = 20

def create_dataset(open_eyes_path, closed_eyes_path):
    data = []
    labels = []
    
    # Xử lý ảnh mắt mở (nhãn 1)
    for img_name in os.listdir(open_eyes_path):
        img_path = os.path.join(open_eyes_path, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  
            data.append(img)
            labels.append(1)  
        except:
            print(f"Lỗi khi xử lý {img_path}")
    
    # Xử lý ảnh mắt đóng (nhãn 0)
    for img_name in os.listdir(closed_eyes_path):
        img_path = os.path.join(closed_eyes_path, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) 
            data.append(img)
            labels.append(0)  
        except:
            print(f"Lỗi khi xử lý {img_path}")
    
    return np.array(data), np.array(labels)

def build_model():
    
    model = Sequential([

        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dropout(0.5),  
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def plot_training_history(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def train_eye_model(open_eyes_path, closed_eyes_path):

    X, y = create_dataset(open_eyes_path, closed_eyes_path)
    X = X / 255.0
    X = X.reshape(X.shape[0], IMG_SIZE, IMG_SIZE, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    model = build_model()
    
    checkpoint = ModelCheckpoint('best_eye_model.h5', monitor='val_accuracy', 
                               save_best_only=True, mode='max', verbose=1)
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint]
    )
    
    score = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {score[1]*100:.2f}%")
    
    plot_training_history(history)
    
    return model, history

open_eyes_dir = "dataset/Open_Eyes"
closed_eyes_dir = "dataset/Close_Eyes"

model, history = train_eye_model(open_eyes_dir, closed_eyes_dir)

model.save("eye_state_model.h5")
print("Mô hình đã được lưu thành công!")