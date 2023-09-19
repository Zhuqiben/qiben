
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import time
import keras
# 加载训练好的多类别分类器模型
start_time = time.time()
model = tf.keras.models.load_model('model/Monster_land_detect_model.h5')


# 图像预处理
def preprocess_image(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (240, 320), interpolation=cv2.INTER_AREA)
    # img = load_img(image_path, target_size=(320, 240))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


# 图像分类
def classify_image(image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    print(f"prediction[0] = {prediction[0]}")
    if any(i >= 0.95 for i in prediction[0]):
        class_index = np.argmax(prediction[0])  # 获取最大概率对应的类别索引，
        class_labels = ['corner_land', 'end_land', 'grass_land',
                        'mine_land', 'pit_land', 'red_brick_land']  # 6个类别，根据实际情况修改
        print(class_index)
        predicted_class = class_labels[class_index]
        return predicted_class
    else:
        return 'not detect'


# 要分类的新图像路径
if __name__ == "__main__":
    new_image_path = 'received_photos/received_photo.jpg'
    result = classify_image(new_image_path)
    print(result)

