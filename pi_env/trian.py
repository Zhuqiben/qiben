from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 数据集路径
train_data_dir = 'training-set'
img_width, img_height = 320, 240
batch_size = 32
num_classes = 6  # 6个类别

# 数据增强（可选）
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',  # 多类别分类
    shuffle=True
)

# 加载预训练模型VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# 冻结预训练模型的权重
for layer in base_model.layers:
    layer.trainable = False

# 添加新的全连接层
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)  # 使用 softmax 激活函数

model = Model(inputs=base_model.input, outputs=output)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
epochs = 10
model.fit(train_generator, epochs=epochs)

# 保存模型
model.save('Monster_land_detect_model.h5')
"""
tensorflow:2.0.0
先装2.5.0版本，让keras模块保存下来，在卸掉2.5.0版本，装2.0.0
h5py要降到2.10.0
pip3 install h5py==2.10.0
"""