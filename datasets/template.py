import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = 'cifar2_datasets/train'
test_dir = 'cifar2_datasets/test'

# 对训练集数据设置数据增强
# one
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest',
                                   )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(32, 32),
                                                    batch_size=32,
                                                    shuffle=True,
                                                    class_mode='binary')

# two
ds_train = tf.data.Dataset.from_tensor_slices((X[0:n * 3 // 4, :], Y[0:n * 3 // 4, :])) \
  .shuffle(buffer_size=1000).batch(20) \
  .prefetch(tf.data.experimental.AUTOTUNE) \



def load_image(img_path, size=(32, 32)):
  label = tf.constant(1, tf.int8) if tf.strings.regex_full_match(img_path, ".*/automobile/.*") \
    else tf.constant(0, tf.int8)
  img = tf.io.read_file(img_path)
  img = tf.image.decode_jpeg(img)  # 注意此处为jpeg格式
  img = tf.image.resize(img, size) / 255.0
  return (img, label)


# 使用并行化预处理num_parallel_calls 和预存数据prefetch来提升性能
ds_train = tf.data.Dataset.list_files("./data/cifar2/train/*/*.jpg") \
  .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
  .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
  .prefetch(tf.data.experimental.AUTOTUNE)

ds_test = tf.data.Dataset.list_files("./data/cifar2/test/*/*.jpg") \
  .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
  .batch(BATCH_SIZE) \
  .prefetch(tf.data.experimental.AUTOTUNE)
