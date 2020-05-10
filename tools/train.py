import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "../datasets/data/clothes_dataset/train"
val_dir = ""
BATCH_SIZE = 8

NUM_GPUS = 2
BS_PER_GPU = 128
NUM_EPOCHS = 60

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 30), (0.01, 45)]

# def augmentation(x, y):
#   x = tf.image.resize_with_crop_or_pad(
#     x, HEIGHT + 8, WIDTH + 8)
#   x = tf.image.random_crop(x, [HEIGHT, WIDTH, 3])
#   x = tf.image.random_flip_left_right(x)
#   return x, y

def load_image(img_path, size=(32, 32)):
  label = tf.constant(1, tf.int8) if tf.strings.regex_full_match(img_path, ".*/masks/.*") \
    else tf.constant(0, tf.int8)
  img = tf.io.read_file(img_path)
  img = tf.image.decode_jpeg(img)  # 注意此处为jpeg格式
  img = tf.image.resize(img, size) / 255.0
  return (img, label)


# 使用并行化预处理num_parallel_calls 和预存数据prefetch来提升性能
ds_train = tf.data.Dataset.list_files(train_dir) \
  .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
  .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
  .prefetch(tf.data.experimental.AUTOTUNE)

def schedule(epoch):
  initial_learning_rate = BASE_LEARNING_RATE * BS_PER_GPU / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate

