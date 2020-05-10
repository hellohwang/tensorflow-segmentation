import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "../datasets/data/clothes_dataset/train"
val_dir = "../datasets/data/clothes_dataset/val"
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

ds_valid = tf.data.Dataset.list_files(train_dir) \
  .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
  .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
  .prefetch(tf.data.experimental.AUTOTUNE)

optimizer = optimizers.Adam()
loss_func = losses.MeanSquaredError()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_metric = tf.keras.metrics.MeanAbsoluteError(name='train_mae')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_metric = tf.keras.metrics.MeanAbsoluteError(name='valid_mae')


@tf.function
def train_step(model, features, labels):
  with tf.GradientTape() as tape:
    predictions = model(features)
    loss = loss_func(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss.update_state(loss)
  train_metric.update_state(labels, predictions)


@tf.function
def valid_step(model, features, labels):
  predictions = model(features)
  batch_loss = loss_func(labels, predictions)
  valid_loss.update_state(batch_loss)
  valid_metric.update_state(labels, predictions)


@tf.function
def train_model(model, ds_train, ds_valid, epochs):
  for epoch in tf.range(1, epochs + 1):
    for features, labels in ds_train:
      train_step(model, features, labels)

    for features, labels in ds_valid:
      valid_step(model, features, labels)

    logs = 'Epoch={},Loss:{},MAE:{},Valid Loss:{},Valid MAE:{}'

    if epoch % 100 == 0:
      # printbar()
      tf.print(tf.strings.format(logs,
                                 (epoch, train_loss.result(), train_metric.result(), valid_loss.result(),
                                  valid_metric.result())))
      tf.print("w=", model.layers[0].kernel)
      tf.print("b=", model.layers[0].bias)
      tf.print("")

    train_loss.reset_states()
    valid_loss.reset_states()
    train_metric.reset_states()
    valid_metric.reset_states()


train_model(model, ds_train, ds_valid, 400)