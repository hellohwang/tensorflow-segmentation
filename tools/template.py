import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics

ds_train = tf.data.Dataset.from_tensor_slices((X[0:n * 3 // 4, :], Y[0:n * 3 // 4, :])) \
  .shuffle(buffer_size=1000).batch(20) \
  .prefetch(tf.data.experimental.AUTOTUNE) \
  .cache()

ds_valid = tf.data.Dataset.from_tensor_slices((X[n * 3 // 4:, :], Y[n * 3 // 4:, :])) \
  .shuffle(buffer_size=1000).batch(20) \
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
