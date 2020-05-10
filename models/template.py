# 先自定义一个残差模块，为自定义Layer

class ResBlock(layers.Layer):
  def __init__(self, kernel_size, **kwargs):
    super(ResBlock, self).__init__(**kwargs)
    self.kernel_size = kernel_size

  def build(self, input_shape):
    self.conv1 = layers.Conv1D(filters=64, kernel_size=self.kernel_size,
                               activation="relu", padding="same")
    self.conv2 = layers.Conv1D(filters=32, kernel_size=self.kernel_size,
                               activation="relu", padding="same")
    self.conv3 = layers.Conv1D(filters=input_shape[-1],
                               kernel_size=self.kernel_size, activation="relu", padding="same")
    self.maxpool = layers.MaxPool1D(2)
    super(ResBlock, self).build(input_shape)  # 相当于设置self.built = True

  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.conv2(x)
    x = self.conv3(x)
    x = layers.Add()([inputs, x])
    x = self.maxpool(x)
    return x

  # 如果要让自定义的Layer通过Functional API 组合成模型时可以序列化，需要自定义get_config方法。
  def get_config(self):
    config = super(ResBlock, self).get_config()
    config.update({'kernel_size': self.kernel_size})
    return config

# 测试ResBlock
resblock = ResBlock(kernel_size=3)
resblock.build(input_shape=(None, 200, 7))
resblock.compute_output_shape(input_shape=(None, 200, 7))


# 自定义模型，实际上也可以使用Sequential或者FunctionalAPI

class ImdbModel(models.Model):
  def __init__(self):
    super(ImdbModel, self).__init__()

  def build(self, input_shape):
    self.embedding = layers.Embedding(MAX_WORDS, 7)
    self.block1 = ResBlock(7)
    self.block2 = ResBlock(5)
    self.dense = layers.Dense(1, activation="sigmoid")
    super(ImdbModel, self).build(input_shape)

  def call(self, x):
    x = self.embedding(x)
    x = self.block1(x)
    x = self.block2(x)
    x = layers.Flatten()(x)
    x = self.dense(x)
    return (x)

tf.keras.backend.clear_session()

model = ImdbModel()
model.build(input_shape =(None,200))
model.summary()

model.compile(optimizer='Nadam',
            loss='binary_crossentropy',
            metrics=['accuracy',"AUC"])
