import argparse
import os
import tensorflow as tf
from tensorflow import keras as K
from cnnf_tf2 import CNN_F


os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--cnnf_weight', type=str,
                    default='/home/aistudio/data/data20371/vgg_net.mat',
                    help="CNN-F weights file path")
parser.add_argument('--data_path', type=str, default="E:/iTom/dataset/")
parser.add_argument('--log_path', type=str, default="log")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_class', type=int, default=10)
parser.add_argument('--epoch', type=int, default=14)
args = parser.parse_args()
# os.system("clear")


mnist = K.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0   # [n, 28, 28], [n]
print("data shape:", type(x_train), x_train.shape, y_test.shape)

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(args.batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.batch_size)


class Net(K.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.cnnf = CNN_F(args.cnnf_weight)
        self.fc = K.layers.Dense(args.n_class, input_shape=[4096])

    def call(self, x, training=False):
        return self.fc(self.cnnf(x, training=training))


model = Net()
criterion = K.losses.SparseCategoricalCrossentropy(from_logits=True)  # `Sparse` for NOT one-hot
optimizer = K.optimizers.Adam()

train_loss = K.metrics.Mean(name='train_loss')
train_accuracy = K.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = K.metrics.Mean(name='test_loss')
test_accuracy = K.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    images = tf.image.resize(images, [224, 224])
    images = tf.tile(images, tf.constant([1, 1, 1, 3]))
    with tf.GradientTape() as tape:
        pred = model(images, True)
        loss = criterion(labels, pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, pred)


@tf.function
def test_step(images, labels):
    images = tf.image.resize(images, [224, 224])
    images = tf.tile(images, tf.constant([1, 1, 1, 3]))
    pred = model(images)
    t_loss = loss_object(labels, pred)

    test_loss(t_loss)
    test_accuracy(labels, pred)


for epoch in range(args.epoch):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        print("image type:", type(images))
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))
