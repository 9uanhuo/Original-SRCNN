from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Input, LeakyReLU


def srcnn(input_channels=1):
    # input single image
    input_shape = Input((None, None, input_channels))

    # convolution
    conv2d_0 = Conv2D(filters=64, kernel_size=(9, 9), padding="same", activation=LeakyReLU(alpha=0.01))(input_shape)
    conv2d_1 = Conv2D(filters=64, kernel_size=(1, 1), padding="same", activation=LeakyReLU(alpha=0.01))(conv2d_0)
    # conv2d_2 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation=LeakyReLU(alpha=0.01))(conv2d_1)
    # conv2d_3 = Conv2D(filters=32, kernel_size=(1, 1), padding="same", activation=LeakyReLU(alpha=0.01))(conv2d_2)
    conv2d_4 = Conv2D(filters=input_channels, kernel_size=(5, 5), padding="same")(conv2d_1)

    model = Model(inputs=input_shape, outputs=[conv2d_4])
    model.summary()

    return model
