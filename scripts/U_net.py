from tensorflow.python.keras import layers
from tensorflow.python.keras import models


def UNet(image_size,kernel_size=(3,3)):
    def bn_act(x, act=True):
        x = layers.BatchNormalization()(x)
        if act == True:
            x = layers.Activation("relu")(x)
        return x

    def conv_block(x, filters, kernel_size=kernel_size, padding="same", strides=1):
        conv = bn_act(x)
        conv = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv

    def stem(x, filters, kernel_size=kernel_size, padding="same", strides=1):
        conv = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

        return conv

    def block(x, filters, kernel_size=kernel_size, padding="same", strides=1):
        res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

        return res

    def upsample_block(x):
        u = layers.UpSampling2D((2, 2))(x)
        return u

    f = [16, 32, 64, 128, 256]
    inputs = layers.Input((image_size, image_size, 3))

    ## Encoder
    e0 = inputs
    print(e0.shape)  # (?, 400, 400, 3)
    e1 = stem(e0, f[0])
    print(e1.shape)  # (?, 400, 400, 16)
    e2 = block(e1, f[1], strides=2)
    print(e2.shape)  # (?, 200, 200, 32)
    e3 = block(e2, f[2], strides=2)
    print(e3.shape)  # (?, 100, 100, 64)
    e4 = block(e3, f[3], strides=2)
    print(e4.shape)  # (?, 50, 50, 128)
    e5 = block(e4, f[4], strides=2)
    print(e5.shape)  # (?, 25, 25, 256)
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    print(b0.shape)  # (?, 25, 25, 256)
    b1 = conv_block(b0, f[4], strides=1)
    print(b1.shape)  # (?, 25, 25, 256)

    ## Decoder
    u1 = upsample_block(b1)
    print(u1.shape)  # (?, 50, 50, 256)
    d1 = block(u1, f[4])
    print(d1.shape)  # (?, 50, 50, 256)

    u2 = upsample_block(d1)
    print(u2.shape)  # (?, 100, 100, 256)
    d2 = block(u2, f[3])
    print(d2.shape)  # (?, 100, 100, 128)

    u3 = upsample_block(d2)
    print(u3.shape)  # (?, 200, 200, 160)
    d3 = block(u3, f[2])
    print(d3.shape)  # (?, 200, 200, 64)

    u4 = upsample_block(d3)
    print(u4.shape)  # (?, 400, 400, 80)
    d4 = block(u4, f[1])
    print(d4.shape)  # (?, 400, 400, 32)

    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    print(outputs.shape)  # (?, 400, 400, 1)
    model = models.Model(inputs, outputs)

    return model

def ResUNet(image_size, kernel_size=(3, 3), dilation_rate=(1,1)):

    def bn_act(x, act=True):
        x = layers.BatchNormalization()(x)
        if act == True:
            x = layers.Activation("relu")(x)
            #x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    def conv_block(x, filters, kernel_size=kernel_size, padding="same", strides=1, dilation_rate = (1,1)):
        conv = bn_act(x)
        conv = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, dilation_rate=dilation_rate)(conv)
        return conv

    def stem(x, filters, kernel_size=kernel_size, padding="same", strides=1):
        conv = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate= dilation_rate)

        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides, dilation_rate= dilation_rate)(x)
        shortcut = bn_act(shortcut, act=False)

        output = layers.Add()([conv, shortcut])
        return output

    def residual_block(x, filters, kernel_size=kernel_size, padding="same", strides=1):# image dimensions are divided by the strides number
        res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = bn_act(shortcut, act=False)

        output = layers.Add()([shortcut, res])
        return output

    def upsample_concat_block(x, xskip): # concatenates a decoder output with and encoder output of same size
        u = layers.UpSampling2D((2, 2))(x)# image dimensions are multiplied by 2
        c = layers.Concatenate()([u, xskip])
        return c

    f = [16, 32, 64, 128, 256]
    inputs = layers.Input((image_size, image_size, 3))

    ## Encoder
    e0 = inputs
    print(e0.shape) #(?, 400, 400, 3)
    e1 = stem(e0, f[0])
    print(e1.shape) #(?, 400, 400, 16)
    e2 = residual_block(e1, f[1], strides=2)
    print(e2.shape)#(?, 200, 200, 32)
    e3 = residual_block(e2, f[2], strides=2)
    print(e3.shape)#(?, 100, 100, 64)
    e4 = residual_block(e3, f[3], strides=2)
    print(e4.shape)#(?, 50, 50, 128)
    e5 = residual_block(e4, f[4], strides=2)
    print(e5.shape)#(?, 25, 25, 256)
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    print(b0.shape)#(?, 25, 25, 256)
    b1 = conv_block(b0, f[4], strides=1)
    print(b1.shape)#(?, 25, 25, 256)

    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    print(u1.shape)#(?, 50, 50, 384) 256+128=384 channels
    d1 = residual_block(u1, f[4])
    print(d1.shape)#(?, 50, 50, 256)

    u2 = upsample_concat_block(d1, e3)
    print(u2.shape)#(?, 100, 100, 320) 256+64=320 channels
    d2 = residual_block(u2, f[3])
    print(d2.shape)#(?, 100, 100, 128)

    u3 = upsample_concat_block(d2, e2)
    print(u3.shape)#(?, 200, 200, 160) 128+32=160 channels
    d3 = residual_block(u3, f[2])
    print(d3.shape)#(?, 200, 200, 64)

    u4 = upsample_concat_block(d3, e1)
    print(u4.shape)#(?, 400, 400, 80)
    d4 = residual_block(u4, f[1])
    print(d4.shape)#(?, 400, 400, 32)

    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    print(outputs.shape)#(?, 400, 400, 1)
    model = models.Model(inputs, outputs)

    return model

def ResUNet_dilation(image_size, kernel_size=(3, 3), dilation_rate=(1,1)):

    def bn_act(x, act=True):
        x = layers.BatchNormalization()(x)
        if act == True:
            x = layers.LeakyReLU(alpha=0.3)(x)
        return x

    def conv_block(x, filters, kernel_size=kernel_size, padding="same", strides=1, dilation_rate = (1,1)):
        conv = bn_act(x)
        conv = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, dilation_rate=dilation_rate)(conv)
        return conv

    def stem(x, filters, kernel_size=kernel_size, padding="same", strides=1):
        conv = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate= dilation_rate)

        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides, dilation_rate= dilation_rate)(x)
        shortcut = bn_act(shortcut, act=False)

        output = layers.Add()([conv, shortcut])
        return output

    def residual_block(x, filters, kernel_size=kernel_size, padding="same", strides=1,dilation_rate=dilation_rate):# image dimensions are divided by the strides number
        res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides,dilation_rate=dilation_rate)
        res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1,dilation_rate=dilation_rate)

        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = bn_act(shortcut, act=False)

        output = layers.Add()([shortcut, res])
        return output

    def upsample_concat_block(x, xskip): # concatenates a decoder output with and encoder output of same size
        u = layers.UpSampling2D((2, 2))(x)# image dimensions are multiplied by 2
        c = layers.Concatenate()([u, xskip])
        return c

    f = [16, 32, 64, 128, 256]
    inputs = layers.Input((image_size, image_size, 3))

    ## Encoder
    e0 = inputs
    print(e0.shape) #(?, 400, 400, 3)
    e1 = stem(e0, f[0])
    print(e1.shape) #(?, 400, 400, 16)
    e1p=layers.MaxPooling2D(pool_size=(2,2))(e1)
    e2 = residual_block(e1p, f[1], strides=1,dilation_rate=dilation_rate)
    print(e2.shape)#(?, 200, 200, 32)
    e2p = layers.MaxPooling2D(pool_size=(2, 2))(e2)
    e3 = residual_block(e2p, f[2], strides=1,dilation_rate=dilation_rate)
    print(e3.shape)#(?, 100, 100, 64)
    e3p = layers.MaxPooling2D(pool_size=(2, 2))(e3)
    e4 = residual_block(e3p, f[3], strides=1,dilation_rate=dilation_rate)
    print(e4.shape)#(?, 50, 50, 128)
    e4p = layers.MaxPooling2D(pool_size=(2, 2))(e4)
    e5 = residual_block(e4p, f[4], strides=1,dilation_rate=dilation_rate)
    print(e5.shape)#(?, 25, 25, 256)
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    print(b0.shape)#(?, 25, 25, 256)
    b1 = conv_block(b0, f[4], strides=1)
    print(b1.shape)#(?, 25, 25, 256)

    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    print(u1.shape)#(?, 50, 50, 384) 256+128=384 channels
    d1 = residual_block(u1, f[4],dilation_rate=dilation_rate)
    print(d1.shape)#(?, 50, 50, 256)

    u2 = upsample_concat_block(d1, e3)
    print(u2.shape)#(?, 100, 100, 320) 256+64=320 channels
    d2 = residual_block(u2, f[3],dilation_rate=dilation_rate)
    print(d2.shape)#(?, 100, 100, 128)

    u3 = upsample_concat_block(d2, e2)
    print(u3.shape)#(?, 200, 200, 160) 128+32=160 channels
    d3 = residual_block(u3, f[2],dilation_rate=dilation_rate)
    print(d3.shape)#(?, 200, 200, 64)

    u4 = upsample_concat_block(d3, e1)
    print(u4.shape)#(?, 400, 400, 80)
    d4 = residual_block(u4, f[1],dilation_rate=dilation_rate)
    print(d4.shape)#(?, 400, 400, 32)

    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    print(outputs.shape)#(?, 400, 400, 1)
    model = models.Model(inputs, outputs)

    return model

def ResUNet_extralayer(image_size):
    def bn_act(x, act=True):
        x = layers.BatchNormalization()(x)
        if act == True:
            x = layers.Activation("relu")(x)
        return x

    def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = bn_act(x)
        conv = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv

    def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        conv = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = bn_act(shortcut, act=False)

        output = layers.Add()([conv, shortcut])
        return output

    def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):# image dimensions are divided by the strides number
        res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        shortcut = bn_act(shortcut, act=False)

        output = layers.Add()([shortcut, res])
        return output

    def upsample_concat_block(x, xskip): # concatenates a decoder output with and encoder output of same size
        u = layers.UpSampling2D((2, 2))(x)# image dimensions are multiplied by 2
        c = layers.Concatenate()([u, xskip])
        return c

    def upsample5_concat_block(x, xskip): # concatenates a decoder output with and encoder output of same size
        u = layers.UpSampling2D((5, 5))(x)# image dimensions are multiplied by 2
        c = layers.Concatenate()([u, xskip])
        return c

    f = [16, 32, 64, 128, 256, 1024]
    inputs = layers.Input((image_size, image_size, 3))

    ## Encoder
    e0 = inputs
    print(e0.shape) #(?, 400, 400, 3)
    e1 = stem(e0, f[0])
    print(e1.shape) #(?, 400, 400, 16)
    e2 = residual_block(e1, f[1], strides=2)
    print(e2.shape)#(?, 200, 200, 32)
    e3 = residual_block(e2, f[2], strides=2)
    print(e3.shape)#(?, 100, 100, 64)
    e4 = residual_block(e3, f[3], strides=2)
    print(e4.shape)#(?, 50, 50, 128)
    e5 = residual_block(e4, f[4], strides=2)
    print(e5.shape)#(?, 25, 25, 256)
    e6 = residual_block(e5, f[5], strides=5)
    print(e6.shape)  # (?, 5, 5, 1024)
    ## Bridge
    b0 = conv_block(e6, f[5], strides=1)
    print(b0.shape)#(?, 5, 5, 1024)
    b1 = conv_block(b0, f[5], strides=1)
    print(b1.shape)#(?, 5, 5, 1024)

    ## Decoder
    u0 = upsample5_concat_block(b1, e5)
    print(u0.shape)  # (?, 25, 25, 1280) 1024+256 channels
    d0 = residual_block(u0, f[5])
    print(d0.shape)  # (?, 25, 25, 1024)

    u1 = upsample_concat_block(d0, e4)
    print(u1.shape)#(?, 50, 50, 1152) 1024+128 channels
    d1 = residual_block(u1, f[4])
    print(d1.shape)#(?, 50, 50, 256)

    u2 = upsample_concat_block(d1, e3)
    print(u2.shape)#(?, 100, 100, 320) 256+64=320 channels
    d2 = residual_block(u2, f[3])
    print(d2.shape)#(?, 100, 100, 128)

    u3 = upsample_concat_block(d2, e2)
    print(u3.shape)#(?, 200, 200, 160) 128+32=160 channels
    d3 = residual_block(u3, f[2])
    print(d3.shape)#(?, 200, 200, 64)

    u4 = upsample_concat_block(d3, e1)
    print(u4.shape)#(?, 400, 400, 80)
    d4 = residual_block(u4, f[1])
    print(d4.shape)#(?, 400, 400, 32)

    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    print(outputs.shape)#(?, 400, 400, 1)
    model = models.Model(inputs, outputs)

    return model