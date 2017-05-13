from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, concatenate, add

def addFireModule(x, squeeze_filters, expand_filters):
    squeeze = Conv2D(16, (3,3), padding='same', activation='elu')(x)
    expand1 = Conv2D(expand_filters, (1,1), padding='same', activation='elu')(squeeze)
    expand3 = Conv2D(expand_filters, (3,3), padding='same', activation='elu')(squeeze)
    c = concatenate([expand1, expand3])

    return c

def addParallelDilatedConvolution(x, num_filters):
    conv1 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=1)(x)
    conv2 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=2)(x)
    conv4 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=4)(x)
    conv8 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=8)(x)
    a = add([conv1, conv2, conv4, conv8])

    return a

def addBypassRefinementModule(high, low, num_filters):
    preConv = Conv2D(num_filters, (3,3), padding='same', activation='elu')(low)
    c = concatenate([preConv, high])
    postConv = Conv2D(num_filters, (3,3), padding='same', activation='elu')(c)

    return postConv

def getModel(input_shape, num_classes, num_filters):
    i = Input(input_shape)
    convI = Conv2D(96, (3,3), padding='same', activation='elu')(i)

    pool1 = MaxPooling2D((2,2))(convI)
    fire1_1 = addFireModule(pool1, 16, 64)
    fire1_2 = addFireModule(fire1_1, 16, 64)

    pool2 = MaxPooling2D((2,2))(fire1_2)
    fire2_1 = addFireModule(pool2, 32, 128)
    fire2_2 = addFireModule(fire2_1, 32, 128)

    pool3 = MaxPooling2D((2,2))(fire2_2)
    fire3_1 = addFireModule(pool3, 48, 192)
    fire3_2 = addFireModule(fire3_1, 48, 192)
    fire3_3 = addFireModule(fire3_2, 64, 256)
    fire3_4 = addFireModule(fire3_3, 64, 256)

    pdc = addParallelDilatedConvolution(fire3_4, num_classes)

    trans_conv1 = Conv2DTranspose(num_classes, (3,3), padding='same', activation='elu', strides=2)(pdc)
    ref1 = addBypassRefinementModule(trans_conv1, fire2_2, num_classes)

    trans_conv2 = Conv2DTranspose(num_classes, (3,3), padding='same', activation='elu', strides=2)(ref1)
    ref2 = addBypassRefinementModule(trans_conv2, fire1_2, num_classes)

    trans_conv3 = Conv2DTranspose(num_classes, (3,3), padding='same', activation='elu', strides=2)(ref2)
    ref3 = addBypassRefinementModule(trans_conv3, convI, num_classes)

    trans_conv4 = Conv2DTranspose(num_classes, (3,3), padding='same', activation='elu')(ref3)

    prediction = Conv2D(num_classes, (1,1), padding='same', activation='softmax')(i)

    model = Model(inputs=i, outputs=prediction)

    return model
