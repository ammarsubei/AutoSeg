from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Dropout, concatenate, add

def addFireModule(x, squeeze_filters, expand_filters, name='fire'):
    squeeze = Conv2D(squeeze_filters, (1,1), padding='same', activation='elu', name=name + '/squeeze1x1')(x)
    expand1 = Conv2D(expand_filters, (1,1), padding='same', activation='elu', name=name + '/expand1x1')(squeeze)
    expand3 = Conv2D(expand_filters, (3,3), padding='same', activation='elu', name=name + '/expand3x3')(squeeze)
    c = concatenate([expand1, expand3])

    return c

def addParallelDilatedConvolution(x, num_filters, name='parallel_dilated_convolution'):
    conv1 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=1, name=name + '/dil_1')(x)
    conv2 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=2, name=name + '/dil_2')(x)
    conv4 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=4, name=name + '/dil_4')(x)
    conv8 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=8, name=name + '/dil_8')(x)
    a = add([conv1, conv2, conv4, conv8])

    return a

def addBypassRefinementModule(high, low, num_filters, name='bypass'):
    preConv = Dropout(0.5)(Conv2D(num_filters, (3,3), padding='same', activation='elu', name=name + '/pre_conv')(low))
    c = concatenate([preConv, high])
    postConv = Conv2D(num_filters, (3,3), padding='same', activation='elu', name=name + '/post_conv')(c)

    return postConv

def getModel(input_shape, num_classes, num_filters):
    i = Input(input_shape)
    convI = Conv2D(64, (3,3), padding='same', activation='elu', name='conv1')(i)

    pool1 = MaxPooling2D((2,2))(convI)
    fire1_1 = addFireModule(pool1, 16, 64, name='fire2')
    fire1_2 = addFireModule(fire1_1, 16, 64, name='fire3')

    pool2 = MaxPooling2D((2,2))(fire1_2)
    fire2_1 = addFireModule(pool2, 32, 128, name='fire4')
    fire2_2 = addFireModule(fire2_1, 32, 128, name='fire5')

    pool3 = MaxPooling2D((2,2))(fire2_2)
    fire3_1 = addFireModule(pool3, 48, 192, name='fire6')
    fire3_2 = addFireModule(fire3_1, 48, 192, name='fire7')
    fire3_3 = addFireModule(fire3_2, 64, 256, name='fire8')
    fire3_4 = addFireModule(fire3_3, 64, 256, name='fire9')

    pdc = addParallelDilatedConvolution(fire3_4, 512, name='parallel_dilated_convolution')

    auxiliary_prediction = Conv2D(num_classes, (1,1), padding='same', activation='softmax', name='aux')(pdc)

    trans_conv1 = Conv2DTranspose(256, (3,3), padding='same', activation='elu', strides=2, name='trans_conv10')(pdc)
    ref1 = addBypassRefinementModule(trans_conv1, fire2_2, 256, name='bypass11')

    trans_conv2 = Conv2DTranspose(128, (3,3), padding='same', activation='elu', strides=2, name='trans_conv12')(ref1)
    ref2 = addBypassRefinementModule(trans_conv2, fire1_2, 128, name='bypass13')

    trans_conv3 = Conv2DTranspose(64, (3,3), padding='same', activation='elu', strides=2, name='trans_conv14')(ref2)
    ref3 = addBypassRefinementModule(trans_conv3, convI, 64, name='bypass15')

    trans_conv4 = Conv2DTranspose(32, (3,3), padding='same', activation='elu', name='trans_conv16')(ref3)

    prediction = Conv2D(num_classes, (1,1), padding='same', activation='softmax', name='main')(trans_conv4)

    model = Model(inputs=i, outputs=[prediction, auxiliary_prediction])

    return model
