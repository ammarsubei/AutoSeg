from keras.models import Model
from keras.layers import *
from keras.regularizers import l2

def addFireModule(x, squeeze_filters, expand_filters, weight_decay, name='fire'):
    squeeze = Conv2D(squeeze_filters, (1,1), padding='same', activation='elu', name=name + '/squeeze1x1', kernel_regularizer=l2(weight_decay))(x)
    expand1 = Conv2D(expand_filters, (1,1), padding='same', activation='elu', name=name + '/expand1x1', kernel_regularizer=l2(weight_decay))(squeeze)
    expand3 = Conv2D(expand_filters, (3,3), padding='same', activation='elu', name=name + '/expand3x3', kernel_regularizer=l2(weight_decay))(squeeze)
    c = concatenate([expand1, expand3])

    return BatchNormalization()(c)

def addParallelDilatedConvolution(x, num_filters, weight_decay, name='parallel_dilated_convolution'):
    conv1 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=1, name=name + '/dil_1', kernel_regularizer=l2(weight_decay))(x)
    conv2 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=2, name=name + '/dil_2', kernel_regularizer=l2(weight_decay))(x)
    conv4 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=4, name=name + '/dil_4', kernel_regularizer=l2(weight_decay))(x)
    conv8 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=8, name=name + '/dil_8', kernel_regularizer=l2(weight_decay))(x)
    a = add([conv1, conv2, conv4, conv8])

    return BatchNormalization()(a)

def addBypassRefinementModule(high, low, num_filters, weight_decay, name='bypass', dropout_rate=0.2):
    preConv = Conv2D(num_filters, (3,3), padding='same', activation='elu', name=name + '/pre_conv', kernel_regularizer=l2(weight_decay))(low)
    c = concatenate([preConv, high])
    postConv = Conv2D(num_filters, (3,3), padding='same', activation='elu', name=name + '/post_conv', kernel_regularizer=l2(weight_decay))(c)

    return Dropout(dropout_rate)(BatchNormalization()(postConv))

def getModel(input_shape, num_classes, residual_encoder_connections=False, dropout_rate=0.2, weight_decay=0.0002):
    i = Input(input_shape)
    convI = Conv2D(64, (3,3), padding='same', activation='elu', name='conv1', kernel_regularizer=l2(weight_decay))(BatchNormalization()(i))

    if residual_encoder_connections:
        pool1 = MaxPooling2D(2)( concatenate( [convI, i] ) )
    else:
        pool1 = MaxPooling2D(2)(convI)
    fire1_1 = addFireModule(pool1, 16, 64, weight_decay, name='fire2')
    fire1_2 = addFireModule(fire1_1, 16, 64, weight_decay, name='fire3')

    if residual_encoder_connections:
        pool2 = MaxPooling2D(2)( concatenate( [fire1_2,pool1] ) )
    else:
        pool2 = MaxPooling2D(2)(fire1_2)
    fire2_1 = addFireModule(pool2, 32, 128, weight_decay, name='fire4')
    fire2_2 = addFireModule(fire2_1, 32, 128, weight_decay, name='fire5')

    if residual_encoder_connections:
        pool3 = MaxPooling2D(2)( concatenate( [fire2_2, pool2] ) )
    else:
        pool3 = MaxPooling2D(2)(fire2_2)
    fire3_1 = addFireModule(pool3, 48, 192, weight_decay, name='fire6')
    fire3_2 = addFireModule(fire3_1, 48, 192, weight_decay, name='fire7')
    fire3_3 = addFireModule(fire3_2, 64, 256, weight_decay, name='fire8')
    fire3_4 = addFireModule(fire3_3, 64, 256, weight_decay, name='fire9')

    if residual_encoder_connections:
        pool4 = Dropout(dropout_rate)(concatenate( [fire3_4, pool3] ))
    else:
        pool4 = Dropout(dropout_rate)(fire3_4)

    pdc = addParallelDilatedConvolution(pool4, 512, weight_decay, name='parallel_dilated_convolution')

    ref10 = addBypassRefinementModule(pdc, pool3, 256, weight_decay, name='bypass10', dropout_rate=dropout_rate)
    trans_conv11 = Conv2DTranspose(256, (3,3), padding='same', activation='elu', strides=2, name='trans_conv11', kernel_regularizer=l2(weight_decay))(Dropout(dropout_rate)(ref10))

    ref12 = addBypassRefinementModule(trans_conv11, pool2, 128, weight_decay, name='bypass12', dropout_rate=dropout_rate)
    trans_conv13 = Conv2DTranspose(128, (3,3), padding='same', activation='elu', strides=2, name='trans_conv13', kernel_regularizer=l2(weight_decay))(Dropout(dropout_rate)(ref12))

    ref14 = addBypassRefinementModule(trans_conv13, pool1, 64, weight_decay, name='bypass14', dropout_rate=dropout_rate)
    trans_conv15 = Conv2DTranspose(64, (3,3), padding='same', activation='elu', strides=2, name='trans_conv15', kernel_regularizer=l2(weight_decay))(Dropout(dropout_rate)(ref14))

    prediction = Conv2D(num_classes, (1,1), padding='same', activation='softmax', name='main', kernel_regularizer=l2(weight_decay))(trans_conv15)

    model = Model(inputs=i, outputs=prediction)

    return model

def getResidualModel(input_shape, num_classes, residual_encoder_connections=False, dropout_rate=0.2, weight_decay=0.0002):
    i = Input(input_shape)
    convI = Conv2D(64, (3,3), padding='same', activation='elu', name='conv1', kernel_regularizer=l2(weight_decay))(BatchNormalization()(i))

    pool1 = MaxPooling2D(2)(convI)
    fire1_1 = addFireModule(pool1, 16, 64, weight_decay, name='fire2')
    fire1_2 = add([fire1_1, addFireModule(fire1_1, 16, 64, weight_decay, name='fire3')])

    pool2 = MaxPooling2D(2)(fire1_2)
    fire2_1 = addFireModule(pool2, 32, 128, weight_decay, name='fire4')
    fire2_2 = add([fire2_1, addFireModule(fire2_1, 32, 128, weight_decay, name='fire5')])

    pool3 = MaxPooling2D(2)(fire2_2)
    fire3_1 = addFireModule(pool3, 48, 192, weight_decay, name='fire6')
    fire3_2 = add([fire3_1, addFireModule(fire3_1, 48, 192, weight_decay, name='fire7')])
    fire3_3 = addFireModule(fire3_2, 64, 256, weight_decay, name='fire8')
    fire3_4 = add([fire3_3, addFireModule(fire3_3, 64, 256, weight_decay, name='fire9')])

    pool4 = Dropout(dropout_rate)(fire3_4)

    pdc = addParallelDilatedConvolution(pool4, 512, weight_decay, name='parallel_dilated_convolution')

    ref10 = addBypassRefinementModule(pdc, pool3, 256, weight_decay, name='bypass10', dropout_rate=dropout_rate)
    trans_conv11 = Conv2DTranspose(256, (3,3), padding='same', activation='elu', strides=2, name='trans_conv11', kernel_regularizer=l2(weight_decay))(Dropout(dropout_rate)(ref10))

    ref12 = addBypassRefinementModule(trans_conv11, pool2, 128, weight_decay, name='bypass12', dropout_rate=dropout_rate)
    trans_conv13 = Conv2DTranspose(128, (3,3), padding='same', activation='elu', strides=2, name='trans_conv13', kernel_regularizer=l2(weight_decay))(Dropout(dropout_rate)(ref12))

    ref14 = addBypassRefinementModule(trans_conv13, pool1, 64, weight_decay, name='bypass14', dropout_rate=dropout_rate)
    trans_conv15 = Conv2DTranspose(64, (3,3), padding='same', activation='elu', strides=2, name='trans_conv15', kernel_regularizer=l2(weight_decay))(Dropout(dropout_rate)(ref14))

    prediction = Conv2D(num_classes, (1,1), padding='same', activation='softmax', name='main', kernel_regularizer=l2(weight_decay))(trans_conv15)

    model = Model(inputs=i, outputs=prediction)

    return model
