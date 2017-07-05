from keras.models import Model
from keras.layers import *
from keras.regularizers import l2

def addFireModule(x, squeeze_filters, expand_filters, weight_decay, name='fire', batch_norm=True):
    squeeze = Conv2D(squeeze_filters, (1,1), padding='same', activation='elu', name=name + '/squeeze1x1', kernel_regularizer=l2(weight_decay))(x)
    expand1 = Conv2D(expand_filters, (1,1), padding='same', activation='elu', name=name + '/expand1x1', kernel_regularizer=l2(weight_decay))(squeeze)
    expand3 = Conv2D(expand_filters, (3,3), padding='same', activation='elu', name=name + '/expand3x3', kernel_regularizer=l2(weight_decay))(squeeze)
    c = concatenate([expand1, expand3])

    if batch_norm:
        c = BatchNormalization()(c)
    return c

def addParallelDilatedConvolution(x, num_filters, weight_decay, name='parallel_dilated_convolution', batch_norm=True):
    conv1 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=1, name=name + '/dil_1', kernel_regularizer=l2(weight_decay))(x)
    conv2 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=2, name=name + '/dil_2', kernel_regularizer=l2(weight_decay))(x)
    conv4 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=4, name=name + '/dil_4', kernel_regularizer=l2(weight_decay))(x)
    conv8 = Conv2D(num_filters, (3,3), padding='same', activation='elu', dilation_rate=8, name=name + '/dil_8', kernel_regularizer=l2(weight_decay))(x)
    a = add([conv1, conv2, conv4, conv8])

    if batch_norm:
        a = BatchNormalization()(a)
    return a

def addBypassRefinementModule(high, low, num_filters, weight_decay, name='bypass', dropout_rate=0.2, batch_norm=True):
    preConv = Conv2D(num_filters, (3,3), padding='same', activation='elu', name=name + '/pre_conv', kernel_regularizer=l2(weight_decay))(low)
    c = concatenate([preConv, high])
    postConv = Conv2D(num_filters, (3,3), padding='same', activation='elu', name=name + '/post_conv', kernel_regularizer=l2(weight_decay))(c)

    b = Dropout(dropout_rate)(postConv)

    if batch_norm:
        b = BatchNormalization()(b)
    return b

def get_SQ(input_shape, num_classes, dropout_rate=0.2, weight_decay=0.0002, batch_norm=True):
    i = Input(input_shape)
    convI = Conv2D(64, (3,3), padding='same', activation='elu', name='conv1', kernel_regularizer=l2(weight_decay))(i)

    pool1 = MaxPooling2D(2)(convI)
    fire1_1 = addFireModule(pool1, 16, 64, weight_decay, name='fire2', batch_norm=batch_norm)
    fire1_2 = add([fire1_1, addFireModule(fire1_1, 16, 64, weight_decay, name='fire3', batch_norm=batch_norm)])

    pool2 = MaxPooling2D(2)(fire1_2)
    fire2_1 = addFireModule(pool2, 32, 128, weight_decay, name='fire4', batch_norm=batch_norm)
    fire2_2 = add([fire2_1, addFireModule(fire2_1, 32, 128, weight_decay, name='fire5', batch_norm=batch_norm)])

    pool3 = MaxPooling2D(2)(fire2_2)
    fire3_1 = addFireModule(pool3, 48, 192, weight_decay, name='fire6', batch_norm=batch_norm)
    fire3_2 = add([fire3_1, addFireModule(fire3_1, 48, 192, weight_decay, name='fire7', batch_norm = batch_norm)])
    fire3_3 = addFireModule(fire3_2, 64, 256, weight_decay, name='fire8', batch_norm=batch_norm)
    fire3_4 = add([fire3_3, addFireModule(fire3_3, 64, 256, weight_decay, name='fire9', batch_norm=batch_norm)])

    pool4 = Dropout(dropout_rate)(fire3_4)

    pdc = addParallelDilatedConvolution(pool4, 512, weight_decay, name='parallel_dilated_convolution', batch_norm=batch_norm)

    ref10 = addBypassRefinementModule(pdc, pool3, 256, weight_decay, name='bypass10', dropout_rate=dropout_rate, batch_norm=batch_norm)
    trans_conv11 = Conv2DTranspose(256, (3,3), padding='same', activation='elu', strides=2, name='trans_conv11', kernel_regularizer=l2(weight_decay))(Dropout(dropout_rate)(ref10))

    ref12 = addBypassRefinementModule(trans_conv11, pool2, 128, weight_decay, name='bypass12', dropout_rate=dropout_rate, batch_norm=batch_norm)
    trans_conv13 = Conv2DTranspose(128, (3,3), padding='same', activation='elu', strides=2, name='trans_conv13', kernel_regularizer=l2(weight_decay))(Dropout(dropout_rate)(ref12))

    ref14 = addBypassRefinementModule(trans_conv13, pool1, 64, weight_decay, name='bypass14', dropout_rate=dropout_rate, batch_norm=batch_norm)
    trans_conv15 = Conv2DTranspose(64, (3,3), padding='same', activation='elu', strides=2, name='trans_conv15', kernel_regularizer=l2(weight_decay))(Dropout(dropout_rate)(ref14))

    if batch_norm:
        trans_conv15 = BatchNormalization()(trans_conv15)

    prediction = Conv2D(num_classes, (1,1), padding='same', activation='softmax', name='main', kernel_regularizer=l2(weight_decay))(trans_conv15)

    model = Model(inputs=i, outputs=prediction)

    return model

def standard_residual_unit(input, num_channels, name='res_unit', double_second=False, dilation_rate=1):
    conv1 = Conv2D(num_channels, (3,3), padding='same', activation='elu', name=name+'/conv1', dilation_rate=dilation_rate)(input)
    batch_norm1 = BatchNormalization(name=name+'/batch_norm1')(conv1)
    if double_second:
        num_channels *= 2
    conv2 = Conv2D(num_channels, (3,3), padding='same', activation='elu', name=name+'/conv2', dilation_rate=dilation_rate)(batch_norm1)
    batch_norm2 = BatchNormalization(name=name+'/batch_norm2')(conv2)

    a = concatenate([input, batch_norm2])
    return a

def bottleneck_residual_unit(input, num_channels, name='res_unit', dilation_rate=1):
    conv1 = Conv2D(num_channels, (1,1), padding='same', activation='elu', name=name+'/conv1', dilation_rate=dilation_rate)(input)
    batch_norm1 = BatchNormalization(name=name+'/batch_norm1')(conv1)
    conv2 = Conv2D(num_channels*2, (3,3), padding='same', activation='elu', name=name+'/conv2', dilation_rate=dilation_rate)(batch_norm1)
    batch_norm2 = BatchNormalization(name=name+'/batch_norm2')(conv2)
    conv3 = Conv2D(num_channels*4, (1,1), padding='same', activation='elu', name=name+'/conv3', dilation_rate=dilation_rate)(batch_norm2)
    batch_norm3 = BatchNormalization(name=name+'/batch_norm3')(conv3)

    a = concatenate([input, batch_norm3])
    return a

def get_rn38(input_shape, num_classes, dropout_rate=0.4):
    input = Input(input_shape)
    x = Conv2D(64, (3,3), padding='same', activation='elu', name='conv_i')(input)
    #x = standard_residual_unit(x, 64, 'B1')
    x = MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = standard_residual_unit(x, 128, 'B2')
    x = MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = standard_residual_unit(x, 256, 'B3')
    #x = MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = standard_residual_unit(x, 512, 'B4', dilation_rate=2)
    #x = MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = standard_residual_unit(x, 512, 'B5', double_second=True, dilation_rate=4)
    #x = MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = bottleneck_residual_unit(x, 512, 'B6', dilation_rate=8)
    x = bottleneck_residual_unit(x, 1024, 'B7', dilation_rate=8)

    x = Conv2DTranspose(512, (3,3), padding='same', activation='elu', strides=2, name='trans_conv1')(x)
    x = Conv2DTranspose(num_classes, (3,3), padding='same', activation='elu', strides=2, name='trans_conv2')(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input, outputs=x)

    return model

def get_rn38_classifier(input_shape, num_classes, dropout_rate=0.4):
    input = Input(input_shape)
    x = Conv2D(64, (3,3), padding='same', activation='elu', name='conv_i')(input)
    #x = standard_residual_unit(x, 64, 'B1')
    x = MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = standard_residual_unit(x, 128, 'B2')
    x = MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = standard_residual_unit(x, 256, 'B3')
    x = MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = standard_residual_unit(x, 512, 'B4')#, dilation_rate=2)
    x = MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = standard_residual_unit(x, 512, 'B5', double_second=True)#, dilation_rate=4)
    x = MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = bottleneck_residual_unit(x, 512, 'B6')#, dilation_rate=8)
    x = bottleneck_residual_unit(x, 1024, 'B7')#, dilation_rate=8)

    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input, outputs=x)

    return model
