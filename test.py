from keras import backend as K
from keras.utils import to_categorical
import numpy as np

test = [[1,2,3],
		[3,2,1],
		[1,1,1]]
test = np.array(test)

K_img = K.eval(K.one_hot(test,4))

print(K_img)

recover = K_img.argmax(2)

print(recover)