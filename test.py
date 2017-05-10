from keras import backend as K
from keras.utils import to_categorical
import numpy as np

test = [[1,2,3],
		[3,2,1],
		[1,1,1]]
test = np.array(test)

K_img = K.one_hot(test,4)

print(K.eval(K_img))

recover = np.zeros(test.shape)
recover 

print(r)