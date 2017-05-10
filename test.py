from keras import backend as K
from keras.utils import to_categorical
import numpy as np
import time

test = [[1,2,3],
		[3,2,1],
		[1,1,1]]
test = np.array(test)

start = time.clock()
K_img = K.eval(K.one_hot(test,4))
end = time.clock()
print(end-start)

start = time.clock()
n_values = np.max(test) + 1
np_img = np.eye(n_values)[test]
end = time.clock()
print(end-start)

recover = np_img.argmax(2)

print(recover)