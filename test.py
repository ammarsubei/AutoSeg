import numpy as np

pretty = [[[1, 1, 1], [2, 2, 2]],
          [[2, 2, 2], [2, 2, 2]]]
pretty = np.array(pretty)
print(pretty.shape)
label = np.zeros((2,2,1))
print(label.shape)
print([1, 1, 1] == [1, 1, 0])
