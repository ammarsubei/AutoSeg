import numpy as np

weights = [1,2]

target = np.array([ [[0.0,1.0],[1.0,0.0]],
                    [[0.0,1.0],[1.0,0.0]]])

output = np.array([ [[0.5,0.5],[0.9,0.1]],
                    [[0.9,0.1],[0.4,0.6]]])

crossentropy_matrix = -np.sum(target * np.log(output), axis=-1)
crossentropy = target * weights * np.log(output)

print(crossentropy)