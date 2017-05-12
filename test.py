import numpy as np

weights = [1,2]

target = np.array([ [[0.0,1.0],[1.0,0.0]],
                    [[0.0,1.0],[1.0,0.0]]])

output = np.array([ [[0.5,0.5],[0.9,0.1]],
                    [[0.9,0.1],[0.4,0.6]]])

crossentropy_matrix = -np.sum(target * np.log(output), axis=-1)
# Multiply each element of crossentropy_matrix by weights[argmax(target)].
indices = np.argmax(target,2)
weights_matrix = np.take(weights, indices)
weighted_crossentropy_matrix = crossentropy_matrix * weights_matrix
crossentropy = np.sum(weighted_crossentropy_matrix)
print(crossentropy_matrix)
print(weights_matrix)
print(weighted_crossentropy_matrix)
print(crossentropy)