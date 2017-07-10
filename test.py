import numpy as np

weights = [1,2]

target = np.array([ [[0.0,1.0],[1.0,0.0]],
                    [[0.0,1.0],[1.0,0.0]]])

output = np.array([ [[0.5,0.5],[0.9,0.1]],
                    [[0.9,0.1],[0.4,0.6]]])

pixelwise_accuracy = np.mean(np.equal(np.argmax(target, axis=2), np.argmax(output, axis=2)).astype(float))

print(pixelwise_accuracy)
