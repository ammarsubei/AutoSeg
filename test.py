import numpy as np
values = np.array([ [1,2,3],
                    [3,2,1],
                    [1,1,1]])
searchval = 1
ii = np.where(values == searchval)[0]
print(len(ii))