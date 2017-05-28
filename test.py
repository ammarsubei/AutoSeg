import numpy as np

# Reduces the 34 CityScapes classes to 9.
def remapClass(arr):
    arr[arr<=6] = 0
    arr[arr==7] = 1

    arr[arr==8] = 2
    arr[arr==9] = 2
    arr[arr==10] = 2

    arr[arr==11] = 3
    arr[arr==12] = 3
    arr[arr==13] = 3
    arr[arr==14] = 3
    arr[arr==15] = 3
    arr[arr==16] = 3

    arr[arr==17] = 4
    arr[arr==18] = 4
    arr[arr==19] = 4
    arr[arr==20] = 4

    arr[arr==21] = 5
    arr[arr==22] = 6
    arr[arr==23] = 7

    arr[arr==24] = 8
    arr[arr==25] = 8
    
    arr[arr>=26] = 9

arr = np.array([[1,2],[23,24]])
print(arr)
remapClass(arr)
print(arr)