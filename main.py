import numpy as np

vec1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
vec2 = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1])
vec3 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
vec4 = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1])

test = np.array([1, 1, 1, 0])

def trashhold_func(bit):
    if bit > 1:
        return 1
    else: return 0

def hopefieldAsync(vector):
    I = np.eye(len(vector))
    transY = (2 * vector - 1).reshape(-1, 1) 
    Y = (2 * vector - 1).reshape(1, -1) 
    W = (transY @ Y) - I




def hopefieldSync(vector):
    I = np.eye(len(vector))
    transY = (2 * vector - 1).reshape(-1, 1) 
    Y = (2 * vector - 1).reshape(1, -1) 
    W = (transY @ Y) - I


hopefieldAsync(test)