import numpy as np

vec1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
vec2 = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1])
vec3 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
vec4 = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1])

test = np.array([0, 1, 1, 0])
corr = np.array([0, 1, 0, 0])
vec11 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#---------------------------------------------------------------------------------
def trashhold_func(bit):
    return 1 if bit >= 1 else 0
#---------------------------------------------------------------------------------
def hopefieldAsync(vector, corr):
    corrupted = corr.copy()
    I = np.eye(len(vector))
    transY = (2 * vector - 1).reshape(-1, 1) 
    Y = (2 * vector - 1).reshape(1, -1) 
    W = (transY @ Y) - I
    temp = 0
    iter = 0
    while not np.array_equal(vector, corrupted):
        temp = corrupted.copy()
        iter += 1
        if iter > 3 and np.array_equal(temp, corrupted):
            print("Unsuccessfull :(")
            return corrupted
        for i in range(len(corrupted)):
            print(i)
            temp = trashhold_func((corrupted @ W[:, i]).item())
            corrupted[i] = temp
            temp = 0
    return corrupted
#---------------------------------------------------------------------------------
def matrix_multiply(vector, matrix, iter_limit=5, iter=0):
    if iter >= iter_limit: 
        return vector
    updated_vector = vector @ matrix
    updated_vector = np.array([trashhold_func(val) for val in updated_vector])
    if np.array_equal(updated_vector, vector): 
        return updated_vector
    return matrix_multiply(updated_vector, matrix, iter_limit, iter + 1)

def hopefieldSync(vector, corr):
    I = np.eye(len(vector))  
    transY = (2 * vector - 1).reshape(-1, 1)
    Y = (2 * vector - 1).reshape(1, -1)
    W = (np.matmul(transY, Y)) - I  
    ans = matrix_multiply(corr, W)
    return ans
#---------------------------------------------------------------------------------

result = hopefieldSync(test, corr)
print(result)