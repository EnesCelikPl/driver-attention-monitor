import numpy as np
# A: 4x4 random integers (0–9), B: 4x4 identity matrix
A = np.random.randint(0, 10, (4, 4))
B = np.eye(4, dtype=int)
print("A:\n", A)
print("B:\n", B)
# a. A + B
print("\na. A + B:\n", A + B)
# b. Sum of all elements in A and B
print("\nb. Sum of all elements:", A.sum() + B.sum())
# c. Max in A
print("\nc. Max in A:", A.max())
# d. Reshape to 8x2 and multiply
A_r = A.reshape(8, 2)
B_r = B.reshape(8, 2)
print("\nd. Reshaped A * B:\n", A_r * B_r)
# e. Sum of 3rd column in A and 3rd row in B
print("\ne. Sum of A[:,2] + B[2,:]:", A[:, 2].sum() + B[2, :].sum())
# f. Square second column in A
A_f = A.copy()
A_f[:, 1] = A_f[:, 1] ** 2
print("\nf. A with 2nd column squared:\n", A_f)
# g. Horizontal join (4x8)
AB = np.hstack((A, B))
print("\ng. A and B joined:\n", AB)
# h. Convert to string and add
A_str = A.astype(str)
B_str = B.astype(str)
print("\nh. A + B (as strings):\n", A_str + B_str)

# Task 2 - Matrix Expressions
X1 = np.array([[2, 2, 2],
               [2, 2, 2],
               [2, 2, 2]])

Y1 = np.array([[3, 4, 5],
               [6, 7, 8],
               [9, 10, 11]])

try:
    res1 = np.dot(X1, Y1.T)
    print("\nTask 2 - First expression result:\n", res1)
except ValueError as e:
    print("First expression failed:", e)

try:
    res2 = np.dot(Y1.T, X1)
    print("\nTask 2 - Second expression result:\n", res2)
except ValueError as e:
    print("Second expression failed:", e)