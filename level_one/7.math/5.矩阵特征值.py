import numpy as np

# Define matrix A
A = np.array([[4, 1],
              [2, 3]])

# Use NumPy to find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("特征值: ", eigenvalues)        # Print eigenvalues
print("特征向量: ", eigenvectors)    # Print eigenvectors
