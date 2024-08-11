import numpy as np

# Load the .npy file
data = np.load('feature_od.npy')

# Print the shape of the array
print("Shape of the data:", data.shape)

# Print the data type of the array
print("Data type of the array:", data.dtype)

# Optionally, print a small sample of the data (e.g., the first few entries)
print("First few entries of the data:")
print(data[:5])  # Modify the slice as needed to view more or less data

# If the data is multidimensional, you might want to print slices of it
# For example, if it's a 3D array, you can look at a specific slice:
print("A specific slice of the data (first element in the first dimension):")
print(data[0])  # Modify the index as needed to explore other parts of the array
