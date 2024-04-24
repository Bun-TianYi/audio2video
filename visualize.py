import numpy as np
import matplotlib.pyplot as plt


face_mesh_path = 'F:/My-repository/audio2motion/data/processed/videos/cut/lms_2d.npy'

mat = np.load(face_mesh_path)
x = mat[0, :, 0]
y = mat[0, :, 1]
plt.plot(x, y, 'o')
plt.show()
dd = 0