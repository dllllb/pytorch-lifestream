from itertools import combinations

import matplotlib.pyplot as plt

from tools.shape_generator import ShapeGenerator

# Plot samples in different modes
n_obj, n_samples = int(2**4 * 2), 8

sgs = [ShapeGenerator(keep_params=keep_params)
       for i in range(5)
       for keep_params in combinations(['color', 'size', 'n_vertex', 'bg_color'], i)
       for j in range(2)]
figire, axs = plt.subplots(n_obj, n_samples, figsize=(2 * n_samples, 2 * n_obj), dpi=74)
for col in range(n_samples):
    for row in range(n_obj):
        ax = axs[row, col]
        ax.axis('off')
        ax.imshow(sgs[row].get_buffer()[0])
plt.show()
