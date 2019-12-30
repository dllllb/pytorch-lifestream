import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Circle


class ShapeGenerator:
    params_pattern = {
        'color': 3,
        'bg_color': 3,
        'size': 1,
        'n_vertex': 1,
    }
    image_shape = (148, 148, 4)

    def __init__(self, keep_params, params=None):
        self.keep_params = keep_params

        start_params = self.generate()
        self.start_params = {
            k: v for k, v in start_params.items() if k in self.keep_params
        }
        if params is not None:
            self.start_params.update(params)

    def generate(self):
        params = {k: np.random.rand(v).astype(np.float32) for k, v in self.params_pattern.items()}
        return params

    def render(self):
        params = self.generate()
        for k, v in self.start_params.items():
            params[k] = v

        # shape, size
        n_vertex = int(params['n_vertex'] * (7 - 3) + 3)
        size = params['size'][0]

        angles = np.linspace(0, 1, n_vertex + 1)[:-1] * 2 * np.pi
        angles = angles + np.random.rand(n_vertex) * 2 * np.pi / n_vertex * 0.95

        rays = np.random.rand(n_vertex)
        rays = rays - rays.min()
        rays = rays / (rays.max() + 1e-5)
        rays = 0.30 + (size * (1 - 0.30 - 0.1)) + rays * 0.1

        x = rays * np.cos(angles)
        y = rays * np.sin(angles)
        shape = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)

        # color
        color = params['color']
        bg_color = params['bg_color']

        return Polygon(shape, color=color), Circle((0, 0), 1.1, color=bg_color), params

    def get_buffer(self):
        fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=74)
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.axis('off')
        img, bg_img, params = self.render()
        ax.add_patch(bg_img)
        ax.add_patch(img)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height() + (4,)).copy()
        plt.close()

        return buf.astype(np.float32) / 255.0, params


def test_shape_generator():
    sg = ShapeGenerator(keep_params=[])
    X, y = sg.get_buffer()
    assert X.shape == ShapeGenerator.image_shape
