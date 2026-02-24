import numpy as np
import matplotlib.pyplot as plt


def get_cycle_colors():
    def hex2rgb(h: str):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [hex2rgb(c)[::-1] for c in colors]  # hex to bgr
    return np.array(colors).astype(np.uint8)


CYCLE_COLORS = get_cycle_colors()


def id_to_color(id_fig: np.ndarray, float_out: bool = False):
    shape = id_fig.shape
    color_fig = CYCLE_COLORS[id_fig.ravel() % len(CYCLE_COLORS)]
    color_fig = color_fig.reshape(*shape, 3)
    
    if float_out:
        color_fig = color_fig.astype(np.float32) / 255.
    
    return color_fig

