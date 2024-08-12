import pickle

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

if __name__ == "__main__":

    values = [i/8 for i in range(8)]
    colors = [(0, 0, 0), (0, 0, 1.0), (0, 1.0, 1.0), (0, 1.0, 0), (1.0, 1.0, 0), (1.0, 0, 0), (1.0, 0, 1.0), (1.0, 1.0, 1.0)]
    norm = plt.Normalize(min(values), max(values))
    cmap = LinearSegmentedColormap.from_list(
        '', [(norm(value), tuple(np.array(color))) for value, color in zip(values, colors)])

    with open("astro_cmap.pickle", "wb") as f:

        pickle.dump(cmap, f)
