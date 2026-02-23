import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
import math
import numpy as np

from file_handler import generate_dfs
from code.src.dreamerv3.analysis.multi_plot import plot_scores

window = 100000
step_limit = 1e6

minigrid_results = {
    "DreamerV3 (BASE)": "wall/SimpleCrossingS9N1/obsnovelty/",
    "DreamerV3 (CBET)": "wall/SimpleCrossingS9N1/obscbet/",
}

df = generate_dfs(
    minigrid_results, mode="intrinsic", window=window, step_limit=step_limit
)

# Plot the scores
plot_scores(df, "Intrinsic", "Tabula Rasa MiniGrid", y_lim=None)

minigrid_transfer_results = {
    "DreamerV3 (CBET)": "dreamerv3/minigrid-pretrain-1M-Doorkey8x8/",
    "IMPALA (CBET)": "impala/minigrid-pretrain-1M-Doorkey8x8/",
}

df = generate_dfs(
    minigrid_transfer_results, mode="intrinsic", window=window, step_limit=step_limit
)

plot_scores(df, "Intrinsic", "Pre-train MiniGrid", y_lim=None)
