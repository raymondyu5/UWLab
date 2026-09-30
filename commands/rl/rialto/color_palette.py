import matplotlib.pyplot as plt
import numpy as np
from colorspacious import cspace_convert

palette = [
    "#808080",  # gray
    "#D9A066",  # muted orange
    "#F2B8CC",  # very light pink
    "#7B5EA7",  # deeper purple
    "#5B9FD4",  # stronger blue
    "#8CC3A3",  # sage green
    "#2D6B4A",  # dark green
]

palette = [
    "#B0B0B0",  # lighter gray
    "#FF7F0E",  # orange
    "#E7298A",  # magenta
    "#7B2D8B",  # purple
    "#5B9BD5",  # steel blue
    "#5EC962",  # light green
    "#1B7C6E",  # dark teal
]

methods = [
    "Base",
    "FPO",
    "PPO-BC",
    "RialTo",
    "Residual",
    "CODE (abl)",
    "CODE"
]

values = [18.3, 19.9, 24.2, 21.5, 17.8, 22.1, 20.0]


# Convert hex to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4)]

rgb = np.array([hex_to_rgb(c) for c in palette])

# Simulate deuteranomaly (common red-green colorblindness)
cvd_space = {
    "name": "sRGB1+CVD",
    "cvd_type": "deuteranomaly",
    "severity": 100
}

simulated = cspace_convert(rgb, cvd_space, "sRGB1")
simulated = np.clip(simulated, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Original
axes[0].bar(methods, values[:len(palette)], color=palette)
axes[0].set_title("Original")

# Simulated colorblind view
axes[1].bar(methods, values[:len(palette)], color=simulated)
axes[1].set_title("Simulated Deuteranomaly")

for ax in axes:
    ax.set_ylim(0, 28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("palette.png")