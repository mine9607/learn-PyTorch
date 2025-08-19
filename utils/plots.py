import matplotlib.pyplot as plt
import numpy as np

# Toy Dataset
x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)

# 1:: Create a Figure
fig, ax = plt.subplots(figsize=(6, 4))
ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.plot(x, y)
# plt.show()

# 2::Basic Styling on a Single Plot
fig, ax = plt.subplots()
ax.plot(
    x,
    y,
    color="tab:blue",
    linestyle="--",  # '-', '--', '-.', ':'
    linewidth="2.0",
    marker="o",
    markersize=4,
    markerfacecolor="white",
)
ax.set_title("Sine wave", fontsize=14, pad=10)
ax.set_xlabel("Angle (rad)")
ax.set_ylabel("Amplitude")

ax.set_xlim(0, 2 * np.pi)  # set x-axis limits
ax.set_ylim(-1.1, 1.1)  # set y-axis limits

ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.legend(["sin(x)"], loc="upper right")

# plt.show()

# 3::Ticks and Tick Labels
fig, ax = plt.subplots()
ax.plot(x, y)

# Set specific tick locations and custom tick labels
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax.set_xticklabels(["0", "$\pi/2$", "$\pi$", "$3\pi/2$", "$2\pi$"])

# Rotate tick labels, adjust size
for label in ax.get_xticklabels():
    label.set_rotation(0)
    label.set_fontsize(10)

# Control y ticks quickly
ax.set_yticks([-1, 0, 1])

ax.grid(True, which="both", linestyle=":", alpha=0.6)
# plt.show()

# 4::Plot multiple lines and legend labels
fig, ax = plt.subplots()

line1 = ax.plot(x, np.sin(x), label="sin", linewidth=2)
line2 = ax.plot(x, np.cos(x), label="cos", linewidth=2, linestyle="--")

ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.legend()  # uses the 'label=' you passed to plot
# plt.show()

# 5:: Subplots - Patterns you'll use
# 5a - Simple 1 x 2 grid
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

axes[0].plot(x, np.sin(x))
axes[0].set_title("sin")
axes[0].grid(True, which="both", linestyle=":", alpha=0.6)

axes[1].plot(x, np.cos(x), color="tab:orange")
axes[1].set_title("cos")
axes[1].grid(True, which="both", linestyle=":", alpha=0.6)

# plt.show()

# 5b - 2x2 grid and indexing

fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)

axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title("sin")
axes[0, 0].grid(True, which="both", linestyle=":", alpha=0.6)

axes[0, 1].plot(x, np.cos(x))
axes[0, 1].set_title("cos")
axes[0, 1].grid(True, which="both", linestyle=":", alpha=0.6)

axes[1, 0].plot(x, np.tan(x))
axes[1, 0].set_title("tan")
axes[1, 0].grid(True, which="both", linestyle=":", alpha=0.6)

axes[1, 1].axis("off")  # turn off unused panel

# plt.show()

# 5c - flatten when you want to loop
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes = axes.ravel()  # or axes.flatten() -> iterator

# NOTE: ravel simply takes a multi-dimensional array and flattens it to a 1D array (more efficient than flatten)

funcs = [np.sin, np.cos, np.tan, np.sinh]
titles = ["sin", "cos", "tan", "sinh"]

for ax, f, t in zip(axes, funcs, titles):
    ax.plot(x, f(x))
    ax.set_title(t)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)

fig.tight_layout()
# plt.show()

# 5d - shared axes (uniform scales)
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(10, 3))
for ax, f, name in zip(axes, [np.sin, np.cos, np.sinh], ["sin", "cos", "sinh"]):
    ax.plot(x, f(x))
    ax.set_title(name)
axes[0].set_ylabel("value")

# plt.show()

# 6::Twin Axes (second y-axis on same subplot)
fig, ax1 = plt.subplots()

ax1.plot(x, np.sin(x), color="tab:blue")
ax1.set_ylabel("sin", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()  # shares x with ax1 - new y axis
ax2.plot(x, np.cos(x), color="tab:red", linestyle="--")
ax2.set_ylabel("cos", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

plt.show()

# 7::Annotations and reference lines

fig, ax = plt.subplots()
ax.plot(x, y)

# vertical/horizontal reference lines
# draws a horizontal line at the given y-value
ax.axhline(0, color="black", linewidth=1)
# draws a vertical line at the given x-value
ax.axvline(np.pi, color="gray", linestyle="--")

# annotate a plot
xm = np.pi / 2
ym = np.sin(xm)
xmid = np.pi
ymid = np.sin(xmid)
ax.plot([xm], [ym], "o")
ax.annotate(
    "peak", xy=(xm, ym), xytext=(xm + 0.5, ym), arrowprops=dict(arrowstyle="->")
)
ax.plot([xmid], [ymid], "o")
ax.annotate(
    "mid", xy=(xmid, ymid), xytext=(xmid + 0.25, ymid), arrowprops=dict(arrowstyle="->")
)

plt.show()

# 8::Layout control: tight vs constrained

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
fig.tight_layout()  # good general fit

plt.show()

fig, axes = plt.subplots(2, 2, constrained_layout=True)

plt.show()

# 9:: Saving Figures
fig, ax = plt.subplots()
ax.plot(x, y)
fig.savefig("figure.png", dpi=300, bbox_inches="tight")

# 10:: MENTAL MAP
"""
Figure (fig) - the whole canvas/page--figsize, savefig, tight_layout
Axes (ax): one plot area--almost all styling lives here
    - data: ax.plot, ax.scatter, ax.bar, etc.
    - text: ax.set_title, ax.set_xlabel, ax.set_ylabel, etc.
    - limits: ax.set_xlim, ax.set_ylim
    - ticks: ax.set_xticks, ax.set_xticklabels, ax.tick_params
    - legend/grid: ax.legend, ax.grid
    - helpers: ax.axhline/axvline, ax.annotate
"""
