import numpy as np
import random
import tkinter as tk
import time
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation

# Function to simulate one robot journey through the maze
def simulate_robot(grid):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    start_position = (2, 0)
    position = start_position
    visited_counts = np.zeros_like(grid)
    path = [start_position]
    path_length = 0
    
    while True:
        visited_counts[position[0], position[1]] += 1
        if grid[position[0]][position[1]] == 2:
            break  # Stop moving when goal is reached
        valid_moves = []
        for dx, dy in directions:
            nx, ny = position[0] + dx, position[1] + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] != 3:
                valid_moves.append((nx, ny))
        if not valid_moves:
            break  # No valid moves
        position = random.choice(valid_moves)
        path.append(position)
        path_length += 1
    return visited_counts, path_length, path

# Define the maze grid
grid = np.array([
    [3, 0, 0, 3, 0, 0, 0, 3, 0, 2],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 2]
])


def get_text_color(value, cmap, thresh=0.5):
    """ Determine the text color based on the colormap and the threshold. """
    color = cmap(Normalize(vmin=total_visits.min(), vmax=total_visits.max())(value))
    # Use the RGB values to calculate luminance assuming sRGB luminance
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return 'black' if luminance > thresh else 'white'

# Run simulations
total_visits = np.zeros_like(grid)
path_lengths = []
paths = []

for _ in range(1000):
    visits, length, path = simulate_robot(grid)
    total_visits += visits
    path_lengths.append(length)
    paths.append(path)

# Heat map of visits
plt.figure(figsize=(10, 5))
ax = plt.gca()
cmap = plt.get_cmap('hot')  # Get the colormap
cax = ax.matshow(total_visits, cmap=cmap, interpolation='nearest')
for (i, j), val in np.ndenumerate(total_visits):
    text_color = get_text_color(val, cmap)  # Determine text color based on cell color
    ax.text(j, i, f"{val}", ha='center', va='center', color=text_color)
plt.colorbar(cax)
plt.title('Heat Map of Robot Visits Over 100 Simulations')
plt.show()

# Define bins for the histogram
bins = range(0, max(path_lengths) + 51, 50)  # Adjust bin size and range as needed

# Create histogram data with actual frequencies
histogram, bin_edges = np.histogram(path_lengths, bins=bins, density=False)

# Prepare bin centers from edges for plotting
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# Calculate KDE for the path lengths
x_d = np.linspace(min(path_lengths), max(path_lengths), 1000)
kde = gaussian_kde(path_lengths)
kde_values = kde(x_d)

# Calculate the scaling factor: this is the total area under the histogram
bin_widths = np.diff(bin_edges)
total_area = np.sum(histogram * bin_widths)

# Scale the KDE values
kde_values_scaled = kde_values * total_area / np.sum(kde(x_d) * (x_d[1] - x_d[0]))

# Create the plot
plt.figure(figsize=(12, 6))
ax = plt.gca()

# Plot histogram
ax.bar(bin_centers, histogram, width=bin_widths, edgecolor='black', alpha=0.7, label='Path Length Distribution (Frequency)')

# Plot KDE - scaled to fit frequencies
ax.plot(x_d, kde_values_scaled, color='red', label='Density Plot Scaled to Frequency')

# Customizing the plot
plt.title('Path Length Distribution and Scaled Density Plot')
plt.xlabel('Path Length')
plt.ylabel('Frequency')
plt.legend()

plt.show()

# Find the shortest and longest paths
shortest_path_index = np.argmin(path_lengths)
longest_path_index = np.argmax(path_lengths)
shortest_path = paths[shortest_path_index]
longest_path = paths[longest_path_index]

# Start the Tkinter GUI setup
root = tk.Tk()
root.title("Random Walk Simulation")

frame = tk.Frame(root, bg="black")
frame.pack(padx=5, pady=5)

# Define colors for the cells
colors = {0: "white", 1: "green", 2: "red", 3: "black"}
trail_colors = ["#ff0000", "#ff4d4d", "#ff9999", "#ffc2c2", "#ffe6e6", "white"]

# Initialize cell_colors and labels
cell_colors = [[colors[grid[i, j]] for j in range(grid.shape[1])] for i in range(grid.shape[0])]
labels = [[None for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        labels[i][j] = tk.Label(frame, bg=cell_colors[i][j], width=10, height=5, relief="solid", borderwidth=1)
        labels[i][j].grid(row=i, column=j, sticky="nsew")

step_counter_label = tk.Label(root, text="Step: 0", font=('Helvetica', 14))
step_counter_label.pack()

def update_cell_colors():
    """Update colors for cells in the fading trail."""
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if cell_colors[i][j] in trail_colors:
                current_color_index = trail_colors.index(cell_colors[i][j])
                if current_color_index < len(trail_colors) - 1:
                    cell_colors[i][j] = trail_colors[current_color_index + 1]
                    labels[i][j].config(bg=cell_colors[i][j])

def visualize_path(path, total_time=None):
    num_steps = len(path)
    time_per_step = total_time / num_steps if total_time else 0.1  # Default to 0.5 seconds if no total time is provided

    previous_position = None
    for step, position in enumerate(path):
        if previous_position is not None:
            # Fade the previous cell using the trail effect
            cell_colors[previous_position[0]][previous_position[1]] = trail_colors[0]
            labels[previous_position[0]][previous_position[1]].config(bg=cell_colors[previous_position[0]][previous_position[1]])

        x, y = position
        step_counter_label.config(text=f"Step: {step + 1}")
        update_cell_colors()  # Update fading of all cells
        labels[x][y].config(bg='blue')  # Highlight current cell
        root.update()
        time.sleep(time_per_step)  # Dynamic sleep time based on total_time and path length

        if grid[x, y] == 2:  # Stop at the goal
            break

        previous_position = position

def start_simulation():
    visualize_path(shortest_path)  # Visualize the shortest path at normal speed
    visualize_path(longest_path, total_time=5)  # Visualize the longest path, aiming to finish in 5 seconds


start_button = tk.Button(root, text="Start Simulation", command=start_simulation)
start_button.pack()

root.mainloop()

# Set up the plot for animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, grid.shape[1])
ax.set_ylim(0, grid.shape[0])

# Initialize the robot position
robot, = ax.plot([], [], marker="o", color="blue", markersize=10)

# Define colors for the cells
colors = {0: "white", 1: "green", 2: "red", 3: "black"}

# Initialize cell_colors
cell_colors = [[colors[grid[i, j]] for j in range(grid.shape[1])] for i in range(grid.shape[0])]

# Update function for animation
def update(frame):
    ax.clear()
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(0, grid.shape[0])
    ax.imshow(np.array(cell_colors), cmap="hot", interpolation="nearest")
    x, y = paths[frame][-1]
    robot.set_data(y, x)

# Create animation
anim = FuncAnimation(fig, update, frames=len(paths), interval=500)

# Save the animation as a video
anim.save("robot_simulation.mp4")

plt.show()