import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio
import os

def initialize_heatmaps(grid, interval, max_steps=1000):
    """ Dynamically initialize a sufficient number of heatmaps for up to 1000 steps. """
    num_heatmaps = (max_steps // interval) + 1
    return [np.zeros_like(grid) for _ in range(num_heatmaps)]

def simulate_robot(grid, heatmaps, interval):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Orthogonal moves
    diagonal_directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # Diagonal moves
    start_position = (2, 0)
    position = start_position
    step_count = 0
    goal_reached = False

    while not goal_reached and step_count < 1000:  # Continue until goal is reached or 1000 steps
        index = step_count // interval
        # Check if the current position is the goal
        if grid[position[0], position[1]] == 2:
            goal_reached = True
            if step_count % interval != 0:
                fill_remaining_heatmaps(heatmaps, position, index, interval+1)
            else:    
                fill_remaining_heatmaps(heatmaps, position, index, interval)
            break

        if step_count % interval == 0:
            # Update the heatmap with the current position
            heatmaps[index][position[0], position[1]] += 1

        # Generate valid moves
        if position == start_position:
            all_directions = directions + diagonal_directions
        else:
            all_directions = directions

        valid_moves = [(position[0] + dx, position[1] + dy) for dx, dy in all_directions
                       if 0 <= position[0] + dx < grid.shape[0] and 0 <= position[1] + dy < grid.shape[1]
                       and grid[position[0] + dx, position[1] + dy] != 3]
        if not valid_moves:
            break  # No valid moves, break out of the loop

        # Choose a new position from valid moves
        position = random.choice(valid_moves)
        step_count += 1


def fill_remaining_heatmaps(heatmaps, position, start_index, interval):
    """Fill all remaining heatmaps with the goal position after it is reached."""
    for i in range(start_index, len(heatmaps)):
        heatmaps[i][position[0], position[1]] += 1

def plot_heatmaps(heatmaps, cmap='hot'):
    """ Plot each heatmap in the list with annotations showing the number of visits. """
    for idx, heatmap in enumerate(heatmaps):
        plt.figure(figsize=(10, 5))
        vmax = np.max(heatmap)
        vmin = np.min(heatmap)
        cmap_instance = plt.get_cmap(cmap)
        im = plt.imshow(heatmap, cmap=cmap_instance, interpolation='nearest')
        plt.title(f"Heat Map of Robot Positions at {idx * 20} Steps")
        plt.colorbar(im)

        # Annotate each cell with the number of visits
        for (i, j), val in np.ndenumerate(heatmap):
            text_color = get_text_color(val, cmap_instance, vmin, vmax)
            plt.text(j, i, str(val), ha='center', va='center', color=text_color, fontsize=8)

        plt.show()

def get_text_color(value, cmap, vmin, vmax):
    """ Determine the text color based on the colormap and the threshold for luminance. """
    norm = Normalize(vmin=vmin, vmax=vmax)
    color = cmap(norm(value))
    # Calculate luminance of the color using the RGB channels
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return 'white' if luminance < 0.5 else 'black'

def save_heatmap_frames(heatmaps, cmap='hot', directory='frames'):
    filenames = []
    cmap_instance = plt.get_cmap(cmap)
    for idx, heatmap in enumerate(heatmaps):
        plt.figure(figsize=(10, 5))
        vmax = np.max(heatmap)
        vmin = np.min(heatmap)
        plt.imshow(heatmap, cmap=cmap, interpolation='nearest')
        plt.colorbar()
        plt.title(f"Heat Map of Robot Positions at Step {idx * 20}")

        # Annotate each cell with the number of visits
        for (i, j), val in np.ndenumerate(heatmap):
            text_color = get_text_color(val, cmap_instance, vmin, vmax)
            plt.text(j, i, str(val), ha='center', va='center', color=text_color, fontsize=8)

        # Save the frame
        filename = f"{directory}/heatmap_{idx}.png"
        plt.savefig(filename)
        plt.close()  # Close the figure to free memory
        filenames.append(filename)

    return filenames

def create_gif(filenames, output_path='robot_heatmaps.gif'):
    with imageio.get_writer(output_path, mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"GIF saved as {output_path}")

# Example grid and simulation parameters
grid = np.array([
    [3, 0, 0, 3, 0, 0, 0, 3, 0, 2],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 2]
])
interval = 20
max_steps = 500  # Estimate of maximum steps you might expect
num_simulations = 100

# Initialize heatmaps
heatmaps = initialize_heatmaps(grid, interval, max_steps)

# Run the simulations
for _ in range(num_simulations):
    simulate_robot(grid, heatmaps, interval)

# Ensure the directory for frames exists
if not os.path.exists('frames'):
    os.makedirs('frames')

# Save frames
filenames = save_heatmap_frames(heatmaps)

# Create a GIF from the frames
create_gif(filenames)