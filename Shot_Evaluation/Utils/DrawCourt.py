import os
import matplotlib.pyplot as plt

def plot_location_court(df_Real, df_Output, player_name):
    os.makedirs('./Shot_Evaluation/Result', exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(18, 6))  # Adjust figsize as needed
        
    # First subplot: Player locations
    axs[0].set_xlim(-177.5, 177.5)
    axs[0].set_ylim(-480, 480)
    axs[0].plot([-177.5, 177.5], [0, 0], 'k-')  # Horizontal middle line
    axs[0].plot([0, 0], [-480, 480], 'k-')  # Vertical middle line
    axs[0].plot([-177.5, -177.5], [-480, 480], 'k-')  # Left line
    axs[0].plot([177.5, 177.5], [-480, 480], 'k-')  # Right line
    axs[0].plot([-177.5, 177.5], [-480, -480], 'k-')  # Bottom line
    axs[0].plot([-177.5, 177.5], [480, 480], 'k-')  # Top line

    df_Real = df_Real[df_Real['player'] == player_name]
    axs[0].scatter(df_Real['player_location_x'], df_Real['player_location_y'], color='red', label='Player')
    axs[0].scatter(df_Real['landing_x'], df_Real['landing_y'], color='blue', label='Ball')
    axs[0].set_title(f'Real Player & Ball Landing Locations')

    axs[0].set_xlabel('Court Length (X)')
    axs[0].set_ylabel('Court Width (Y)')

    # Second subplot: Ball landing locations
    axs[1].set_xlim(-177.5, 177.5)
    axs[1].set_ylim(-480, 480)
    axs[1].plot([-177.5, 177.5], [0, 0], 'k-')  # Horizontal middle line
    axs[1].plot([0, 0], [-480, 480], 'k-')  # Vertical middle line
    axs[1].plot([-177.5, -177.5], [-480, 480], 'k-')  # Left line
    axs[1].plot([177.5, 177.5], [-480, 480], 'k-')  # Right line
    axs[1].plot([-177.5, 177.5], [-480, -480], 'k-')  # Bottom line
    axs[1].plot([-177.5, 177.5], [480, 480], 'k-')  # Top line

    df_Output = df_Output[df_Output['player'] == 'A']
    axs[1].scatter(df_Output['player_location_x'], df_Output['player_location_y'], color='red', label='Player')
    axs[1].scatter(df_Output['landing_x'], df_Output['landing_y'], color='blue', label='Ball')
    axs[1].set_title(f'Output Player & Ball Landing Locations')

    axs[1].set_xlabel('Court Length (X)')
    axs[1].set_ylabel('Court Width (Y)')
    
    # Adjust layout to avoid overlap
    plt.tight_layout()
    
    filename = f'./Shot_Evaluation/location_distribution.png'
    plt.savefig(filename)
    plt.close()

# Create a badminton court layout
def draw_half_court(ax=None, color='black'):
    if ax is None:
        ax = plt.gca()
    # Draw outer lines
    ax.plot([50, 50], [0, 330], color=color)
    ax.plot([305, 305], [0, 330], color=color)
    ax.plot([27.4, 327.6], [330, 330], color=color)
    ax.plot([27.4, 327.6], [0, 0], color=color)
    ax.plot([27.4, 27.4], [0, 330], color=color)
    ax.plot([327.6, 327.6], [0, 330], color=color)
    # Draw the middle line
    ax.plot([177.5, 177.5], [0, 330], color=color)
    # Draw the service lines
    ax.plot([27.4, 327.6], [114, 114], color=color)
    ax.plot([27.4, 327.6], [276, 276], color=color)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 355)
    ax.set_ylim(0, 480)
    ax.set_aspect(1)

# Create a badminton court layout
def draw_full_court(ax=None, color='black'):
    if ax is None:
        ax = plt.gca()
    # Draw outer lines
    ax.plot([50, 50], [150, 810], color=color)
    ax.plot([305, 305], [150, 810], color=color)
    ax.plot([27.4, 327.6], [810, 810], color=color)
    ax.plot([27.4, 327.6], [480, 480], color=color)
    ax.plot([27.4, 327.6], [150, 150], color=color)
    ax.plot([27.4, 27.4], [150, 810], color=color)
    ax.plot([327.6, 327.6], [150, 810], color=color)
    # Draw the middle line
    ax.plot([177.5, 177.5], [150, 810], color=color)
    # Draw the service lines
    ax.plot([27.4, 327.6], [594, 594], color=color)
    ax.plot([27.4, 327.6], [756, 756], color=color)
    ax.plot([27.4, 327.6], [366, 366], color=color)
    ax.plot([27.4, 327.6], [204, 204], color=color)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 355)
    ax.set_ylim(0, 960)
    ax.set_aspect(1)
