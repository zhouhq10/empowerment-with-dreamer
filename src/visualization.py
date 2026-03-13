import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import validate_probability_distribution
from typing import Tuple, List, Optional

def visualize_mutual_info_calculation(current_state, 
                                      p_next_state_given_state_action: np.ndarray,
                                      p_action_given_state: np.ndarray,
                                      action_names: Optional[List[str]] = None,
                                      state_names: Optional[List[str]] = None,
                                      figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
    """
    Visualize the different components of the mutual information calculation for a given state, and how they are combined to yield the empowerment value. 
    Does not work well for large action spaces (roughly > 20 actions) or state spaces (roughly > 100 states).
    
    Args:
        current_state: Current state (can be state index, coordinates, image, etc.)
        p_next_state_given_state_action: NumPy array of shape (num_actions, num_states), containing p(S_{t+n} | A_t, S_t=s)
        p_action_given_state: NumPy array of shape (num_actions,) containing p(A_t | S_t=s)
        action_names: Optional list of action names
        state_names: Optional list of state names
        figsize: Figure size for the plot
    """
    
    # validate probability distributions
    validate_probability_distribution(p_next_state_given_state_action, axis=1)
    validate_probability_distribution(p_action_given_state)

    num_actions, num_states = p_next_state_given_state_action.shape
    
    if num_actions > 20 or num_states > 100:
        # Warn user that visualization may not work well for large action or state spaces
        print("Warning: Visualization may not work well for large action or state spaces.")
    
    
    if action_names is None:
        action_names = [f"a{i}" for i in range(num_actions)]
    if state_names is None:
        state_names = [f"s{i}" for i in range(num_states)]
        
    # Create figure with subplots - added extra space at bottom for calculation text
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(4, 2, height_ratios=[1, 2, 1, 0.5]) 
    
    # 1. Plot current state information
    ax_state = fig.add_subplot(gs[0, 0])
    ax_state.axis('off')
    ax_state.text(0.5, 0.5, f"Current State: {state_names[current_state]}", 
                  horizontalalignment='center', verticalalignment='center')
    
    # TODO: Maybe add support for showing visualizing current state in some other way here, e.g. as an image
    
    # 2. Plot p(s'|s,a) for each action and their entropies
    ax_transitions = fig.add_subplot(gs[1, 0])
    entropies = []
    bar_positions = []
    bar_centers = []
    
    for i, (action_dist, action_name) in enumerate(zip(p_next_state_given_state_action, action_names)):
        # Calculate positions for this group of bars
        positions = np.arange(num_states) + i * (num_states + 2)
        bar_positions.extend(positions)
        bar_centers.append(positions.mean())
        
        plt.bar(positions, action_dist, label=f'{action_name}')
        
        # Calculate entropy
        entropy = -np.sum(action_dist * np.log2(action_dist + 1e-10))
        entropies.append(entropy)
    
    plt.title('p(s\'|s,a) for each action')
    plt.legend()
    
    # Set x-axis ticks to show entropy values at the center of each action's bars
    plt.xticks(bar_centers, [f'H_a{i}={entropy:.2f}' for i, entropy in enumerate(entropies)])
    
    # 3. Plot policy p(a|s)
    ax_policy = fig.add_subplot(gs[1, 1])
    plt.bar(range(num_actions), p_action_given_state)
    plt.title('Source Policy p(a|s)')
    plt.xticks(range(num_actions), action_names)
    
    # 4. Calculate and plot marginalized p(s'|s)
    ax_marginal = fig.add_subplot(gs[2, :])
    p_next_state_given_state = p_next_state_given_state_action.T @ p_action_given_state
    
    plt.bar(range(num_states), p_next_state_given_state)
    plt.title('Marginalized p(s\'|s)')
    plt.xticks(range(num_states), state_names)
    
    # Calculate marginal entropy
    h_marginal = -np.sum(p_next_state_given_state * 
                        np.log2(p_next_state_given_state + 1e-10))
    
    # Calculate conditional entropy term
    conditional_entropy = np.sum(p_action_given_state * entropies)
    
    # Calculate empowerment
    empowerment = h_marginal - conditional_entropy
    
    # Add empowerment calculation text in the bottom row
    ax_text = fig.add_subplot(gs[3, :])
    ax_text.axis('off')
    calculation_text = (
        f"Mutual Information = H(S'|S=s) - Σ_a p(A=a|S=s)H(S'|A=a,S=s)\n"
        f"            = {h_marginal:.2f} - ({' + '.join(f'{p:.2f}×{h:.2f}' for p, h in zip(p_action_given_state, entropies))})\n"
        f"            = {h_marginal:.2f} - {conditional_entropy:.2f}\n"
        f"            = {empowerment:.2f}"
    )
    ax_text.text(0.0, 0.5, calculation_text, fontfamily='monospace', verticalalignment='center')
    
    plt.tight_layout()
    return fig


def plot_gridworld_and_heatmap(env, heatmap, title, colorbar_label, ax=None):
    # Set NaN values to grey in the colormap
    cmap = mpl.colormaps.get_cmap('viridis')  # viridis is the default colormap for imshow
    cmap.set_bad(color='grey')
    
    if ax is None:
        plt.figure(figsize=(7.5, 6))
        # Note: Transpose the heatmap to match the gridworld layout
        plt.imshow(heatmap.T, origin='lower', interpolation='nearest', cmap=cmap)
        
        # add grid lines to clearly separate grid cells (minor ticks)
        # minor ticks are not enabled by default, so we need to enable them
        plt.gca().set_xticks(np.arange(-.5, env.width, 1), minor=True)
        plt.gca().set_yticks(np.arange(-.5, env.height, 1), minor=True)
        # plt.gca().grid(which="minor", color="black", linestyle='-', linewidth=1)
        
        plt.colorbar(label=colorbar_label)
        plt.title(title)
            
        
        # Plot icons for obstacles, ice, death states, and portals
        for x, y in env.obstacles:
            plt.text(x, y, '#', color='black', ha='center', va='center', fontsize=30)
            
        for x, y in env.ice_states:
            plt.text(x, y, '❄️', color='white', ha='center', va='center', fontsize=30)
            
        for x, y in env.death_states:
            plt.text(x, y, '☠️', color='white', ha='center', va='center', fontsize=30)
            
        for x, y in env.portals:
            plt.text(x, y, 'O', color='white', ha='center', va='center', fontsize=30)
        
        plt.show()
    else:
        im = ax.imshow(heatmap.T, origin='lower', interpolation='nearest', cmap=cmap)
        
        # add grid lines to clearly separate grid cells (minor ticks)
        # minor ticks are not enabled by default, so we need to enable them
        ax.set_xticks(np.arange(-.5, env.width, 1), minor=True)
        ax.set_yticks(np.arange(-.5, env.height, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        
        plt.colorbar(im, label=colorbar_label, ax=ax)
        ax.set_title(title)
        
        # Plot icons for obstacles, ice, death states, and portals
        for x, y in env.obstacles:
            ax.text(x, y, '#', color='black', ha='center', va='center', fontsize=30)
            
        for x, y in env.ice_states:
            ax.text(x, y, '❄️', color='white', ha='center', va='center', fontsize=30)
            
        for x, y in env.death_states:
            ax.text(x, y, '☠️', color='white', ha='center', va='center', fontsize=30)
            
        for x, y in env.portals:
            ax.text(x, y, 'O', color='white', ha='center', va='center', fontsize=30)
        
        return ax

def plot_learning_curves(history):
    """Plot learning curves showing state/action discovery and transition model error over time."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    # Plot state discovery
    eval_steps = history['eval_steps']
    ax1.plot(eval_steps, history['visited_states'], label='States')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('# States')
    ax1.legend()
    ax1.set_title('Total Discovered States')
    
    # Plot episode count (if the agent is actually learning, it should die less often, so this should flatten out)
    ax2.plot(eval_steps, history['eval_episodes'])
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Episodes')
    ax2.set_title('Episodes')
    
    # Plot average error over time
    errors = history['errors']
    ax3.plot(eval_steps, errors)
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Average Error')
    ax3.set_title('Transition Model Error')
        
    plt.tight_layout()
    return fig