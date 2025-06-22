#!/usr/bin/env python3
"""
Configuration file for ice cream scatter plot
All parameters, filenames, and data in one place
"""

import numpy as np

# File Configuration
OUTPUT_FILENAME = "ice_cream_scatter"  # Will generate .png and .pdf
SCRIPT_NAME = "ice_cream_plot.py"

# Plot Configuration
PLOT_CONFIG = {
    'figsize': (10, 7),
    'dpi': 300,
    'facecolor': 'white',
    'title': 'Ice Cream Consumed by Temp',
    'xlabel': 'Temperature (°F)',
    'ylabel': 'Number of Ice Creams Eaten',
    'title_fontsize': 16,
    'label_fontsize': 14
}

# XKCD Style Parameters
XKCD_PARAMS = {
    'scale': 0.5,
    'length': 50,
    'randomness': 1
}

# Scatter Plot Style
SCATTER_STYLE = {
    'size': 200,
    'color': '#FF6B9D',
    'alpha': 0.9,
    'edge_color': '#FF6B9D',
    'edge_width': 2,
    'marker': 'o',
    'zorder': 5
}

# Axis Configuration
AXIS_CONFIG = {
    'xlim': (25, 115),  # Extended range for more data
    'ylim': (0, 55),    # Increased to accommodate higher values
    'grid_alpha': 0.2,
    'grid_linestyle': '-',
    'grid_linewidth': 0.5,
    'spine_linewidth': 1.5
}

# Data Configuration
DATA_CONFIG = {
    # Extended temperature range with more data points
    'temperatures': np.array([28, 35, 42, 48, 55, 62, 68, 72, 76, 80, 85, 88, 92, 96, 100, 105, 108]),
    # Exponential/curved relationship with more data points
    'ice_creams_base': np.array([0.2, 0.4, 0.7, 1.2, 2.0, 3.2, 4.8, 6.5, 8.8, 11.5, 15.2, 18.5, 23.2, 28.8, 35.2, 42.5, 48.0]),
    'random_seed': 42,
    'noise_std': 2.0  # Reduced noise slightly with more data points
}

# Trend Line Configuration
TREND_LINE_CONFIG = {
    'show_linear': False,
    'show_polynomial': False,
    'polynomial_degree': 2,  # Quadratic curve
    'linear_color': '#2E86AB',  # Blue color
    'polynomial_color': '#F77F00',  # Orange color
    'linewidth': 3,
    'alpha': 0.5,
    'linestyle': '-',
    'zorder': 3,  # Behind scatter points but above grid
    'extend_to_border': True  # Extend line to plot borders
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'show_prediction': True,
    'temperature': 90,  # Temperature to predict at
    'vertical_line_color': '#E63946',  # Red color
    'vertical_line_style': '--',
    'vertical_line_width': 2,
    'prediction_point_color': '#E63946',  # Red color
    'prediction_point_size': 250,
    'prediction_point_marker': 'D',  # Diamond shape
    'prediction_point_edge_color': '#B91C1C',
    'prediction_point_edge_width': 2,
    'zorder': 6  # Above everything else
}

# Neural Network Configuration
NEURAL_NET_CONFIG = {
    'show_neural_net': True,
    'hidden_layers': [4],  # Simple network: 4 neurons
    'activation': 'relu',     # ReLU activation function
    'epochs': 1000,          # Training iterations
    'learning_rate': 0.01,   # Learning rate
    'color': '#9D4EDD',      # Purple color for neural net line
    'linewidth': 3,
    'alpha': 0.9,
    'linestyle': '-',
    'zorder': 4,             # Above trend lines but below scatter points
    'label': 'Neural Network (Linear)'
}

# Neural Network Visualization Configuration
NEURAL_VIZ_CONFIG = {
    'show_colored_segments': True,
    'segment_colors': ['#FF6B35', '#F7931E', '#FFD23F', '#06FFA5', '#118AB2', '#073B4C'],
    'segment_alpha': 0.8,
    'segment_linewidth': 4,
    'analyze_breakpoints': True
}

# Legend Configuration
LEGEND_CONFIG = {
    'show_legend': True,
    'location': 'upper left',  # 'upper left', 'upper right', 'lower left', 'lower right', 'center', etc.
    'fontsize': 10,
    'frameon': True,
    'fancybox': True,
    'shadow': True,
    'framealpha': 0.9,
    'facecolor': 'white',
    'edgecolor': 'gray'
}

# Formula Display Configuration
FORMULA_CONFIG = {
    'show_formula': False,
    'show_linear_formula': False,
    'show_polynomial_formula': False,
    'position_x': 0.05,  # Relative position (0-1)
    'position_y': 0.75,  # Moved down to avoid legend overlap
    'fontsize': 10,
    'bbox_props': {
        'boxstyle': 'round,pad=0.5',
        'facecolor': 'white',
        'alpha': 0.8,
        'edgecolor': 'gray'
    }
}

# Font Configuration
FONT_CONFIG = {
    'xkcd_font_path': '/Users/kabdolha/Library/Fonts/xkcd-script.ttf',
    'font_family': 'xkcd Script'
}

# Environment Configuration
ENV_CONFIG = {
    'matplotlib_backend': 'Agg',
    'clear_font_cache': True
}
