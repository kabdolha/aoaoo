# Ice Cream Scatter Plot - Multi-Model Comparison

Statistical visualization comparing linear regression, polynomial regression, and neural networks with colored segments and formula extraction.

## Quick Start

```bash
pip install -r requirements.txt
python ice_cream_plot.py
```

## Files

### Core

- `config.py` - All configuration parameters
- `ice_cream_plot.py` - Main plotting script
- `requirements.txt` - Dependencies (matplotlib, numpy, scikit-learn)

### Generated

- `ice_cream_scatter.png/pdf` - Multi-model plot
- `neural_network_segments.txt` - Colored segment analysis

## Features

### Three Models

- **Blue**: Linear regression
- **Orange**: Polynomial with LaTeX formula
- **Colored Segments**: Neural network with activation regions

### Neural Network Analysis

- **Multi-colored line**: Shows neuron activation states
- **Automatic breakpoints**: AI discovers temperature thresholds (63°F, 81°F)
- **Mathematical extraction**: Complete ReLU formulas

## Configuration (config.py)

- **DATA_CONFIG**: Temperature/ice cream data with noise
- **NEURAL_NET_CONFIG**: 4-neuron network, ReLU activation
- **TREND_LINE_CONFIG**: Linear/polynomial settings
- **PREDICTION_CONFIG**: Red prediction line at 90°F
- **PLOT_CONFIG**: Titles, fonts, styling
- **FORMULA_CONFIG**: LaTeX display settings

## Key Insights

Neural network discovers behavioral regimes:

- **Orange segment** (25-63°F): No ice cream consumption
- **Yellow-Orange** (63-81°F): Gentle increase starts
- **Yellow** (81-115°F): Rapid consumption growth

Demonstrates why neural networks excel at pattern recognition through automatic feature discovery.
