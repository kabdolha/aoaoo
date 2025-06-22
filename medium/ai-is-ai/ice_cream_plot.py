#!/usr/bin/env python3
"""
Ice cream scatter plot using configuration from config.py
Now includes neural network prediction overlay
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import matplotlib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from config import *

def setup_environment():
    """Setup matplotlib environment"""
    matplotlib.use(ENV_CONFIG['matplotlib_backend'])
    
    if ENV_CONFIG['clear_font_cache']:
        fm._load_fontmanager(try_read_cache=False)

def setup_xkcd_font():
    """Configure matplotlib to use XKCD font"""
    try:
        fm.fontManager.addfont(FONT_CONFIG['xkcd_font_path'])
        plt.rcParams['font.family'] = FONT_CONFIG['font_family']
        print("XKCD font registered successfully")
        return True
    except Exception as e:
        print(f"Could not register XKCD font: {e}")
        return False

def analyze_neural_network_segments(nn_model, scaler_X, scaler_y, temp_range):
    """Analyze where each neuron activates and create colored segments"""
    
    # Get the neural network components
    weights = nn_model.coefs_
    biases = nn_model.intercepts_
    
    # Scale temperature range
    temp_scaled = scaler_X.transform(temp_range.reshape(-1, 1)).ravel()
    
    # Calculate hidden layer outputs for each temperature
    w1 = weights[0]  # Input to hidden weights
    b1 = biases[0]   # Hidden biases
    
    # Calculate when each neuron activates (ReLU > 0)
    activation_points = []
    neuron_info = []
    
    for i in range(len(b1)):
        # h1_i = ReLU(w1[0,i] * x_scaled + b1[i])
        # Neuron activates when w1[0,i] * x_scaled + b1[i] > 0
        # So x_scaled > -b1[i] / w1[0,i] (if w1[0,i] > 0)
        
        weight = w1[0, i]
        bias = b1[i]
        
        if abs(weight) > 1e-6:  # Avoid division by zero
            activation_temp_scaled = -bias / weight
            activation_temp = scaler_X.inverse_transform([[activation_temp_scaled]])[0][0]
            activation_points.append((i, activation_temp, weight, bias))
            
            neuron_info.append({
                'neuron': i + 1,
                'weight': weight,
                'bias': bias,
                'activation_temp': activation_temp,
                'activation_temp_scaled': activation_temp_scaled
            })
    
    # Sort by activation temperature
    activation_points.sort(key=lambda x: x[1])
    neuron_info.sort(key=lambda x: x['activation_temp'])
    
    return activation_points, neuron_info

def create_colored_neural_network_plot(ax, nn_model, scaler_X, scaler_y, temp_range):
    """Create a multi-colored neural network plot showing different segments"""
    
    # Analyze segments
    activation_points, neuron_info = analyze_neural_network_segments(nn_model, scaler_X, scaler_y, temp_range)
    
    # Get full prediction
    temp_scaled = scaler_X.transform(temp_range.reshape(-1, 1))
    nn_pred_scaled = nn_model.predict(temp_scaled)
    nn_predictions = scaler_y.inverse_transform(nn_pred_scaled.reshape(-1, 1)).ravel()
    
    # Define colors for different segments
    colors = ['#FF6B35', '#F7931E', '#FFD23F', '#06FFA5', '#118AB2', '#073B4C']
    
    # Create segments based on activation points
    segment_temps = [temp_range[0]]
    for _, temp, _, _ in activation_points:
        if temp_range[0] <= temp <= temp_range[-1]:
            segment_temps.append(temp)
    segment_temps.append(temp_range[-1])
    segment_temps = sorted(list(set(segment_temps)))
    
    # Plot each segment with different colors
    for i in range(len(segment_temps) - 1):
        start_temp = segment_temps[i]
        end_temp = segment_temps[i + 1]
        
        # Find indices for this segment
        start_idx = np.argmin(np.abs(temp_range - start_temp))
        end_idx = np.argmin(np.abs(temp_range - end_temp)) + 1
        
        # Plot this segment
        segment_temps_plot = temp_range[start_idx:end_idx]
        segment_preds = nn_predictions[start_idx:end_idx]
        
        color = colors[i % len(colors)]
        ax.plot(segment_temps_plot, segment_preds,
               color=color,
               linewidth=4,
               alpha=0.9,
               zorder=4)
    
    return activation_points, neuron_info

def save_neural_network_analysis(nn_model, scaler_X, scaler_y, temp_range, filename="neural_network_analysis.txt"):
    """Save detailed analysis of neural network segments"""
    
    activation_points, neuron_info = analyze_neural_network_segments(nn_model, scaler_X, scaler_y, temp_range)
    
    weights = nn_model.coefs_
    biases = nn_model.intercepts_
    
    with open(filename, 'w') as f:
        f.write("NEURAL NETWORK SEGMENT ANALYSIS\n")
        f.write("="*50 + "\n\n")
        
        f.write("NEURON ACTIVATION ANALYSIS\n")
        f.write("-" * 30 + "\n")
        
        for info in neuron_info:
            f.write(f"Neuron {info['neuron']}:\n")
            f.write(f"  Formula: h{info['neuron']} = ReLU({info['weight']:.4f} * x_scaled + {info['bias']:.4f})\n")
            f.write(f"  Activates when temperature > {info['activation_temp']:.1f}°F\n")
            
            if info['weight'] > 0:
                f.write(f"  Effect: INCREASES ice cream consumption as temperature rises\n")
            else:
                f.write(f"  Effect: DECREASES ice cream consumption as temperature rises\n")
            f.write("\n")
        
        f.write("TEMPERATURE SEGMENTS\n")
        f.write("-" * 20 + "\n")
        
        # Create segments
        segment_temps = [temp_range[0]]
        for _, temp, _, _ in activation_points:
            if temp_range[0] <= temp <= temp_range[-1]:
                segment_temps.append(temp)
        segment_temps.append(temp_range[-1])
        segment_temps = sorted(list(set(segment_temps)))
        
        colors = ['Orange', 'Yellow-Orange', 'Yellow', 'Green', 'Blue', 'Dark Blue']
        
        for i in range(len(segment_temps) - 1):
            start_temp = segment_temps[i]
            end_temp = segment_temps[i + 1]
            color = colors[i % len(colors)]
            
            f.write(f"Segment {i+1} ({color} line): {start_temp:.1f}°F to {end_temp:.1f}°F\n")
            
            # Determine which neurons are active in this segment
            active_neurons = []
            for info in neuron_info:
                if info['activation_temp'] <= start_temp:
                    active_neurons.append(info['neuron'])
            
            if active_neurons:
                f.write(f"  Active neurons: {', '.join(map(str, active_neurons))}\n")
            else:
                f.write(f"  Active neurons: None (baseline only)\n")
            
            f.write(f"  Behavior: ")
            if len(active_neurons) == 0:
                f.write("Minimal ice cream consumption\n")
            elif len(active_neurons) == 1:
                f.write("Linear increase begins\n")
            else:
                f.write(f"Multiple neurons contributing - accelerated growth\n")
            f.write("\n")
        
        f.write("INTERPRETATION\n")
        f.write("-" * 15 + "\n")
        f.write("Each color segment represents a different 'regime' where\n")
        f.write("different combinations of neurons are active. The neural\n")
        f.write("network creates a piecewise linear function that approximates\n")
        f.write("the curved relationship between temperature and ice cream consumption.\n")
    
    print(f"Neural network segment analysis saved to '{filename}'")

def train_neural_network(temperatures, ice_creams):
    """Train a simple neural network on the data"""
    # Prepare data
    X = temperatures.reshape(-1, 1)  # Reshape for sklearn
    y = ice_creams
    
    # Scale the features for better neural network performance
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Create and train neural network
    nn = MLPRegressor(
        hidden_layer_sizes=tuple(NEURAL_NET_CONFIG['hidden_layers']),
        activation=NEURAL_NET_CONFIG['activation'],
        max_iter=NEURAL_NET_CONFIG['epochs'],
        learning_rate_init=NEURAL_NET_CONFIG['learning_rate'],
        random_state=42,  # For reproducible results
        alpha=0.001  # Small regularization
    )
    
    # Train the network
    nn.fit(X_scaled, y_scaled)
    
    # Print the neural network formula
    print("\n" + "="*60)
    print("NEURAL NETWORK FORMULA BREAKDOWN")
    print("="*60)
    
    # Get weights and biases
    weights = nn.coefs_
    biases = nn.intercepts_
    
    print(f"Network Architecture: {[1] + list(nn.hidden_layer_sizes) + [1]}")
    print(f"Activation Function: {nn.activation}")
    
    # Layer 1: Input to Hidden Layer 1
    print(f"\nLayer 1 (Input → Hidden Layer 1):")
    print(f"Input: x (temperature, scaled)")
    w1 = weights[0]  # Shape: (1, 8)
    b1 = biases[0]   # Shape: (8,)
    
    for i in range(len(b1)):
        print(f"  h1_{i+1} = ReLU({w1[0,i]:.4f} * x + {b1[i]:.4f})")
    
    # Layer 2: Hidden Layer 1 to Hidden Layer 2 (if exists)
    if len(weights) > 2:
        print(f"\nLayer 2 (Hidden Layer 1 → Hidden Layer 2):")
        w2 = weights[1]  # Shape: (8, 4)
        b2 = biases[1]   # Shape: (4,)
        
        for i in range(len(b2)):
            formula = f"  h2_{i+1} = ReLU("
            terms = []
            for j in range(w2.shape[0]):
                terms.append(f"{w2[j,i]:.4f}*h1_{j+1}")
            formula += " + ".join(terms) + f" + {b2[i]:.4f})"
            print(formula)
    
    # Final Layer: Hidden to Output
    print(f"\nFinal Layer (Hidden → Output):")
    w_final = weights[-1]  # Shape: (4, 1) or (8, 1)
    b_final = biases[-1]   # Shape: (1,)
    
    formula = "  y_scaled = "
    terms = []
    hidden_size = w_final.shape[0]
    layer_name = "h2" if len(weights) > 2 else "h1"
    
    for i in range(hidden_size):
        terms.append(f"{w_final[i,0]:.4f}*{layer_name}_{i+1}")
    formula += " + ".join(terms) + f" + {b_final[0]:.4f}"
    print(formula)
    
    print(f"\nFinal step: y = inverse_scale(y_scaled)")
    print(f"Where inverse_scale converts back to original ice cream units")
    
    print("\nNote: ReLU(x) = max(0, x)")
    print("="*60)
    
    return nn, scaler_X, scaler_y

def generate_data():
    """Generate ice cream consumption data with realistic noise"""
    temperatures = DATA_CONFIG['temperatures'].copy().astype(float)
    ice_creams = DATA_CONFIG['ice_creams_base'].copy().astype(float)
    
    # Add realistic variation
    np.random.seed(DATA_CONFIG['random_seed'])
    
    # Add noise to ice cream consumption
    ice_creams += np.random.normal(0, DATA_CONFIG['noise_std'], len(ice_creams))
    
    # Add some temperature variation too
    temperatures += np.random.normal(0, 3, len(temperatures))
    
    # Add some outliers to make it more realistic
    outlier_indices = [2, 6, 8]  # Add some outliers
    for i in outlier_indices:
        if i < len(ice_creams):
            ice_creams[i] += np.random.choice([-3, 3]) * np.random.uniform(1, 2)
    
    # Ensure no negative ice creams
    ice_creams = np.maximum(ice_creams, 0)
    
    return temperatures, ice_creams

def save_neural_network_segments(nn_model, scaler_X, scaler_y, temp_range, segment_temps, colors, filename="neural_network_segments.txt"):
    """Save detailed analysis of neural network segments"""
    
    weights = nn_model.coefs_[0]  # Input to hidden weights
    biases = nn_model.intercepts_[0]  # Hidden biases
    
    with open(filename, 'w') as f:
        f.write("NEURAL NETWORK COLORED SEGMENTS ANALYSIS\n")
        f.write("="*50 + "\n\n")
        
        f.write("NEURON ACTIVATION POINTS\n")
        f.write("-" * 25 + "\n")
        
        neuron_info = []
        for i in range(len(biases)):
            weight = weights[0, i]
            bias = biases[i]
            if abs(weight) > 1e-6:
                activation_temp_scaled = -bias / weight
                activation_temp = scaler_X.inverse_transform([[activation_temp_scaled]])[0][0]
                neuron_info.append({
                    'neuron': i + 1,
                    'weight': weight,
                    'bias': bias,
                    'activation_temp': activation_temp
                })
        
        neuron_info.sort(key=lambda x: x['activation_temp'])
        
        for info in neuron_info:
            f.write(f"Neuron {info['neuron']}:\n")
            f.write(f"  Formula: h{info['neuron']} = ReLU({info['weight']:.4f} * x_scaled + {info['bias']:.4f})\n")
            f.write(f"  Activates when temperature > {info['activation_temp']:.1f}°F\n")
            
            if info['weight'] > 0:
                f.write(f"  Effect: INCREASES ice cream consumption\n")
            else:
                f.write(f"  Effect: DECREASES ice cream consumption\n")
            f.write("\n")
        
        f.write("COLORED SEGMENTS ON GRAPH\n")
        f.write("-" * 25 + "\n")
        
        color_names = ['Orange', 'Yellow-Orange', 'Yellow', 'Green', 'Blue', 'Dark Blue']
        
        for i in range(len(segment_temps) - 1):
            start_temp = segment_temps[i]
            end_temp = segment_temps[i + 1]
            color_name = color_names[i % len(color_names)]
            
            f.write(f"Segment {i+1} ({color_name}): {start_temp:.1f}°F to {end_temp:.1f}°F\n")
            
            # Determine which neurons are active in this segment
            active_neurons = []
            for info in neuron_info:
                if info['activation_temp'] <= start_temp:
                    active_neurons.append(info['neuron'])
            
            if active_neurons:
                f.write(f"  Active neurons: {', '.join(map(str, active_neurons))}\n")
            else:
                f.write(f"  Active neurons: None (baseline only)\n")
            
            # Calculate approximate slope in this segment
            temp_start_idx = np.argmin(np.abs(temp_range - start_temp))
            temp_end_idx = np.argmin(np.abs(temp_range - end_temp))
            
            if temp_end_idx > temp_start_idx:
                temp_scaled = scaler_X.transform(temp_range[temp_start_idx:temp_end_idx+1].reshape(-1, 1))
                nn_pred_scaled = nn_model.predict(temp_scaled)
                nn_predictions = scaler_y.inverse_transform(nn_pred_scaled.reshape(-1, 1)).ravel()
                
                if len(nn_predictions) > 1:
                    slope = (nn_predictions[-1] - nn_predictions[0]) / (temp_range[temp_end_idx] - temp_range[temp_start_idx])
                    f.write(f"  Slope: {slope:.2f} ice creams per °F\n")
            
            f.write("\n")
        
        f.write("HOW TO READ THE COLORED LINE\n")
        f.write("-" * 30 + "\n")
        f.write("Each color represents a different mathematical 'regime':\n")
        f.write("- When temperature crosses an activation point, a new neuron 'turns on'\n")
        f.write("- This changes the slope of the line (piecewise linear function)\n")
        f.write("- More active neurons = steeper slope = faster ice cream growth\n")
        f.write("- The neural network creates 'smart breakpoints' in the data\n")
    
    print(f"Neural network segment analysis saved to '{filename}'")

def save_neural_network_formula(nn_model, scaler_X, scaler_y, filename="neural_network_formula.txt"):
    """Save the complete neural network formula to a text file"""
    
    weights = nn_model.coefs_
    biases = nn_model.intercepts_
    
    with open(filename, 'w') as f:
        f.write("COMPLETE NEURAL NETWORK FORMULA\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Network Architecture: {[1] + list(nn_model.hidden_layer_sizes) + [1]}\n")
        f.write(f"Activation Function: {nn_model.activation}\n\n")
        
        # Input scaling
        f.write("Step 1: Input Scaling\n")
        f.write("-" * 20 + "\n")
        mean_x = scaler_X.mean_[0]
        scale_x = scaler_X.scale_[0]
        f.write(f"x_scaled = (temperature - {mean_x:.4f}) / {scale_x:.4f}\n\n")
        
        # Layer 1
        f.write("Step 2: First Hidden Layer (8 neurons)\n")
        f.write("-" * 35 + "\n")
        w1 = weights[0]
        b1 = biases[0]
        
        for i in range(len(b1)):
            f.write(f"h1_{i+1} = ReLU({w1[0,i]:.6f} * x_scaled + {b1[i]:.6f})\n")
        f.write("\n")
        
        # Layer 2 (if exists)
        if len(weights) > 2:
            f.write("Step 3: Second Hidden Layer (4 neurons)\n")
            f.write("-" * 36 + "\n")
            w2 = weights[1]
            b2 = biases[1]
            
            for i in range(len(b2)):
                terms = [f"{w2[j,i]:.6f}*h1_{j+1}" for j in range(w2.shape[0])]
                f.write(f"h2_{i+1} = ReLU({' + '.join(terms)} + {b2[i]:.6f})\n")
            f.write("\n")
        
        # Final layer
        step_num = 4 if len(weights) > 2 else 3
        f.write(f"Step {step_num}: Output Layer\n")
        f.write("-" * 20 + "\n")
        w_final = weights[-1]
        b_final = biases[-1]
        
        hidden_size = w_final.shape[0]
        layer_name = "h2" if len(weights) > 2 else "h1"
        terms = [f"{w_final[i,0]:.6f}*{layer_name}_{i+1}" for i in range(hidden_size)]
        f.write(f"y_scaled = {' + '.join(terms)} + {b_final[0]:.6f}\n\n")
        
        # Output scaling
        f.write(f"Step {step_num + 1}: Output Scaling\n")
        f.write("-" * 22 + "\n")
        mean_y = scaler_y.mean_[0]
        scale_y = scaler_y.scale_[0]
        f.write(f"y = y_scaled * {scale_y:.6f} + {mean_y:.6f}\n\n")
        
        f.write("Where ReLU(x) = max(0, x)\n")
        f.write("="*50 + "\n")
    
    print(f"Complete neural network formula saved to '{filename}'")

def create_scatter_plot():
    """Create the ice cream scatter plot with trend line"""
    setup_environment()
    setup_xkcd_font()
    
    # Generate data
    temperatures, ice_creams = generate_data()
    
    # Create plot with XKCD style
    with plt.xkcd(**XKCD_PARAMS):
        fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
        fig.patch.set_facecolor(PLOT_CONFIG['facecolor'])
        ax.set_facecolor(PLOT_CONFIG['facecolor'])
        
        # Initialize trend functions storage
        trend_functions = {}  # Store for prediction calculations
        formula_text = []  # Store formula text for display
        
        # Train neural network if enabled
        if NEURAL_NET_CONFIG.get('show_neural_net', False):
            print("Training neural network...")
            nn_model, scaler_X, scaler_y = train_neural_network(temperatures, ice_creams)
            
            # Save the complete formula
            save_neural_network_formula(nn_model, scaler_X, scaler_y)
            
            # Generate predictions for smooth curve
            if TREND_LINE_CONFIG.get('extend_to_border', False):
                temp_range = np.linspace(AXIS_CONFIG['xlim'][0], AXIS_CONFIG['xlim'][1], 100)
            else:
                temp_range = np.linspace(temperatures.min(), temperatures.max(), 100)
            
            # Create colored neural network visualization
            colors = ['#FF6B35', '#F7931E', '#FFD23F', '#06FFA5', '#118AB2', '#073B4C']
            
            # Get full prediction
            temp_scaled = scaler_X.transform(temp_range.reshape(-1, 1))
            nn_pred_scaled = nn_model.predict(temp_scaled)
            nn_predictions = scaler_y.inverse_transform(nn_pred_scaled.reshape(-1, 1)).ravel()
            
            # Analyze activation points
            weights = nn_model.coefs_[0]  # Input to hidden weights
            biases = nn_model.intercepts_[0]  # Hidden biases
            
            activation_temps = []
            for i in range(len(biases)):
                weight = weights[0, i]
                bias = biases[i]
                if abs(weight) > 1e-6:
                    activation_temp_scaled = -bias / weight
                    activation_temp = scaler_X.inverse_transform([[activation_temp_scaled]])[0][0]
                    if temp_range[0] <= activation_temp <= temp_range[-1]:
                        activation_temps.append(activation_temp)
            
            # Create segments
            segment_temps = [temp_range[0]] + sorted(activation_temps) + [temp_range[-1]]
            segment_temps = sorted(list(set(segment_temps)))
            
            # Plot each segment with different colors
            for i in range(len(segment_temps) - 1):
                start_temp = segment_temps[i]
                end_temp = segment_temps[i + 1]
                
                # Find indices for this segment
                start_idx = np.argmin(np.abs(temp_range - start_temp))
                end_idx = np.argmin(np.abs(temp_range - end_temp)) + 1
                
                # Plot this segment
                segment_temps_plot = temp_range[start_idx:end_idx]
                segment_preds = nn_predictions[start_idx:end_idx]
                
                color = colors[i % len(colors)]
                ax.plot(segment_temps_plot, segment_preds,
                       color=color,
                       linewidth=4,
                       alpha=0.9,
                       zorder=4,
                       label=f'Neural Net Segment {i+1}' if i == 0 else "")
            
            # Save segment analysis
            save_neural_network_segments(nn_model, scaler_X, scaler_y, temp_range, segment_temps, colors)
            
            # Store for prediction if needed
            trend_functions['neural_net'] = lambda x: scaler_y.inverse_transform(
                nn_model.predict(scaler_X.transform(np.array(x).reshape(-1, 1))).reshape(-1, 1)
            ).ravel()
        
        # Add trend lines if enabled
        
        # Linear trend line
        if TREND_LINE_CONFIG.get('show_linear', False):
            # Calculate linear regression
            z_linear = np.polyfit(temperatures, ice_creams, 1)  # Linear fit
            p_linear = np.poly1d(z_linear)
            trend_functions['linear'] = p_linear
            
            # Store linear formula
            if FORMULA_CONFIG.get('show_linear_formula', False):
                slope, intercept = z_linear
                formula_text.append(f"Linear: $y = {slope:.2f}x + {intercept:.2f}$")
            
            # Create line range - extend to plot borders if enabled
            if TREND_LINE_CONFIG.get('extend_to_border', False):
                temp_range = np.linspace(AXIS_CONFIG['xlim'][0], AXIS_CONFIG['xlim'][1], 100)
            else:
                temp_range = np.linspace(temperatures.min(), temperatures.max(), 100)
            
            linear_trend = p_linear(temp_range)
            
            # Plot linear trend line
            ax.plot(temp_range, linear_trend,
                   color=TREND_LINE_CONFIG['linear_color'],
                   linewidth=TREND_LINE_CONFIG['linewidth'],
                   alpha=TREND_LINE_CONFIG['alpha'],
                   linestyle=TREND_LINE_CONFIG['linestyle'],
                   zorder=TREND_LINE_CONFIG['zorder'],
                   label='Linear Trend')
        
        # Polynomial trend line
        if TREND_LINE_CONFIG.get('show_polynomial', False):
            # Calculate polynomial regression
            degree = TREND_LINE_CONFIG.get('polynomial_degree', 2)
            z_poly = np.polyfit(temperatures, ice_creams, degree)
            p_poly = np.poly1d(z_poly)
            trend_functions['polynomial'] = p_poly
            
            # Store polynomial formula
            if FORMULA_CONFIG.get('show_polynomial_formula', False):
                if degree == 2:
                    a, b, c = z_poly
                    formula_text.append(f"Polynomial: $y = {a:.4f}x^2 + {b:.2f}x + {c:.2f}$")
                elif degree == 3:
                    a, b, c, d = z_poly
                    formula_text.append(f"Polynomial: $y = {a:.6f}x^3 + {b:.4f}x^2 + {c:.2f}x + {d:.2f}$")
            
            # Create line range
            if TREND_LINE_CONFIG.get('extend_to_border', False):
                temp_range = np.linspace(AXIS_CONFIG['xlim'][0], AXIS_CONFIG['xlim'][1], 100)
            else:
                temp_range = np.linspace(temperatures.min(), temperatures.max(), 100)
            
            poly_trend = p_poly(temp_range)
            
            # Plot polynomial trend line
            ax.plot(temp_range, poly_trend,
                   color=TREND_LINE_CONFIG['polynomial_color'],
                   linewidth=TREND_LINE_CONFIG['linewidth'],
                   alpha=TREND_LINE_CONFIG['alpha'],
                   linestyle=TREND_LINE_CONFIG['linestyle'],
                   zorder=TREND_LINE_CONFIG['zorder'],
                   label=f'Polynomial Trend (degree {degree})')
        
        # Display formulas if enabled
        if FORMULA_CONFIG.get('show_formula', False) and formula_text:
            formula_string = '\n'.join(formula_text)
            ax.text(FORMULA_CONFIG['position_x'], FORMULA_CONFIG['position_y'], 
                   formula_string,
                   transform=ax.transAxes,
                   fontsize=FORMULA_CONFIG['fontsize'],
                   verticalalignment='top',
                   bbox=FORMULA_CONFIG['bbox_props'])
        
        # Add prediction visualization if enabled
        if PREDICTION_CONFIG.get('show_prediction', False):
            pred_temp = PREDICTION_CONFIG['temperature']
            
            # Use neural network prediction if available, otherwise polynomial, otherwise linear
            if 'neural_net' in trend_functions:
                pred_value = trend_functions['neural_net'](pred_temp)[0]
                pred_color = NEURAL_NET_CONFIG['color']
            elif 'polynomial' in trend_functions:
                pred_value = trend_functions['polynomial'](pred_temp)
                pred_color = TREND_LINE_CONFIG['polynomial_color']
            elif 'linear' in trend_functions:
                pred_value = trend_functions['linear'](pred_temp)
                pred_color = TREND_LINE_CONFIG['linear_color']
            else:
                pred_value = 0  # Fallback
                pred_color = PREDICTION_CONFIG['prediction_point_color']
            
            # Draw vertical line from x-axis to prediction point
            ax.plot([pred_temp, pred_temp], [0, pred_value],
                   color=PREDICTION_CONFIG['vertical_line_color'],
                   linestyle=PREDICTION_CONFIG['vertical_line_style'],
                   linewidth=PREDICTION_CONFIG['vertical_line_width'],
                   alpha=0.8,
                   zorder=PREDICTION_CONFIG['zorder'] - 1)
            
            # Draw prediction point
            ax.scatter([pred_temp], [pred_value],
                      s=PREDICTION_CONFIG['prediction_point_size'],
                      c=pred_color,
                      marker=PREDICTION_CONFIG['prediction_point_marker'],
                      edgecolors=PREDICTION_CONFIG['prediction_point_edge_color'],
                      linewidth=PREDICTION_CONFIG['prediction_point_edge_width'],
                      zorder=PREDICTION_CONFIG['zorder'],
                      alpha=0.9)
        
        # Create scatter plot
        ax.scatter(temperatures, ice_creams, 
                  s=SCATTER_STYLE['size'],
                  c=SCATTER_STYLE['color'],
                  alpha=SCATTER_STYLE['alpha'],
                  edgecolors=SCATTER_STYLE['edge_color'],
                  linewidth=SCATTER_STYLE['edge_width'],
                  marker=SCATTER_STYLE['marker'],
                  zorder=SCATTER_STYLE['zorder'],
                  label='Data Points')
        
        # Configure plot
        ax.set_xlabel(PLOT_CONFIG['xlabel'], fontsize=PLOT_CONFIG['label_fontsize'])
        ax.set_ylabel(PLOT_CONFIG['ylabel'], fontsize=PLOT_CONFIG['label_fontsize'])
        ax.set_title(PLOT_CONFIG['title'], fontsize=PLOT_CONFIG['title_fontsize'])
        
        # Set axis limits and grid
        ax.set_xlim(AXIS_CONFIG['xlim'])
        ax.set_ylim(AXIS_CONFIG['ylim'])
        ax.grid(True, 
               alpha=AXIS_CONFIG['grid_alpha'],
               linestyle=AXIS_CONFIG['grid_linestyle'],
               linewidth=AXIS_CONFIG['grid_linewidth'])
        
        # Style spines
        for spine in ax.spines.values():
            spine.set_linewidth(AXIS_CONFIG['spine_linewidth'])
        
        # Add legend if enabled
        if LEGEND_CONFIG.get('show_legend', False):
            legend = ax.legend(
                loc=LEGEND_CONFIG['location'],
                fontsize=LEGEND_CONFIG['fontsize'],
                frameon=LEGEND_CONFIG['frameon'],
                fancybox=LEGEND_CONFIG['fancybox'],
                shadow=LEGEND_CONFIG['shadow'],
                framealpha=LEGEND_CONFIG['framealpha'],
                facecolor=LEGEND_CONFIG['facecolor'],
                edgecolor=LEGEND_CONFIG['edgecolor']
            )
        
        plt.tight_layout()
        
        # Save files
        png_filename = f"{OUTPUT_FILENAME}.png"
        pdf_filename = f"{OUTPUT_FILENAME}.pdf"
        
        plt.savefig(png_filename, 
                   dpi=PLOT_CONFIG['dpi'], 
                   bbox_inches='tight',
                   facecolor=PLOT_CONFIG['facecolor'], 
                   edgecolor='none')
        plt.savefig(pdf_filename, 
                   bbox_inches='tight',
                   facecolor=PLOT_CONFIG['facecolor'], 
                   edgecolor='none')
        
        plt.close()
        
        print(f"Plot saved as '{png_filename}' and '{pdf_filename}'")
    
    # Reset font settings
    plt.rcParams.update(plt.rcParamsDefault)

if __name__ == "__main__":
    print("Creating ice cream scatter plot from configuration...")
    create_scatter_plot()
    print("Done!")
