import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOT_WIDTH = 800
PLOT_HEIGHT = 600


def plot_results(predictions, actuals, timestamps, horizon_len):
    if isinstance(predictions, tuple):
        plot_probabilistic_results(predictions, actuals, timestamps, horizon_len)
    else:
        plot_deterministic_results(predictions, actuals, timestamps, horizon_len)

def plot_probabilistic_results(predictions, actuals, timestamps, horizon_len):
    mean_predictions, std_predictions = predictions
    mean_predictions = np.array(mean_predictions)
    std_predictions = np.array(std_predictions)
    actuals = np.array(actuals)
    timestamps = timestamps[:len(mean_predictions)]

    fig = make_subplots()
    
    
    fig.add_trace(go.Scatter(
        x=timestamps, 
        y=mean_predictions + 3 * std_predictions, 
        fill=None, 
        mode='lines', 
        line_color='gray', 
        opacity=0.3, 
        name='99.7% (3σ)'
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, 
        y=mean_predictions - 3 * std_predictions, 
        fill='tonexty', 
        mode='lines', 
        line_color='gray', 
        opacity=0.3, 
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps, 
        y=mean_predictions + 2 * std_predictions, 
        fill=None, 
        mode='lines', 
        line_color='gray', 
        opacity=0.5, 
        name='95% (2σ)'
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, 
        y=mean_predictions - 2 * std_predictions, 
        fill='tonexty', 
        mode='lines', 
        line_color='gray', 
        opacity=0.5, 
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps, 
        y=mean_predictions + std_predictions, 
        fill=None, 
        mode='lines', 
        line_color='gray', 
        opacity=0.8, 
        name='68% (σ)'
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, 
        y=mean_predictions - std_predictions, 
        fill='tonexty', 
        mode='lines', 
        line_color='gray', 
        opacity=0.8, 
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(line_color='blue', x=timestamps, y=mean_predictions, mode='lines', name='Prediction'))

    fig.add_trace(go.Scatter(line_color='red', x=timestamps, y=actuals, mode='lines', name='Actual'))

    fig.add_trace(go.Scatter(
        x=timestamps[::horizon_len], 
        y=mean_predictions[::horizon_len], 
        mode='markers', 
        name='Prediction Start Points', 
        marker=dict(size=4, symbol='circle', line_width=1, color='lightgreen')
    ))


    fig.update_layout(
        title="Predictions vs Actuals with Uncertainty",
        xaxis_title="Time",
        yaxis_title="Indoor Temperature",
        legend_title="Legend",
        hovermode="x",
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT
    )

    fig.show()

def plot_deterministic_results(predictions, actuals, timestamps, horizon_len):
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    timestamps = timestamps[:len(predictions)]

    fig = make_subplots()
    
    fig.add_trace(go.Scatter(x=timestamps, y=predictions, mode='lines', name='Prediction'))
    fig.add_trace(go.Scatter(
        x=timestamps[::horizon_len], 
        y=predictions[::horizon_len], 
        mode='markers', 
        name='Prediction Start Points', 
        marker=dict(size=4, symbol='circle', line_width=1)
    ))
    fig.add_trace(go.Scatter(x=timestamps, y=actuals, mode='lines', name='Actual'))

    fig.update_layout(
        title="Predictions vs Actuals",
        xaxis_title="Time",
        yaxis_title="Indoor Temperature",
        legend_title="Legend",
        hovermode="x",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT
    )

    fig.show()

def plot_comparative_results(predictions1, predictions2, actuals, timestamps, horizon_len, width=800, height=600):
    predictions1 = np.array(predictions1)
    predictions2 = np.array(predictions2)
    actuals = np.array(actuals)
    timestamps = timestamps[:len(predictions1)]
    fig = make_subplots()
    
    fig.add_trace(go.Scatter(x=timestamps, y=predictions1, mode='lines', name='Copy Strategy', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=timestamps, y=predictions2, mode='lines', name='Lag Strategy', line=dict(color='blue')))
    fig.add_trace(go.Scatter(
        x=timestamps[::horizon_len], 
        y=predictions1[::horizon_len], 
        mode='markers', 
        name='Prediction 1 Start Points', 
        marker=dict(size=4, symbol='circle', line=dict(width=1, color='darkgreen'))
    ))
    fig.add_trace(go.Scatter(
        x=timestamps[::horizon_len], 
        y=predictions2[::horizon_len], 
        mode='markers', 
        name='Prediction 2 Start Points', 
        marker=dict(size=4, symbol='circle', line=dict(width=1, color='darkblue'))
    ))
    fig.add_trace(go.Scatter(x=timestamps, y=actuals, mode='lines', name='Actual', line=dict(color='red')))
    fig.update_layout(
        title="Comparative Predictions vs Actuals",
        xaxis_title="Time",
        yaxis_title="Indoor Temperature",
        legend_title="Legend",
        hovermode="x",
        width=width,
        height=height,
    )
    fig.show()

def plot_flex_probabilities(flex_probabilities, confidence):
    fig = make_subplots()

    for i, probabilities in enumerate(flex_probabilities):
        probabilities.insert(0, 1)
        fig.add_trace(go.Scatter(
            x=list(range(len(probabilities))),
            y=probabilities,
            mode='lines+markers',
            name=f'Forecast {i + 1} Probability'
        ))
    
    fig.add_trace(go.Scatter(x=list(range(len(probabilities))), y=[confidence]*len(probabilities), mode='lines', name=f'Confidence Level ({confidence})', line=dict(dash='dash', color='red')))
    
    fig.update_layout(
        title="Probability of Forecast Falling Within Bounds",
        xaxis_title="Predicted Flexibility",
        yaxis_title="Probability",
        legend_title="Legend",
        hovermode="x",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT
    )   
    
    fig.show()

def plot_loss(loss):
    epochs = np.arange(1, len(loss) + 1)
    loss = np.array(loss)

    fig = make_subplots()
        
    fig.add_trace(go.Scatter(x=epochs, y=loss, mode='lines+markers', name='Loss'))

    fig.update_layout(
        title="Training Loss Over Epochs",
        xaxis_title="Epochs",
        yaxis_title="Loss",
        legend_title="Legend",
        hovermode="x",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT
    )

    fig.show()


def create_pillar_plot(keys, *dictionaries, title='Comparison of Values', x_title='Categories', y_title='Values'):
    """
    Create a Plotly bar plot comparing values across multiple dictionaries.
    
    Parameters:
    - keys: List of keys to be used for comparison
    - dictionaries: Unlimited number of dictionaries to compare
    - title: Optional title for the plot
    - x_title: Optional x-axis title
    - y_title: Optional y-axis title
    
    Returns:
    - Plotly figure object
    """
    # Validate input
    if not keys:
        raise ValueError("At least one key must be provided")
    if not dictionaries:
        raise ValueError("At least one dictionary must be provided")
    
    # Prepare data for plotting
    traces = []
    for i, dictionary in enumerate(dictionaries):
        # Extract values for specified keys, defaulting to 0 if key not found
        values = [dictionary.get(key, 0) for key in keys]
        
        # Create a bar trace for each dictionary
        trace = go.Bar(
            x=keys,
            y=values,
            name=f'Dictionary {i+1}'
        )
        traces.append(trace)
    
    # Create the layout
    layout = go.Layout(
        title=title,
        xaxis=dict(title=x_title),
        yaxis=dict(title=y_title),
        barmode='group'  # This allows side-by-side comparison
    )
    
    # Create figure and return
    fig = go.Figure(data=traces, layout=layout)
    return fig