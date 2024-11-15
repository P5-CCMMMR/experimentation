import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    fig.add_trace(go.Scatter(x=timestamps, y=mean_predictions, mode='lines', name='Prediction'))
    
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

    fig.add_trace(go.Scatter(
        x=timestamps[::horizon_len], 
        y=mean_predictions[::horizon_len], 
        mode='markers', 
        name='Prediction Start Points', 
        marker=dict(size=4, symbol='circle', line_width=1)
    ))

    fig.add_trace(go.Scatter(x=timestamps, y=actuals, mode='lines', name='Actual'))

    fig.update_layout(
        title="Predictions vs Actuals with Uncertainty",
        xaxis_title="Time",
        yaxis_title="Indoor Temperature",
        legend_title="Legend",
        hovermode="x",
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey')
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
        hovermode="x"
    )

    fig.show()

def plot_flex_probabilities(probabilities, confidence):
    fig = make_subplots()
    
    fig.add_trace(go.Scatter(x=list(range(len(probabilities))), y=probabilities, mode='lines+markers', name='Probability within bounds'))
    fig.add_trace(go.Scatter(x=list(range(len(probabilities))), y=[confidence]*len(probabilities), mode='lines', name=f'Confidence Level ({confidence})', line=dict(dash='dash', color='red')))

    fig.update_layout(
        title="Probability of Forecast Falling Within Bounds",
        xaxis_title="Time Step",
        yaxis_title="Probability",
        legend_title="Legend",
        hovermode="x"
    )

    fig.show()