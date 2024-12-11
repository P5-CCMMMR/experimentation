import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOT_WIDTH = 800
PLOT_HEIGHT = 600


colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

def plot_results(predictions, actuals, timestamps, horizon_len, time_interval=None, temp_interval=None, titles=None):
    if isinstance(predictions, tuple):
        plot_probabilistic_results(predictions, actuals, timestamps, horizon_len, time_interval, temp_interval, titles)
    else:
        plot_deterministic_results(predictions, actuals, timestamps, horizon_len, time_interval, temp_interval, titles)

def plot_deterministic_results(predictions, actuals, timestamps, horizon_len, time_interval=None, temp_interval=None, titles=None):
    if not isinstance(predictions, list):
        predictions = [predictions]

    fig = make_subplots(rows=len(predictions), cols=1, shared_xaxes=True, vertical_spacing=0.02)

    for i, prediction in enumerate(predictions):
        prediction = np.array(prediction)
        actuals = np.array(actuals)
        timestamps = timestamps[:len(prediction)]

        if time_interval:
            timestamps = timestamps[time_interval[0]:time_interval[1]]
            prediction = prediction[time_interval[0]:time_interval[1]]
            actuals = actuals[time_interval[0]:time_interval[1]]

        if temp_interval:
            prediction = prediction + temp_interval * i

        fig.add_trace(go.Scatter(x=timestamps, y=prediction, mode='lines', name=titles[i] if titles else f'Prediction {i+1}', line=dict(color=colors[i % len(colors)])), row=i+1, col=1)
        fig.add_trace(go.Scatter(
            x=timestamps[::horizon_len], 
            y=prediction[::horizon_len], 
            mode='markers', 
            name=f'Prediction {i+1} Start Points', 
            marker=dict(size=4, symbol='circle', line_width=1, color=colors[i % len(colors)])
        ), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=timestamps, y=actuals, mode='lines', name='Actual', line=dict(color='red')), row=i+1, col=1)

    fig.update_layout(
        title="Predictions vs Actuals",
        xaxis_title="Time",
        yaxis_title="Indoor Temperature",
        legend_title="Legend",
        hovermode="x",
        height=300 * len(predictions)
    )

    fig.show()

def plot_boxplots(predictions, titles=None):
    if not isinstance(predictions, list):
        predictions = [predictions]

    fig = make_subplots(rows=1, cols=len(predictions), subplot_titles=titles)

    for i, prediction in enumerate(predictions):
        prediction = np.array(prediction)
        fig.add_trace(go.Box(y=prediction, name=titles[i] if titles else f'Prediction {i+1}', marker_color=colors[i % len(colors)]), row=1, col=i+1)

    fig.update_layout(
        title="Boxplots of Predictions",
        yaxis_title="Values",
        legend_title="Legend",
        hovermode="x"
    )

    fig.show()

def plot_probabilistic_results(predictions, actuals, timestamps, horizon_len, time_interval=None, temp_interval=None, titles=None):
    if not isinstance(predictions, list):
        predictions = [predictions]

    fig = make_subplots(rows=len(predictions), cols=1, shared_xaxes=True, vertical_spacing=0.02)

    for i, (mean_predictions, std_predictions) in enumerate(predictions):
        mean_predictions = np.array(mean_predictions)
        std_predictions = np.array(std_predictions)
        actuals = np.array(actuals)
        timestamps = timestamps[:len(mean_predictions)]

        if time_interval:
            timestamps = timestamps[time_interval[0]:time_interval[1]]
            mean_predictions = mean_predictions[time_interval[0]:time_interval[1]]
            std_predictions = std_predictions[time_interval[0]:time_interval[1]]
            actuals = actuals[time_interval[0]:time_interval[1]]

        if temp_interval:
            mean_predictions = mean_predictions + temp_interval * i

        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=mean_predictions + 3 * std_predictions, 
            fill=None, 
            mode='lines', 
            line_color='gray', 
            opacity=0.3, 
            name=f'99.7% (3σ) {i+1}'
        ), row=i+1, col=1)
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=mean_predictions - 3 * std_predictions, 
            fill='tonexty', 
            mode='lines', 
            line_color='gray', 
            opacity=0.3, 
            showlegend=False
        ), row=i+1, col=1)
        
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=mean_predictions + 2 * std_predictions, 
            fill=None, 
            mode='lines', 
            line_color='gray', 
            opacity=0.5, 
            name=f'95% (2σ) {i+1}'
        ), row=i+1, col=1)
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=mean_predictions - 2 * std_predictions, 
            fill='tonexty', 
            mode='lines', 
            line_color='gray', 
            opacity=0.5, 
            showlegend=False
        ), row=i+1, col=1)
        
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=mean_predictions + std_predictions, 
            fill=None, 
            mode='lines', 
            line_color='gray', 
            opacity=0.8, 
            name=f'68% (σ) {i+1}'
        ), row=i+1, col=1)
        fig.add_trace(go.Scatter(
            x=timestamps, 
            y=mean_predictions - std_predictions, 
            fill='tonexty', 
            mode='lines', 
            line_color='gray', 
            opacity=0.8, 
            showlegend=False
        ), row=i+1, col=1)
        
        fig.add_trace(go.Scatter(line_color=colors[i % len(colors)], x=timestamps, y=mean_predictions, mode='lines', name=titles[i] if titles else f'Prediction {i+1}'), row=i+1, col=1)

        fig.add_trace(go.Scatter(
            x=timestamps[::horizon_len], 
            y=mean_predictions[::horizon_len], 
            mode='markers', 
            name=f'Prediction {i+1} Start Points', 
            marker=dict(size=4, symbol='circle', line_width=1, color=colors[i % len(colors)])
        ), row=i+1, col=1)

        fig.add_trace(go.Scatter(line_color='red', x=timestamps, y=actuals, mode='lines', name='Actual'), row=i+1, col=1)

    fig.update_layout(
        title="Predictions vs Actuals with Uncertainty",
        xaxis_title="Time",
        yaxis_title="Indoor Temperature",
        legend_title="Legend",
        hovermode="x",
        height=300 * len(predictions)
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
            name=f'Forecast {i + 1} Probability',
            line=dict(color=colors[i % len(colors)])
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

def plot_loss(loss_arr, path, titles=None):
    if not isinstance(loss_arr[0], list):
        loss_arr = [loss_arr]

    max_epoch = 0
    for loss in loss_arr:
        max_epoch = max(max_epoch, len(loss))
    epochs = np.arange(1, max_epoch + 1)

    fig = make_subplots()

    for i, loss in enumerate(loss_arr):
        loss = np.array(loss)
        fig.add_trace(go.Scatter(x=epochs, y=loss, mode='lines', 
                                 name=titles[i] if titles[i] != None else "loss"))

    fig.update_layout(
        title="Loss Over Epochs",
        xaxis_title="Epochs",
        yaxis_title="Loss",
        legend_title="Legend",
        hovermode="x",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT
    )

    fig.show()
    fig.write_image(path + ".png")

def plot_pillar_diagrams(keys, dicts_array, group_names=None, y_max=1.5):
    if group_names is None:
        group_names = [f'Group {i+1}' for i in range(len(dicts_array))]
    
    fig = make_subplots(rows=1, cols=len(dicts_array), subplot_titles=group_names)

    for i, dicts in enumerate(dicts_array):
        for j, d in enumerate(dicts):
            x = []
            y = []
            
            for key in keys:
                x.append(key)
                y.append(round(d[key], 4))

            fig.add_trace(
                    go.Bar(
                        x=x,
                        y=y,
                        text=y,
                        textposition='auto',
                        name=d["title"],
                        marker_color=colors[j % len(colors)],
                        width=0.25
                    ),
                    row=1,
                    col=i+1
                )
    
    fig.update_layout(
        title="Pillar Diagrams",
        xaxis_title="Metrics",
        yaxis_title="Values",
        legend_title="Legend",
        barmode='group'
    )

    if y_max is not None:
        fig.update_yaxes(range=[0, y_max])

    fig.show()
    
def plot_metric_comparison(keys, dicts_array, titles=None):
    if titles is None:
        titles = [f'Dictionary {i+1}' for i in range(len(dicts_array))]

    fig = make_subplots()

    for i, d in enumerate(dicts_array):
        x = d[keys[0]]
        print(f"x {x}")
        y = d[keys[1]]
        print(f"y {y}")
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers+text',
            name=titles[i],
            text=titles[i],
            textposition='top center',
            marker=dict(size=10, color=colors[i % len(colors)])
        ))

    fig.update_layout(
        title="Metric Comparison",
        xaxis_title=keys[0],
        yaxis_title=keys[1],
        legend_title="Legend",
        hovermode="closest"
    )

    fig.show()