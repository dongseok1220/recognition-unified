import pandas as pd
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def frame2time(unified, fps):
    seconds = unified['idx'] / fps
    return seconds

def sec2min(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{int(minutes):02d}:{int(seconds):02d}"

def plot(video_name):
    csv_path = f'./data/{video_name}.csv'
    video_path = f'./videos/{video_name}.mp4'

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    unified = pd.read_csv(csv_path)
    unified['Time'] = frame2time(unified, fps)
    unified['MinSec'] = unified['Time'].apply(lambda x: sec2min(x))

    activities = ['handclap', 'headbanging', 'sitstand']
    colors = ['red', 'green', 'blue']

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=activities)

    for i, (activity, color) in enumerate(zip(activities, colors), start=1):
        activity_data = unified[unified[activity] == True]
        times = activity_data['Time']
        min_secs = activity_data['MinSec']

        for time, min_sec in zip(times, min_secs):
            fig.add_trace(go.Scatter(x=[time, time], y=[0, 1], mode="lines",
                                     line=dict(color=color, width=2),
                                     showlegend=False, hoverinfo='text',
                                     text=f"{min_sec}", name=activity),
                          row=i, col=1)

    fig.update_layout(
        title_text="Activity Timeline",
        autosize=True,  # Enable autosize to make the plot responsive
        template="plotly_white",
    )
    fig.update_yaxes(tickvals=[], showgrid=False)

    fig.show()
