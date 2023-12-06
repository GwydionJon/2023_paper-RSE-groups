# author: Gwydion Daskalakis <daskalakis@uni-heidelberg.de>
# license: CC0

import numpy as np
from dash import dcc, html, Input, Output, Dash, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import os
import glob
import pathlib
import json


# settings
dash_port = 8015
data_dir = "./submissions/2023-12-06_submissions"


def get_data(data_dir):
    """
    Load the data from the json file
    """
    # load the data
    data = {}
    for filepath in glob.glob(os.path.join(data_dir, "*.json")):
        filepath = pathlib.Path(filepath)
        filename = filepath.stem
        data[str(filename)] = str(filepath)
    return data


def create_layout(json_dicts):
    layout = html.Div(
        id="main",
        style={"margin": "1%"},
        children=[
            dbc.Row(
                html.H3("Select a group to visualize the results:"),
            ),
            dbc.Row(
                [
                    dcc.Dropdown(
                        options=[
                            {"label": k, "value": v} for k, v in json_dicts.items()
                        ],
                        id="dropdown_selection",
                    ),
                ],
                style={"width": "30%"},
            ),
            dbc.Row(id="output_row"),
        ],
    )
    return layout


def create_result_text(
    instition_name, institution_citation, institution_contact, institution_text
):
    text = html.Div(
        [
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Institution name", style={"width": "40%"}),
                    dbc.Input(value=instition_name, readonly=True),
                ]
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Institution citation", style={"width": "40%"}),
                    dbc.Input(value=institution_citation, readonly=True),
                ]
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Institution contact", style={"width": "40%"}),
                    dbc.Input(value=institution_contact, readonly=True),
                ]
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Institution text", style={"width": "40%"}),
                    dbc.Textarea(value=institution_text, readonly=True),
                ]
            ),
        ], style={"width": "30%", "margin-top": "1%"}
    )
    return text


def show_results(json_dict):
    with open(json_dict) as f:
        data = json.load(f)
    
    instition_name = data.get("institution_name", "Nothing written here") 
    institution_citation =  data.get("institution_citation", "Nothing written here") 
    institution_contact =  data.get("institution_contact", "Nothing written here") 
    institution_text =  data.get("institution_text", "Nothing written here") 
    activity_names = data["activity_names"]
    activity_weights = data["activity_weights"]
    activity_widths = data["activity_widths"]
    
    text = create_result_text(instition_name, institution_citation, institution_contact, institution_text)
    
    fig = update_graph(activity_names, activity_weights, activity_widths)
    graph = dcc.Graph(figure=fig, style={"width": "60%", "margin-top": "1%"})
    
    
    return html.Div([text, graph])





# Callback function to draw a new graph based on the input widgets.
def update_graph(names, heights, widths):
    # end points for the different plot regions
    proportions = [0.2, 0.3, 1]

    # norm heights to 1
    heights = heights / np.sum(heights)

    # calculate the total height of each section
    total_height = []
    for i in np.arange(len(heights)):
        total_height.append(np.sum(heights[: i + 1]))

    # create the figure
    fig = go.Figure(
        layout={
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "xaxis": {"title": "x-label", "visible": False, "showticklabels": False},
            "yaxis": {"title": "y-label", "visible": False, "showticklabels": False},
        },
    )

    ## setup the left side
    # this does not change when we change the height widgets.
    y_left = np.arange(1, len(heights) + 1) / len(heights)

    # here we draw the parallel lines based only on the amount of categories to display.
    for y, name in zip(y_left, names):
        fig.add_scatter(
            x=[0, proportions[0]],
            y=[y, y],
            mode="lines",
            line=dict(color="black", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
        # add the names as labels on the left

        fig.add_annotation(
            x=0.1,
            y=(y) - (1 / len(names) / 2),
            text=name,
            align="left",  # doesn't work
            showarrow=False,
            font=dict(size=14),
        )

    # add bounding box on the left
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=proportions[0],
        y1=1,
        line=dict(color="black", width=1),
    )

    ## setup the middle section
    # we transition from the previous lines to the new weighted distribution
    for y_start, y_end, name in zip(y_left, total_height, names):
        fig.add_scatter(
            x=[proportions[0], proportions[1]],
            y=[y_start, y_end],
            mode="lines",
            line=dict(color="black", width=1),
            showlegend=False,
            hoverinfo="skip",
        )

    fig.add_shape(
        type="rect",
        x0=proportions[0],
        y0=0,
        x1=proportions[1],
        y1=1,
        line=dict(color="black", width=1),
    )

    ## setup the right side

    # bounding box
    fig.add_shape(
        type="rect",
        x0=proportions[1],
        y0=0,
        x1=proportions[2],
        y1=1,
        line=dict(color="black", width=1),
    )

    # here we need to do a split according to the set widths
    x_length = proportions[2] - proportions[1]

    y_prev = 0

    # we draw two boxes, each based on the weighted distribution.
    # one for the amount of centralized and one for decentralized percentage.
    for y, x_split, name in zip(total_height, widths, names):
        # add pre split region
        fig.add_scatter(
            # we basically draw a rectangle including the starting point as end point and fill in the center
            x=[
                proportions[1],
                proportions[1],
                proportions[1] + x_split * x_length,
                proportions[1] + x_split * x_length,
                proportions[1],
            ],
            y=[y, y_prev, y_prev, y, y],
            mode="lines",
            fill="toself",
            line=dict(color="black", width=1),
            fillpattern=go.scatter.Fillpattern(
                shape="/", bgcolor="white", fillmode="overlay"
            ),
            showlegend=False,
            hoverinfo="skip",
        )
        # add post split region
        fig.add_scatter(
            # we basically draw a rectangle including the starting point as end point and fill in the center
            x=[
                proportions[1] + x_split * x_length,
                proportions[1] + x_split * x_length,
                proportions[2],
                proportions[2],
                proportions[1] + x_split * x_length,
            ],
            y=[y, y_prev, y_prev, y, y],
            mode="lines",
            fill="toself",
            line=dict(color="black", width=1),
            fillpattern=go.scatter.Fillpattern(
                shape="\\", bgcolor="white", fillmode="overlay"
            ),
            showlegend=False,
            hoverinfo="skip",
        )

        # set the y_prev values for the next iteration
        y_prev = y

    return fig

if __name__ == "__main__":
    json_dicts = get_data(data_dir)

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = create_layout(json_dicts)

    app.callback(
        Output("output_row", "children"),
        [Input("dropdown_selection", "value")],
        prevent_initial_call=True,
    )(show_results)

    app.run_server(debug=True, port=dash_port)
