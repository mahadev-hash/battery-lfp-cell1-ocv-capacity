import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ================= USER INPUT =================
CSV_PATH = "LFP-1.csv"
DV = 0.005
ICA_RANGE = [-200, 200]

# ================= FUNCTIONS =================
def interpolate_and_dqdv(V, Q, dv):
    idx = np.argsort(V)
    V, Q = V[idx], Q[idx]
    Vn = np.arange(V.min(), V.max(), dv)
    Qi = np.interp(Vn, V, Q)
    return Vn, np.gradient(Qi, Vn)

# ================= LOAD DATA =================
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

n_cells = len(df.columns) // 4
cell_names = [f"Cell-{i+1}" for i in range(n_cells)]
colors = px.colors.qualitative.Dark24

# ================= DASH APP =================
app = Dash(__name__)
server = app.server
app.title = "Battery Analysis Dashboard"

# ================= LAYOUT =================
app.layout = html.Div([
    html.H2("Battery Analysis Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Div([
            html.Label("Cell A"),
            dcc.Dropdown(
                options=[{"label": c, "value": i} for i, c in enumerate(cell_names)],
                value=0,
                id="cell-a"
            )
        ], style={"width": "30%"}),

        html.Div([
            html.Label("Cell B (Compare)"),
            dcc.Dropdown(
                options=[{"label": c, "value": i} for i, c in enumerate(cell_names)],
                value=1,
                id="cell-b"
            )
        ], style={"width": "30%"}),

        html.Div([
            html.Label("Mode"),
            dcc.Dropdown(
                options=[
                    {"label": "Single Cell", "value": "single"},
                    {"label": "Compare Cells", "value": "compare"}
                ],
                value="single",
                id="mode"
            )
        ], style={"width": "30%"})
    ], style={"display": "flex", "gap": "20px"}),

    dcc.Graph(id="battery-graph", style={"height": "650px"})
], style={"margin": "20px"})

# ================= CALLBACK =================
@app.callback(
    Output("battery-graph", "figure"),
    Input("cell-a", "value"),
    Input("cell-b", "value"),
    Input("mode", "value")
)
def update_graph(cell_a, cell_b, mode):

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        subplot_titles=["Voltage vs Capacity", "dQ/dE vs Voltage"]
    )

    def get_cell_data(cell_idx):

        c = 4 * cell_idx

        Vc = df.iloc[:, c].dropna().to_numpy()
        Qc = df.iloc[:, c+1].dropna().to_numpy()
        Vd = df.iloc[:, c+2].dropna().to_numpy()
        Qd = df.iloc[:, c+3].dropna().to_numpy()

        Qc = -(Qc.max() - Qc)
        Qd = -Qd
        Qd, Vd = Qd[::-1], Vd[::-1]

        Vci, dQci = interpolate_and_dqdv(Vc, Qc, DV)
        Vdi, dQdi = interpolate_and_dqdv(Vd, Qd[::-1], DV)

        return Qc, Vc, Qd, Vd, dQci, Vci, dQdi, Vdi

    # ===== Background cells =====
    for i in range(n_cells):

        Qc, Vc, Qd, Vd, dQci, Vci, dQdi, Vdi = get_cell_data(i)
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=Qc, y=Vc,
            line=dict(color=color, width=2),
            opacity=0.2,
            showlegend=False
        ), 1, 1)

        fig.add_trace(go.Scatter(
            x=Qd, y=Vd,
            line=dict(color=color, width=2),
            opacity=0.2,
            showlegend=False
        ), 1, 1)

        fig.add_trace(go.Scatter(
            x=dQci, y=Vci,
            line=dict(color=color, width=2),
            opacity=0.2,
            showlegend=False
        ), 1, 2)

        fig.add_trace(go.Scatter(
            x=dQdi, y=Vdi,
            line=dict(color=color, width=2),
            opacity=0.2,
            showlegend=False
        ), 1, 2)

    # ===== Highlight selected cells =====
    def highlight_cell(cell_idx, label):

        Qc, Vc, Qd, Vd, dQci, Vci, dQdi, Vdi = get_cell_data(cell_idx)
        color = colors[cell_idx % len(colors)]

        fig.add_trace(go.Scatter(
            x=Qc, y=Vc,
            name=f"{label} Charge",
            line=dict(color=color, width=2.5)
        ), 1, 1)

        fig.add_trace(go.Scatter(
            x=Qd, y=Vd,
            name=f"{label} Discharge",
            line=dict(color=color, width=2.5)
        ), 1, 1)

        fig.add_trace(go.Scatter(
            x=dQci, y=Vci,
            showlegend=False,
            line=dict(color=color, width=2.5)
        ), 1, 2)

        fig.add_trace(go.Scatter(
            x=dQdi, y=Vdi,
            showlegend=False,
            line=dict(color=color, width=2.5)
        ), 1, 2)

    highlight_cell(cell_a, cell_names[cell_a])

    if mode == "compare" and cell_b != cell_a:
        highlight_cell(cell_b, cell_names[cell_b])

    fig.update_layout(
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", y=-0.2)
    )

    fig.update_xaxes(title="Capacity (mAh)", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title="dQ/dE (mAh/V)", range=ICA_RANGE, row=1, col=2)
    fig.update_yaxes(title="Voltage (V)")

    return fig

# ================= RUN =================
if __name__ == "__main__":

    app.run()
