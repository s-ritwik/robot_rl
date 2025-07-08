#!/usr/bin/env python3
"""
Creates a final, clean, and readable interactive parallel coordinates plot
by manually constructing it with Plotly Scatter traces to allow for full
aesthetic control, including line width and a consistent color scheme.
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from plotly.colors import sample_colorscale


def create_manual_parcoords_plot(input_filepath: str, df: pd.DataFrame):
    """
    Manually builds an interactive parallel coordinates plot to gain full
    control over aesthetics like line width and a dynamic color scheme.

    Args:
        input_filepath (str): Path to the original CSV file for naming the output.
        df (pd.DataFrame): The DataFrame containing the policy data.
    """
    # --- 1. Parse Hyperparameters ---
    print("⚙️  Parsing and normalizing hyperparameter values...")
    params_to_extract = {
        'CLF Dec. Weight': r"clf_decreasing_condition_weight_(-?\d+p?\d*)",
        'CLF Reward': r"clf_reward_weight_(-?\d+p?\d*)",
        'Holo. Vel. Weight': r"holonomic_constraint_vel_weight_(-?\d+p?\d*)",
        'Holo. Pos. Weight': r"holonomic_constraint_(?!vel_weight_)(-?\d+p?\d*)",
    }
    
    dims = list(params_to_extract.keys())
    parsed_df = pd.DataFrame(index=df.index)

    for dim_name, pattern in params_to_extract.items():
        extracted_series = df['Policy'].str.extract(pattern, expand=False)
        parsed_df[dim_name] = pd.to_numeric(extracted_series.str.replace('p', '.'), errors='coerce')

    # --- 2. Clean Data and Synchronize DataFrames ---
    failed_rows = parsed_df.isnull().any(axis=1)
    if failed_rows.any():
        num_failed = failed_rows.sum()
        print(f"⚠️  Warning: Could not parse hyperparameters for {num_failed} policies. They will be excluded.")
        df = df[~failed_rows].reset_index(drop=True)
        parsed_df = parsed_df[~failed_rows].reset_index(drop=True)

    if df.empty:
        print("❌ Error: No policies could be parsed successfully. Aborting.")
        return

    # --- 3. Normalize Data for Plotting ---
    print("⚙️  Remapping data to visually spaced axes...")
    remapped_df = pd.DataFrame(index=parsed_df.index, columns=dims)
    for dim_name in dims:
        min_val = parsed_df[dim_name].min()
        max_val = parsed_df[dim_name].max()
        original_values = parsed_df[dim_name]
        unique_vals = sorted(parsed_df[dim_name].unique())
        if len(unique_vals) <= 5:
            tick_texts = unique_vals
        else:
            tick_texts = np.linspace(min_val, max_val, num=5)
        num_ticks = len(tick_texts)
        if num_ticks == 1:
            tick_ys = [0.5]
        else:
            tick_ys = np.linspace(0, 1, num=num_ticks)
        remapped_df[dim_name] = np.interp(original_values, tick_texts, tick_ys)
    normalized_df = remapped_df
    
    # --- 4. Generate Plot with Dynamic Colors ---
    print("🎨 Generating interactive plot with dynamic colors...")
    master_colorscale = 'Plasma_r'
    metrics = {
        "Position RMSE": 'Average Position RMSE',
        "Velocity RMSE": 'Average Velocity RMSE',
        "Fall Rate": 'Fall Rate'
    }

    def get_dynamic_colors(metric_series, colorscale):
        metric_scaler = MinMaxScaler(feature_range=(0, 1))
        norm_values = metric_scaler.fit_transform(
            metric_series.values.reshape(-1, 1)
        ).flatten()
        norm_values = np.clip(norm_values, 0, 0.999999)
        return sample_colorscale(colorscale, norm_values.tolist())

    initial_metric_label = "Position RMSE"
    initial_metric_col = metrics[initial_metric_label]
    line_colors = get_dynamic_colors(df[initial_metric_col], master_colorscale)
    fig = go.Figure()

    for i in range(len(df)):
        fig.add_trace(go.Scatter(
            x=dims,
            y=normalized_df.iloc[i].values,
            mode='lines',
            line=dict(width=4, color=line_colors[i]),
            customdata=parsed_df.iloc[i].values,
            hovertemplate='%{x}: %{customdata:.3f}<extra></extra>'
        ))

    # --- 5. Add Invisible Trace for the Color Bar ---
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(
            colorscale=master_colorscale,
            showscale=True,
            color=df[initial_metric_col].values,
            cmin=df[initial_metric_col].min(),
            cmax=df[initial_metric_col].max(),
            colorbar=dict(title=initial_metric_label, thickness=15)
        ),
        hoverinfo='none'
    ))

    # --- 6. Manually Create Axes and Labels ---
    for i, dim_name in enumerate(dims):
        fig.add_shape(type="line", x0=i, y0=-0.05, x1=i, y1=1.05, line=dict(color="grey", width=1))
        fig.add_annotation(x=i, y=1.05, text=dim_name, showarrow=False, font=dict(size=14), yshift=10)
        min_val = parsed_df[dim_name].min()
        max_val = parsed_df[dim_name].max()
        unique_vals = sorted(parsed_df[dim_name].unique())
        if len(unique_vals) <= 5:
            tick_values = unique_vals
        else:
            tick_values = np.linspace(min_val, max_val, num=5)
        num_ticks = len(tick_values)
        if num_ticks == 1:
            y_positions = [0.5]
        else:
            y_positions = np.linspace(0, 1, num=num_ticks)
        for tick_val, y_pos in zip(tick_values, y_positions):
            fig.add_annotation(
                x=i, y=y_pos, text=f"{tick_val:g}",
                showarrow=False, xshift=-10, xanchor="right", font=dict(size=10)
            )

    # --- 7. Configure the Dropdown Menu for Dynamic Updates ---
    buttons = []
    for label, metric_col in metrics.items():
        new_line_colors = get_dynamic_colors(df[metric_col], master_colorscale)
        num_line_traces = len(df)
        line_color_update = new_line_colors + [None] 
        marker_color_update = [None] * num_line_traces + [df[metric_col].values]
        marker_cmin_update = [None] * num_line_traces + [df[metric_col].min()]
        marker_cmax_update = [None] * num_line_traces + [df[metric_col].max()]
        colorbar_title_update = [None] * num_line_traces + [label]
        buttons.append(dict(
            label=f"Color by {label}",
            method="update",
            args=[{
                'line.color': line_color_update,
                'marker.color': marker_color_update,
                'marker.cmin': marker_cmin_update,
                'marker.cmax': marker_cmax_update,
                'marker.colorbar.title.text': colorbar_title_update
            }]
        ))
    fig.update_layout(updatemenus=[dict(active=0, buttons=buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.01, xanchor="left", y=1.2, yanchor="top")])

    # --- 8. Finalize Layout and Style ---
    fig.update_layout(
        title_text='Top 10% Policies: Analyze by Performance Metric',
        template='plotly_dark',
        height=700,
        margin=dict(l=80, r=80, b=80, t=180),
        xaxis=dict(tickfont=dict(size=12), showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        showlegend=False,
        hovermode='x',
        clickmode='event+select',    # ➜ enable click events
    )

    # ------------------------------------------------------------------
    # ✨  Add click-to-highlight behaviour (raw JS, no <script> tag)  ✨
    # ------------------------------------------------------------------
    num_line_traces = len(df)              # exclude the dummy colour-bar trace
    highlight_js = f"""
    document.addEventListener('DOMContentLoaded', () => {{
        const gd = document.querySelector('.js-plotly-plot');
        const n  = {num_line_traces};

        gd.on('plotly_click', ev => {{
            const k = ev.points[0].curveNumber;
            if (k >= n) return;                       // skip colour-bar trace
            for (let i = 0; i < n; i++) {{
                const on = (i === k);
                Plotly.restyle(gd, {{opacity: on ? 1   : 0.15}}, [i]);
                Plotly.restyle(gd, {{'line.width': on ? 6 : 2}},   [i]);
            }}
        }});

        gd.on('plotly_doubleclick', () => {{
            for (let i = 0; i < n; i++) {{
                Plotly.restyle(gd, {{opacity: 1}}, [i]);
                Plotly.restyle(gd, {{'line.width': 4}}, [i]);
            }}
        }});
    }});
    """

    # --- 9. Save to HTML & Show ---
    base_filename = os.path.splitext(os.path.basename(input_filepath))[0]
    output_path = os.path.join(os.path.dirname(input_filepath),
                                 f"{base_filename}_interactive_plot.html")

    fig.write_html(
        output_path,
        include_plotlyjs='cdn',
        full_html=True,
        post_script=highlight_js   # inject the JS
    )
    print(f"✔️ Interactive plot saved to: {output_path}")


    # We are removing fig.show() to prevent the script from hanging or opening
    # a browser window unnecessarily, as the primary output is the saved file.
    # fig.show()


def main():
    """Main function to read CSV and generate plot."""
    parser = argparse.ArgumentParser(description="Visualize policy hyperparameters from a CSV file.")
    parser.add_argument("input_filepath", help="Path to the input CSV file.")
    args = parser.parse_args()

    try:
        # Use the filepath from the command-line arguments
        df = pd.read_csv(args.input_filepath)
        create_manual_parcoords_plot(args.input_filepath, df)
    except FileNotFoundError:
        print(f"❌ ERROR: The file was not found at: {args.input_filepath}")
    except Exception as e:
        print(f"❌ ERROR during script execution. Reason: {e}")


if __name__ == "__main__":
    main()