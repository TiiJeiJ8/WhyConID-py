"""
3D trajectory visualization tools.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def export_trajectories_3d_html(trajectories_3d: Dict[int, List[Tuple[float, float, float]]],
                                output_path: str,
                                title: str = "3D Trajectory Visualization"):
    """
    Export 3D trajectories as interactive HTML using Plotly.
    
    Args:
        trajectories_3d: Dict mapping track_id to list of (X, Y, Z) world coordinates
        output_path: Output HTML file path
        title: Visualization title
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Warning: plotly not installed. Install with: pip install plotly")
        print("Skipping HTML 3D visualization export.")
        return
    
    fig = go.Figure()
    
    # Define color palette
    colors = [
        'rgb(31, 119, 180)',   # Blue
        'rgb(255, 127, 14)',   # Orange
        'rgb(44, 160, 44)',    # Green
        'rgb(214, 39, 40)',    # Red
        'rgb(148, 103, 189)',  # Purple
        'rgb(140, 86, 75)',    # Brown
        'rgb(227, 119, 194)',  # Pink
        'rgb(127, 127, 127)',  # Gray
        'rgb(188, 189, 34)',   # Olive
        'rgb(23, 190, 207)',   # Cyan
    ]
    
    # Plot each trajectory
    for track_id, trajectory in sorted(trajectories_3d.items()):
        if not trajectory:
            continue
        
        trajectory = np.array(trajectory)
        x_coords = trajectory[:, 0]
        y_coords = trajectory[:, 1]
        z_coords = trajectory[:, 2]
        
        color = colors[track_id % len(colors)]
        
        # Trajectory line
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines+markers',
            name=f'Track {track_id}',
            line=dict(color=color, width=4),
            marker=dict(size=3, color=color),
            hovertemplate='<b>Track %{text}</b><br>' +
                         'X: %{x:.3f}m<br>' +
                         'Y: %{y:.3f}m<br>' +
                         'Z: %{z:.3f}m<br>' +
                         '<extra></extra>',
            text=[track_id] * len(x_coords)
        ))
        
        # Start point (larger marker)
        fig.add_trace(go.Scatter3d(
            x=[x_coords[0]],
            y=[y_coords[0]],
            z=[z_coords[0]],
            mode='markers',
            name=f'Track {track_id} Start',
            marker=dict(size=8, color=color, symbol='diamond'),
            showlegend=False,
            hovertemplate=f'<b>Track {track_id} - Start</b><br>' +
                         'X: %{x:.3f}m<br>' +
                         'Y: %{y:.3f}m<br>' +
                         'Z: %{z:.3f}m<br>' +
                         '<extra></extra>'
        ))
        
        # End point (larger marker)
        fig.add_trace(go.Scatter3d(
            x=[x_coords[-1]],
            y=[y_coords[-1]],
            z=[z_coords[-1]],
            mode='markers',
            name=f'Track {track_id} End',
            marker=dict(size=8, color=color, symbol='square'),
            showlegend=False,
            hovertemplate=f'<b>Track {track_id} - End</b><br>' +
                         'X: %{x:.3f}m<br>' +
                         'Y: %{y:.3f}m<br>' +
                         'Z: %{z:.3f}m<br>' +
                         '<extra></extra>'
        ))
    
    # Add ground plane
    if trajectories_3d:
        all_points = np.vstack([np.array(traj) for traj in trajectories_3d.values()])
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        
        # Expand bounds
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.2
        x_max += x_range * 0.2
        y_min -= y_range * 0.2
        y_max += y_range * 0.2
        
        # Ground grid
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 10),
            np.linspace(y_min, y_max, 10)
        )
        zz = np.zeros_like(xx)
        
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=zz,
            colorscale=[[0, 'rgba(200,200,200,0.3)'], [1, 'rgba(200,200,200,0.3)']],
            showscale=False,
            name='Ground Plane',
            hoverinfo='skip'
        ))
    
    # Layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (forward, m)',
            yaxis_title='Y (left, m)',
            zaxis_title='Z (up, m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        hovermode='closest',
        showlegend=True
    )
    
    # Save
    fig.write_html(output_path)
    print(f"3D trajectory visualization saved: {output_path}")


def export_trajectories_3d_matplotlib(trajectories_3d: Dict[int, List[Tuple[float, float, float]]],
                                     output_path: str,
                                     title: str = "3D Trajectory",
                                     dpi: int = 150):
    """
    Export 3D trajectories as static image using Matplotlib.
    
    Args:
        trajectories_3d: Dict mapping track_id to list of (X, Y, Z) world coordinates
        output_path: Output image file path (PNG/PDF/SVG)
        title: Plot title
        dpi: Image resolution
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Warning: matplotlib not installed. Install with: pip install matplotlib")
        print("Skipping static 3D visualization export.")
        return
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot each trajectory
    for track_id, trajectory in sorted(trajectories_3d.items()):
        if not trajectory:
            continue
        
        trajectory = np.array(trajectory)
        x_coords = trajectory[:, 0]
        y_coords = trajectory[:, 1]
        z_coords = trajectory[:, 2]
        
        color = colors[track_id % len(colors)]
        
        # Trajectory line
        ax.plot(x_coords, y_coords, z_coords, 
               label=f'Track {track_id}', 
               color=color, 
               linewidth=2, 
               marker='o', 
               markersize=3,
               alpha=0.8)
        
        # Start point
        ax.scatter(x_coords[0], y_coords[0], z_coords[0], 
                  color=color, s=100, marker='D', 
                  edgecolor='black', linewidth=1, zorder=10)
        
        # End point
        ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], 
                  color=color, s=100, marker='s', 
                  edgecolor='black', linewidth=1, zorder=10)
    
    # Labels and title
    ax.set_xlabel('X (forward, m)', fontsize=12)
    ax.set_ylabel('Y (left, m)', fontsize=12)
    ax.set_zlabel('Z (up, m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Set aspect ratio
    if trajectories_3d:
        all_points = np.vstack([np.array(traj) for traj in trajectories_3d.values()])
        max_range = np.array([
            all_points[:, 0].max() - all_points[:, 0].min(),
            all_points[:, 1].max() - all_points[:, 1].min(),
            all_points[:, 2].max() - all_points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # View angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"3D trajectory image saved: {output_path}")
