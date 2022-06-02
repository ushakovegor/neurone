import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
from nucleidet.data.keypoints import rescale_keypoints

def visualize_heatmaps_3d(heatmaps, num_classes=3, resize_to=None, z_range=3.):

    heatmaps = heatmaps.detach().cpu().numpy()
    _, x_heatmap, y_heatmap = heatmaps.shape
        
    fig, axs = plt.subplots(1, num_classes, subplot_kw={"projection": "3d"}, figsize=(18,9))
    
            
    for heatmap, ax in zip(heatmaps, axs):
        if resize_to:
            heatmap = cv2.resize(heatmap, resize_to)
            
        X = np.arange(x_heatmap)
        Y = np.arange(y_heatmap)
        X, Y = np.meshgrid(X, Y)
    
        ax.set_zlim(0, z_range)
        surf = ax.scatter(X, Y, heatmap, c=heatmap, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    
    fig.colorbar(surf, shrink=0.4, aspect=10)
    fig.tight_layout()

    plt.show()

def visualize_heatmaps_plotly(heatmap, image=None, keypoints=None, resize_to=None, z_range=2.):
    heatmap = heatmap.detach().cpu().numpy()
    if resize_to:
        heatmap = cv2.resize(heatmap, resize_to)

    x_heatmap, y_heatmap = heatmap.shape
    X = np.arange(x_heatmap)
    Y = np.arange(y_heatmap)
    X, Y = np.meshgrid(X, Y)

    X = X.flatten()
    Y = Y.flatten()
    Z = heatmap.flatten()

    go_objects = [
            go.Scatter3d(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z,
                mode="markers",
                marker=dict(size=3, color=Z, opacity=1),
        )
    ]

    if not keypoints is None:
        if resize_to:
            target_size = resize_to
        else:
            target_size = (x_heatmap, y_heatmap)
        _, x_size, y_size = image.shape
        keypoints = rescale_keypoints(keypoints, (x_size, y_size), target_size)
        
        for kp_x, kp_y in zip(keypoints.x_coords(), keypoints.y_coords()):
            go_objects.append(
                go.Scatter3d(
                    x=[kp_x, kp_x],
                    y=[kp_y, kp_y],
                    z=[0, z_range],
                    marker=dict(size=2, color="red"),
                line=dict(color="red", width=2)
            ),
            )
                
    fig = go.Figure(go_objects)

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4,range=[0, x_heatmap]),
            yaxis=dict(nticks=4,range=[0, y_heatmap]),
            zaxis=dict(nticks=4,range=[0, z_range]),
        ),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10),
    )

    fig.show()