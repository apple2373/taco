# utility functions for visualization of hand pose, camera, and 3D objects, etc
# Satoshi
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
import plotly.graph_objs as go
import matplotlib.animation as animation
from IPython.display import HTML

color_hand_joints = [
    [1.0, 0.0, 0.0],
    [0.0, 0.4, 0.0],
    [0.0, 0.6, 0.0],
    [0.0, 0.8, 0.0],
    [0.0, 1.0, 0.0],  # thumb
    [0.0, 0.0, 0.6],
    [0.0, 0.0, 1.0],
    [0.2, 0.2, 1.0],
    [0.4, 0.4, 1.0],  # index
    [0.0, 0.4, 0.4],
    [0.0, 0.6, 0.6],
    [0.0, 0.8, 0.8],
    [0.0, 1.0, 1.0],  # middle
    [0.4, 0.4, 0.0],
    [0.6, 0.6, 0.0],
    [0.8, 0.8, 0.0],
    [1.0, 1.0, 0.0],  # ring
    [0.4, 0.0, 0.4],
    [0.6, 0.0, 0.6],
    [0.8, 0.0, 0.8],
    [1.0, 0.0, 1.0],
]  # little


def visualize_joints_2d(pose_uv, ax, color_hand_joints=color_hand_joints, marker_sz=5,line_wd=1.5):
    """
    Draws a 2D skeleton on an existing Matplotlib axis.

    :param pose_uv: A 21 x 2 numpy array with joint coordinates (x, y)
    :param ax: A Matplotlib axis object where the skeleton will be drawn
    :param color_hand_joints: A list of RGB tuples for joint colors (optional)
    :return: The Matplotlib axis object with the drawn skeleton
    """
    assert pose_uv.shape[0] == 21

    # marker_sz = 5
    # line_wd = 1.5

    # Default color for joints (red)
    if color_hand_joints is None:
        color_hand_joints = [(255, 0, 0)] * 21

    def draw_circle(ax, center, radius, color):
        circle = plt.Circle(center, radius, color=color, fill=True)
        ax.add_patch(circle)

    def draw_line(ax, start, end, color, width):
        line = plt.Line2D(
            [start[0], end[0]], [start[1], end[1]], color=color, linewidth=width
        )
        ax.add_line(line)

    for joint_ind in range(pose_uv.shape[0]):
        joint = tuple(pose_uv[joint_ind].astype("int32"))
        color = np.array(
            color_hand_joints[joint_ind]
        )  # Convert to [0, 1] for Matplotlib

        # Draw the joint as a circle
        draw_circle(ax, joint, marker_sz, color)

        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            root_joint = tuple(pose_uv[0].astype("int32"))
            # Draw line from root joint to the current joint
            draw_line(ax, root_joint, joint, color, line_wd)
        else:
            joint_2 = tuple(pose_uv[joint_ind - 1].astype("int32"))
            # Draw line from the previous joint to the current joint
            draw_line(ax, joint_2, joint, color, line_wd)

    return ax


def visualize_joints_3d(pose_cam_xyz, ax, color_hand_joints=color_hand_joints):
    # citation: https://github.com/3d-hand-shape/hand-graph-cnn/blob/master/hand_shape_pose/util/vis.py
    """
    :param pose_cam_xyz: 21 x 3
    :param ax:
    :return:
    """
    assert pose_cam_xyz.shape[0] == 21

    marker_sz = 15
    line_wd = 2

    for joint_ind in range(pose_cam_xyz.shape[0]):
        ax.plot(
            pose_cam_xyz[joint_ind : joint_ind + 1, 0],
            pose_cam_xyz[joint_ind : joint_ind + 1, 1],
            pose_cam_xyz[joint_ind : joint_ind + 1, 2],
            ".",
            c=color_hand_joints[joint_ind],
            markersize=marker_sz,
        )
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(
                pose_cam_xyz[[0, joint_ind], 0],
                pose_cam_xyz[[0, joint_ind], 1],
                pose_cam_xyz[[0, joint_ind], 2],
                color=color_hand_joints[joint_ind],
                linewidth=line_wd,
            )
        else:
            ax.plot(
                pose_cam_xyz[[joint_ind - 1, joint_ind], 0],
                pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
                pose_cam_xyz[[joint_ind - 1, joint_ind], 2],
                color=color_hand_joints[joint_ind],
                linewidth=line_wd,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return ax


def visualize_cube_2d(cube_corners_proj, cube_edges, ax, color="lightgreen",linewidth=1.5, alpha=0.9,markersize=5):
    # Draw the edges of the cube
    for edge in cube_edges:
        x = [cube_corners_proj[edge[0], 0], cube_corners_proj[edge[1], 0]]
        y = [cube_corners_proj[edge[0], 1], cube_corners_proj[edge[1], 1]]
        ax.plot(x, y, color=color, linestyle="-", linewidth=linewidth, alpha=alpha)
    #add point at the corner of the cube
    for corner in cube_corners_proj:
        ax.plot(corner[0], corner[1], 'o', color=color, markersize=markersize)
    return ax


def visualize_cube_3d(cube_corners, cube_edges, ax):
    # Draw the edges of the cube
    for edge in cube_edges:
        x = [cube_corners[edge[0], 0], cube_corners[edge[1], 0]]
        y = [cube_corners[edge[0], 1], cube_corners[edge[1], 1]]
        z = [cube_corners[edge[0], 2], cube_corners[edge[1], 2]]
        ax.plot(x, y, z, color="lightgreen", linestyle="-", linewidth=1.5, alpha=0.9)
    return ax


def visualize_cam_3d(cam_extr, cam_intr, ax, virtual_image_distance=None):
    # cam_intr is camera-to-image matrix!!!
    # https://dfki-ric.github.io/pytransform3d/_auto_examples/plots/plot_camera_3d.html
    # https://github.com/demul/extrinsic2pyramid
    import pytransform3d.camera as pc
    import pytransform3d.transformations as pt

    if cam_extr.shape == (3, 4):
        cam_extr = np.vstack([cam_extr, [0, 0, 0, 1]])

    R = cam_extr[:3, :3]
    t = cam_extr[:3, 3]
    sensor_size = np.array([cam_intr[0, 2], cam_intr[1, 2]]) * 2
    # sensor_size = img.size
    if virtual_image_distance is None:
        virtual_image_distance = 500

    pc.plot_camera(
        ax,
        cam2world=cam_extr,
        M=cam_intr,
        sensor_size=sensor_size,
        virtual_image_distance=virtual_image_distance,
    )
    ax.scatter(t[0], t[1], t[2], c="r", alpha=0.5)
    return ax


def plotly_joints(hand_pose, color_hand_joints=color_hand_joints):
    bones = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),  # Thumb
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),  # Index
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),  # Middle
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),  # Ring
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),  # Little
    ]

    # Plotly traces for the bones
    traces = []
    for bone in bones:
        x = [hand_pose[bone[0], 0], hand_pose[bone[1], 0]]
        y = [hand_pose[bone[0], 1], hand_pose[bone[1], 1]]
        z = [hand_pose[bone[0], 2], hand_pose[bone[1], 2]]
        color = f"rgb({color_hand_joints[bone[1]][0]*255}, {color_hand_joints[bone[1]][1]*255}, {color_hand_joints[bone[1]][2]*255})"

        # Add a trace for the bone with colored markers and lines
        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines+markers",
                marker=dict(size=4, color=color),
                line=dict(color=color, width=2),
            )
        )

    return traces


def plotly_cube(
    cube_corners,
    cube_edges,
    color="rgb(144, 238, 144)",
    opacity=1,
    mode="lines",
    line_dash="solid",
    width=2,
):
    # Plotly traces for the cube edges
    traces = []
    for edge in cube_edges:
        x = [cube_corners[edge[0], 0], cube_corners[edge[1], 0]]
        y = [cube_corners[edge[0], 1], cube_corners[edge[1], 1]]
        z = [cube_corners[edge[0], 2], cube_corners[edge[1], 2]]

        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode=mode,
                line=dict(
                    color=color, width=width, dash=line_dash
                ),  # Set line_dash here
                name="Object Box",
                opacity=opacity,
            )
        )

    return traces


def plotly_line(
    point1,
    point2,
    color="rgb(144, 238, 144)",
    opacity=1,
    mode="lines",
    line_dash="solid",
    width=2,
):
    # Extract coordinates from the two points
    x = [point1[0], point2[0]]
    y = [point1[1], point2[1]]
    z = [point1[2], point2[2]]

    # Create a 3D scatter plot trace to represent the line between the points
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode=mode,
        line=dict(color=color, width=width, dash=line_dash),  # Set line style and color
        name="Line between points",
        opacity=opacity,
    )

    # Return the trace for the line
    return [trace]


def plotly_points(points, color="rgb(255, 0, 0)", size=5, opacity=1):
    """
    Create a 3D scatter plot for a set of points.

    Parameters:
    - points: A NumPy array or list of shape (N, 3) where N is the number of points.
    - color: Color of the points in RGB format (e.g., "rgb(255, 0, 0)").
    - size: Size of the points.
    - opacity: Opacity of the points (0 to 1).

    Returns:
    - A list containing a Plotly Scatter3d trace for the points.
    """
    # Check if points is a numpy array and convert if necessary
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    # Create the scatter plot trace for the points
    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(size=size, color=color, opacity=opacity),
        name="Points",
    )

    return [trace]


def plotly_points_with_text(points, labels=None, color="rgb(255, 0, 0)", size=5, opacity=1):
    """
    Create a 3D scatter plot for a set of points with optional text annotations.

    Parameters:
    - points: A NumPy array or list of shape (N, 3) where N is the number of points.
    - labels: A list of strings of length N for text annotations. Defaults to None.
    - color: Color of the points in RGB format (e.g., "rgb(255, 0, 0)").
    - size: Size of the points.
    - opacity: Opacity of the points (0 to 1).

    Returns:
    - A list containing a Plotly Scatter3d trace for the points.
    """
    # Check if points is a numpy array and convert if necessary
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    # Ensure labels are provided and match the number of points
    if labels is not None and len(labels) != points.shape[0]:
        raise ValueError("The length of labels must match the number of points.")

    # Create the scatter plot trace for the points
    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers+text",  # Include text annotations
        marker=dict(size=size, color=color, opacity=opacity),
        text=labels,  # Add the text for each point
        textposition="top center",  # Position of the text relative to the points
        name="Points",
    )

    return [trace]


def plotly_arrows(
    start,
    end,
    line_color="rgb(0, 0, 255)",
    head_color="rgb(255, 0, 0)",
    width=3,
    opacity=1,
    head_size=0.2,
):
    """
    Create a 3D plot with arrows between start and end points.

    Parameters:
    - start: A NumPy array or list of shape (N, 3) where N is the number of arrows' starting points.
    - end: A NumPy array or list of shape (N, 3) where N is the number of arrows' ending points.
    - line_color: Color of the arrow lines.
    - head_color: Color of the arrowheads.
    - width: Width of the arrow lines.
    - opacity: Opacity of the arrows (0 to 1).
    - head_size: Size of the arrowheads.

    Returns:
    - A list containing Plotly Scatter3d traces for the arrows and cone traces for arrowheads.
    """
    if not isinstance(start, np.ndarray):
        start = np.array(start)
    if not isinstance(end, np.ndarray):
        end = np.array(end)

    if start.shape != end.shape or start.shape[1] != 3:
        raise ValueError("Start and end must be of shape (N, 3).")

    x_coords = []
    y_coords = []
    z_coords = []

    # Calculate directional vectors for arrowheads
    arrow_vectors = end - start

    # Add line segments for arrows
    for i in range(start.shape[0]):
        x_coords.extend(
            [start[i, 0], end[i, 0], None]
        )  # None to break line between arrows
        y_coords.extend([start[i, 1], end[i, 1], None])
        z_coords.extend([start[i, 2], end[i, 2], None])

    # Line trace for arrow bodies
    arrow_body_trace = go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode="lines",
        line=dict(color=line_color, width=width),
        opacity=opacity,
        name="Arrow Bodies",
    )

    # Cone traces for arrowheads
    arrow_heads = go.Cone(
        x=end[:, 0],
        y=end[:, 1],
        z=end[:, 2],
        u=arrow_vectors[:, 0],
        v=arrow_vectors[:, 1],
        w=arrow_vectors[:, 2],
        colorscale=[[0, head_color], [1, head_color]],
        showscale=False,
        sizemode="absolute",
        sizeref=head_size,
        anchor="tip",
        name="Arrow Heads",
    )

    return [arrow_body_trace, arrow_heads]


def plotly_cam(cam_extr, cam_intr, near_plane_z=100, far_plane_z=500):
    # cam_extr is camera-to-world matrix!!!

    # Extract focal length (in pixels)
    focal_length_x = cam_intr[0, 0]
    focal_length_y = cam_intr[1, 1]

    # Extract principal point (in pixels)
    principal_point_x = cam_intr[0, 2]
    principal_point_y = cam_intr[1, 2]

    # Sensor size calculation (in pixels)
    sensor_size_x = 2 * (principal_point_x)
    sensor_size_y = 2 * (principal_point_y)

    # Calculate horizontal and vertical FOV
    fov_x = (
        2 * np.arctan(sensor_size_x / (2 * focal_length_x)) * (180 / np.pi)
    )  # Convert radians to degrees
    fov_y = (
        2 * np.arctan(sensor_size_y / (2 * focal_length_y)) * (180 / np.pi)
    )  # Convert radians to degrees

    aspect_ratio = sensor_size_x / sensor_size_y

    # print(f"Horizontal FOV: {fov_x:.2f} degrees")
    # print(f"Vertical FOV: {fov_y:.2f} degrees")

    fov_x = 2 * np.arctan(np.tan(np.radians(fov_y) / 2) * aspect_ratio)
    near_half_width = np.tan(fov_x / 2) * near_plane_z
    near_half_height = np.tan(np.radians(fov_y) / 2) * near_plane_z
    far_half_width = np.tan(fov_x / 2) * far_plane_z
    far_half_height = np.tan(np.radians(fov_y) / 2) * far_plane_z

    near_plane = np.array(
        [
            [-near_half_width, -near_half_height, near_plane_z],
            [near_half_width, -near_half_height, near_plane_z],
            [near_half_width, near_half_height, near_plane_z],
            [-near_half_width, near_half_height, near_plane_z],
        ]
    )

    far_plane = np.array(
        [
            [-far_half_width, -far_half_height, far_plane_z],
            [far_half_width, -far_half_height, far_plane_z],
            [far_half_width, far_half_height, far_plane_z],
            [-far_half_width, far_half_height, far_plane_z],
        ]
    )

    def transform_to_world(coords, extrinsic_matrix):
        coords_hom = np.hstack([coords, np.ones((coords.shape[0], 1))])
        transformed_coords = extrinsic_matrix.dot(coords_hom.T).T
        return transformed_coords[:, :3]

    near_plane_world = transform_to_world(near_plane, cam_extr)
    far_plane_world = transform_to_world(far_plane, cam_extr)

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    # Initialize traces for the camera pyramid
    camera_traces = []
    for edge in edges[:4]:
        x = [near_plane_world[edge[0], 0], near_plane_world[edge[1], 0]]
        y = [near_plane_world[edge[0], 1], near_plane_world[edge[1], 1]]
        z = [near_plane_world[edge[0], 2], near_plane_world[edge[1], 2]]
        camera_traces.append(
            go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="blue", width=2))
        )

    for edge in edges[4:8]:
        x = [far_plane_world[edge[0] - 4, 0], far_plane_world[edge[1] - 4, 0]]
        y = [far_plane_world[edge[0] - 4, 1], far_plane_world[edge[1] - 4, 1]]
        z = [far_plane_world[edge[0] - 4, 2], far_plane_world[edge[1] - 4, 2]]
        camera_traces.append(
            go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="red", width=2))
        )

    for edge in edges[8:]:
        x = [near_plane_world[edge[0], 0], far_plane_world[edge[1] - 4, 0]]
        y = [near_plane_world[edge[0], 1], far_plane_world[edge[1] - 4, 1]]
        z = [near_plane_world[edge[0], 2], far_plane_world[edge[1] - 4, 2]]
        camera_traces.append(
            go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="green", width=2))
        )

    camera_center = transform_to_world(np.array([[0, 0, 0]]), cam_extr)
    camera_traces.append(
        go.Scatter3d(
            x=camera_center[:, 0],
            y=camera_center[:, 1],
            z=camera_center[:, 2],
            mode="markers",
            marker=dict(size=5, color="black", symbol="x"),
            name="Camera Center",
        )
    )

    # Arrow properties
    arrow_length = near_half_width * 0.2  # Adjust the arrow length
    arrow_head_size = 30  # Adjust the size of the arrowhead

    # Arrow for X direction (→) in the imaeg space.
    arrow_x = go.Cone(
        x=[near_plane_world[1, 0]],  # Start position
        y=[near_plane_world[1, 1]],
        z=[near_plane_world[1, 2]],
        u=[arrow_length],  # Direction and length
        v=[0],
        w=[0],
        sizemode="absolute",
        sizeref=arrow_head_size,
        anchor="tip",
        colorscale=[[0, "blue"], [1, "blue"]],  # Color of the arrow
        showscale=False,
    )

    # Arrow for Y direction (↓) in the image space.
    arrow_y = go.Cone(
        x=[near_plane_world[2, 0]],  # Start position
        y=[near_plane_world[2, 1]],
        z=[near_plane_world[2, 2]],
        u=[0],
        v=[arrow_length],  # Direction and length
        w=[0],
        sizemode="absolute",
        sizeref=arrow_head_size,
        anchor="tip",
        colorscale=[[0, "blue"], [1, "blue"]],  # Color of the arrow
        showscale=False,
    )

    # Add the arrows to the camera traces
    camera_traces.append(arrow_x)
    camera_traces.append(arrow_y)

    return camera_traces


def axes2video(axes_time, interval=100, repeat_delay=1000):
    # Create a figure for the animation
    fig = plt.figure(figsize=(5, 5))

    # Function to update the frame in the animation
    def update_frame(i):
        fig.clear()  # Clear the figure, not just the axis
        ax = fig.add_subplot(111)  # Add a new axis to the figure
        saved_ax = axes_time[i]
        saved_ax.figure = fig
        fig.add_axes(saved_ax)
        # Optionally, hide ticks or axes if desired
        ax.set_xticks([])  # Hide x-axis ticks
        ax.set_yticks([])  # Hide y-axis ticks
        ax.axis("off")  # Hide the axis

    # Create an animation
    ani = animation.FuncAnimation(
        fig,
        update_frame,
        frames=len(axes_time),
        interval=interval,
        repeat_delay=repeat_delay,
    )

    # Display the animation in the notebook
    return HTML(ani.to_jshtml())


# # source : https://github.com/3d-hand-shape/hand-graph-cnn/blob/master/hand_shape_pose/util/vis.py
# def fig2data(fig):
#     """
#     @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
#     @param fig a matplotlib figure
#     @return a numpy 3D array of RGBA values
#     """
#     # draw the renderer
#     fig.canvas.draw()

#     # Get the RGBA buffer from the figure
#     w, h = fig.canvas.get_width_height()
#     buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
#     buf.shape = (w, h, 4)

#     # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
#     buf = np.roll(buf, 3, axis=2)
#     return buf
