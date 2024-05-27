import numpy as np
from PIL import Image
import plotly.graph_objs as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from sklearn.cluster import KMeans


def find_brightest_points(image):
    brightest_points = np.argwhere(image > image.max() * 0.955)
    return brightest_points


def read_tif_stack(file_path):
    image_stack = Image.open(file_path)
    frames = []
    try:
        while True:
            frame = np.array(image_stack)
            frames.append(frame)
            image_stack.seek(image_stack.tell() + 1)
    except EOFError:
        pass
    return np.array(frames)


def find_stem_contour(image):
    thresh = threshold_otsu(image)
    binary = image > thresh
    contours = find_contours(binary, level=0.5)
    if contours:
        contour = max(contours, key=len)

        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # ax = axes.ravel()
        #
        # ax[0].imshow(image, cmap=plt.cm.gray)
        # ax[0].set_title('Original Image')
        # ax[0].axis('off')
        #
        # ax[1].imshow(binary, cmap=plt.cm.gray)
        # if contour is not None:
        #     ax[1].plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
        # ax[1].set_title('Binary Image with Contour')
        # ax[1].axis('off')
        #
        # plt.tight_layout()
        # plt.show()
        return contour

    return None


def clustering(image_stack, z_distance=0.05):
    curves = []
    # Clustering points by intensity for all frames
    points_list = []
    for z, image in enumerate(image_stack):
        points = find_brightest_points(image)
        points_list.extend([(x, y, z * z_distance) for x, y in points])
    points_list = np.array(points_list)

    # Clustering points by x and y coordinates
    kmeans = KMeans(n_clusters=150)
    kmeans.fit(points_list[:, :2])
    labels = kmeans.predict(points_list[:, :2])

    # Dividing clusters into layers and selecting central points
    clustered_points = {}
    for i, label in enumerate(labels):
        z = points_list[i, 2]
        if label not in clustered_points:
            clustered_points[label] = []
        clustered_points[label].append((points_list[i, 0], points_list[i, 1], z))

    for cluster_points in clustered_points.values():
        cluster_points.sort(key=lambda p: p[2])  # Sorting points by z coordinate
        # Initialize list to store coordinates for the curve
        curve = []
        for z_coord in np.unique([p[2] for p in cluster_points]):
            # Filter points for the current z coordinate
            points_z = [p for p in cluster_points if p[2] == z_coord]
            # Find the center of the cluster at this z coordinate
            center = np.mean(np.array(points_z)[:, :2], axis=0)
            curve.append((center[0], center[1], z_coord))
        curves.append(curve)

    return curves


def extract_channels_with_curves(image_stack, z_distance=0.05):
    channels = []
    intensities = []
    curves = clustering(image_stack)
    for z, image in enumerate(image_stack):
        print(f"Processing frame {z + 1}/{len(image_stack)}")
        print(f'image №{z}: max val = {image.max()}, min val = {image.min()}')
        points = find_brightest_points(image)
        for point in points:
            x, y = point
            channels.append((x, y, z * z_distance))
            intensities.append(image[int(x), int(y)])

        # Adding channels for contour
        contour = find_stem_contour(image)
        if contour is not None:
            for point in contour:
                x, y = point
                channels.append((x, y, z * z_distance))  # Distance between layers
                intensities.append(image[int(x), int(y)])
        else:
            print(f"No contour found for frame {z + 1}")

    return channels, curves, intensities


def plot_3d_channels_with_curves(channels, curves, intensities, x_range, y_range, z_range):
    x_coords_channels = np.array([p[0] for p in channels], dtype=np.float32)
    y_coords_channels = np.array([p[1] for p in channels], dtype=np.float32)
    z_coords_channels = np.array([p[2] for p in channels], dtype=np.float32)

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    # Plotting channels as scatter points
    trace_channels = go.Scatter3d(
        x=x_coords_channels, y=y_coords_channels, z=z_coords_channels,
        mode='markers',
        marker=dict(
            size=2,
            color=intensities,
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Channels'
    )
    fig.add_trace(trace_channels)

    # Plotting curves
    for i, curve in enumerate(curves):
        x_coords_curve = np.array([p[0] for p in curve], dtype=np.float32)
        y_coords_curve = np.array([p[1] for p in curve], dtype=np.float32)
        z_coords_curve = np.array([p[2] for p in curve], dtype=np.float32)

        trace_curve = go.Scatter3d(
            x=x_coords_curve, y=y_coords_curve, z=z_coords_curve,
            mode='lines',
            line=dict(
                color='red',
                width=4
            ),
            name=f'Curve {i + 1}'
        )
        fig.add_trace(trace_curve)

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='X axis',
                range=x_range
            ),
            yaxis=dict(
                title='Y axis',
                range=y_range
            ),
            zaxis=dict(
                title='Z axis',
                range=z_range
            )
        )
    )

    fig.show()


def main(file_path):
    image_stack = read_tif_stack(file_path)

    if image_stack.size == 0:
        print("Error: The image stack is empty or could not be read.")
        return

    x_range = [160, 300]  # Scale for X axis
    y_range = [160, 300]  # Scale for Y axis
    z_range = [0, len(image_stack) * 0.05]  # Scale for Z axis (z_distance * number of frames)

    channels, curves, intensities = extract_channels_with_curves(image_stack)

    if len(channels) == 0:
        print("No channels found.")
        return

    plot_3d_channels_with_curves(channels, curves, intensities, x_range, y_range, z_range)


if __name__ == "__main__":
    file_path = 'MRI_maize.tif'
    main(file_path)
