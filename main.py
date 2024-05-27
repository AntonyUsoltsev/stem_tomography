import numpy as np
from PIL import Image
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from skimage.filters import threshold_otsu
from skimage.measure import find_contours


def find_brightest_points(image):
    brightest_points = np.argwhere(image > image.min() * 1.8)
    print(f'brightest_points: {brightest_points.size} ')
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
        contour = max(contours, key=len)  # Выбрать самый длинный контур
        return contour
    return None


def extract_channels(image_stack, z_distance=0.05):
    channels = []
    intensities = []
    for z, image in enumerate(image_stack):
        print(f"Processing frame {z + 1}/{len(image_stack)}")
        contour = find_stem_contour(image)
        if contour is not None:
            for point in contour:
                x, y = point
                channels.append((x, y, z * z_distance))  # Distance between layers
                intensities.append(image[int(x), int(y)])
        else:
            print(f"No contour found for frame {z + 1}")

        points = find_brightest_points(image)
        for point in points:
            x, y = point
            channels.append((x, y, z * z_distance))
            intensities.append(image[int(x), int(y)])

    return channels, intensities


def plot_3d_channels_interactive(channels, intensities, x_range, y_range, z_range):
    x_coords = np.array([p[0] for p in channels], dtype=np.float32)
    y_coords = np.array([p[1] for p in channels], dtype=np.float32)
    z_coords = np.array([p[2] for p in channels], dtype=np.float32)
    intensities = np.array(intensities, dtype=np.float32)

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    trace = go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(
            size=2,
            color=intensities,
            colorscale='Viridis',
            opacity=0.8
        )
    )

    fig.add_trace(trace)

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

    channels, intensities = extract_channels(image_stack)

    if len(channels) == 0:
        print("No channels found.")
        return

    plot_3d_channels_interactive(channels, intensities, x_range, y_range, z_range)


if __name__ == "__main__":
    file_path = 'MRI_maize.tif'
    main(file_path)
