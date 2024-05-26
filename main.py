import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def find_brightest_points(image):
    # Находим индексы самых ярких точек в изображении
    brightest_points = np.argwhere(image > image.min() * 1.05)
    return brightest_points


def read_tif_stack(file_path):
    # Чтение многослойного TIFF файла
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


def extract_channels(image_stack):
    channels = []
    min_val = image_stack.min() * 1.05
    intensities = []
    for z, image in enumerate(image_stack):
        points = find_brightest_points(image)
        for point in points:
            x, y = point
            if image[x, y] > min_val:
                channels.append((x, y, z))
                intensities.append(image[x, y])
    return channels, intensities


def plot_3d_channels(channels, intensities):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_coords = [p[0] for p in channels]
    y_coords = [p[1] for p in channels]
    z_coords = [p[2] for p in channels]

    # Нормализуем интенсивности для использования в цветовой карте
    norm = plt.Normalize(min(intensities), max(intensities))
    colors = cm.viridis(norm(intensities))

    ax.scatter(x_coords, y_coords, z_coords, c=colors, marker='o')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()


def plot_image_slices(image_stack):
    num_slices = len(image_stack)
    print(f'num slices: {num_slices}')
    fig, axes = plt.subplots(1, num_slices // 50, figsize=(50, 15))
    for i in range(0, num_slices-50, 50):
        ax = axes[i // 50]
        # Преобразуем изображение в float, чтобы поддерживать NaN
        image = image_stack[i].astype(float)
        image[image <= image.min() * 1.05] = np.nan  # Устанавливаем минимальные значения как NaN
        ax.imshow(image, cmap='gray', vmin=np.nanmin(image_stack), vmax=np.nanmax(image_stack))
        ax.axis('off')  # Отключаем оси для лучшего восприятия
    plt.show()


def main(file_path):
    image_stack = read_tif_stack(file_path)
    channels, intensities = extract_channels(image_stack)
    plot_3d_channels(channels, intensities)
    plot_image_slices(image_stack)


if __name__ == "__main__":
    file_path = 'MRI_maize.tif'
    main(file_path)
