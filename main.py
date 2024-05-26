# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
#
#
# def find_brightest_points(image):
#     # Находим индексы самых ярких точек в изображении
#     brightest_points = np.argwhere(image > image.min() * 1.1)
#     return brightest_points
#
#
# def read_tif_stack(file_path):
#     # Чтение многослойного TIFF файла
#     image_stack = Image.open(file_path)
#     frames = []
#     try:
#         while True:
#             frame = np.array(image_stack)
#             frames.append(frame)
#             image_stack.seek(image_stack.tell() + 1)
#     except EOFError:
#         pass
#     return np.array(frames)
#
#
# def extract_channels(image_stack):
#     channels = []
#     min_val = image_stack.min() * 1.05
#     intensities = []
#     for z, image in enumerate(image_stack):
#         points = find_brightest_points(image)
#         for point in points:
#             x, y = point
#             if image[x, y] > min_val:
#                 channels.append((x, y, z))
#                 intensities.append(image[x, y])
#     return channels, intensities
#
#
# def plot_3d_channels(channels, intensities):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     x_coords = [p[0] for p in channels]
#     y_coords = [p[1] for p in channels]
#     z_coords = [p[2] for p in channels]
#
#     # Нормализуем интенсивности для использования в цветовой карте
#     norm = plt.Normalize(min(intensities), max(intensities))
#     colors = cm.viridis(norm(intensities))
#
#     ax.scatter(x_coords, y_coords, z_coords, c=colors, marker='o')
#     ax.set_xlabel('X axis')
#     ax.set_ylabel('Y axis')
#     ax.set_zlabel('Z axis')
#
#     plt.show()
#
#
# def plot_image_slices(image_stack):
#     num_slices = len(image_stack)
#     print(f'num slices: {num_slices}')
#     fig, axes = plt.subplots(1, num_slices // 50, figsize=(50, 15))
#     for i in range(0, num_slices-50, 50):
#         ax = axes[i // 50]
#         # Преобразуем изображение в float, чтобы поддерживать NaN
#         image = image_stack[i].astype(float)
#         image[image <= image.min() * 1.05] = np.nan  # Устанавливаем минимальные значения как NaN
#         ax.imshow(image, cmap='gray', vmin=np.nanmin(image_stack), vmax=np.nanmax(image_stack))
#         ax.axis('off')  # Отключаем оси для лучшего восприятия
#     plt.show()
#
#
# def main(file_path):
#     image_stack = read_tif_stack(file_path)
#     channels, intensities = extract_channels(image_stack)
#     plot_3d_channels(channels, intensities)
#     plot_image_slices(image_stack)
#
#
# if __name__ == "__main__":
#     file_path = 'MRI_maize.tif'
#     main(file_path)
#
#
#


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image


def find_brightest_points(image, threshold_ratio=1.2):
    threshold = image.min() * threshold_ratio
    brightest_points = np.argwhere(image > threshold)
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


def extract_channels(image_stack):
    channels = []
    intensities = []
    for z, image in enumerate(image_stack):
        points = find_brightest_points(image)
        for point in points:
            x, y = point
            channels.append((x, y, z))
            intensities.append(image[x, y])
    return channels, intensities


def plot_3d_channels(channels, intensities, exclude_area=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if exclude_area is not None:
        x_min, x_max, y_min, y_max = exclude_area
        filtered_channels = [(x, y, z) for (x, y, z) in channels if not (x_min <= x <= x_max and y_min <= y <= y_max)]
        filtered_intensities = [intensities[i] for i in range(len(intensities)) if
                                not (x_min <= channels[i][0] <= x_max and y_min <= channels[i][1] <= y_max)]
    else:
        filtered_channels = channels
        filtered_intensities = intensities
    print(filtered_intensities)
    x_coords = [p[0] for p in filtered_channels]
    y_coords = [p[1] for p in filtered_channels]
    z_coords = [p[2] for p in filtered_channels]

    norm = plt.Normalize(min(filtered_intensities), max(filtered_intensities))
    colors = cm.viridis(norm(filtered_intensities))

    ax.scatter(x_coords, y_coords, z_coords, c=colors, marker='o')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()


def find_stick_coordinates(channels):
    x_coords = [p[0] for p in channels]
    y_coords = [p[1] for p in channels]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(x_coords, bins=100)
    plt.xlabel('X Coordinates')
    plt.ylabel('Frequency')
    plt.title('Histogram of X Coordinates')

    plt.subplot(1, 2, 2)
    plt.hist(y_coords, bins=100)
    plt.xlabel('Y Coordinates')
    plt.ylabel('Frequency')
    plt.title('Histogram of Y Coordinates')

    plt.tight_layout()
    plt.show()


def main(file_path):
    image_stack = read_tif_stack(file_path)
    channels, intensities = extract_channels(image_stack)
    find_stick_coordinates(channels)

    exclude_area = (120, 200, 300, 375)  # Замените на координаты вашего стержня после анализа гистограмм

    plot_3d_channels(channels, intensities, exclude_area)
    plot_image_slices(image_stack)


def plot_image_slices(image_stack):
    num_slices = len(image_stack)
    step = max(num_slices // 50, 1)
    fig, axes = plt.subplots(1, num_slices // step, figsize=(50, 15))
    for i in range(0, num_slices, step):
        ax = axes[i // step]
        image = image_stack[i].astype(float)
        image[image == image.min()] = np.nan
        ax.imshow(image, cmap='gray', vmin=np.nanmin(image_stack), vmax=np.nanmax(image_stack))
        ax.axis('off')
    plt.show()


if __name__ == "__main__":
    file_path = 'MRI_maize.tif'
    main(file_path)
