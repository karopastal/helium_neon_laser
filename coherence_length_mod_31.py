import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.draw import line_aa
from skimage.measure import profile_line
from skimage import io
from coherence_length_data import data

modes = data()
mode = modes['31']

# set profile line for each image
# take the avg fringe for each image


def plot_image(item, image):
    fig, ax = plt.subplots()

    plt.title('TEM_00 fringes distance: %s cm' % (item['x']), fontsize=14)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)

    ax.imshow(image)
    ax.set_ylim(0, image.shape[0])
    plt.show()


def plot_profile(item, profile):
    plt.title('TEM_00 fringes profile delta_x: %s cm' % (item['x'],), fontsize=14)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)

    x = np.arange(profile.shape[0])
    plt.plot(x, profile, 'bo')
    plt.show()


def get_profile_line(scan, distance):
    profile_lines = {
        '0.0': [(2453, 1197), (2353, 1118)],
        '0.5': [(1643, 2450), (1537, 2403)],
        '4.0': [(1698, 1458), (1645, 1403)],
        '5.0': [(1665, 1456), (1608, 1427)],
        '7.0': [(4259, 1420), (4212, 1368)],
        '9.0': [(4208, 1324), (4170, 1275)],
    }

    return profile_lines[distance]


def plot_scans():
    for item in mode:
        print(item)

        scan = io.imread(item['path'], as_gray=False)
        print(scan.shape)

        scan_line = get_profile_line(scan, str(item['x']))
        scan_profile = profile_line(scan, scan_line[0], scan_line[1])
        cv2.line(scan, scan_line[0], scan_line[1], (255, 0, 0), 10)

        plot_image(item, scan)
        plot_profile(item, scan_profile)


def calculate_dist(points):
    return np.linalg.norm(np.array(points[0]) - np.array(points[1]))


def plot_graph():
    data_points = {
        '0.0': [(2453, 1197), (2353, 1118)],
        '0.5': [(1643, 2450), (1537, 2403)],
        '4.0': [(1698, 1458), (1645, 1403)],
        '5.0': [(1665, 1456), (1608, 1427)],
        '7.0': [(4259, 1420), (4212, 1368)],
        '9.0': [(4208, 1324), (4170, 1275)],
    }

    distances = []
    fringes = []

    for x in data_points:
        distances.append(float(x))
        fringes.append(calculate_dist(data_points[x]))

    print(fringes)
    print(distances)

    """ add fit to linear model """

    plt.title('fringe vs distance' % (), fontsize=14)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)

    plt.plot(distances, fringes, 'bo')
    plt.show()


plot_graph()
# plot_scans()




