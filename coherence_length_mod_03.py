import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.draw import line_aa
from skimage.measure import profile_line
from skimage import io
from coherence_length_data import data

modes = data()
mode = modes['3']

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
        '1.0': [(4058, 1624), (4289, 1483)],
        '3.0': [(3964, 1542), (3956, 1300)],
        '5.0': [(1705, 1541), (1784, 1413)],
        '7.0': [(2548, 1656), (2567, 1505)],
        '8.0': [(2512, 2020), (2515, 1915)],
        '9.0': [(2500, 1827), (2495, 1738)],
        '10.0': [(2384, 2375), (2328, 2295)],
        '12.0': [(3093, 1690), (3093, 1638)],
        '19.0': [(2491, 1303), (2489, 1262)],
        '38.0': [(2435, 2075), (2427, 2029)]
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
        '1.0': [(4058, 1624), (4289, 1483)],
        '3.0': [(3964, 1542), (3956, 1300)],
        '5.0': [(1705, 1541), (1784, 1413)],
        '7.0': [(2548, 1656), (2567, 1505)],
        '8.0': [(2512, 2020), (2515, 1915)],
        '9.0': [(2500, 1827), (2495, 1738)],
        '10.0': [(2384, 2375), (2328, 2295)],
        '12.0': [(3093, 1690), (3093, 1638)],
        '19.0': [(2491, 1303), (2489, 1262)],
        '38.0': [(2435, 2075), (2427, 2029)]
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







