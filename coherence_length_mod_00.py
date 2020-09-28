import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.measure import profile_line
from skimage import io
from coherence_length_data import data
from lmfit import Model


modes = data()
mode = modes['0']

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
        '2.0': [(2445, 1981), (2568, 2190)],
        '3.0': [(2380, 2155), (2462, 2074)],
        '4.5': [(2220, 2132), (2300, 2025)],
        '7.0': [(2548, 1447), (2619, 1318)],
        '10.0': [(2147, 1713), (2152, 1627)],
        '14.0': [(2251, 1805), (2249, 1750)],
        '18.0': [(2070, 2090), (2064, 2025)]
    }

    return profile_lines[distance]


def plot_scans():
    for item in mode:
        print(item)

        scan = io.imread(item['path'], as_gray=False)
        print(scan.shape)

        scan_line = get_profile_line(scan, str(item['x']))
        scan_profile = profile_line(scan, scan_line[0], scan_line[1])
        cv2.line(scan, scan_line[0], scan_line[1], (255, 0, 0), 30)

        plot_image(item, scan)
        # plot_profile(item, scan_profile)


def linear(x, a, b):
    return a*x + b


def calculate_dist(points):
    return np.linalg.norm(np.array(points[0]) - np.array(points[1]))


def plot_graph():
    data_points = {
        '2.0': [(2445, 1981), (2568, 2190)],
        '3.0': [(2380, 2155), (2462, 2074)],
        '4.5': [(2220, 2132), (2300, 2025)],
        '7.0': [(2548, 1447), (2619, 1318)],
        '10.0': [(2147, 1713), (2152, 1627)],
        '14.0': [(2251, 1805), (2249, 1750)],
        '18.0': [(2070, 2090), (2064, 2025)]
    }

    distances = []
    fringes = []

    for x in data_points:
        distances.append(float(x))
        fringes.append(calculate_dist(data_points[x]))

    distances = np.array(distances)
    fringes = np.array(fringes)

    print(fringes)
    print(distances)

    """ add fit to linear model """

    linear_model = Model(linear)

    result = linear_model.fit(fringes, x=distances, a=1, b=1)

    a = result.params['a'].value
    b = result.params['b'].value

    a_err = result.params['a'].stderr

    print(result.fit_report())
    print(result.chisqr)

    """  ---------------------- """

    plt.title('fringe vs distance' % (), fontsize=14)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)

    plt.plot(distances, fringes, 'bo')
    plt.plot(distances, linear(distances, 1, 1), 'k--', label='initial fit')
    plt.plot(distances, linear(distances, a, b), 'r-', label='best fit')
    plt.show()


plot_graph()
# plot_scans()
