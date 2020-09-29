import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.draw import line_aa
from skimage.measure import profile_line
from skimage import io
from coherence_length_data import data
from lmfit import Model

modes = data()
mode = modes['31']


def plot_image(item, image):
    fig, ax = plt.subplots()

    plt.title('TEM_00 fringes distance: %s cm' % (item['x']), fontsize=14)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)

    ax.imshow(image, cmap='pink')
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

        scan = io.imread(item['path'], as_gray=True)
        print(scan.shape)

        # scan_line = get_profile_line(scan, str(item['x']))
        # scan_profile = profile_line(scan, scan_line[0], scan_line[1])
        # cv2.line(scan, scan_line[0], scan_line[1], (255, 0, 0), 10)

        plot_image(item, scan)
        # plot_profile(item, scan_profile)


def calculate_dist(points):
    return np.linalg.norm(np.array(points[0]) - np.array(points[1]))


def linear(x, a, b):
    return a*x + b


def get_visibility(intensity):
    return np.abs((intensity[0] - intensity[1])/(intensity[0] + intensity[1]))
    # return np.abs((intensity[0] - intensity[1]))


def plot_graph():
    intensities = {
        '0.0': [0.678, 0.180],
        '0.5': [0.34, 0.0940],
        '4.0': [0.25, 0.095],
        '5.0': [0.15, 0.05],
        '7.0': [0.188, 0.063],
        '9.0': [0.0863, 0.055],
    }

    distances = []
    fringes = []

    for item in mode:
        distances.append(float(item['x']))
        fringes.append(get_visibility(intensities[str(item['x'])]))

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
    print('coherence length : ', np.abs(b / a))

    """  ---------------------- """

    distances_err = np.ones_like(distances) / 2

    plt.title('Visibility vs Length HG31' % (), fontsize=14)
    plt.xlabel('Length [cm]', fontsize=15)
    plt.ylabel('Visibility', fontsize=15)

    plt.plot(distances, fringes, 'bo', label='data points')
    plt.errorbar(distances, fringes, xerr=distances_err, fmt='.k', capthick=2, label='uncertainties')
    plt.plot(distances, linear(distances, a, b), 'r-', label='fit')
    plt.legend(loc='best')
    plt.show()


plot_graph()
# plot_scans()



