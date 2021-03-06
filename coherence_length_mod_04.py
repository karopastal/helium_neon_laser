import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.draw import line_aa
from skimage.measure import profile_line
from skimage import io
from coherence_length_data import data
from lmfit import Model

modes = data()
mode = modes['4']

# set profile line for each image
# take the avg fringe for each image


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
        '0.5': [(1305, 2052), (1097, 1927)],
        '1.0': [(2850, 1773), (2705, 1636)],
        '3.0': [(2243, 1861), (2331, 1757)],
        '5.5': [(2202, 2005), (2272, 1888)],
        '7.0': [(1440, 1988), (1452, 1819)],
        '8.0': [(2857, 1879), (2881, 1769)],
        '10.0': [(2153, 1944), (2151, 1847)],
        '12.0': [(3506, 1984), (3511, 1926)],
        '15.0': [(3452, 1934), (3459, 1873)],
        '17.0': [(2228, 1946), (2224, 1897)],
        '23.0': [(2149, 1928), (2146, 1880)],
        '28.0': [(2090, 1545), (2076, 1510)],
        '31.5': [(1134, 1241), (1108, 1198)],
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


def linear(x, a, b):
    return a*x + b


def calculate_dist(points):
    return np.linalg.norm(np.array(points[0]) - np.array(points[1]))


def get_visibility(intensity):
    return np.abs((intensity[0] - intensity[1])/(intensity[0] + intensity[1]))
    # return np.abs((intensity[0] - intensity[1]))


def plot_graph():
    intensities = {
        '0.5': [1, 0.122],
        '1.0': [0.804, 0.07],
        '3.0': [1, 0.129],
        '5.5': [1, 0.290],
        '7.0': [1, 0.300],
        '8.0': [1, 0.380],
        '10.0': [1, 0.463],
        '12.0': [1, 0.580],
        '15.0': [1, 0.670],
        '17.0': [1, 0.635],
        '23.0': [0.961, 0.682],
        '28.0': [0.914, 0.7],
        '31.5': [0.667, 0.522]
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

    plt.title('Visibility vs Length HG04' % (), fontsize=14)
    plt.xlabel('Length [cm]', fontsize=15)
    plt.ylabel('Visibility', fontsize=15)

    plt.plot(distances, fringes, 'bo', label='data points')
    plt.errorbar(distances, fringes, xerr=distances_err, fmt='.k', capthick=2, label='uncertainties')
    plt.plot(distances, linear(distances, a, b), 'r-', label='fit')
    plt.legend(loc='best')
    # plt.show()


plot_graph()
# plot_scans()
