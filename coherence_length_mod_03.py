import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.measure import profile_line
from skimage import io
from coherence_length_data import data
from lmfit import Model

modes = data()
mode = modes['3']

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
        '1.0': [1, 0.102],
        '3.0': [1, 0.278],
        '5.0': [1, 0.373],
        '7.0': [1, 0.391],
        '8.0': [1, 0.314],
        '9.0': [1, 0.345],
        '10.0': [0.945, 0.345],
        '12.0': [1, 0.416],
        '19.0': [0.9, 0.584],
        '38.0': [0.514, 0.350]
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

    plt.title('Visibility vs Length HG03' % (), fontsize=14)
    plt.xlabel('Length [cm]', fontsize=15)
    plt.ylabel('Visibility', fontsize=15)

    plt.plot(distances, fringes, 'bo', label='data points')
    plt.errorbar(distances, fringes, xerr=distances_err, fmt='.k', capthick=2, label='uncertainties')
    plt.plot(distances, linear(distances, a, b), 'r-', label='fit')
    plt.legend(loc='best')
    # plt.show()


plot_graph()
# plot_scans()







