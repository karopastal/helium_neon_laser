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
        '2.0': [(2758, 2391), (2687, 2121)],
        '3.0': [(2113, 1895), (2145, 1790)],
        '4.5': [(2220, 2132), (2300, 2025)],
        '7.0': [(2195, 2104), (1974, 1663)],
        '10.0': [(2329, 1579), (2337, 1496)],
        '14.0': [(2321, 1695), (2314, 1641)],
        '18.0': [(2284, 1818), (2186, 1770)]
    }

    return profile_lines[distance]


def plot_scans():
    for item in mode:
        print(item)

        scan = io.imread(item['path'], as_gray=True)

        print(scan.shape)

        scan_line = get_profile_line(scan, str(item['x']))
        scan_profile = profile_line(scan, scan_line[0], scan_line[1])
        # cv2.line(scan, scan_line[0], scan_line[1], (255, 0, 0), 30)
        # cv2.line(scan, scan_line[0], scan_line[1], (255,), 30)
        plot_image(item, scan)
        # plot_profile(item, scan_profile)


def linear(x, a, b):
    return a*x + b


def calculate_dist(scan, points):
    # return np.abs((scan[points[0]] - scan[points[1]])/(scan[points[0]] + scan[points[1]]))
    print(scan[points[0]], scan[points[1]])
    print(np.abs((scan[points[0]] - scan[points[1]]))/(scan[points[0]] + scan[points[1]]))
    return np.abs((scan[points[0]] - scan[points[1]]))

    # return np.linalg.norm(np.array(points[0]) - np.array(points[1]))


def get_visibility(intensity):
    return np.abs((intensity[0] - intensity[1])/(intensity[0] + intensity[1]))
    # return np.abs((intensity[0] - intensity[1]))


def plot_graph():
    intensities = {
        '2.0': [1, 0.0471],
        '3.0': [1, 0.251],
        '4.5': [1, 0.286],
        '7.0': [1, 0.345],
        '10.0': [1, 0.390],
        '14.0': [1, 0.488],
        '18.0': [1, 0.570]
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
    a_err = result.params['a'].stderr

    b = result.params['b'].value
    b_err = result.params['b'].stderr

    a_err = result.params['a'].stderr

    print(result.fit_report())
    print('coherence length : ', np.abs(b/a))

    """  ---------------------- """

    distances_err = np.ones_like(distances)/2

    plt.title('Visibility vs Length HG00' % (), fontsize=14)
    plt.xlabel('Length [cm]', fontsize=15)
    plt.ylabel('Visibility', fontsize=15)

    plt.plot(distances, fringes, 'bo', label='data points')
    plt.errorbar(distances, fringes, xerr=distances_err, fmt='.k', capthick=2, label='uncertainties')
    plt.plot(distances, linear(distances, a, b), 'r-', label='fit')
    plt.legend(loc='best')
    # plt.show()


plot_graph()
# plot_scans()
