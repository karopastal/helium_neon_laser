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
        '0.0': [(2650, 0), (2650, scan.shape[0])],
        '0.5': [(0, 2500), (scan.shape[1], 2500)],
        '4.0': [(1279, 780), (2012, 1712)],
        '5.0': [(992, 1145), (2000, 1613)],
        '7.0': [(3795, 885), (4517, 1723)],
        '9.0': [(3783, 790), (4566, 1778)],
    }

    return profile_lines[distance]


for item in mode:
    print(item)

    scan = io.imread(item['path'], as_gray=False)
    print(scan.shape)

    scan_line = get_profile_line(scan, str(item['x']))
    scan_profile = profile_line(scan, scan_line[0], scan_line[1])
    cv2.line(scan, scan_line[0], scan_line[1], (255, 0, 0), 10)

    plot_image(item, scan)
    plot_profile(item, scan_profile)







