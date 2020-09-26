import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from skimage.measure import profile_line
from skimage import io
from scipy.special import eval_hermite as hermiteH


# plt.plot(x, y, 'bo')
# plt.plot(x, result.init_fit, 'k--', label='initial fit')
# plt.plot(x, result.best_fit, 'r-', label='best fit')
# plt.legend(loc='best')
# plt.show()
#
# plt.plot(x, mod_01_profile, '.')
# plt.ylabel('intensity')
# plt.xlabel('line path')
# plt.show()


def plot_mod_analysis(x, y, mod):
    plt.plot(x, y, 'o')
    plt.plot(x, mod_01_function(x, a_init, b_init, w_init), 'k--', label='initial fit')
    plt.plot(x, mod_01_function(x, a, b, w), 'r-', label='best fit')
    plt.legend(loc='best')
    plt.show()


def fit_to_model(model, data):
    pass


# ------------
# mods images
# ------------

mod01 = io.imread('imgs/1nnew.tiff', as_gray=True)
# mod02 = io.imread('imgs/02 2.tiff', as_gray=True)
# mod03 = io.imread('imgs/02 2.tiff', as_gray=True)
# mod04 = io.imread('imgs/02 2.tiff', as_gray=True)
# mod05 = io.imread('imgs/02 2.tiff', as_gray=True)
# mod22 = io.imread('imgs/02 2.tiff', as_gray=True)

print(mod01.shape)

# mods lines
# ------------

mod_01_line = [(0, mod01.shape[1]/2), (mod01.shape[0], mod01.shape[1]/2)]
# mod_02_line = [(0, 0), (0, 0)]
# mod_03_line = [(0, 0), (0, 0)]
# mod_04_line = [(0, 0), (0, 0)]
# mod_05_line = [(0, 0), (0, 0)]
# mod_22_line = [(0, 0), (0, 0)]

# mods profiles
# ---------------

mod_01_profile = profile_line(mod01, mod_01_line[0], mod_01_line[1])
# mod_02_profile = profile_line(mod01, mod_01_line[0], mod_01_line[1])
# mod_03_profile = profile_line(mod01, mod_01_line[0], mod_01_line[1])
# mod_04_profile = profile_line(mod01, mod_01_line[0], mod_01_line[1])
# mod_05_profile = profile_line(mod01, mod_01_line[0], mod_01_line[1])
# mod_22_profile = profile_line(mod01, mod_01_line[0], mod_01_line[1])

print(mod_01_profile.shape)
# x = np.linspace(0, mod_01_profile.shape[0], mod_01_profile.shape[0])


def mod_01_function(x, a, b, w):
    # return a*((x-b)*(np.exp(-1*((x-b)/w)**2)))**2 + d
    return a*((np.sqrt(2)*((x-b)/w))*(np.exp(-1*((x-b)/w)**2)))**2


mod_01_model = Model(mod_01_function)
# model_02 =
# model_03 =
# model_04 =
# model_05 =
# model_22 =

mods = [mod_01_profile]
#         mod_02_profile,
#         mod_03_profile,
#         mod_04_profile,
#         mod_05_profile,
#         mod_22_profile]

# models = [model_01]
#           model_02,
#           model_03,
#           model_04,
#           model_05,
#           model_22]

# plot_mod_analysis(mod_01_profile)

a_init = 1
b_init = 2000
w_init = 1000

x = np.arange(mod_01_profile.shape[0])
x_mod_01_outliers = np.append(np.arange(700, 1900), np.arange(2070, 3200))
x_mod_01 = np.delete(x, x_mod_01_outliers)
result = mod_01_model.fit(mod_01_profile[x_mod_01], x=x[x_mod_01], a=a_init, b=b_init, w=w_init)

print(result.fit_report())

a = result.params['a'].value
b = result.params['b'].value
w = result.params['w'].value

plt.plot(x[x_mod_01], mod_01_profile[x_mod_01], 'o')
plt.plot(x, mod_01_function(x, a_init, b_init, w_init), 'k--', label='initial fit')
plt.plot(x, mod_01_function(x, a, b, w), 'r-', label='best fit')
plt.legend(loc='best')
plt.show()

# result = mod_01_model.fit(mod_01_profile, x=x, a=1/5000, b=2000, w=100, d=0)
# y = mod_01_function(x)
# plt.plot(x, mod_01_function(x, 1/50000, 1800, 500), '*')

# plt.plot(x[x_mod_01], mod_01_profile[x_mod_01], 'o')
# plt.plot(x[x_mod_01], result.init_fit, 'k--', label='initial fit')
# plt.plot(x[x_mod_01], result.best_fit, 'r-', label='best fit')
# plt.legend(loc='best')
# plt.show()
