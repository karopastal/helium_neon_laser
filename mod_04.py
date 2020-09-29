import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from skimage.measure import profile_line
from skimage import io

mod = io.imread('imgs/laser_modes/4nnew.tiff', as_gray=True)

# mod_line = [(mod.shape[0]//2, 0), (mod.shape[0]//2, mod.shape[1])]
mod_line = [(0, mod.shape[1]/2), (mod.shape[0], mod.shape[1]/2)]

mod_profile = profile_line(mod, mod_line[0], mod_line[1])

print(mod.shape)
print(mod_profile.shape)


def mod_function(x, a, b, w, d):
    var_x = ((x-b)/w)
    part_1 = ((np.sqrt(2)*var_x)**4 - 6*(np.sqrt(2)*var_x)**2 + 3)
    part_2 = np.exp(-1*var_x**2)

    expression = a*(part_1**2)*part_2 + d

    return expression


a_init = 0.5
b_init = 2000
w_init = 500
d_init = 0

mod_model = Model(mod_function)

x = np.arange(mod_profile.shape[0])

x_mod_outliers = np.concatenate([np.arange(595, 1045),
                                 np.arange(1315, 1580),
                                 np.arange(1790, 2115),
                                 np.arange(2285, 2705),
                                 np.arange(2920, 3560)])
# x_mod_outliers = []

print(mod_profile)
print("outliers: ", x_mod_outliers)
x_mod = np.delete(x, x_mod_outliers)

result = mod_model.fit(mod_profile[x_mod], x=x[x_mod], a=a_init, b=b_init, w=w_init, d=d_init)

print(result.fit_report())

a = result.params['a'].value
b = result.params['b'].value
w = result.params['w'].value
d = result.params['d'].value

# plt.plot(x[x_mod], mod_profile[x_mod], 'o')
# # plt.plot(x, mod_function(x, a_init, b_init, w_init, d_init), 'k--', label='initial fit')
# plt.plot(x, mod_function(x, a, b, w, d), 'r-', label='best fit')
# plt.legend(loc='best')
# plt.show()


plt.plot(x[x_mod], mod_profile[x_mod], 'bo', label='profile line')
plt.plot(x[x_mod_outliers], mod_profile[x_mod_outliers], 'ko', label='excluded saturation')
plt.plot(x, mod_function(x, a, b, w, d), 'r-', label='fit')
plt.title('mode 40' % (), fontsize=14)
plt.xlabel('X (pixels)', fontsize=15)
plt.ylabel('Intensity (A.U)', fontsize=15)
plt.legend(loc='best')
plt.show()