import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from skimage.measure import profile_line
from skimage import io

# mod02 = io.imread('imgs/02 2.tiff', as_gray=True)
mod02 = io.imread('imgs/laser_modes/2 new2.tiff', as_gray=True)
# print(mod02[0], mod02[-1])

mod_02_line = [(mod02.shape[0]//2, 0), (mod02.shape[0]//2, mod02.shape[1])]
# mod_02_line = [(0, mod02.shape[1]/2), (mod02.shape[0], mod02.shape[1]/2)]
mod_02_profile = profile_line(mod02, mod_02_line[0], mod_02_line[1])

print(mod02.shape)
print(mod_02_profile.shape)


def mod_02_function(x, a, b, w, d):
    return a*(((np.sqrt(2)*((x-b)/w))**2 - 1)**2)*(np.exp(-1*((x-b)/w)**2)) + d


a_init = 1
b_init = 2500
w_init = 300
d_init = 0

mod_02_model = Model(mod_02_function)

x = np.arange(mod_02_profile.shape[0])
# x_mod_02_outliers = np.argwhere(mod_02_profile == np.amax(mod_02_profile)).flatten()

x_mod_02_outliers = np.concatenate([np.arange(1430, 2030), np.arange(2400, 2780), np.arange(3110, 3640)])
print(mod_02_profile)
print("outliers: ", x_mod_02_outliers)
x_mod_02 = np.delete(x, x_mod_02_outliers)

result = mod_02_model.fit(mod_02_profile[x_mod_02], x=x[x_mod_02], a=a_init, b=b_init, w=w_init, d=d_init)

print(result.fit_report())

a = result.params['a'].value
b = result.params['b'].value
w = result.params['w'].value
d = result.params['d'].value

plt.plot(x[x_mod_02], mod_02_profile[x_mod_02], 'o')
plt.plot(x, mod_02_function(x, a_init, b_init, w_init, d_init), 'k--', label='initial fit')
plt.plot(x, mod_02_function(x, a, b, w, d), 'r-', label='best fit')
plt.legend(loc='best')
plt.show()
