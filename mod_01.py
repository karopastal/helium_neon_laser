import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from skimage.measure import profile_line
from skimage import io


mod01 = io.imread('imgs/laser_modes/1nnew.tiff', as_gray=True)
mod_01_line = [(0, mod01.shape[1]/2), (mod01.shape[0], mod01.shape[1]/2)]
mod_01_profile = profile_line(mod01, mod_01_line[0], mod_01_line[1])

print(mod01.shape)
print(mod_01_profile.shape)


def mod_01_function(x, a, b, w):
    return a*((np.sqrt(2)*((x-b)/w))**2)*(np.exp(-1*((x-b)/w)**2))


a_init = 10
b_init = 2000
w_init = 500

mod_01_model = Model(mod_01_function)

x = np.arange(mod_01_profile.shape[0])
x_mod_01_outliers = np.append(np.arange(700, 1900), np.arange(2070, 3200))
x_mod_01 = np.delete(x, x_mod_01_outliers)

result = mod_01_model.fit(mod_01_profile[x_mod_01], x=x[x_mod_01], a=a_init, b=b_init, w=w_init)

print(result.fit_report())

a = result.params['a'].value
b = result.params['b'].value
w = result.params['w'].value


plt.plot(x[x_mod_01], mod_01_profile[x_mod_01], 'bo', label='profile line')
plt.plot(x[x_mod_01_outliers], mod_01_profile[x_mod_01_outliers], 'ko', label='excluded saturation')
plt.plot(x, mod_01_function(x, a, b, w), 'r-', label='fit')
plt.title('mode 10' % (), fontsize=14)
plt.xlabel('X (pixels)', fontsize=15)
plt.ylabel('Intensity (A.U)', fontsize=15)
plt.legend(loc='best')
plt.show()
