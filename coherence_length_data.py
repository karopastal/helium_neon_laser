import os
import pprint

pp = pprint.PrettyPrinter(indent=2)
path = 'imgs/coherence_length'

image_paths = os.listdir(path)

modes = {
    '0': [],
    '3': [],
    '4': [],
    '31': []
}


def data():
    for image_path in image_paths:
        mode = {}
        data = image_path.split('.')
        mode['x'] = float(data[1].replace('_', '.'))
        mode['path'] = path + '/' + image_path

        modes[data[0]].append(mode)

    return modes


