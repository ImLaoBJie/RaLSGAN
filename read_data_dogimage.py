import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import os


ROOT = os.path.dirname(__file__)
PATH_IMAGES = ROOT + '/all-dogs/'
PATH_ANNO = ROOT + '/Annotation/'
IMAGES = os.listdir(PATH_IMAGES)
breeds = os.listdir(PATH_ANNO)


def load_data():
    idxIn = 0
    namesIn = []
    imagesIn = np.zeros((22000, 64, 64, 3))

    for breed in breeds:
        for dog in os.listdir(PATH_ANNO + breed):
            try:
                img = Image.open(PATH_IMAGES + dog + '.jpg')
            except:
                continue
            tree = ET.parse(PATH_ANNO + breed + '/' + dog)
            root = tree.getroot()
            for o in root.iter('object'):
                bndbox = o.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                w = np.min((xmax - xmin, ymax - ymin))
                img2 = img.crop((xmin, ymin, xmin + w, ymin + w))
                img2 = img2.resize((64, 64), Image.ANTIALIAS)
                imagesIn[idxIn, :, :, :] = np.asarray(img2)
                if idxIn % 1000 == 0:
                    print(idxIn)
                namesIn.append(breed)
                idxIn += 1
                if idxIn > 21000:
                    break

        if idxIn > 21000:
            break
    idx = np.arange(idxIn)
    np.random.shuffle(idx)
    imagesIn = imagesIn[idx, :, :, :]
    namesIn = np.array(namesIn)[idx]

    return imagesIn, namesIn