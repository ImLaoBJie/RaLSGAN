import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import os
import albumentations as albu

ROOT = os.path.dirname(__file__)
PATH_IMAGES = ROOT + '/all-dogs/'
PATH_ANNO = ROOT + '/Annotation/'
IMAGES = os.listdir(PATH_IMAGES)
breeds = os.listdir(PATH_ANNO)

transform = albu.Compose([
    albu.Rotate(limit=(-5, 5), p=0.3),
    albu.HorizontalFlip(p=0.5)
])


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
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            for o in root.iter('object'):
                bndbox = o.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                # update ↓
                w = np.max((xmax - xmin, ymax - ymin, 64))
                w = np.min((w, width, height))
                xmin = min(max(0, xmin - int((w - (xmax - xmin)) / 2)), width - w)
                xmax = xmin + w
                ymin = min(max(0, ymin - int((w - (ymax - ymin)) / 2)), height - w)
                ymax = ymin + w
                img2 = img.crop((xmin, ymin, xmax, ymax))
                img2 = img2.resize((64, 64), Image.ANTIALIAS)
                img2 = transform(image=np.asarray(img2))['image']
                # update ↑
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