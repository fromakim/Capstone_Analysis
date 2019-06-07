# %% In[0]: Packages & Settings
import cv2
import os
import math
import numpy as np
heatDataPath = './data/Sub1/HeatCamera/'

class Area:
    def __init__(self, source):
        self.source = cv2.cvtColor(source.copy(), cv2.COLOR_GRAY2BGR)
        self.TL = (0, 0)
        self.TR = (0, 0)
        self.BL = (0, 0)
        self.BR = (0, 0)

    def setPoints(self, point):
        self.TL = point[0]
        self.TR = point[1]
        self.BL = point[2]
        self.BR = point[3]

    def drawPoints(self):
        temp = cv2.polylines(self.source, [np.array([list(self.TL), list(self.TR), list(self.BR), list(self.BL)]).reshape(-1, 1, 2)], True, (0, 255, 0))
        result = cv2.resize(temp, dsize=(640, 480))
        return result

# %% In[1]: Retrieve List of Image
bmp = list(filter(lambda x : x[-3:] == 'bmp', os.listdir(heatDataPath)))
bmp.sort()



# %% In[2]: Image Normalization & Binarization
original = cv2.split(cv2.imread(heatDataPath + bmp[0]))[0]
show = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
img = original.copy()

THUMB_A = Area(original)
THUMB_B = Area(original)
INDEX_A = Area(original)
INDEX_B = Area(original)
INDEX_C = Area(original)
MIDDLE_A = Area(original)
MIDDLE_B = Area(original)
MIDDLE_C = Area(original)
RING_A = Area(original)
RING_B = Area(original)
RING_C = Area(original)
LITTLE_A = Area(original)
LITTLE_B = Area(original)
LITTLE_C = Area(original)

img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)



# %% In[3]: Contours, Hull, Defect Raw Data
c, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
h = cv2.convexHull(c[0], returnPoints = False)
hp = list(filter(lambda x: x[0][1] > img.shape[0] / 4, cv2.convexHull(c[0])))
df = cv2.convexityDefects(c[0], h)



# %% In[4]: Find Between Points via centroid
M = cv2.moments(img)
centroid = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

between = list(map(lambda x : np.array(c[0][x[0][2]][0]), df))
between = list(filter(lambda x: x[1] > img.shape[0] / 3, between))
between.sort(key=lambda x : (centroid[0] - x[0]) ** 2 + (centroid[1] - x[1]) ** 2)
between = between[:4]
between.sort(key=lambda x : x[0])



# %% In[5]: Find FingerTips
hp.sort(key=lambda x : x[0][0])
tips = [np.array(hp[0][0])]
for p in hp:
    if np.linalg.norm(p[0] - tips[-1]) > 10:
        tips.append(p[0])



# %% In[6]: Set Points of Area
MIDDLE_A.setPoints([
    np.rint((tips[2] * 2 / 3 + between[1] / 6 + between[2] / 6) - between[2] * 0.5 + between[1] * 0.5).astype(int),
    np.rint((tips[2] * 2 / 3 + between[1] / 6 + between[2] / 6) + between[2] * 0.5 - between[1] * 0.5).astype(int),
    np.rint(tips[2] - between[2] * 0.5 + between[1] * 0.5).astype(int),
    np.rint(tips[2] + between[2] * 0.5 - between[1] * 0.5).astype(int)
])
MIDDLE_B.setPoints([
    np.rint((tips[2] * 1 / 3 + between[1] / 3 + between[2] / 3) - between[2] * 0.5 + between[1] * 0.5).astype(int),
    np.rint((tips[2] * 1 / 3 + between[1] / 3 + between[2] / 3) + between[2] * 0.5 - between[1] * 0.5).astype(int),
    np.rint((tips[2] * 2 / 3 + between[1] / 6 + between[2] / 6) - between[2] * 0.5 + between[1] * 0.5).astype(int),
    np.rint((tips[2] * 2 / 3 + between[1] / 6 + between[2] / 6) + between[2] * 0.5 - between[1] * 0.5).astype(int)
])
MIDDLE_C.setPoints([
    between[1],
    between[2],
    np.rint((tips[2] * 1 / 3 + between[1] / 3 + between[2] / 3) - between[2] * 0.5 + between[1] * 0.5).astype(int),
    np.rint((tips[2] * 1 / 3 + between[1] / 3 + between[2] / 3) + between[2] * 0.5 - between[1] * 0.5).astype(int)
])
RING_A.setPoints([
    np.rint((tips[3] * 2 / 3 + between[2] / 6 + between[3] / 6) - between[3] * 0.5 + between[2] * 0.5).astype(int),
    np.rint((tips[3] * 2 / 3 + between[2] / 6 + between[3] / 6) + between[3] * 0.5 - between[2] * 0.5).astype(int),
    np.rint(tips[3] - between[3] * 0.5 + between[2] * 0.5).astype(int),
    np.rint(tips[3] + between[3] * 0.5 - between[2] * 0.5).astype(int)
])
RING_B.setPoints([
    np.rint((tips[3] * 1 / 3 + between[2] / 3 + between[3] / 3) - between[3] * 0.5 + between[2] * 0.5).astype(int),
    np.rint((tips[3] * 1 / 3 + between[2] / 3 + between[3] / 3) + between[3] * 0.5 - between[2] * 0.5).astype(int),
    np.rint((tips[3] * 2 / 3 + between[2] / 6 + between[3] / 6) - between[3] * 0.5 + between[2] * 0.5).astype(int),
    np.rint((tips[3] * 2 / 3 + between[2] / 6 + between[3] / 6) + between[3] * 0.5 - between[2] * 0.5).astype(int)
])
RING_C.setPoints([
    between[2],
    between[3],
    np.rint((tips[3] * 1 / 3 + between[2] / 3 + between[3] / 3) - between[3] * 0.5 + between[2] * 0.5).astype(int),
    np.rint((tips[3] * 1 / 3 + between[2] / 3 + between[3] / 3) + between[3] * 0.5 - between[2] * 0.5).astype(int)
])

index_approx = between[1] - between[2] + between[1]
index_approx = c[0][np.linalg.norm((c - index_approx), axis=-1).argmin()][0]

INDEX_A.setPoints([
    np.rint((tips[1] * 2 / 3 + index_approx / 6 + between[1] / 6) - between[1] * 0.5 + index_approx * 0.5).astype(int),
    np.rint((tips[1] * 2 / 3 + index_approx / 6 + between[1] / 6) + between[1] * 0.5 - index_approx * 0.5).astype(int),
    np.rint(tips[1] - between[1] * 0.5 + index_approx * 0.5).astype(int),
    np.rint(tips[1] + between[1] * 0.5 - index_approx * 0.5).astype(int)
])
INDEX_B.setPoints([
    np.rint((tips[1] * 1 / 3 + index_approx / 3 + between[1] / 3) - between[1] * 0.5 + index_approx * 0.5).astype(int),
    np.rint((tips[1] * 1 / 3 + index_approx / 3 + between[1] / 3) + between[1] * 0.5 - index_approx * 0.5).astype(int),
    np.rint((tips[1] * 2 / 3 + index_approx / 6 + between[1] / 6) - between[1] * 0.5 + index_approx * 0.5).astype(int),
    np.rint((tips[1] * 2 / 3 + index_approx / 6 + between[1] / 6) + between[1] * 0.5 - index_approx * 0.5).astype(int),
])
INDEX_C.setPoints([
    index_approx,
    between[1],
    np.rint((tips[1] * 1 / 3 + index_approx / 3 + between[1] / 3) - between[1] * 0.5 + index_approx * 0.5).astype(int),
    np.rint((tips[1] * 1 / 3 + index_approx / 3 + between[1] / 3) + between[1] * 0.5 - index_approx * 0.5).astype(int),
])

little_approx = between[3] + between[3] - between[2]
little_approx = c[0][np.linalg.norm((c - little_approx), axis=-1).argmin()][0]

LITTLE_A.setPoints([
    np.rint((tips[4] * 2 / 3 + between[3] / 6 + little_approx / 6) - little_approx * 0.5 + between[3] * 0.5).astype(int),
    np.rint((tips[4] * 2 / 3 + between[3] / 6 + little_approx / 6) + little_approx * 0.5 - between[3] * 0.5).astype(int),
    np.rint(tips[4] - little_approx * 0.5 + between[3] * 0.5).astype(int),
    np.rint(tips[4] + little_approx * 0.5 - between[3] * 0.5).astype(int)
])
LITTLE_B.setPoints([
    np.rint((tips[4] * 1 / 3 + between[3] / 3 + little_approx / 3) - little_approx * 0.5 + between[3] * 0.5).astype(int),
    np.rint((tips[4] * 1 / 3 + between[3] / 3 + little_approx / 3) + little_approx * 0.5 - between[3] * 0.5).astype(int),
    np.rint((tips[4] * 2 / 3 + between[3] / 6 + little_approx / 6) - little_approx * 0.5 + between[3] * 0.5).astype(int),
    np.rint((tips[4] * 2 / 3 + between[3] / 6 + little_approx / 6) + little_approx * 0.5 - between[3] * 0.5).astype(int),
])
LITTLE_C.setPoints([
    between[3],
    little_approx,
    np.rint((tips[4] * 1 / 3 + between[3] / 3 + little_approx / 3) - little_approx * 0.5 + between[3] * 0.5).astype(int),
    np.rint((tips[4] * 1 / 3 + between[3] / 3 + little_approx / 3) + little_approx * 0.5 - between[3] * 0.5).astype(int),
])

thumb_approx = c[0][0][0]
for cp in c[0]:
    if between[0][1] < thumb_approx[1]:
        break
    thumb_approx = cp[0]
    pass

# cv2.imshow('MA', MIDDLE_A.drawPoints())
# cv2.imshow('MB', MIDDLE_B.drawPoints())
# cv2.imshow('MC', MIDDLE_C.drawPoints())
# cv2.imshow('RA', RING_A.drawPoints())
# cv2.imshow('RB', RING_B.drawPoints())
# cv2.imshow('RC', RING_C.drawPoints())
cv2.imshow('IA', INDEX_A.drawPoints())
cv2.imshow('IB', INDEX_B.drawPoints())
cv2.imshow('IC', INDEX_C.drawPoints())

cv2.imshow('LA', LITTLE_A.drawPoints())
cv2.imshow('LB', LITTLE_B.drawPoints())
cv2.imshow('LC', LITTLE_C.drawPoints())

cv2.waitKey()
cv2.destroyAllWindows()




# %% In[99]: Test
show = cv2.drawContours(show, list(map(lambda x : np.array([[x[0], x[1]]]), between)), -1, (0, 255, 255), 2)
show = cv2.drawContours(show, c[0], -1, (0, 255, 0), 1)
show = cv2.drawContours(show, hp, -1, (0, 0, 255), 1)

result = cv2.resize(show, dsize=(640, 480))

cv2.imshow('show', show)
cv2.imshow('result', result)
cv2.waitKey()
cv2.destroyAllWindows()
