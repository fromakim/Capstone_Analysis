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
        temp = self.source.copy()
        temp = cv2.polylines(temp, [np.array([list(self.TL), list(self.TR), list(self.BR), list(self.BL)]).reshape(-1, 1, 2)], True, (0, 255, 0))
        # result = cv2.resize(temp, dsize=(640, 480))
        return temp

    def getMask(self):
        mask = np.zeros((self.source.shape[0], self.source.shape[1]), np.int8)
        mask = cv2.fillPoly(mask, [np.array([list(self.TL), list(self.TR), list(self.BR), list(self.BL)]).reshape(-1, 1, 2)], 255)
        return mask

    def getMean(self, mask):
        filtered = self.source.copy()

        intersection = np.logical_and(self.getMask(), mask)
        intersection = intersection.astype(np.uint8)
        intersection *= 255

        filtered = np.bitwise_and(cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY), intersection)
        filtered = filtered.astype(np.uint8)
        filtered *= 255

        filtered = filtered.flatten()
        filtered = filtered.astype(np.float16)

        count = 0
        sum = 0
        for f in filtered:
            if f != 0:
                sum += f
                count += 1
        return math.floor(sum / count)

# %% In[1]: Retrieve List of Image
bmp = list(filter(lambda x : x[-3:] == 'bmp', os.listdir(heatDataPath)))
bmp.sort()

index = 1
# pic = bmp[1]
# if True:
for pic in bmp:
    try:
        # [2]: Image Normalization & Binarization
        original = cv2.split(cv2.imread(heatDataPath + pic))[0]
        show = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        mask = original.copy()

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
        UPPER_PALM = Area(original)
        MIDDLE_PALM = Area(original)
        LITTLE_PALM = Area(original)
        THUMB_PALM = Area(original)

        mask = cv2.normalize(mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # [3]: Contours, Hull, Defect Raw Data
        _, c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        c.sort(key=lambda x: len(x), reverse = True)

        h = cv2.convexHull(c[0], returnPoints = False)
        hp = list(filter(lambda x: x[0][1] > mask.shape[0] / 4, cv2.convexHull(c[0])))
        df = cv2.convexityDefects(c[0], h)





        # [4]: Find Between Points via centroid
        M = cv2.moments(mask)
        centroid = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        between = list(map(lambda x : np.array(c[0][x[0][2]][0]), df))
        # between = list(filter(lambda x: x[1] > mask.shape[0] / 3, between))
        between.sort(key=lambda x : (centroid[0] - x[0]) ** 2 + (centroid[1] - x[1]) ** 2)
        between = between[:4]
        between.sort(key=lambda x : x[0])



        # [5]: Find FingerTips
        hp.sort(key=lambda x : x[0][0])
        tips = [np.array(hp[0][0])]
        for p in hp:
            if np.linalg.norm(p[0][0] - tips[-1][0]) > 10:
                tips.append(p[0])



        # [6]: Set Points of Area
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
        index_approx = c[0][np.linalg.norm((c[0] - index_approx), axis=-1).argmin()][0]

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
        little_approx = c[0][np.linalg.norm((c[0] - little_approx), axis=-1).argmin()][0]

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

        thumb_points = np.argwhere(c[0] == thumb_approx)
        thumb_points = list(filter(None.__ne__, list(map(lambda x: x if (c[0][x[0]][0][0] == thumb_approx[0] and c[0][x[0]][0][1] == thumb_approx[1]) else None, thumb_points))))[0]
        thumb_part = c[0][0:thumb_points[0] + 1]

        thumb_approx = thumb_part[np.linalg.norm((thumb_part - between[0]), axis=-1).argmin()][0]

        THUMB_A.setPoints([
            np.rint(tips[0] / 2 + between[0] / 4 + thumb_approx / 4 - between[0] * 0.5 + thumb_approx * 0.5).astype(int),
            np.rint(tips[0] / 2 + between[0] / 4 + thumb_approx / 4 + between[0] * 0.5 - thumb_approx * 0.5).astype(int),
            np.rint(tips[0] - between[0] * 0.5 + thumb_approx * 0.5).astype(int),
            np.rint(tips[0] + between[0] * 0.5 - thumb_approx * 0.5).astype(int)
        ])
        THUMB_B.setPoints([
            thumb_approx,
            between[0],
            np.rint(tips[0] / 2 + between[0] / 4 + thumb_approx / 4 - between[0] * 0.5 + thumb_approx * 0.5).astype(int),
            np.rint(tips[0] / 2 + between[0] / 4 + thumb_approx / 4 + between[0] * 0.5 - thumb_approx * 0.5).astype(int),
        ])

        start = c[0][0][0]
        end = c[0][-1][0]
        middle = np.rint((start + end) / 2).astype(int)
        center = np.rint((start + end + index_approx + little_approx) / 4).astype(int)

        UPPER_PALM.setPoints([
            np.rint(start / 4 + index_approx / 4 * 3).astype(int),
            np.rint(end / 4 + little_approx / 4 * 3).astype(int),
            index_approx,
            little_approx
        ])
        MIDDLE_PALM.setPoints([
            center - np.array([1, 0]),
            center + np.array([1, 0]),
            np.rint(start / 4 + index_approx / 4 * 3).astype(int),
            np.rint(end / 4 + little_approx / 4 * 3).astype(int)
        ])
        THUMB_PALM.setPoints([
            start,
            middle,
            np.rint(start / 4 + index_approx / 4 * 3).astype(int),
            center
        ])
        LITTLE_PALM.setPoints([
            middle,
            end,
            center,
            np.rint(end / 4 + little_approx / 4 * 3).astype(int)
        ])

        cv2.namedWindow('MA')
        cv2.namedWindow('MB')
        cv2.namedWindow('MC')
        cv2.namedWindow('RA')
        cv2.namedWindow('RB')
        cv2.namedWindow('RC')

        cv2.namedWindow('IA')
        cv2.namedWindow('IB')
        cv2.namedWindow('IC')
        cv2.namedWindow('LA')
        cv2.namedWindow('LB')
        cv2.namedWindow('LC')

        cv2.namedWindow('TA')
        cv2.namedWindow('TB')

        cv2.namedWindow('PU')
        cv2.namedWindow('PM')
        cv2.namedWindow('PT')
        cv2.namedWindow('PL')

        cv2.moveWindow('TA', 0, 0)
        cv2.moveWindow('TB', 400, 0)

        cv2.moveWindow('IA', 0, 200)
        cv2.moveWindow('IB', 400, 200)
        cv2.moveWindow('IC', 800, 200)

        cv2.moveWindow('MA', 0, 400)
        cv2.moveWindow('MB', 400, 400)
        cv2.moveWindow('MC', 800, 400)

        cv2.moveWindow('RA', 0, 600)
        cv2.moveWindow('RB', 400, 600)
        cv2.moveWindow('RC', 800, 600)

        cv2.moveWindow('LA', 0, 800)
        cv2.moveWindow('LB', 400, 800)
        cv2.moveWindow('LC', 800, 800)

        cv2.moveWindow('PU', 1400, 0)
        cv2.moveWindow('PM', 1400, 200)
        cv2.moveWindow('PT', 1400, 400)
        cv2.moveWindow('PL', 1400, 600)

        cv2.imshow('MA', MIDDLE_A.drawPoints())
        cv2.imshow('MB', MIDDLE_B.drawPoints())
        cv2.imshow('MC', MIDDLE_C.drawPoints())
        cv2.imshow('RA', RING_A.drawPoints())
        cv2.imshow('RB', RING_B.drawPoints())
        cv2.imshow('RC', RING_C.drawPoints())

        cv2.imshow('IA', INDEX_A.drawPoints())
        cv2.imshow('IB', INDEX_B.drawPoints())
        cv2.imshow('IC', INDEX_C.drawPoints())
        cv2.imshow('LA', LITTLE_A.drawPoints())
        cv2.imshow('LB', LITTLE_B.drawPoints())
        cv2.imshow('LC', LITTLE_C.drawPoints())

        cv2.imshow('TA', THUMB_A.drawPoints())
        cv2.imshow('TB', THUMB_B.drawPoints())

        cv2.imshow('PU', UPPER_PALM.drawPoints())
        cv2.imshow('PM', MIDDLE_PALM.drawPoints())
        cv2.imshow('PT', THUMB_PALM.drawPoints())
        cv2.imshow('PL', LITTLE_PALM.drawPoints())

        res = cv2.waitKey()
        cv2.destroyAllWindows()

        if res == 122:
            print(index, ':', THUMB_A.getMean(mask), THUMB_B.getMean(mask), INDEX_A.getMean(mask), INDEX_B.getMean(mask), INDEX_C.getMean(mask), MIDDLE_A.getMean(mask), MIDDLE_B.getMean(mask), MIDDLE_C.getMean(mask), RING_A.getMean(mask), RING_B.getMean(mask), RING_C.getMean(mask), LITTLE_A.getMean(mask), LITTLE_B.getMean(mask), LITTLE_C.getMean(mask), UPPER_PALM.getMean(mask), MIDDLE_PALM.getMean(mask), THUMB_PALM.getMean(mask), LITTLE_PALM.getMean(mask))
        else:
            print(index, ':', 'invalid data')
    except:
        print(index, ':', 'invalid data with error')
        pass
    finally:
        index = index + 1

# %% In[99]: Test
show = cv2.circle(show, centroid, 3, (255, 255, 255), -1)

show = cv2.drawContours(show, list(map(lambda x : np.array([[x[0], x[1]]]), between)), -1, (0, 255, 255), 2)
show = cv2.drawContours(show, c[0], -1, (0, 255, 0), 1)
show = cv2.drawContours(show, hp, -1, (0, 0, 255), 1)

result = cv2.resize(show, dsize=(640, 480))

cv2.imshow('show', show)
cv2.imshow('result', result)
cv2.waitKey()
cv2.destroyAllWindows()
