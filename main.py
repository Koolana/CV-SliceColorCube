import cv2
import numpy as np
import math
import time
import multiprocessing as mp
import imageio

# output: transformation matrix between the coordinate system
#         of the colored cube and the slice plane, shape: (3, 3)
def getInvTransformMat():
    e1 = np.array([1, 1, 1])
    e1n = e1 / np.linalg.norm(e1)

    rPoint = np.array([3, 0, 0])
    e2 = rPoint - e1
    e2n = e2 / np.linalg.norm(e2)

    e3 = np.array([np.linalg.det(np.array([[e1[1], e1[2]], [e2[1], e2[2]]])),
                   -np.linalg.det(np.array([[e1[0], e1[2]], [e2[0], e2[2]]])),
                   np.linalg.det(np.array([[e1[0], e1[1]], [e2[0], e2[1]]]))])
    e3n = e3 / np.linalg.norm(e3)

    invE = np.linalg.inv(np.array([e2n, e3n, e1n]))

    return invE

# input: grayValue - value of gray color intensive [0; 255]
#        invE - transformation matrix between the coordinate system
#               of the colored cube and the slice plane
# output: image of the slice plane as numpy array, shape: (width, height, 3)
def createCubeSlice(grayValue, invE):
    width = int(255 * math.sqrt(3))
    height = int(255 * math.sqrt(3))
    grayValue = grayValue * math.sqrt(3)

    img = np.zeros([width, height, 3])

    for w in range (0, width):
        for h in range (0, height):
            vec_new = invE.dot(np.array([w - width / 2, h - height / 2, grayValue])) / 255

            if vec_new[0] < 0 or vec_new[1] < 0 or vec_new[2] < 0 or \
                vec_new[0] > 1 or vec_new[1] > 1 or vec_new[2] > 1:
                continue

            img[w, h, 0] = vec_new[0]
            img[w, h, 1] = vec_new[1]
            img[w, h, 2] = vec_new[2]

    return img

if __name__ == "__main__":
    invE = getInvTransformMat()

    # create pool of process
    multiple_results = [mp.Pool().apply_async(createCubeSlice, (grayValue, invE,)) for grayValue in range(0, 256, 5)]

    time.sleep(1)

    imgs = []

    # show and save to list
    for res in multiple_results:
        img = res.get()
        img = (img * 255).astype(np.uint8)

        cv2.imshow('Cube slice', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        imgs.append(img)
        cv2.waitKey(100)

    # save as gif
    imageio.mimsave('img/example.gif', imgs, 'GIF-FI', **{'fps':20.0, 'quantizer':'nq'})
    print('gif saved\n')

    # save as png
    grayValues = [50, 130, 210]

    for grayValue in grayValues:
        cv2.imwrite('img/gray' + str(grayValue) + '.png',
                    cv2.cvtColor((createCubeSlice(grayValue, invE) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        print('for gray: ' + str(grayValue) + ' saved\n')
