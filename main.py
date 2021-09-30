import cv2
import numpy as np
import math
import time
import tkinter

def createCubeSlice(grayValue):
    width = int(255 * math.sqrt(3))
    height = int(255 * math.sqrt(3))
    grayValue = grayValue * math.sqrt(3)

    img = np.zeros([width, height, 3])

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

    for w in range (0, width):
        for h in range (0, height):
            vec_new = invE.dot(np.array([w - width / 2, h - height / 2, grayValue])) / 255

            if vec_new[0] < 0 or vec_new[1] < 0 or vec_new[2] < 0 or \
                vec_new[0] > 1 or vec_new[1] > 1 or vec_new[2] > 1:
                continue

            # print(vec_new)
            img[w, h, 0] = vec_new[0]
            img[w, h, 1] = vec_new[1]
            img[w, h, 2] = vec_new[2]

    return img

def drawSlide(slideVal):
    slideVal = int(slideVal)
    img = createCubeSlice(slideVal)
    cv2.imshow('Slice', img)
    print(slideVal)


if __name__ == "__main__":
    # root = tkinter.Tk()
    # tkinter.Scale(orient='horizontal', from_=0, to=255, command=drawSlide).pack()
    drawSlide(150)
    # root.mainloop()

    cv2.waitKey(0)
