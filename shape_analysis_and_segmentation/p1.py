import cv2 as cv
import numpy as np


def read_data():
    image = cv.imread("2.png")
    cv.imshow('sdf',image)
    cv.waitKey(0)
    image = cv.resize(image,(300,300))
    return image


def closing(img):
    mask = np.ones((3,3))
    img = cv.dilate(img, mask)
    img = cv.erode(img,mask)
    return img


def opening(img):
    mask = np.ones((13,13))
    img = cv.erode(img, mask)
    img = cv.dilate(img, mask)
    return img


def preprocess(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV)
    return img


def fill(data, start_coords, fill_value):
    """
    Flood fill algorithm

    Parameters
    ----------
    data : (M, N) ndarray of uint8 type
    Image with flood to be filled. Modified inplace.
    start_coords : tuple
    Length-2 tuple of ints defining (row, col) start coordinates.
    fill_value : int
    Value the flooded area will take after the fill.

    Returns
    -------
    None, ``data`` is modified inplace.
    """
    fill_value = np.array(fill_value)
    ndata = np.copy(data)
    print(fill_value)
    ndata = cv.cvtColor(ndata,cv.COLOR_GRAY2RGB)
    #cv.imshow("unfilled",ndata)
    ndata = np.array(ndata)
    xsize, ysize = ndata.shape[0],ndata.shape[1]
    #print(xsize,ysize)
    orig_value = ndata[start_coords[0], start_coords[1]]
    #print('orig',orig_value)
    stack = set(((start_coords[0], start_coords[1]),))
    #print('fill',fill_value)
    v = fill_value == orig_value
    if v.all():
        raise ValueError("Filling region with same value "
                         "already present is unsupported. "
                         "Did you already fill this region?")

    while stack:

        x, y = stack.pop()
        #print(data[x,y])
        v = (ndata[x,y]==orig_value)
        #print(v)
        if v.all():
            #print('pre change',data[x,y])
            ndata[x,y] = fill_value
            print('changed color',ndata[x,y])
            if x > 0:
                stack.add((x - 1, y))
            if x < (xsize - 1):
                stack.add((x + 1, y))
            if y > 0:
                stack.add((x, y - 1))
            if y < (ysize - 1):
                stack.add((x, y + 1))
    print('this is what happens',ndata[62,183])
    cv.imshow("filled_color",ndata)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Maveshi istyle
def getCoord(data):
    cords = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if(data[i][j]==255):
                return i,j


def getCont(cont):
    c_list = []
    for i in range(len(cont)):
        temp = np.array(cont[i][0,0])
        #print(temp)
        data = temp[1],temp[0]
        #print(data)
        c_list.append(data)
    return c_list


def runFlood(data,list):
    value = [255,0,0]
    for item in list:
        fill(data, item, value)
        #value = value+90



def main():
    #cv
    data = read_data()
    data = preprocess(data)
    data = closing(data)
    data = cv.medianBlur(data,5)
    #cv.imshow("preprocessed",data)
    contours, hierarchy = cv.findContours(data, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #cont_drawn = cv.drawContours(data,contours,-1,128,0)
    #cv.imshow("contours drawn",data)
    contours = np.array(contours)
    c_list = getCont(contours)
    runFlood(data,c_list)
    #l = np.array(contours[1][0,0])            #Read cotours pixel values



    #m = [l[0]+1,l[1]+1]
    #print(l[0])                    #contours y values
    #print(l[1])                       # contours x value
    #print(data[269,212])

    #cord = getCoord(data)              #Maveshi istyle of getting pixel
    #print(cord)


    cv.waitKey(0)
    cv.destroyAllWindows()




if __name__ == "__main__":
    main()
