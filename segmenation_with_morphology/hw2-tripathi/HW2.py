import numpy as np
import cv2 as cv
import imutils


def preprocess(img):
    img = cv.resize(img, (300, 300))
    template = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, template = cv.threshold(template, 128, 255, cv.THRESH_BINARY)
    return template


firstframe=0

backSub = cv.createBackgroundSubtractorKNN()



template=[]
cap = cv.VideoCapture(0)

temp = cv.imread('template.jpg')
temp = preprocess(temp)
cv.imwrite("Hand_binary.jpg",temp)
template.append(temp)

temp2 = cv.imread('L.jpg')
temp2 = preprocess(temp2)
cv.imwrite("Loser_binary.jpg",temp2)
template.append(temp2)

temp3 = cv.imread('gun.jpg')
temp3 = preprocess(temp3)
cv.imwrite("gun_binary.jpg",temp3)
template.append(temp3)


temp4 = cv.imread('peace.jpg')
temp4 = preprocess(temp4)
cv.imwrite("peace_binary.jpg",temp4)
template.append(temp4)





#cv.imshow("template", template)

#w, h = template.shape[::-1]
(height, width) = template[0].shape[:2]

if(cap.isOpened()==False):
        print("Error opening camera")

while(cap.isOpened()):
    """if (firstframe==0):
        ret,frame = cap.read()
        firstframe=1
        frame_0 = cv.resize(frame, (800, 600))
        frame_0 = cv.resize(frame, (800, 600))
        frame_0 = cv.cvtColor(frame_0, cv.COLOR_BGR2GRAY)
        #frame = backSub.apply(frame)
"""
    ret, frame = cap.read()
    #cv.resize(frame,(1024,800))
    if ret == True:
        frame = cv.resize(frame, (800, 600))
        gray_image = cv.resize(frame, (800, 600))
        gray_image = cv.cvtColor(gray_image,cv.COLOR_BGR2GRAY)
        #gray_image = cv.absdiff(frame_0,gray_image)
        #gray_image = cv.GaussianBlur(gray_image,(5,5),0)
        gray_image = backSub.apply(frame)
        _,gray_image = cv.threshold(gray_image, 50, 255, cv.THRESH_BINARY)


        temp_found = None
        temp_type=0
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
   #resize the image and store the ratio
            resized_img = imutils.resize(gray_image, width = int(gray_image.shape[1] * scale))
            ratio = gray_image.shape[1] / float(resized_img.shape[1])
            if resized_img.shape[0] < height or resized_img.shape[1] < width:
                break


            for item in template:
                match = cv.matchTemplate(resized_img, item, cv.TM_CCOEFF_NORMED)
                (_, val_max, _, loc_max) = cv.minMaxLoc(match)
                if temp_found is None or val_max>temp_found[0]:
                    temp_found = (val_max, loc_max, ratio)
                    item_type = item
                    cv.imshow("Template Matched",item)

    #Get information from temp_found to compute x,y coordinate
            (_, loc_max, r) = temp_found
            (x_start, y_start) = (int(loc_max[0]), int(loc_max[1]))
            (x_end, y_end) = (int((loc_max[0] + width)), int((loc_max[1] + height)))

    #Draw rectangle around the template
        if(temp_type==0):
            cv.rectangle(frame, (x_start, y_start), (x_end, y_end), (255,255,255), 5)
        if (temp_type == 1):
            cv.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 0), 5)
        if (temp_type == 2):
            cv.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 5)
        if (temp_type == 3):
            cv.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 5)

        cv.imshow("image",frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()


