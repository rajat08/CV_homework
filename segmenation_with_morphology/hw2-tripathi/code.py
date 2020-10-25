import numpy as np
import cv2 as cv

# Filter for morphological transformation(Closing)
kernel = np.ones((5,5), np.uint8)

#Method to preprocess the Template.
def preprocess(img):
    img = cv.resize(img, (300, 300))
    template = cv.cvtColor(img, cv.COLOR_BGR2GRAY)                  #Convert BGR to grayscale
    _, template = cv.threshold(template, 128, 255, cv.THRESH_BINARY)            # Thresholding to convert image to binary
    template = cv.dilate(template,kernel,iterations=1)                  # Performing closing to remove holes from the template
    template = cv.erode(template,kernel,iterations=1)
    template = cv.GaussianBlur(template,(5,5),0)
    return template

def readTemplates():
    template = []  # List of templates

    temp = cv.imread('template.jpg')
    temp = preprocess(temp)
    cv.imwrite("Hand_binary.jpg", temp)
    template.append(temp)

    temp2 = cv.imread('L.jpg')
    temp2 = preprocess(temp2)
    cv.imwrite("Loser_binary.jpg", temp2)
    template.append(temp2)

    temp3 = cv.imread('gun.jpg')
    temp3 = preprocess(temp3)
    cv.imwrite("gun_binary.jpg", temp3)
    template.append(temp3)

    temp4 = cv.imread('peace.jpg')
    temp4 = preprocess(temp4)
    cv.imwrite("peace_binary.jpg", temp4)
    template.append(temp4)

    return template

 # FirstFrame is the flag that helps me capture the first frame. We then Subtract every subsequent frame from this first frame which helps us remove background
def readFirstFrame(cap):
    firstframe=0
    if (cap.isOpened() == False):
        print("Error opening camera")

    while (cap.isOpened()):
        if (firstframe == 0):
            ret, frame = cap.read()
            firstframe = 1
            frame_0 = cv.resize(frame, (800, 600))
            frame_0 = cv.cvtColor(frame_0, cv.COLOR_BGR2GRAY)  # Convert to grayscale
            frame_0 = cv.dilate(frame_0, kernel, iterations=1)  # Performing closing to remove holes from the frames
            frame_0 = cv.erode(frame_0, kernel, iterations=1)
            frame_0 = cv.GaussianBlur(frame_0, (5, 5), 0)
            return frame_0

# Method to reaad every subsequent frame after the first frame
def subsequestFrames(cap,frame_0):
    ret, frame = cap.read()
    # cv.resize(frame,(1024,800))
    if ret == True:
        frame = cv.resize(frame, (800, 600))
        gray_image = cv.resize(frame, (800, 600))
        gray_image = cv.cvtColor(gray_image, cv.COLOR_BGR2GRAY)  # Convert to grayscale
        # gray_image = cv.GaussianBlur(gray_image,(5,5),0)
        gray_image = cv.absdiff(frame_0,gray_image)  # Using absolute difference to substact every frame from the first frame
        # gray_image = backSub.apply(frame)
        _, img = cv.threshold(gray_image, 50, 255, cv.THRESH_BINARY)  # Thresholding based on skin color to binarize
        gray_image = cv.dilate(gray_image, kernel, iterations=1)
        gray_image = cv.erode(gray_image, kernel, iterations=1)
        return img,frame




def matchTempalte(template,img,frame,w,h):
        temp_found = None
        for item in template:
            res = cv.matchTemplate(img, item, cv.TM_CCOEFF_NORMED)                  #Using normalized coefficient corelation to match template
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)                  #MinMaxLoc() gives me the values of Minimum and Maximum of the normalized coefficient and the location for that.
            if temp_found is None or max_val>temp_found[0]:
                temp_found = [max_val,max_loc]
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv.imshow("Template Matched", item)                         # Show the template matched.

        cv.rectangle(frame, top_left, bottom_right, 255, 2)                     #cv.rectangle creates the bounding box around the detected object
        cv.putText(frame,"Coefficient : " + str(max_val),top_left,cv.FONT_HERSHEY_COMPLEX_SMALL,2,(255,255,255),1)

        cv.imshow("image", frame)


def main():
    cap = cv.VideoCapture(0)
    while(cap.isOpened()):
        template = readTemplates()
        w, h = template[0].shape[::-1]
        frame_0 = readFirstFrame(cap)
        img,frame = subsequestFrames(cap,frame_0)
        matchTempalte(template,img,frame,w,h)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__=="__main__":
    main()