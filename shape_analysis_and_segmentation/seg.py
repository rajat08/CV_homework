import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img = cv2.resize(img,(1000,1000))
            #img = img[100:400,200:350]
            images.append(img)
    return images

#def find_roi():
#https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
# lower = np.array([0, 48, 80], dtype = "uint8")
# upper = np.array([20, 255, 255], dtype = "uint8")
# 	converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# 	skinMask = cv2.inRange(converted, lower, upper)
# 	# apply a series of erosions and dilations to the mask
# 	# using an elliptical kernel
# 	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
# 	skinMask = cv2.erode(skinMask, kernel, iterations = 2)
# 	skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
# 	# blur the mask to help remove noise, then apply the
# 	# mask to the frame
# 	skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
# 	skin = cv2.bitwise_and(frame, frame, mask = skinMask)
# 	# show the skin in the image along with the mask
# 	cv2.imshow("images", np.hstack([frame, skin]))
def skin_thresh(img):
    conv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower = np.array([20,107,76])
    upper = np.array([25,177,234])
    skinMask = cv2.inRange(conv,lower,upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    skinMask = cv2.erode(skinMask,kernel,iterations=2)
    skinMask = cv2.dilate(skinMask,kernel,iterations=2)
    skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
    skin = cv2.bitwise_and(img,img,mask=skinMask)
    cv2.imshow("image",skin)
    cv2.waitKey(0)


#def find_roi():
#https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
# lower = np.array([0, 48, 80], dtype = "uint8")
# upper = np.array([20, 255, 255], dtype = "uint8")
# 	converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# 	skinMask = cv2.inRange(converted, lower, upper)
# 	# apply a series of erosions and dilations to the mask
# 	# using an elliptical kernel
# 	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
# 	skinMask = cv2.erode(skinMask, kernel, iterations = 2)
# 	skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
# 	# blur the mask to help remove noise, then apply the
# 	# mask to the frame
# 	skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
# 	skin = cv2.bitwise_and(frame, frame, mask = skinMask)
# 	# show the skin in the image along with the mask
# 	cv2.imshow("images", np.hstack([frame, skin]))
def skin_thresh(img):
    conv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower = np.array([20,107,76])
    upper = np.array([25,177,234])
    skinMask = cv2.inRange(conv,lower,upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    skinMask = cv2.erode(skinMask,kernel)
    skinMask = cv2.dilate(skinMask,kernel)
    skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
    skin = cv2.bitwise_and(img,img,mask=skinMask)
    cv2.imshow("image",skin)
    cv2.waitKey(0)

def main():
    folder = 'CS585-PianoImages/'
    images = load_images_from_folder(folder)
    median_frame = np.median(images, axis=0).astype(dtype=np.uint8)
    #diff_frame = images[0]-median_frame
    #diff_frame = cv2.cvtColor(diff_frame,cv2.COLOR_BGR2GRAY)
    #_,diff_frame = cv2.threshold(diff_frame,128,200,cv2.THRESH_BINARY)
    #cv2.imshow("diff",diff_frame)
    #cv2.waitKey(0)
    #median_frame = cv2.cvtColor(median_frame,cv2.COLOR_BGR2HSV)
    for frame in images:
        # Z = frame.reshape((-1,3))
        #
        #
        # Z = np.float32(Z)
        #
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # K = 8
        # ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        #
        #
        # center = np.uint8(center)
        # res = center[label.flatten()]
        # res2 = res.reshape((frame.shape))
        #
        # cv2.imshow('res2',res2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        frame = frame - median_frame
        frame = cv2.dilate(frame,kernel)
        frame = frame[:,:350]
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                val = frame[i,j]
                b,g,r = val[0],val[1],val[2]
                if((r>95 and g>40 and b>20) and ((max(r,g,b)-min(r,g,b)>15)) and (abs(r-g)>15 and r>g and r>b)):
                    frame[i,j] = 255
                else:
                    frame[i,j] = 0
        cv2.imshow('chala kya',frame)
        cv2.waitKey(0)
        #frame = cv2.inRange(frame,upper,lower)

        #skin_thresh(frame)
    #     #skin_thresh(frame)
    #     #subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
    #     #frame = subtractor.apply(frame)
    #     #frame = cv2.absdiff(median_frame,frame)
    #     #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #     #lower_red = np.array([0, 50, 100])
    #     #upper_red = np.array([12, 255, 255])
    #     #frame = frame-median_frame
    #     #frame = cv2.GaussianBlur(frame,(7,7),0)
    #     #frame = cv2.absdiff(median_frame,frame)
    #     #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #     #mask = cv2.inRange(hsv, lower_red,upper_red)
    #
    #     # Bitwise-AND mask and original image
    #     #res = cv2.bitwise_and(frame, frame, mask=mask)
    #
    #     #cv2.imshow('frame', frame)
    #     #cv2.imshow('mask', mask)
    #     #cv2.imshow('res', res)
    #
    #     #subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
    #     #frame = subtractor.apply(frame)

        kernel = np.ones((3,3),np.uint8)

        diff_frame = cv2.subtract(median_frame,frame)

        #diff_frame = cv2.cvtColor(diff_frame,cv2.COLOR_BGR2GRAY)
        #_,diff_frame = cv2.threshold(diff_frame,70,180,cv2.THRESH_BINARY)
        diff_frame = cv2.GaussianBlur(diff_frame,(7,7),0)
        diff_frame = cv2.morphologyEx(diff_frame, cv2.MORPH_CLOSE, kernel)
    #
    #     #diff_frame = subtractor.apply(diff_frame)
    #     #diff_frame = cv2.adaptiveThreshold(diff_frame,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

    #     #diff_frame = diff_frame[:,:500]
    #     #contours, hierarchy = cv2.findContours(diff_frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #     #cnt = sorted(contours, key=cv2.contourArea)
    #     #print(cnt)
    #     #cnt = cnt[:-5]
    #     #cv2.drawContours(diff_frame, cnt, -1, (0, 255, 0), 3)
    #     #upper_bounds = [235,172,136]
    #     #lower_bounds = [113,75,60]
    #     #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #     #mask = cv2.inRange(frame,np.array(lower_bounds),np.array(upper_bounds))
    #     #mask_rgb = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    #     #frame = frame & mask_rgb
    #     #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #     #_frame = cv2.threshold(frame,128,255,cv2.THRESH_BINARY)
    #     #r = cv2.selectROI(diff_frame)
    #     #imCrop = diff_frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    #     cv2.imshow('diff_frame',diff_frame)
    #
    #     #skin_color(frame)
        cv2.imshow('skin thresh',diff_frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
