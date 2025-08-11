import cv2

haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Loading Algorithm

cam = cv2.VideoCapture(0)  # 0- for image capture from default webcam

while True:  # infinite loop

    _, img = cam.read()  # reading frame from camera & _ stores the return value (True/False)

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting color image to grayscale image

    # faces = haar_cascade.detectMultiScale(src, scaleFactor :- how much image size is reduced at each image scale, minNeighbours :- how many neighbors each rectangle should have to be retained)
    faces = haar_cascade.detectMultiScale(grayImg, 1.3, 4)  # getting coordinates

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # drawing a rectangle

    cv2.imshow("FaceDetection", img)  # display the frame

    key = cv2.waitKey(10)
    print(key)
    if key == 27:  # escape key to exit
        break

cam.release()
cv2.destroyAllWindows()
