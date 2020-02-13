import cv2
import numpy as np
import pyttsx3


def nothing(x):
    pass


image_x, image_y = 64, 64

from keras.models import load_model
classifier = load_model('Trained_model.h5')


def predictor():
    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img('1.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    if result[0][0] == 1:
        return 'Excuse'
    elif result[0][1] == 1:
        return 'Hello'
    elif result[0][2] == 1:
        return 'Me'
    elif result[0][3] == 1:
        return 'Good Morning'
    elif result[0][4] == 1:
        return 'Mam'
    elif result[0][5] == 1:
        return 'You'
    elif result[0][6] == 1:
        return 'Thank'
    elif result[0][7] == 1:
        return 'Tell'
    elif result[0][8] == 1:
        return 'Can'
    elif result[0][9] == 1:
        return 'Way'
    elif result[0][10] == 1:
        return 'Bank'
    elif result[0][11] == 1:
        return 'From'
    elif result[0][12] == 1:
        return 'Traffic Signal'
    elif result[0][13] == 1:
        return 'After'
    elif result[0][14] == 1:
        return 'Okay'
    elif result[0][15] == 1:
        return 'Round About'
    elif result[0][16] == 1:
        return 'Left'
    elif result[0][17] == 1:
        return 'Right'
    elif result[0][18] == 1:
        return 'Bye'
    elif result[0][19] == 1:
        return 'Turn'
    elif result[0][20] == 1:
        return 'Upwards'
    elif result[0][21] == 1:
        return 'Go'
    elif result[0][22] == 1:
        return 'Downwards'
    elif result[0][23] == 1:
        return 'Guide'
    elif result[0][24] == 1:
        return 'Far'
    elif result[0][25] == 1:
        return 'Near'


cam = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cv2.namedWindow("test")

img_counter = 0

img_text = ''
# -----------------------------------------------------------------------
temp = 0
prelabel = None
pretext = " "
label = "Happy"
engine = pyttsx3.init()
#-------------------------------------------------------------------------
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0),
                        thickness=2,
                        lineType=8,
                        shift=0)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    if (label != None):
        if (temp == 0):
            prelabel = label

        if (prelabel == label):
            label = prelabel
            temp += 1
        else:
            temp = 0

        if (temp == 30):
            img_text = img_text + " " + label

        if (len(img_text) > 50):
            engine.say(img_text)
            engine.runAndWait()
            img_text = " "

    cv2.putText(frame, img_text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                (0, 255, 0))
    cv2.putText(frame, label, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5,
                (0, 255, 0))
    cv2.imshow("test", frame)
    cv2.imshow("mask", mask)

    #if cv2.waitKey(1) == ord('c'):
    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    print("{} written!".format(img_name))
    label = predictor()

    if cv2.waitKey(1) == 10:
        break

cam.release()
cv2.destroyAllWindows()