from keras.models import load_model

classifier = load_model('Trained_model.h5')
classifier.evaluate()

#Prediction of single image
import numpy as np
from keras.preprocessing import image
img_name = input('Enter Image Name: ')
image_path = './predicting_data/{}'.format(img_name)
print('')

test_image = image.load_img(image_path, target_size=(200, 200))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
#training_set.class_indices
print('Predicted Sign is:')
print('')

if result[0][0] == 1:
    print('Excuse')
elif result[0][1] == 1:
    print('Hello')
elif result[0][2] == 1:
    print('Me')
elif result[0][3] == 1:
    print('Good Morning')
elif result[0][4] == 1:
    print('Mam')
elif result[0][5] == 1:
    print('You')
elif result[0][6] == 1:
    print('Thank')
elif result[0][7] == 1:
    print('Tell')
elif result[0][8] == 1:
    print('Can')
elif result[0][9] == 1:
    print('Way')
elif result[0][10] == 1:
    print('Bank')
elif result[0][11] == 1:
    print('From')
elif result[0][12] == 1:
    print('Traffic Signal')
elif result[0][13] == 1:
    print('After')
elif result[0][14] == 1:
    print('Okay')
elif result[0][15] == 1:
    print('Round About')
elif result[0][16] == 1:
    print('Left')
elif result[0][17] == 1:
    print('Right')
elif result[0][18] == 1:
    print('Bye')
elif result[0][19] == 1:
    print('Bye')
elif result[0][20] == 1:
    print('Upwards')
elif result[0][21] == 1:
    print('Go')
elif result[0][22] == 1:
    print('Downwards')
elif result[0][23] == 1:
    print('Guide')
elif result[0][24] == 1:
    print('Far')
elif result[0][25] == 1:
    print('Near')
