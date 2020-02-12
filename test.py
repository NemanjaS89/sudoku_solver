import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import os


while True:
    img = cv2.imread('cetri.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, contours, _, = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, 0, (0, 255, 0), 2)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 50 and h < 120 and w < 120:
            region = bin[y:y + h + 1, x:x + w + 1]
            cv2.imshow('region', region)

    region = cv2.resize(region, (28, 28))
    test = region.reshape(1, 1, 28, 28)

    #plt.imshow(region)
    #plt.show()
    cv2.imshow('img', img)
    quit = cv2.waitKey(1)
    if quit == 27:
        break
cv2.destroyAllWindows()

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

loaded_model.load_weights('model.h5')
loaded_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

prediction = loaded_model.predict(test)
result = np.argmax(prediction)
print(result)

