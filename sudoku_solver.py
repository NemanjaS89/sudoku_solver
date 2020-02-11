import cv2
import numpy as np
import operator
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import tensorflow as tf

# take webcam stream as an input
# convert the stream to grayscale
# apply gaussian blur
# apply adaptive gaussian thresholding
# use Hough line transform to detect and draw over the straight lines
# use linear segments to find the contours
# select the contour with the biggest area or an approximate encompasing polygon
# keep only the selected part of the image and apply perspective transformation
# using Keras, train a CNN model on the Chars74K dataset, to achieve number recognition
# puzzle solving, using the recursive backtracking algorithm


cap = cv2.VideoCapture(0)


def warp(frame, square):
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in square]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in square]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in square]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in square]), key=operator.itemgetter(1))

    reference_pts = np.array([(0, 0), (449, 0), (449, 449), (0, 449)], np.float32)
    corners = [square[top_left][0], square[top_right][0], square[bottom_right][0], square[bottom_left][0]]

    mask = cv2.getPerspectiveTransform(np.float32(corners), reference_pts)
    return cv2.warpPerspective(frame, np.float32(mask), (450, 450))


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def invert(image):
    return 255 - image


def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')


def dilate(image):
    kernel = np.ones((3, 3))
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def select_roi(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 100 and h < 100 and h > 10 and w > 20:
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]

    return image_orig, sorted_regions


def scale_to_range(image):
    return image/255


def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))

    return ready_for_ann


def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)

    return np.array(nn_outputs)


def create_ann():
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))
    return ann


def train_ann(ann, x_train, y_train):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    ann.fit(x_train, y_train, epochs=500, batch_size=1, verbose=0, shuffle=False)

    return ann


def winner(output):
    return max(enumerate(output), key=lambda X: X[1])[0]


def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

new_model = tf.keras.models. load_model('sudoku_solver_model')

while True:
    ret, frame = cap.read()
    x, y, _ = frame.shape

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    agt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, contours, _ = cv2.findContours(agt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    square = []

    for c in contours:
        c_len = cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, c_len * 0.02, True)
        if len(c) == 4 and cv2.contourArea(c) > 1200:
            square = c

    cv2.drawContours(frame, square, -1, [0, 255, 0], 3)

    try:
        warped = warp(frame, square)

        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped = cv2.GaussianBlur(warped, (9, 9), 0)
        warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        cv2.imshow('warped', warped)
        i = 2
        j = 2
        y_1 = i * 50
        x_1 = j * 50
        y_2 = (i + 1) * 50
        x_2 = (j + 1) * 50
        field_i_j = warped[y_1:y_2, x_1:x_2]
        field_i_j = field_i_j[5:45, 5:45]
        field_i_j = cv2.resize(field_i_j, (80, 80), interpolation=cv2.INTER_NEAREST)
        malo_boje = cv2.cvtColor(field_i_j, cv2.COLOR_GRAY2BGR)

        selected_test, test_numbers = select_roi(malo_boje, field_i_j)
        display_image(selected_test)
        test_inputs = prepare_for_ann(test_numbers)
        cv2.imshow('selected_test', selected_test)
        cv2.imshow('field_i_j', field_i_j)
        prediction = new_model.predict([test_inputs])
        print(np.argmax(prediction[0]))
    except:
        pass

    cv2.imshow('frame', frame)



    quit = cv2.waitKey(1)
    if quit == 27:
        break

cv2.destroyAllWindows()
cap.release()
"""
array = [
        [0,0,0,0,0,9,6,3,0],
        [0,4,0,6,0,0,8,2,0],
        [0,0,0,0,0,4,0,1,0],
        [0,0,8,9,0,0,0,0,0],
        [2,1,0,0,3,0,0,9,7],
        [0,0,0,0,0,5,3,0,0],
        [0,7,0,4,0,0,0,0,0],
        [0,6,9,0,0,3,0,7,0],
        [0,2,3,5,0,0,0,0,0]
]

def print_table(sudoku):
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print('--------------------')
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print('| ', end='')
            if j == 8:
                print(str(sudoku[i][j]))
            else:
                print(str(sudoku[i][j]) + " ", end='')


def free_square(sudoku, k):
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] == 0:
                k[0] = i
                k[1] = j
                return True
    return False


def row_collision(sudoku, num, row):
    for i in range(9):
        if sudoku[row][i] == num:
            return True
    return False


def column_collision(sudoku, num, col):
    for i in range(9):
        if sudoku[i][col] == num:
            return True
    return False


def square_collision(sudoku, num, row, col):
    for i in range(3):
        for j in range(3):
            if sudoku[i + row][j + col] == num:
                return True
    return False


def valid_square(sudoku, num, row, col):
    return not row_collision(sudoku, num, row) and not column_collision(sudoku, num, col)\
    and not square_collision(sudoku, num, row - row % 3, col - col % 3)

def solve(sudoku):

    k = [0, 0]

    if not free_square(sudoku, k):
        return True

    row = k[0]
    col = k[1]

    for num in range(1, 10):
        if valid_square(sudoku, num, row, col):
            sudoku[row][col] = num

            if solve(sudoku):
                return True

            sudoku[row][col] = 0

    return False

solve(array)
print_table(array)
"""
