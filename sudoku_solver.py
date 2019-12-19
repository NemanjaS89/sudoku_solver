import cv2
import numpy as np
import operator

#take webcam stream as an input
#convert the stream to grayscale
#apply gaussian blur
#apply adaptive gaussian thresholding
#use Hough line transform to detect and draw over the straight lines
#use linear segments to find the contours
#select the contour with the biggest area or an approximate encompasing polygon
#keep only the selected part of the image and apply perspective transformation
#using Keras, train a CNN model on the Chars74K dataset, to achieve number recognition
#puzzle solving, using the recursive backtracking algorithm


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



while True:
    ret, frame = cap.read()
    x, y, _ = frame.shape
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    agt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    """
    preprocess = cv2.bitwise_not(agt, agt)
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
    dilate = cv2.dilate(preprocess, kernel)
    """
    contours, _ = cv2.findContours(agt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:2]
    square = []
    
    for c in contours:
        c_len = cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, c_len * 0.02, True)
        if len(c) == 4 and cv2.contourArea(c) > 1200:
            square = c

    cv2.drawContours(frame, square, -1, [0,255,0], 3)

    try:
        warped = warp(frame, square)
        cv2.imshow('warped', warped)
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