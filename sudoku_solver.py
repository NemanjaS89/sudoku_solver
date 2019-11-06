import cv2
import numpy as np


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

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    agt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    #edges = cv2.Canny(agt, 50, 150, apertureSize = 3)
    
    agt_prepared = cv2.cvtColor(agt, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(agt, 1, np.pi/180, 150, 100, 10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(agt_prepared, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
    #cv2.createTrackbar('')
    """
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_draw = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    """
    
    
    cv2.imshow('agt_prepared', agt_prepared)
    
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