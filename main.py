import cv2
from tensorflow import keras
#from keras.models import load_model
import numpy as np

model = keras.models.load_model('model.h5')


def draw_rectangle(frame, start_point, end_point):
    color = (0, 0, 255)
    thickness = 2
    return cv2.rectangle(frame, start_point, end_point, color, thickness)


def process(frame, start_point, end_point):
    color = (0, 0, 255)
    thickness = 2
    rect = draw_rectangle(frame, start_point, end_point)
    #rect1 = draw_rectangle(frame, [20, 20], [225, 70])
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_sensivity = 15
    s_h = 255
    v_h = 255
    s_l = 50
    v_l = 50
    blue_upper = np.array([115 + h_sensivity, s_h, v_h])
    blue_lower = np.array([115 - h_sensivity, s_l, v_l])
    red_lower1 = np.array([0, s_l, v_l])
    red_lower2 = np.array([170, s_l, v_l])
    red_upper1 = np.array([15, s_h, v_h])
    red_upper2 = np.array([180, s_h, v_h])
    green_lower = np.array([42, s_l, v_l])
    green_upper = np.array([72, s_h, v_h])
    white_lower = np.array([230, 230, 230])
    white_upper = np.array([255, 255, 255])
    mask_frame = hsv_frame[start_point[1]:end_point[1] + 1, start_point[0]:end_point[0] + 1]
    mask_red = cv2.inRange(mask_frame, red_lower1, red_upper1)+cv2.inRange(mask_frame, red_lower2, red_upper2)
    mask_blue = cv2.inRange(mask_frame, blue_lower, blue_upper)
    mask_green = cv2.inRange(mask_frame, green_lower, green_upper)
    mask_white = cv2.inRange(mask_frame, white_lower, white_upper)
    red_rate = np.count_nonzero(mask_red) / ((end_point[0] - start_point[0]) * (end_point[1] - start_point[1]))
    green_rate = np.count_nonzero(mask_green) / ((end_point[0] - start_point[0]) * (end_point[1] - start_point[1]))
    blue_rate = np.count_nonzero(mask_blue) / ((end_point[0] - start_point[0]) * (end_point[1] - start_point[1]))
    white_rate = np.count_nonzero(mask_white) / ((end_point[0] - start_point[0]) * (end_point[1] - start_point[1]))

    cropped_frame = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    input_data = np.reshape(resized, (1, 28, 28, 1))
    input_data = input_data.astype('float32') / 255
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    number_string = ""
    if predicted_class == 0 or predicted_class == 1 or predicted_class == 2:
        number_string = str(predicted_class)
    else:
        number_string = "undefined"

    org = end_point
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    if red_rate > 0.45:
        cv2.putText(frame, number_string, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(rect, ' red ', org, font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    elif green_rate > 0.45:
        cv2.putText(frame, number_string, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(rect, ' green ', org, font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    elif blue_rate > 0.45:
        cv2.putText(frame, number_string, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(rect, ' blue ', org, font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
    else:
        cv2.putText(frame, number_string, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(rect, ' undefined ', org, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return rect, resized


def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, dragging, resizing

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_point = (x, y)
        end_point = (x+1, y+1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            end_point = (x + 1, y + 1)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        end_point = (x + 1, y + 1)


cap = cv2.VideoCapture(0)
rect_size = 100
cv2.namedWindow('Cam')
cv2.setMouseCallback('Cam', mouse_callback)
dragging = False
resizing = False
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
start_point = (int(height / 2 - rect_size / 2), int(width / 2 - rect_size / 2))
end_point = (int(height / 2 + rect_size / 2), int(width / 2 + rect_size / 2))

while cap.isOpened():
    ret, frame = cap.read()
    processed, resized = process(frame, start_point, end_point)
    cv2.imshow('Cam', processed)

    k = cv2.waitKey(1) & 0xFFF
    if k == 52:
        break
    elif k == 97:
        start_point = (max(start_point[0] - 5, 0), start_point[1])
        end_point = (max(end_point[0] - 5, rect_size), end_point[1])
    elif k == 119:
        start_point = (start_point[0], max(start_point[1] - 5, 0))
        end_point = (end_point[0], max(end_point[1] - 5, rect_size))
    elif k == 100:
        start_point = (min(start_point[0] + 5, height - rect_size), start_point[1])
        end_point = (min(end_point[0] + 5, height), end_point[1])
    elif k == 115:
        start_point = (start_point[0], min(start_point[1] + 5, width - rect_size))
        end_point = (end_point[0], min(end_point[1] + 5, width))

cap.release()
cv2.destroyAllWindows()
