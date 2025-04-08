#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import pyautogui
from collections import Counter
from collections import deque

import time
import win32api
import win32con
import ctypes
import ctypes.wintypes

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
from model import SkelettHistoryClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

def clickDown(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)

def clickUp(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    screen_width, screen_height = pyautogui.size() #FJ: added this line
    aspect_ratio = 16/10 ### change this to 16/9 if you have a 16:9 monitor #FJ added this line
    
    downclick = False #FJ added this line
    semaphore = True


    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True


    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier() # static gestures

    skelett_history_classifier = SkelettHistoryClassifier() # dynamic gestures #FJ

    point_history_classifier = PointHistoryClassifier() #dynamic gestures / Finger movment
    ########
    ########
    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]
    with open(
            'model/skelett_history_classifier/skelett_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        skelett_history_classifier_labels = csv.reader(f)
        skelett_history_classifier_labels = [
            row[0] for row in skelett_history_classifier_labels
        ]    

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16 # FJ behöver kanske träna om modellen med en annan history length
    point_history = deque(maxlen=history_length)
    
    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    # Skelett gesture history ################################################
    skelett_history = deque(maxlen=history_length)
    skelett_gesture_history = deque(maxlen=history_length)
    
    #  ########################################################################
    mode = 0
    timer = 0
    while True:
        fps = cvFpsCalc.get()
        
        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True


        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)


                #print("landmarklist: " , landmark_list)
                skelett_history.append(landmark_list)
                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                pre_processed_skelett_history_list = pre_process_skelett_history(debug_image, skelett_history,landmark_list)
                # Write to the dataset file
                
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list,pre_processed_skelett_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                Xpos = (landmark_list[8][0]/cap_width)*aspect_ratio*screen_width
                Ypos = (landmark_list[8][1]/cap_height)*screen_height

                # Skelett gesture classification
                #skelett_gesture_id = skelett_history_classifier(pre_processed_skelett_history_list)
                ##TODO Add actions to skelett_gestures_id //F


                # actualtime = time.time()
                # if semaphore == False and actualtime >= currtime + 1: #Delay before pasteing and copying. FJ added this line 
                #     semaphore = True

                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                    ctypes.windll.user32.SetCursorPos(int(Xpos), int(Ypos))
                    #pyautogui.moveTo((landmark_list[8][0]/cap_width)*aspect_ratio*screen_width, (landmark_list[8][1]/cap_height)*screen_height, 0.1) #FJ added this line
                    #pyautogui.sleep(0.01) #FJ added this line
                
                # if hand_sign_id == 3:  # OK sign #FJ added this line
                #     if downclick == False:
                #         clickDown(int(Xpos), int(Ypos))
                #         downclick = True
                #     ctypes.windll.user32.SetCursorPos(int(Xpos), int(Ypos))
                    
                # if hand_sign_id != 3 and downclick == True: #FJ added this line                           
                #     clickUp(int(Xpos), int(Ypos))
                #     #clickUp(convertedX, convertedY) #FJ added this line
                #     downclick = False

                
                # if hand_sign_id == 4:  # Back sign #FJ added this line
                #     pyautogui.hotkey('alt', 'left')
                #     pyautogui.PAUSE = 0.2
                # if hand_sign_id == 5:  # RocknRoll sign #FJ added this line
                #     currentPosX, currentPosY = win32api.GetCursorPos()
                #     win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, currentPosX, currentPosY, 50, 0)

                #     #pyautogui.scroll(50, pyautogui.position().x, pyautogui.position().y)
                #     #pyautogui.PAUSE = 0.2
                # if hand_sign_id == 6 and semaphore == True:  # Copy #FJ added this line
                #     pyautogui.hotkey('ctrl', 'c')
                #     semaphore = False
                #     currtime = time.time()

                # if hand_sign_id == 7 and semaphore == True: # Paste / Peace sign #FJ added this line
                #     pyautogui.hotkey('ctrl', 'v')
                #     semaphore = False
                #     currtime = time.time()
                # if hand_sign_id == 8:  # Turn off #FJ added this line
                #     break

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
              #print("point history length", point_history_len)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()


                skelett_gesture_id = 0
                skelett_history_len = len(pre_processed_skelett_history_list)
              #print("skelett history length", skelett_history_len)
                if skelett_history_len == (history_length * 2 * 21):
                    skelett_gesture_id = skelett_history_classifier(pre_processed_skelett_history_list)
                #print("skelett gesture id", skelett_gesture_id)
                skelett_gesture_history.append(skelett_gesture_id)
                #print("skelett history", skelett_gesture_history)
                most_common_skelett_id = Counter(skelett_gesture_history).most_common()


                # Drawing part

                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                    skelett_history_classifier_labels[most_common_skelett_id[0][0]],
                )
        else:
            point_history.append([0, 0])
            skelett_history.append([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    if key == 109:  # m #Added for skelett gesture mode #FJ
        mode = 3 
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    #print("landmark list", landmark_list)
    temp_landmark_list = copy.deepcopy(landmark_list)
    #print(temp_landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            #print(landmark_point)  
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


        #först tar vi palmens koordinater i första frame och sätter det som bas
        #sen tar vi i punkter på handen och konverterar de gentemot basen.
        #I de nästa frames så sätter vi vad palmens koordinater i den framen är och konverterar de gentemot palmen i första framen
        #efter det så konverterar vi de andra punkterna gentemot palmen i den framen
        #vi får in en deque med 16 frames. Vi får deque med 16 listor, csv = 16frames * 2coordinater *21 punkter ((+1class))

def pre_process_skelett_history(image, skelett_history,landmarklist):
    image_width, image_height = image.shape[1], image.shape[0]
    #print("skelett history", skelett_history)
    temp_skelett_history = copy.deepcopy(skelett_history)
    temp_landmarklist = copy.deepcopy(landmarklist)
    #print("temp_skelett_history", temp_skelett_history, " temp_landmarklist" , temp_landmarklist)

    # print("Length of deque:", len(temp_skelett_history))  # Should be ≤16 (history_length)
    # print("Whole deque:", temp_skelett_history) 
    # print("First frame:", temp_skelett_history[0])       # Should be a list of 21 [x,y] pairs
    # print("First landmark:", temp_skelett_history[0][0]) # Should be [x, y], not an int



    base_x, base_y = 0,0
    #jag tror att skelett är tom för att dens queue har inte hunnit fyllas på
    
    for frames,_ in enumerate(temp_skelett_history):
        if frames == 0:
            base_x, base_y =  temp_skelett_history[0][0][0] , temp_skelett_history[0][0][1]
            #print("base x: ", base_x , "base y: ",base_y)           
        else:  
            temp_skelett_history[frames][0][0] = (temp_skelett_history[frames][0][0] - base_x) / image_width
            temp_skelett_history[frames][0][1] = (temp_skelett_history[frames][0][1] - base_y) / image_height
            for point in range(1, 20):
                #print("point",point)
                temp_skelett_history[frames][point][0] = temp_skelett_history[frames][point][0] - temp_skelett_history[frames][0][0]
                temp_skelett_history[frames][point][1] = temp_skelett_history[frames][point][1] - temp_skelett_history[frames][0][1]

    # Convert to a one-dimensional list
    temp_skelett_history = list(
            itertools.chain.from_iterable(temp_skelett_history))
    temp_skelett_history = list(
            itertools.chain.from_iterable(temp_skelett_history))
    
    max_skelett = max(1e-2, max(map(abs, temp_skelett_history)))
    
    def normalize_skelett(n):
        return n / max_skelett

    temp_skelett_history = list(map(normalize_skelett, temp_skelett_history))

    return temp_skelett_history  ## Vill returnera en deque med alla punkter i rad

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list, skelett_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    if mode == 3 and (0 <= number <= 9):        
        csv_path = 'model/skelett_history_classifier/skelett_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *skelett_history_list])
    return

def timern (start_time):
    start_time = time.time()
    return start_time

def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text,skelett_history_classifier_labels):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)
        
    if skelett_history_classifier_labels !="":
        cv.putText(image, "Skelett Gesture:" + skelett_history_classifier_labels, (10, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
