from ultralytics import YOLO
import cv2

modelForObjects = YOLO('')
modelForHands = YOLO('')

cap = cv2.VideoCapture(1)

