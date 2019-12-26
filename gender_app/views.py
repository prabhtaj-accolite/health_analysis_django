from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from django.core import serializers
from django.conf import settings
import json
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
import base64
import numpy as np
import cv2


@api_view(["POST"])
def get_gender_post(img_data):
    try:
        img_b64 = img_data.body
        decoded_img = base64.b64decode(img_b64)
        with open('img.jpg', 'wb') as img:
            img.write(decoded_img)
        face = get_detected_face()
        if not isinstance(face, int):
            gender = get_gender(face)
            age = get_age(face)
            return JsonResponse(f"Gender: {gender}, Age: {age}", safe=False)
        else:
            return JsonResponse(f"No Faces Found", safe=False)
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)


def get_detected_face():
    net = cv2.dnn.readNetFromCaffe('models/face_detection/gil_levi_tal_hassner/faces_detect.prototxt',
                                   'models/face_detection/gil_levi_tal_hassner/faces_detect.caffemodel')

    image = cv2.imread('img.jpg')
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    crop_img = image
    face_found = False
    max_confidence = 0
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.9 and confidence > max_confidence:
            max_confidence = confidence
            face_found = True
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            crop_img = image[startY: endY, startX: endX]
    # cv2.imshow("Output", crop_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if face_found:
        return image
    else:
        return 0


def get_age(face):
    age_net = cv2.dnn.readNetFromCaffe('models/age_prediction/DEX/DEX_age.prototxt',
                                       'models/age_prediction/DEX/DEX_age.caffemodel')
    age_list = list(range(101))
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    blob = cv2.dnn.blobFromImage(cv2.resize(face, (224, 224)), 1.0, (224, 224), MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(blob)
    predictions = age_net.forward()
    age = age_list[predictions[0].argmax()]
    return age


def get_gender(face):
    gender_net = cv2.dnn.readNetFromCaffe('models/gender_prediction/DEX/DEX_gender.prototxt',
                                          'models/gender_prediction/DEX/DEX_gender.caffemodel')
    gender_list = ['Female', 'Male']
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    blob = cv2.dnn.blobFromImage(cv2.resize(face, (224, 224)), 1.0, (224, 224), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    predictions = gender_net.forward()
    gender = gender_list[predictions[0].argmax()]
    return gender
