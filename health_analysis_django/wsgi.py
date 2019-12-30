"""
WSGI config for health_analysis_django project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""

import os

import cv2
import dlib
from django.core.wsgi import get_wsgi_application

age_list = list(range(101))

gender_list = ['female', 'male']

input_size = (224, 224)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_net = cv2.dnn.readNetFromCaffe('models/DEX_real_age.prototxt', 'models/DEX_real_age.caffemodel')

gender_net = cv2.dnn.readNetFromCaffe('models/DEX_gender.prototxt', 'models/DEX_gender.caffemodel')

landmark_detector = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

face_detector_dlib = dlib.get_frontal_face_detector()


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'health_analysis_django.settings')

application = get_wsgi_application()
