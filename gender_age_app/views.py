import base64
import dlib
import cv2
import math
import numpy as np
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from health_analysis_django.wsgi import age_net, gender_net, face_detector_dlib, landmark_detector, MODEL_MEAN_VALUES, \
    input_size, age_list, gender_list


@api_view(["POST"])
def get_gender_age(img_data):
    try:
        img_b64 = img_data.body
        decoded_img = base64.b64decode(img_b64)
        with open('img.jpg', 'wb') as img:
            img.write(decoded_img)
        image = cv2.imread('img.jpg')
        blob = process_image(image)
        age = get_age(blob)
        gender = get_gender(blob)
        return JsonResponse(f"Gender: {gender}, Age: {age}", safe=False)
    except Exception as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)


def get_age(blob):
    age_net.setInput(blob)
    predictions = age_net.forward()
    age = age_list[predictions[0].argmax()]
    return age


def get_gender(blob):
    gender_net.setInput(blob)
    predictions = gender_net.forward()
    gender = gender_list[predictions[0].argmax()]
    return gender


def detect_face_dlib(image):
    faces = face_detector_dlib(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if len(faces) == 1:
        return (faces[0].left(), faces[0].top()), (faces[0].right(), faces[0].bottom())
    else:
        return None


def get_landmarks(img, visualize=False, face=None):
    if not face:
        face = detect_face_dlib(img)
        if face is None:
            return None
    else:
        (x1, y1), (x2, y2) = face
        face = dlib.rectangle(x1, y1, x2, y2)

    landmarks = landmark_detector(img, face)
    if visualize:
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, (x, y), 4, (255, 0, 0), 1)

        cv2.imshow("Output", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return landmarks


def get_face_with_margin(image, margin=0.4, face=None):
    row, col = image.shape[:2]
    bottom = image[row - 2:row, 0:col]
    mean = cv2.mean(bottom)[0]
    top_bottom = int(image.shape[0] * margin)
    left_right = int(image.shape[1] * margin)
    border = cv2.copyMakeBorder(
        image,
        top=top_bottom,
        bottom=top_bottom,
        left=left_right,
        right=left_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[mean, mean, mean]
    )
    if face:
        (x1, y1), (x2, y2) = face
    else:
        (x1, y1), (x2, y2) = detect_face_dlib(image)

    (x1, y1), (x2, y2) = (x1 + left_right, y1 + top_bottom), (x2 + left_right, y2 + top_bottom)

    w = x2 - x1
    h = y2 - y1
    _x1 = int(x1 - w * margin)
    _x2 = int(x2 + w * margin)
    _y1 = int(y1 - h * margin)
    _y2 = int(y2 + h * margin)
    return border[_y1: _y2, _x1: _x2]


def get_orientation_constants(landmarks, size):
    image_points = np.array([
        (landmarks.part(33).x, landmarks.part(33).y),  # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),  # Chin
        (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
        (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corne
        (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
        (landmarks.part(54).x, landmarks.part(54).y)
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-165.0, 170.0, -135.0),  # Left eye left corner
        (165.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Camera internals

    center = (size[1] / 2, size[0] / 2)
    focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    axis = np.float32([[500, 0, 0],
                       [0, 500, 0],
                       [0, 0, 500]])

    return image_points, model_points, camera_matrix, dist_coeffs, axis


def check_orientation(roll, pitch, yaw, roll_thresh=5, pitch_thresh=10, yaw_thresh=10):

    is_good_orientation = True
    if not (-roll_thresh <= roll <= roll_thresh):
        is_good_orientation = False
    if not (-yaw_thresh <= yaw <= yaw_thresh):
        is_good_orientation = False
    if not (-pitch_thresh <= pitch <= pitch_thresh):
        is_good_orientation = False
    if is_good_orientation:
        return roll, pitch, yaw
    else:
        return None


def face_orientation(frame, landmarks, PITCH_ADJUSTMENT=15):
    if landmarks:
        size = frame.shape  # (height, width, color_channel)

        image_points, model_points, camera_matrix, dist_coeffs, axis = get_orientation_constants(landmarks, size)

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix,
                                           dist_coeffs)
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in euler_angles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        return imgpts, modelpts, (int(roll), int(pitch) + PITCH_ADJUSTMENT, int(yaw))
    else:
        return None, None, None


def process_image(img):
    if img.shape[0] < 224 or img.shape[1] < 224:
        raise Exception('Error: Image Resolution too Low')

    face = detect_face_dlib(img)

    if not face:
        raise Exception('Error: No face or Multiple Faces Found')

    face_img = get_face_with_margin(img, face=face)

    margin_img_face = detect_face_dlib(face_img)

    if face_img.shape[0] < 224 or face_img.shape[1] < 224:
        raise Exception('Error: Image Resolution too Low. Try coming nearer to the camera')

    if not margin_img_face:
        raise Exception('Error: No face or Multiple Faces Found')

    landmarks = get_landmarks(face_img, face=margin_img_face)

    if not landmarks:
        raise Exception('Error: Landmark Detection Failed')

    orientation = face_orientation(face_img, landmarks)[2]

    if not check_orientation(orientation[0], orientation[1], orientation[2]):
        raise Exception('Error: Bad Facial Orientation. Look towards the camera.')

    return cv2.dnn.blobFromImage(cv2.resize(face_img, input_size), 1.0, input_size, MODEL_MEAN_VALUES, swapRB=False)
