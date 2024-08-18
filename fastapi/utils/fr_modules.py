import numpy as np


def face_distance(face_encodings, face_to_compare):
    """
    주어진 얼굴 인코딩과 비교할 얼굴 인코딩 간의 유클리드 거리를 계산합니다.
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


# 유사도비교 함수 추가(0818)
def compare_faces_with_similarity(
    known_face_encodings, face_encoding_to_check, tolerance=0.6
):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.
    Also return the similarity score (Euclidean distance).

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of tuples with a bool indicating match and the similarity score (distance)
    """
    distances = face_distance(known_face_encodings, face_encoding_to_check)
    results = [(bool(distance <= tolerance), distance) for distance in distances]
    return results
