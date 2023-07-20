import numpy as np, scipy as sp


def calculate_vector_length(vector):
    return np.sqrt(np.sum(np.square(vector)))


def calculate_distance_between_2p(PointA, PointB) -> np.array:
    r = calculate_vector_length(np.array(PointB, ndmin=1) - np.array(PointA, ndmin=1))
    return np.squeeze(r)


def calculate_angle_between_3p(PointA, PointB, PointC):  # Bug
    ab = calculate_vector_length(np.array(PointA, ndmin=1) - np.array(PointB, ndmin=1))
    cb = calculate_vector_length(np.array(PointC, ndmin=1) - np.array(PointB, ndmin=1))
    ac = calculate_vector_length(np.array(PointA, ndmin=1) - np.array(PointC, ndmin=1))

    alpha = np.rad2deg(np.arcsin(ab / ac))
    return alpha


def calculate_angle_between_3p(PointA, PointB, PointC):  # Bug
    ab = calculate_vector_length(np.array(PointA, ndmin=1) - np.array(PointB, ndmin=1))
    cb = calculate_vector_length(np.array(PointC, ndmin=1) - np.array(PointB, ndmin=1))
    ac = calculate_vector_length(np.array(PointA, ndmin=1) - np.array(PointC, ndmin=1))

    alpha = np.rad2deg(np.arcsin(ab / ac))
    return alpha


def calculate_distance_matrix_row(pointA, all_points):
    return all_points - pointA


def calculate_distance_matrix(all_points):
    all_points = np.array([np.array(p) for p in all_points])
    f = lambda x: all_points - x
    get_dmat = np.vectorize(f)

    dmat = np.array(list(map(lambda x: calculate_distance_matrix_row(x, all_points), all_points)))

    return dmat
