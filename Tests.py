
import pypoman
import numpy
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sf = [0.127283, 0.050662, 0.1, 0.05, 0.1, 0.050703, 0.126484, 0.076621]
    A = numpy.array([
        [1, 1],
        [0, 1],
        [-1, 1],
        [-1, 0],
        [-1, -1],
        [0, -1],
        [1, -1],
        [1, 0]])

    b = numpy.array(sf)
    vertices = pypoman.compute_polytope_vertices(A, b)
    print("vertices length: ", len(vertices))
    print(vertices[0])
    plt.figure()
    pypoman.polygon.plot_polygon(vertices)
    plt.show()
