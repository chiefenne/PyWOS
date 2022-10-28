'''
Monte Carlo estimator for using the walk on spheres method
by Sawhney & Crane

Based on:
https://www.cs.cmu.edu/~kmcrane/Projects/MonteCarloGeometryProcessing/WoSPoisson2D.cpp.html

Author: Andreas Ennemoser
Date: 2020-05-01

'''

import numpy as np


def sample_domain():
    """Example polygon points"""
    p1 = (-9.06, 4.28)
    p2 = (-3.28, 6.68)
    p3 = (-2.00, 3.00)
    p4 = (8.52, 4.78)
    p5 = (12.42, -4.36)
    p6 = (5.38, -8.60)
    p7 = (-1.0, -9.66)
    p8 = (-7.38, -6.68)
    p9 = (-13.36, -1.50)
    p10 = (-14.12, 4.64)

    # polygon from points
    polygon = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])

    return polygon

def G(r, R):
    """Harmonic Green's function for a 2D ball of radius R"""
    return np.log(R / r) / (2.0 * np.pi)

def closest_point_on_line_segment(p, a, b):
    """Return the closest point on the line segment ab to point p in 2D."""
    # Vector from a to p
    ap = p - a
    # Vector from a to b
    ab = b - a
    ab2 = np.dot(ab, ab)
    ap_ab = np.dot(ap, ab)
    t = ap_ab / ab2
    # clip ensures that t is between 0 and 1
    # i.e., the closest point is on the line segment
    t = np.clip(t, 0.0, 1.0)
    return a + ab * t

def closest_point_on_domain(domain, x):
    """Return the closest point on the domain to point x in 2D."""
    R = np.finfo(np.double).max
    closest_point = None
    # closest distance of x to the domain boundary
    ld = len(domain)
    for i, _ in enumerate(domain):
        cp = closest_point_on_line_segment(x, domain[i], domain[(i+1)%ld])
        dist = np.linalg.norm(x - cp)
        if dist < R:
            R = dist
            closest_point = cp
    return R, closest_point

def sample_point_inside_circle(R, x):
    """Sample a point uniformly inside a circle of radius R
    centered at x in 2D.
    """
    # sample a point uniformly inside the unit circle
    r = R * np.random.uniform(0.0, 1.0)
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    # scale the point to the circle of radius R
    return x + np.array([r * np.cos(theta), r * np.sin(theta)])

def sample_point_on_circle(R, x):
    """Sample a point uniformly on a circle of radius R
    centered at x in 2D.
    """
    # sample a point uniformly on the unit circle
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    # scale the point to the circle of radius R
    return x + np.array([R * np.cos(theta), R * np.sin(theta)])

def walk_on_spheres(domain, x, eps, wos=None, recursions=0):
    """Recursive walk on spheres implementation"""

    if recursions == 0:
        print('Starting walk on spheres')
    else:
        print(f'Recursion level: {recursions:2d}')

    # radius of the circle centered at x
    R, cp = closest_point_on_domain(domain, x)

    # sample a point uniformly inside the circle
    y = sample_point_inside_circle(R, x)

    if wos is None:
        wos = list()
        wos.append((x, y, R, cp))
    else:
        wos.append((x, y, R, cp))

    # sample next point on the walk on the circle
    x = sample_point_on_circle(R, x)

    if R > eps:
        # recurse
        return walk_on_spheres(domain, x, eps, wos, recursions+1)
    else:
        print('Stopping walk on spheres')
        # return the point on the boundary
        return wos

def solver(domain, walks):
    """Monte Carlo estimator for the Poisson equation using
    the walk on spheres method
    """
    pass
    return


if __name__ == '__main__':

    domain = sample_domain()

    x0 = np.array([0.0, 0.0])
    eps = 0.1
    wos = walk_on_spheres(domain, x0, eps)

    # plot the domain
    import matplotlib.pyplot as plt

    domain = np.append(domain, [domain[0]], axis=0)

    plt.plot(domain[:, 0], domain[:, 1], 'r-', lw=2)
    plt.plot(domain[:, 0], domain[:, 1], 'ro')
    plt.axis('equal')

    # plot the walk on spheres
    col = list()
    for x, y, R, cp in wos:
        #random color
        color = np.random.rand(3,)
        col.append(color)
        circle = plt.Circle(x, R, color=color, fill=False)
        plt.gca().add_patch(circle)
        plt.plot(x[0], x[1], color=color, marker='o', ms=3)
        plt.plot(cp[0], cp[1], color=color, marker='x', ms=3)
        plt.plot(y[0], y[1], color=color, marker='*')
    
    # plot last closest point, i.e., the boundary point
    plt.plot(wos[-1][3][0], wos[-1][3][1], color=col[-1], marker='D', ms=4)

    # plot arrow between each x
    for i in range(len(wos)-1):
        x, _, _, _ = wos[i]
        xn, _, _, _ = wos[i+1]
        plt.arrow(x[0], x[1], xn[0]-x[0], xn[1]-x[1],
                  head_width=0.1, head_length=0.2, fc=col[i], ec=col[i], length_includes_head=True)
    plt.axis('equal')
    # toggle fullscreen mode
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()

