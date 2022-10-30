'''
Monte Carlo estimator for using the walk on spheres method
by Sawhney & Crane

Based on:
https://www.cs.cmu.edu/~kmcrane/Projects/MonteCarloGeometryProcessing/WoSPoisson2D.cpp.html

Author: Andreas Ennemoser
Date: 28.10. 2022

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

def sample_point_on_circle(R, x):
    """Sample a point uniformly on a circle of radius R
    centered at x in 2D.
    """
    # sample a point uniformly on the unit circle
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    # scale the point to the circle of radius R
    return x + np.array([R * np.cos(theta), R * np.sin(theta)])

def sample_point_inside_circle(R, x):
    """Sample a point uniformly inside a circle of radius R
    centered at x in 2D.
    """
    # sample a point uniformly inside the unit circle
    r = R * np.random.uniform(0.0, 1.0)
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    # scale the point to the circle of radius R
    return x + np.array([r * np.cos(theta), r * np.sin(theta)])

def walk_on_spheres(domain, x, source, solution, steps, eps, wos=None, verbose=False, recursions=0):
    """Recursive walk on spheres implementation"""

    if recursions == 0:
        if verbose:
            print('Starting walk on spheres')
            print('Solution at x0: {}'.format(solution))
    else:
        if verbose:
            print(f'Recursion level: {recursions:03d}')

    # radius of the circle centered at x
    R, cp = closest_point_on_domain(domain, x)

    # sample a point uniformly inside the circle
    y = sample_point_inside_circle(R, x)

    # update solution
    r = np.linalg.norm(x - y)
    solution += np.pi * source(y) * G(r, R)

    if wos is None:
        wos = list()
        wos.append((x, y, R, cp, solution))
    else:
        wos.append((x, y, R, cp, solution))

    # sample next point on the walk on the circle
    x = sample_point_on_circle(R, x)

    if R > eps and recursions < steps:
        # recurse
        return walk_on_spheres(domain, x, source, solution, steps, eps, wos, verbose, recursions+1)
    else:
        if verbose:
            print('Walk on spheres finished')
        # return the walk on spheres path and sample points
        return wos

def solver(domain, x0, boundary_conditions, source, solution, walks, steps, eps, verbose):
    """Monte Carlo estimator for the Poisson equation using
    the walk on spheres method
    """

    for walk in range(walks):
        wos = walk_on_spheres(domain, x0, source, solution, steps, eps, verbose=verbose)

        # update solution with boundary conditions
        cp = wos[-1][3]
        solution += boundary_conditions(cp)
    
    return solution


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # get argunments for drawing options
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action=argparse.BooleanOptionalAction,
        default=False, help='Print recursion levels')
    parser.add_argument('-w', '--walks', type=int, default=32, help='Number of walks')
    parser.add_argument('-s', '--steps', type=int, default=16, help='Maximum number of steps per walk')
    parser.add_argument('-e', '--eps', type=float, default=0.1,
        help='Accuracy parameter')
    parser.add_argument('-d', '--draw_single', action=argparse.BooleanOptionalAction,
        default=False, help='Draw only a single walk (do not solve PDE)')
    args = parser.parse_args()

    domain = sample_domain()

    # start point
    x0 = np.array([0.0, 0.0])

    # boundary conditions
    def boundary_conditions(point):
        
        x, y = point
        
        if y >= 0.0:
            return 1.0
        else:
            if x <= 0.0:
                return 0.0
            else:
                return -1.0

    def source(point):
        """Source term"""

        x, y = point

        if args.draw_single:
            return 0.0
        else:
            if y >= 0.0:
                return 1.0
            else:
                return 0.0
        
    if args.draw_single:
        solution = 0.0
        wos = walk_on_spheres(domain, x0, source, solution, args.steps, args.eps, verbose=args.verbose)

        # plot the domain
        domain = np.append(domain, [domain[0]], axis=0) # close the polygon
        plt.plot(domain[:, 0], domain[:, 1], 'r-', lw=2)
        plt.plot(domain[:, 0], domain[:, 1], 'ro')
        plt.axis('equal')

        #random color
        col = list()
        for i in range(len(wos)):
            color = np.random.rand(3,)
            col.append(color)

        # plot the spheres
        for i, (x, y, R, cp, solution) in enumerate(wos):
            plt.plot(x[0], x[1], color=col[i], marker='o', ms=3)
            plt.plot(cp[0], cp[1], color=col[i], marker='x', ms=3)
            circle = plt.Circle(x, R, color=col[i], fill=False)
            plt.gca().add_patch(circle)
            plt.plot(y[0], y[1], color=col[i], marker='*', ms=3)
        
        # plot last closest point, i.e., the boundary point
        plt.plot(wos[-1][3][0], wos[-1][3][1], color='m', marker='o', mfc='none', mew=1.5, ms=6)

        # plot arrow between each x
        for i in range(len(wos)-1):
            x, _, _, _, _ = wos[i]
            xn, _, Rn, _, _ = wos[i+1]
            plt.arrow(x[0], x[1], xn[0]-x[0], xn[1]-x[1],
                    head_width=0.05*(R+Rn)/2., head_length=0.15*(R+Rn)/2.,
                    fc=col[i], ec=col[i], length_includes_head=True)

        plt.axis('equal')
        plt.show()

    else:
        # solve the Poisson equation
        solution = 0.0
        solution = solver(domain, x0, boundary_conditions, source, solution,
            args.walks, args.steps, args.eps, args.verbose)

        print(f'Estimated solution: {solution:0.4f}')
