import numpy as np
import math

# functions to compute quadcopter state
# x = { x, y, z, dt(x, y, z), gam, beta, alpha, dt(gam, beta, alpha) }
# u = { f1, f2, f3, f4 }
# dt = scalar time increment for forward euler integration
# equations of motion are standard results


def state_advance(x, u, dt):

    f_max = 30

    u = np.array([min(max(h, 0), f_max) for h in u])

    p = x[0:3]
    pdot = x[3:6]
    e_angles = x[6:9]
    e_dot = x[9:]

    trig = get_trig(e_angles)

    impulses = vect(get_M().dot(u))

    thrust = impulses.item(0)

    body_rates = vect(np.dot(get_T(trig), e_dot))

    m = 10
    g = 9.81
    J = np.diag([10, 10, 10])  # intertia matrix
    invJ = np.diag([1/10, 1/10, 1/10])

    pdd = 1/m * \
        vect(get_R(trig).dot(np.array([0, 0, thrust])) - np.array([0, 0, m*g]))

    p += pdot*dt
    pdot += pdd*dt

    body_acc = np.matmul(
        invJ, (impulses[1:] - np.cross(body_rates, np.matmul(J, body_rates))))

    e_dd = vect(np.matmul(get_T_inv(trig), vect(
        body_acc - np.matmul(get_dotT(trig, e_dot), e_dot))))

    e_angles += e_dot*dt
    e_dot += e_dd*dt


def get_T(trig):
    sg, cg, sb, cb, tanb, sa, ca = trig

    return np.matrix([[1, 0, -sb],
                      [0, cg, sg*cb],
                      [0, -sg, cg*cb]])


def get_dotT(trig, e_dot):
    gam_dot, beta_dot, alpha_dot = e_dot

    sg, cg, sb, cb, tanb, sa, ca = trig

    return np.matrix([[1, 0, -beta_dot*cb],
                      [0, -gam_dot*sg, gam_dot*cg*cb - beta_dot*sg*sb],
                      [0, -gam_dot*cg, -gam_dot*sg*cb - beta_dot*cg*sb]])


def get_T_inv(trig):
    sg, cg, sb, cb, tanb, sa, ca = trig

    secb = 1
    if cb != 0:
        secb = 1/cb

    return np.matrix([[1, sg*tanb, cg*tanb],
                      [0, cg, -sg],
                      [0, sg*secb, cg*secb]])


def get_trig(e_angles):
    gamma, beta, alpha = e_angles

    sg = math.sin(gamma)
    cg = math.cos(gamma)
    sb = math.sin(beta)
    cb = math.cos(beta)
    tanb = math.tan(beta)
    sa = math.sin(alpha)
    ca = math.cos(alpha)

    return [sg, cg, sb, cb, tanb, sa, ca]


def get_R(trig):
    sg, cg, sb, cb, tanb, sa, ca = trig

    return np.matrix([[ca*cb, -sa*cg + ca*sb*sg, sa*sg + ca*sb*cg],
                      [sa*cb, ca*cg + sa*sb*sg, -ca*sg + sa*sb*cg],
                      [-sb, cb*sg, cb*cg]])


def get_M():
    ydist = 0.3
    xdist = 0.3
    c = 6e-3

    return np.matrix([[1, 1, 1, 1],
                      [u*xdist for u in [1, -1, -1,  1]],
                      [u*ydist for u in [-1, -1,  1,  1]],
                      [u*c for u in [-1,  1, -1,  1]]])


def vect(should_be_vec):
    return np.array(should_be_vec).squeeze()
