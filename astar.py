import numpy as np
import matplotlib.pyplot as plt
import time
import json
import copy
import pickle

show_animation = True
sa_count = 1

# initial conditions
a0 = 0
s0 = 0.0
v0 = 8 
v_ref = 9
a_max, a_min = 5, -5

T = 4  # planning horizon
dT = 0.5  # a star time step
time_step = 0.1  

dg_X = dT
dg_Y = 1
Y_max = 40
safety = 3.5 

motions = [
    # discrete accelerations in m/s^2
    [1, -2],
    [1, -1],
    [1, 0],
    [1, 1],
]

backup_motions = [
    [1, -3],
    [1, -4],
    [1, -5],
    [1, 2],
    [1, 3]
]

grid_x = int(T/dT + 1)
grid_y = int(Y_max / dg_Y + 1)

x = np.arange(0, grid_x, 1)  # grid index
y = np.arange(0, grid_y, 1)
t = np.arange(0, T+time_step, time_step)

v_u, v_l = v0 + a_max * t,  v0 + a_min * t
s_min = np.zeros(t.size)
s_max = np.zeros(t.size)
s_min[1:] = np.cumsum(np.clip(v_l + 0.5 * a_min * time_step, 0, None))[:-1]*time_step
s_max[1:] = np.cumsum(v_u + 0.5 * a_max * time_step)[:-1]*time_step

# obstacle info is sobstacle_tred as a list of numpy arrays: 
# list[np.array(timestamp), list[np.array(obstacle pos.)]] 
obstacle_info = [np.array([2. , 2.1, 2.2, 2.3, 2.4, 2.5, 3., 3.1, 3.2, 3.3, 3.4, 3.5 ]),
                [np.array([11.7]),
                np.array([11.1, 12.]),
                np.array([10.4, 11.3, 12.2 ]),
                np.array([10.7, 11.5]),
                np.array([10.1, 10.9]),
                np.array([10.3]),
                np.array([25.7]),
                np.array([25.1, 26.]),
                np.array([24.5, 25.3, 26.2]),
                np.array([24.7, 25.5]),
                np.array([24.1, 24.9]),
                np.array([24.3])]]

if show_animation:
    plt.figure(1)

    plt.grid(True)
    for i in range(obstacle_info[0].shape[0]):
        for j in range(obstacle_info[1][i].shape[0]):
            plt.plot(obstacle_info[0][i], obstacle_info[1][i][j], 'ok')
    plt.plot(t, s_min, 'g', label='acc. limit')
    plt.plot(t, s_max, 'g')
    plt.xlabel('time')
    plt.ylabel('distance')
    plt.pause(0.001)

class Node:
    def __init__(self, xid, yid, v, cost, parent_id):
        self.t = xid  
        self.s = yid
        self.v = v
        self.cost = cost
        self.parent_id = parent_id

    def __state__(self):
        return self.s, self.v, self.t, self.cost, self.parent_id
        
def grid_id(n: Node):
    return str(round(100 * n.s)) + str(round(10 * n.t)) + str(round(1000 * n.v))

     
def cost_fun(s_n: float, v_n: float, a_n: float):
    """
    Calculate the cost for node expansion
    """
    Kv, Ka = 1, 1
    cost_v = (v_n - v_ref)**2
    cost_a = a_n**2
    return Kv * cost_v + Ka * cost_a


def calc_heuristic(n: Node):
    """
    Calculate the heuristic cost
    """
    v = n.v
    t0 = n.t
    tg = np.linspace(0, T-t0, int((T-t0)/0.5)+1)

    t_m = motions + backup_motions
    s = np.zeros((len(t_m), tg.size))
    acc = np.array(t_m)[:, 1].reshape(-1, 1)
    v1 = v + acc * tg
    v11 = np.clip(v1, 0, None)
    delta_s = np.clip(v1 + 0.5 * acc * dT, 0, None)[:, :-1] * dT
    s[:, 1:] = np.cumsum(delta_s, axis=1)
    s += n.s

    a_cost = (acc**2).reshape(-1,)
    v_cost = np.sum((v11 - v_ref)**2, axis=1)
    h_cost = a_cost + v_cost

    return np.min(h_cost)


def check_collision(x1, x2, obstacle):
    """
    Check a trajectory for safety
    """
    if isinstance(x2, list):
        x2 = np.array(x2).reshape(len(x2), -1)

    if len(obstacle) == 0:
        return np.zeros(x2.shape[0]).astype(bool)

    obstacle_t = np.linspace(x1[0], x1[-1], int((x1[-1]-x1[0])/0.1)+1)
    obstacle_s = np.array([np.interp(obstacle_t, x1, x2[i]) for i in range(x2.shape[0])])

    tid = np.isin(obstacle_t.astype(np.float32), obstacle[0].astype(np.float32), assume_unique=True)
    oids, = np.where(np.isin(obstacle[0].astype(np.float32), obstacle_t.astype(np.float32), assume_unique=True))

    if oids.size == 0:
        return np.zeros(x2.shape[0]).astype(bool)

    sid = obstacle_s[:, tid]
    obs_temp = obstacle[1]

    safe_arr = np.zeros(x2.shape[0]).astype(bool)
    safe_id = np.arange(0, x2.shape[0], 1)

    t_id = (obstacle_t / time_step).astype(int)


    for i, oid in enumerate(oids):

        o_d = obs_temp[oid][:, None] - sid[:, i]

        o_du = o_d.copy()
        o_du[~(o_du >= 0)] = 99999
        min_d_u = o_du[o_du.argmin(axis=0), np.arange(o_du.shape[1])]

        o_dl = o_d.copy()
        o_dl[~(o_dl < 0)] = -99999
        min_d_l = o_dl[o_dl.argmax(axis=0), np.arange(o_dl.shape[1])]

        safe_i = (min_d_u >= safety) & (min_d_l <= -(safety))

        safe_id = safe_id[safe_i]
        
        if safe_id.size == 0:
            break
        sid = sid[safe_i, :]
    safe_arr[safe_id] = np.ones(safe_id.size).astype(bool)

    return ~safe_arr 

def extend_node(m, c_n, c_n_id):
    n_x = c_n.t + m[0] * dT
    a = m[1]
    n_v = np.clip(c_n.v + a * dT, 0, None)

    n_s = c_n.s + np.clip(c_n.v * dT + 0.5 * a * dT ** 2, 0, None)
    n_cost = cost_fun(n_s, n_v, a)

    n_node = Node(n_x, n_s, n_v,
                  c_n.cost + n_cost,
                  c_n_id)

    return grid_id(n_node), n_node

start_node = Node(0, s0, v0, 0.0, -1)
goal_node = None

open_set, closed_set = dict(), dict()
open_set[grid_id(start_node)] = start_node

counter = 0
closeidlist = []

t_m = motions + backup_motions

while len(open_set) != 0:
    counter += 1

    c_id = min(open_set, key=lambda o: (-open_set[o].t, open_set[o].cost + calc_heuristic(open_set[o])))

    current = open_set[c_id]
    current_id = c_id

    if np.isclose(current.t, T):
        goal_node = current
        break

    del open_set[current_id]

    closed_set[current_id] = current

    n_temp_list = []
    d_temp_list = []


    for motion in t_m:

        n_id, node = extend_node(motion, current, current_id)

        check = check_collision([current.t, current.t + dT], np.array([[current.s, node.s]]), obstacle_info)

        if check:
            continue

        if n_id in closed_set:
            continue

        if n_id not in open_set:
            open_set[n_id] = node

            if show_animation:
                if sa_count == 1:
                    plt.plot([current.t, node.t], [current.s, node.s], 'r', label='extended nodes')
                else:
                    plt.plot([current.t, node.t], [current.s, node.s], 'r')
                sa_count += 1
                plt.pause(0.1)
        else:
            if open_set[n_id].cost > node.cost:
                open_set[n_id] = node

                if show_animation:
                    if sa_count == 1:
                        plt.plot([current.t, node.t], [current.s, node.s], 'r', label='extended nodes')
                    else:
                        plt.plot([current.t, node.t], [current.s, node.s], 'r')
                    sa_count += 1
                    plt.pause(0.1)

    if goal_node is not None:
        break

rx, ry = [goal_node.t], [goal_node.s]
p_id = goal_node.parent_id

while p_id != -1:
    nd = closed_set[p_id]
    rx.append(nd.t)
    ry.append(nd.s)
    p_id = nd.parent_id

if show_animation:
    plt.plot(rx, ry, 'b', label='solution')
    plt.pause(0.001)
    plt.show()


