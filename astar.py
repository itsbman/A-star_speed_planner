import numpy as np
import matplotlib.pyplot as plt
import time
import json
import copy
import pickle

show_animation = True
sa_count = 1

plist = []
with open('pred_list2.txt') as file:
    data = json.load(file)
    for p in data['fut']:
        if len(p) != 0:
            for post in p:
                for pos in post[0]:
                    plist.append([pos, post[1]])


def grid_id(n):
    # return round(10 * n.s) * grid_x + 10 * n.t * n.v
    return str(round(100 * n.s)) + str(round(10 * n.t)) + str(round(1000 * n.v))


# initial conditions
v0 = 9
a0 = 0
s0 = 0
v_ref = 9
a_max, a_min = 5, -5

T = 4
dT = 0.5
time_step = 0.1

dg_X = dT
dg_Y = 1
Y_max = 40
# safety = 3 #3.9686932853499925 # 3.5
safety = np.array([3.6, 3.6077, 3.6168, 3.6273, 3.6392, 3.6525, 3.6672, 3.6833,
                        3.7008, 3.7197, 3.74, 3.7617, 3.7848, 3.8093, 3.8352, 3.8625,
                        3.8912, 3.9213, 3.9528, 3.9857, 4.02, 4.0557, 4.0928, 4.1313,
                        4.1712, 4.2125, 4.2552, 4.2993, 4.3448, 4.3917, 4.44, 4.4897,
                        4.5408, 4.5933, 4.6472, 4.7025, 4.7592, 4.8173, 4.8768, 4.9377,
                        5.])

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

t_motions = [
    [1, -2],
    [1, -1],
    [1, 0],
    [1, 1],
    [1, -3],
    [1, -4],
    [1, -5],
    [1, 2]
]
grid_x = int(T/dT + 1)
grid_y = int(Y_max / dg_Y + 1)


x = np.arange(0, grid_x, 1)  # grid index
y = np.arange(0, grid_y, 1)
t = np.arange(0, T+time_step, time_step)


def convert_obs2(obs_i):
    # return: [[time], [obs(time)]]
    if obs_i.size == 0:
        return []
    obs_i = obs_i.transpose()
    to = np.unique(obs_i[1])
    obs_temp = [obs_i[0, np.where(np.isclose(tt, obs_i[1, :]))].reshape(-1, ) for tt in to]
    # print(obs_temp)
    obs_new = [to*time_step, obs_temp]
    return obs_new


# obs_list = [[20, 2], [20, 2.5], [20, 3]]
obs1 = np.array(plist)[3:]
obs2 = copy.deepcopy(obs1)
# problematic
obs1[:, 0] += 1
obs1[:, 1] += 10
obs2[:, 0] += 15
obs2[:, 1] += 5
# obs1[:, 0] += 10
# obs1[:, 1] += 10
# obs2[:, 0] += 20
# obs2[:, 1] += 10
obs111 = np.concatenate((obs1, obs2))
obs11 = convert_obs2(obs111)

obs21 = convert_obs2(obs2)
obs_list = obs11

obstacle_map = np.zeros((grid_x, grid_y, 2), dtype=bool)

file = open('run_debug_198.pckl', 'rb')
done_state = pickle.load(file)
es_list = pickle.load(file)
efs_list = pickle.load(file)
obs_list = pickle.load(file)
observationp = pickle.load(file)

init_state = [0.0, es_list[-1][2]]
s0 = init_state[0]
v0 = init_state[1]

v_u, v_l = v0 + a_max * t,  v0 + a_min * t
s_min = np.zeros(t.size)
s_min1 = np.zeros(t.size)
s_max = np.zeros(t.size)
s_min[1:] = np.cumsum(np.clip(v_l + 0.5 * a_min * time_step, 0, None))[:-1]*time_step
v_l1 = np.clip(v_l, 0, None)
s_min1[1:] = np.cumsum(np.clip(v_l1 + 0.5 * a_min * time_step, 0, None))[:-1]*time_step
s_max[1:] = np.cumsum(v_u + 0.5 * a_max * time_step)[:-1]*time_step

obs_list = obs_list[-1]
# version 1
for ix in x:
    xx = ix * dg_X
    for iy in y:
        yy = iy * dg_Y
        if not (s_min[ix*int(dg_X/0.1)] <= yy):  # <= s_max[ix*int(dg_X/0.1)]):
            obstacle_map[ix, iy, 0] = True
            # obstacle_map[ix, iy, 1] = True
            continue

# version 3
# safe_index = round(safety / dg_Y)

# plt.figure(2)
# plt.imshow(obstacle_map[:, :, 0].transpose(), origin='lower')
if show_animation:
    plt.figure(1)
    # for obs_id in obs_list:
    #     obs_1 = obs_id.transpose()
    plt.grid(True)
    # plt.plot(obs1[:, 1]*0.1, obs1[:, 0], 'ok', label='obstacle')
    for i in range(obs_list[0].shape[0]):
        for j in range(obs_list[1][i].shape[0]):
            plt.plot(obs_list[0][i], obs_list[1][i][j], 'ok')
    # plt.plot(obs_list[0], obs_list_t, 'ok', label='obstacle')
    # plt.plot(obs2[:, 1]*0.1, obs2[:, 0], 'ok')
    plt.plot(t, s_min, 'g', label='acc. limit')
    plt.plot(t, s_min1, 'r--', label='acc. limit')
    plt.plot(t, s_max, 'g')
    # plt.show()


def cost_fun(s_n, v_n, a_n):
    Kv, Ka = 1, 1
    cost_v = (v_n - v_ref)**2
    cost_a = a_n**2
    return Kv * cost_v + Ka * cost_a


def cost_fun2(s_n, v_n, a_n):
    Kv, Ka = 1, 1
    cost_v = np.sum((v_n - v_ref)**2)
    # cost_v = np.sum(cost_v)
    cost_a = np.sum(a_n**2)
    return Kv * cost_v + Ka * cost_a


def valid(xid, yid):
    if yid >= grid_y:
        return False, False

    om = obstacle_map[xid, yid, :]
    if om[0]:
        return om[0], om[1]

    return True, False


# def ccw(x1, y1, x2, y2, x3, y3):
#     # https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
#     return (y3-y1)*(x2-x1) > (y2-y1)*(x3-x1)


def calc_heuristic(nnode):
    v = nnode.v
    t0 = nnode.t
    tg = np.arange(0, T-t0+dT, dT)
    coll_checks = 0

    for _motion in motions:
        s = np.zeros(tg.size)
        acc = _motion[1]
        # delta_s = np.clip(v + acc * tg, 0, None)[1:] * dT
        ###
        v1 = v + a.reshape(-1, 1)*tg
        delta_s = np.clip(v1 + 0.5 * acc * dT)*dT
        np.cumsum(delta_s, axis=1)[:, :-1]
        ###
        vt = v + acc * tg
        delta_s = np.clip(vt + 0.5 * acc * dT, 0, None)[:-1] * dT
        s[1:] = np.cumsum(delta_s)
        s += nnode.s
        if show_animation:
            # print(tg+t0, s)
            l1 = plt.plot(tg+t0, s, 'y--')
            plt.pause(0.01)
        if check_collision2(tg+t0, s, obs_list):
            coll_checks += 1
            l3 = l1.pop(0)
            l3.remove()
        else:
            # print(_motion)
            return False

    if coll_checks >= len(motions):
        for _b_motion in backup_motions:
            s = np.zeros(tg.size)
            acc = _b_motion[1]

            # delta_s = np.clip(v + acc * tg, 0, None)[:-1] * dT
            # delta_s = np.clip(v + 0.5 * acc * tg, 0, None)[1:] * dT

            vt = v + acc * tg
            delta_s = np.clip(vt + 0.5 * acc * dT, 0, None)[:-1] * dT
            s[1:] = np.cumsum(delta_s)
            s += nnode.s
            if show_animation:
                # print(tg + t0, s)
                l1 = plt.plot(tg+t0, s, 'y--')
                plt.pause(0.01)
            if check_collision2(tg+t0, s, obs_list):
                coll_checks += 1
                l2 = l1.pop(0)
                l2.remove()
            else:
                # print(_b_motion)
                return False

        return True
    else:
        return False


def calc_heuristic3(nnode):
    '''
    :param nnode: node type
    :return: True if there is no possible heuristic path
    '''
    v = nnode.v
    t0 = nnode.t
    # tt = min(T - t0, 2)
    # tg = np.arange(0, T-t0+dT, dT)
    # tg = np.arange(0, tt + dT, dT)
    tg = np.linspace(0, T-t0, int((T-t0)/0.5)+1)

    t_m = motions + backup_motions
    s = np.zeros((len(t_m), tg.size))
    acc = np.array(t_m)[:, 1].reshape(-1, 1)
    v1 = v + acc * tg
    v11 = np.clip(v1, 0, None)
    delta_s = np.clip(v1 + 0.5 * acc * dT, 0, None)[:, :-1] * dT
    s[:, 1:] = np.cumsum(delta_s, axis=1)
    s += nnode.s

    # h_bool = check_collision2(tg + t0, s, obs_list)
    a_cost = (acc**2).reshape(-1,)
    v_cost = np.sum((v11 - v_ref)**2, axis=1)
    # scost
    h_cost = a_cost + v_cost

    return np.min(h_cost)


def calc_heuristic22(nnode):
    '''
    :param nnode: node type
    :return: True if there is no possible heuristic path
    '''
    v = nnode.v
    t0 = nnode.t
    tt = min(T-t0, 2)
    # tg = np.arange(0, T-t0+dT, dT)
    
    tg = np.arange(0, tt+dT, dT)

    # for _motion in motions:
    # ssst = time.time()
    # v_u, v_l = v0 + a_max * t, v0 + a_min * t
    # s_min = np.zeros(t.size)
    # s_max = np.zeros(t.size)
    # s_min[1:] = np.cumsum(np.clip(v_l + 0.5 * a_min * time_step, 0, None))[:-1] * time_step

    t_m = motions #+ backup_motions
    s = np.zeros((len(t_m), tg.size))
    acc = np.array(t_m)[:, 1].reshape(-1, 1)
    v1 = v + acc * tg
    delta_s = np.clip(v1 + 0.5 * acc * dT, 0, None)[:, :-1]*dT
    s[:, 1:] = np.cumsum(delta_s, axis=1)
    s += nnode.s
    # print('ch')
    # print(time.time() - ssst)
    # ssst = time.time()
    # print(s)
    h_bool = check_collision2(tg+t0, s, obs_list)
    # print(time.time() - ssst)
    if False in h_bool:

        idx, = np.where(h_bool == False)
        if show_animation:
            for i in idx.tolist():
                ll = plt.plot(tg + t0, s[i, :], 'y--')
            plt.pause(0.01)

        return False, None

    else:

        s = np.zeros((len(backup_motions), tg.size))
        acc = np.array(backup_motions)[:, 1].reshape(-1, 1)
        v1 = v + acc * tg
        delta_s = np.clip(v1 + 0.5 * acc * dT, 0, None) * dT
        s[:, 1:] = np.cumsum(delta_s, axis=1)[:, :-1]
        s += nnode.s

        h_bool1 = check_collision2(tg + t0, s, obs_list)

        if False in h_bool1:

            idx, = np.where(h_bool1 == False)
            if show_animation:

                for i in idx.tolist():
                    ll = plt.plot(tg + t0, s[i, :], 'y--')
                plt.pause(0.01)

            return False, np.array(backup_motions)[idx, :].tolist()

        else:
            return True, None


def check_collision2(x1, x2, obstacle):
    '''
    :param x1:
    :param x2:
    :param obstacle:
    :return: True: violates safety
    '''

    if isinstance(x2, list):
        x2 = np.array(x2).reshape(len(x2), -1)

    if len(obstacle) == 0:
        return np.zeros(x2.shape[0]).astype(bool)

    # to = np.arange(x1[0], x1[-1] + time_step, time_step)
    to = np.linspace(x1[0], x1[-1], int((x1[-1]-x1[0])/0.1)+1)
    so2 = np.array([np.interp(to, x1, x2[i]) for i in range(x2.shape[0])])
    # for obstacle in obs_map:
    tid = np.isin(to.astype(np.float32), obstacle[0].astype(np.float32), assume_unique=True)
    oids, = np.where(np.isin(obstacle[0].astype(np.float32), to.astype(np.float32), assume_unique=True))
    # print(time.time() - ssst)
    if oids.size == 0:
        return np.zeros(x2.shape[0]).astype(bool)

    sid = so2[:, tid]
    obs_temp = obstacle[1]

    safe_arr = np.zeros(x2.shape[0]).astype(bool)
    safe_id = np.arange(0, x2.shape[0], 1)

    t_id = (to / time_step).astype(int)
    safety1 = safety[t_id]
    safety1 = safety1[tid]

    for i, oid in enumerate(oids):
        # ssst = time.time()
        # print('method1', time.time() - ssst)

        # ssst = time.time()
        # mind = np.min([np.square((o - sid[:, i])) for o in obs_temp[oid]], axis=0)

        o_d = obs_temp[oid][:, None] - sid[:, i]

        o_du = o_d.copy()
        o_du[~(o_du >= 0)] = 99999
        min_d_u = o_du[o_du.argmin(axis=0), np.arange(o_du.shape[1])]

        o_dl = o_d.copy()
        o_dl[~(o_dl < 0)] = -99999
        min_d_l = o_dl[o_dl.argmax(axis=0), np.arange(o_dl.shape[1])]

        safe_i = (min_d_u >= safety1[i]) & (min_d_l <= -(safety1[i] + 5))

        safe_id = safe_id[safe_i]
        
        # old
        # min_id = np.abs(o_d).argmin(axis=0)
        # mind = o_d[min_id, np.arange(o_d.shape[1])]
        # # print('method2', time.time() - ssst)
        #
        # safe_i = (mind >= safety1[i]) | (mind <= -(safety1[i] + 5))
        # safe_id = safe_id[safe_i]
        if safe_id.size == 0:
            break
        sid = sid[safe_i, :]
    safe_arr[safe_id] = np.ones(safe_id.size).astype(bool)

    return ~safe_arr  # safe_bool.any(1)


def extend_node(m, c_node, c_node_id):
    n_x = c_node.t + m[0] * dT
    a = m[1]
    n_v = np.clip(c_node.v + a * dT, 0, None)

    # if n_v < 0:
    #     return None, None

    # n_s = c_node.s + c_node.v * dT + 0.5 * a * dT ** 2
    n_s = c_node.s + np.clip(c_node.v * dT + 0.5 * a * dT ** 2, 0, None)
    n_cost = cost_fun(n_s, n_v, a)

    n_node = Node(n_x, n_s, n_v,
                  c_node.cost + n_cost,
                  c_node_id)

    return grid_id(n_node), n_node


def get_constraints(ssoln, tsoln):

    if len(obs_list) == 0:
        return [], []

    to = np.arange(tsoln[0], tsoln[-1] + time_step, time_step)
    so = np.interp(to, tsoln, ssoln)

    # for obstacle in obs_map:
    tid = np.isin(to.astype(np.float32), obs_list[0].astype(np.float32), assume_unique=True)
    oids, = np.where(np.isin(obs_list[0].astype(np.float32), to.astype(np.float32), assume_unique=True))
    # print(time.time() - ssst)
    if oids.size == 0:
        return [], []

    sid = so[tid]

    obs_temp = obs_list[1]

    u_bound, l_bound = [], []

    for i, oid in enumerate(oids):
        d = np.array([o - sid[i] for o in obs_temp[oid]])
        u = d[d > 0] + sid[i]
        l = d[d < 0] + sid[i]
        if len(u) != 0:
            u_bound.append([np.min(u), obs_list[0][oid]])
        if len(l) != 0:
            l_bound.append([np.max(l), obs_list[0][oid]])

    return u_bound, l_bound


class Node:
    def __init__(self, xid, yid, v, cost, parent_id):
        # TODO: convert t from index to actual t
        self.t = xid  # index
        self.s = yid
        self.v = v
        self.cost = cost
        self.parent_id = parent_id

    def __state__(self):
        return self.s, self.v, self.t, self.cost, self.parent_id


start_node = Node(0, s0, v0, 0.0, -1)
goal_node = None

open_set, closed_set = dict(), dict()
open_set[grid_id(start_node)] = start_node

st = time.time()
counter = 0
closeidlist = []

while len(open_set) != 0:
    counter += 1

    # sst = time.time()

    # c_id_list = sorted(open_set, key=lambda o: open_set[o].cost + calc_heuristic3(open_set[o]))
    c_id = min(open_set, key=lambda o: (-open_set[o].t, open_set[o].cost + calc_heuristic3(open_set[o])))

    # current_nodes = []
    # for c_id in c_id_list:
    #     sst = time.time()
    #     check_h, b_motions = calc_heuristic3(open_set[c_id])
    #     if check_h:
    #         continue
    #     else:
    #
    #         closeidlist.append(c_id)
    #         current_nodes.append([open_set[c_id], c_id, b_motions])
    #
    #         if show_animation:
    #             pid = open_set[c_id].parent_id
    #             if pid != -1:
    #                 plt.plot([open_set[c_id].t, closed_set[pid].t],
    #                          [open_set[c_id].s, closed_set[pid].s], 'b--')
    #
    #         target_nodes = current_nodes
    #     if len(current_nodes) == 2:
    #         break

    # print(time.time() - sst)
    # sst = time.time()
    current = open_set[c_id]
    current_id = c_id

    coll_counter2 = 0
    # print(current.cost)
    if np.isclose(current.t, T):
        goal_node = current
        break

    del open_set[current_id]

    closed_set[current_id] = current

    n_temp_list = []
    d_temp_list = []

    for motion in t_motions:

        n_id, node = extend_node(motion, current, current_id)

        check = check_collision2([current.t, current.t + dT], np.array([[current.s, node.s]]), obs_list)

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

    # print(time.time() - sst)
    # print('end loop')
    if goal_node is not None:
        break
# TODO: else no solution when None
rx, ry = [goal_node.t], [goal_node.s]
p_id = goal_node.parent_id

while p_id != -1:
    nd = closed_set[p_id]
    rx.append(nd.t)
    ry.append(nd.s)
    p_id = nd.parent_id

print(time.time()-st)

uu, ll = get_constraints(ry[::-1], rx[::-1])

if show_animation:
    plt.plot(rx, ry, 'b', label='solution')
    for u in uu:
        plt.plot(u[1], u[0], 'y*')
    for l in ll:
        plt.plot(l[1], l[0], 'y*')

    plt.legend()
    plt.xlabel('time [s]')
    plt.ylabel('distance travelled [m]')
    plt.pause(0.001)
    plt.show()


