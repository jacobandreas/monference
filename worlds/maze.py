from collections import namedtuple
import numpy as np
import sys

#BOARD_SIZE = 7
#TRUE_SIZE = BOARD_SIZE * 2 - 1
MAX_LEN = 20
#MAX_LEN = 8

def shortest_path(board, init, goal):
    def neighbors(node):
        r, c = node
        out = []
        if r > 0 and not board[r-1, c]:
            out.append((r-1, c))
        if r < board.shape[0] - 1 and not board[r+1, c]:
            out.append((r+1, c))
        if c > 0 and not board[r, c-1]:
            out.append((r, c-1))
        if c < board.shape[1] - 1 and not board[r, c+1]:
            out.append((r, c+1))
        return out

    stack = [(init,)]
    visited = set()
    while len(stack) > 0:
        history = stack.pop()
        node = history[-1]
        if node == goal:
            return history
        if node in visited:
            continue
        visited.add(node)
        for neighbor in neighbors(node):
            stack.append(history + (neighbor,))
    assert False

def random_maze(size):
    # crappy random spanning tree
    vert_edges = [((i,j), (i+1,j)) for i in range(size-1) for j in range(size)]
    horiz_edges = [((i,j), (i,j+1)) for i in range(size) for j in range(size-1)]
    edges = vert_edges + horiz_edges
    np.random.shuffle(edges)
    tree = []
    nodes = set(sum(edges, ()))
    groups = {n: {n} for n in nodes}
    while len(tree) < len(nodes) - 1:
        fr, to = edge = edges.pop()
        if fr in groups[to]:
            continue
        tree.append(edge)
        groups[fr] |= groups[to]
        for to_ in groups[to]:
            groups[to_] = groups[fr]

        groups[to] = groups[fr]
    assert groups[to] == nodes
    assert set(sum(tree, ())) == nodes
    return tree


class MazeWorld(object):
    def __init__(self, config):
        self.config = config.world
        self.n_actions = 2
        self.size = self.config.size
        self.true_size = self.config.size * 2 - 1
        self.n_features = self.true_size * self.true_size * 3 + 4 + 2

    def sample_scenario(self):
        config = self.config

        while True:
            board = np.zeros((self.true_size, self.true_size, 3))

            # create walls
            board[1::2, :, 0] = 1
            board[:, 1::2, 0] = 1

            # knock out walls
            sp_tree = random_maze(self.size)
            for fr, to in sp_tree:
                if fr[0] == to[0]:
                    r = fr[0] * 2
                    c = fr[1] * 2 + 1
                else:
                    assert fr[1] == to[1]
                    r = fr[0] * 2 + 1
                    c = fr[1] * 2
                assert board[r, c, 0] == 1
                board[r, c, 0] = 0

            # solve
            init = tuple(np.random.randint(self.size, size=2))
            goal = None
            while goal is None:
                g = tuple(np.random.randint(self.size, size=2))
                if g != init:
                    goal = g
            i_init = init[0] * self.size + init[1]
            i_goal = goal[0] * self.size + goal[1]
            raw_init = (init[0] * 2, init[1] * 2) 
            raw_goal = (goal[0] * 2, goal[1] * 2)
            board[raw_init + (1,)] = 1
            board[raw_goal + (2,)] = 1
            demo = shortest_path(board[..., 0], raw_init, raw_goal)
            demo = demo[::2]
            demo = [(r / 2, c / 2) for r, c in demo]
            #demo = [(r + 0.5 * np.random.random() - 0.25,
            #         c + 0.5 * np.random.random() - 0.25)
            #        for r, c in demo]

            if len(demo) > MAX_LEN:
                continue

            demo = tuple(demo)

            #features = board.ravel()
            features = np.concatenate((board.ravel(), init, goal))
            #features = board.ravel()
            return MazeScenario(board, features, demo, init, goal)

class MazeScenario(object):
    def __init__(self, board, features, demo, start, goal):
        self.board = board
        self.features = features
        self.demo = demo
        self.start = start
        self.goal = goal

    def init(self):
        return MazeState(self, self.demo[0])

    def get_demonstration(self):
        return self.demo[1:]

class MazeState(object):
    def __init__(self, scenario, pos):
        self.scenario = scenario
        self.pos = pos

    def features(self):
        return np.concatenate((self.scenario.features, self.pos))

    def step(self, action, verify=False):
        #action = np.round(action).astype(int)
        action = np.maximum(action, 0)
        action = np.minimum(action, self.scenario.board.shape[0]/2)
        raction = np.round(action).astype(int)
        rpos = np.round(self.pos).astype(int)

        true_pos = np.asarray(rpos) * 2
        true_action = raction * 2

        diff = np.abs(raction - rpos).sum()
        mid = (true_pos + true_action) / 2

        action = tuple(action)

        if diff > 1 or self.scenario.board[mid[0], mid[1], 0]:
            if verify: assert False
            return 0, self, True
        if self.scenario.board[true_action[0], true_action[1], 2]:
            reward = 1
            terminate = True
        else:
            reward = 0
            terminate = False
        return reward, MazeState(self.scenario, action), terminate


#class Datum(namedtuple("Datum", ["features", "init", "goal", "demonstration", "task_data"])):
#    def inject_state_features(self, state):
#        return np.concatenate((self.features, state))
#
#def load_batch(n_batch):
#    data = []
#    while len(data) < n_batch:
#        board = np.zeros((TRUE_SIZE, TRUE_SIZE, 3))
#
#        # create walls
#        board[1::2, :, 0] = 1
#        board[:, 1::2, 0] = 1
#
#        # knock out walls
#        sp_tree = random_maze(BOARD_SIZE)
#        for fr, to in sp_tree:
#            if fr[0] == to[0]:
#                r = fr[0] * 2
#                c = fr[1] * 2 + 1
#            else:
#                assert fr[1] == to[1]
#                r = fr[0] * 2 + 1
#                c = fr[1] * 2
#            assert board[r, c, 0] == 1
#            board[r, c, 0] = 0
#
#        # solve
#        init = tuple(np.random.randint(BOARD_SIZE, size=2))
#        goal = tuple(np.random.randint(BOARD_SIZE, size=2))
#        i_init = init[0] * BOARD_SIZE + init[1]
#        i_goal = goal[0] * BOARD_SIZE + goal[1]
#        raw_init = (init[0] * 2, init[1] * 2) 
#        raw_goal = (goal[0] * 2, goal[1] * 2)
#        board[raw_init + (1,)] = 1
#        board[raw_goal + (2,)] = 1
#        demo = shortest_path(board[..., 0], raw_init, raw_goal)
#        demo = demo[::2]
#        demo = [(r / 2, c / 2) for r, c in demo]
#        #demo = [d[0] * BOARD_SIZE + d[1] for d in demo]
#
#        l1dist = np.sum(np.abs(np.asarray(init) - np.asarray(goal)))
#        if len(demo) <= l1dist + 1:
#            continue
#        if len(demo) > MAX_LEN:
#            continue
#
#        demo = tuple(demo)
#
#        #features = board.ravel()
#        features = np.concatenate((board.ravel(), init, goal))
#        datum = Datum(features, init, goal, demo, board)
#        data.append(datum)
#
#    return data
#
#def evaluate(prediction, datum):
#    #return 1. if tuple(prediction) == datum.demonstration else 0.
#    rounded = np.round(prediction).astype(int)
#    if len(datum.demonstration) > len(prediction):
#        print >>sys.stderr, "WARNING: length mismatch"
#        return 0.
#    for t in range(len(datum.demonstration)):
#        if datum.demonstration[t] != tuple(rounded[t, :]):
#            return 0.
#    return 1.
#
#def random_maze(size):
#    # crappy random spanning tree
#    vert_edges = [((i,j), (i+1,j)) for i in range(size-1) for j in range(size)]
#    horiz_edges = [((i,j), (i,j+1)) for i in range(size) for j in range(size-1)]
#    edges = vert_edges + horiz_edges
#    np.random.shuffle(edges)
#    tree = []
#    nodes = set(sum(edges, ()))
#    groups = {n: {n} for n in nodes}
#    while len(tree) < len(nodes) - 1:
#        fr, to = edge = edges.pop()
#        if fr in groups[to]:
#            continue
#        tree.append(edge)
#        groups[fr] |= groups[to]
#        for to_ in groups[to]:
#            groups[to_] = groups[fr]
#
#        groups[to] = groups[fr]
#    assert groups[to] == nodes
#    assert set(sum(tree, ())) == nodes
#    return tree
#
#def shortest_path(board, init, goal):
#    def neighbors(node):
#        r, c = node
#        out = []
#        if r > 0 and not board[r-1, c]:
#            out.append((r-1, c))
#        if r < board.shape[0] - 1 and not board[r+1, c]:
#            out.append((r+1, c))
#        if c > 0 and not board[r, c-1]:
#            out.append((r, c-1))
#        if c < board.shape[1] - 1 and not board[r, c+1]:
#            out.append((r, c+1))
#        return out
#
#    stack = [(init,)]
#    visited = set()
#    while len(stack) > 0:
#        history = stack.pop()
#        node = history[-1]
#        if node == goal:
#            return history
#        if node in visited:
#            continue
#        visited.add(node)
#        for neighbor in neighbors(node):
#            stack.append(history + (neighbor,))
#    assert False
#
#BLOCK = u"\u2593"
#CRUMB = u"o"
#
#def visualize(path, datum, prefix):
#    #path = [(r*2, c*2) for r, c in path]
#    #path = [(p / BOARD_SIZE, p % BOARD_SIZE) for p in path]
#    path = [(np.round(r) * 2, np.round(c) * 2) for r, c in path]
#    board = datum.task_data[..., 0]
#    print BLOCK * 2 * (board.shape[1] + 2)
#    for r in range(board.shape[0]):
#        sys.stdout.write(BLOCK * 2)
#        for c in range(board.shape[1]):
#            h = int(board[r,c])
#            p = {
#                0: " ",
#                1: BLOCK
#            }[h]
#            if (r, c) in path:
#                p = CRUMB
#            sys.stdout.write(p)
#            sys.stdout.write(p)
#        sys.stdout.write(BLOCK * 2)
#        sys.stdout.write("\n")
#    print BLOCK * 2 * (board.shape[1] + 2)
#
#def score(datum, path):
#    npath = [np.asarray(p) for p in path]
#    board = datum.task_data[..., 0]
#    rewards = [0 for p in path]
#    transitions = [1 for p in path]
#    for t in range(1, len(path)):
#        prev = np.asarray(path[t-1])
#        curr = np.asarray(path[t])
#        if np.linalg.norm(curr - prev) >= 1.5:
#            transitions[t] = 0
#            continue
#
#        pd = (np.round(prev[0]) * 2, np.round(prev[1] * 2))
#        cd = (np.round(curr[0]) * 2, np.round(curr[1] * 2))
#        betw = (int((pd[0] + cd[0]) / 2), int((pd[1] + cd[1]) / 2))
#        try:
#            if board[betw] > 0:
#                transitions[t] = 0
#        except:
#            transitions[t] = 0
#
#    if np.linalg.norm(npath[-1] - np.asarray(datum.goal)) < 0.5:
#        rewards[-1] = 1
#
#    return rewards, transitions
