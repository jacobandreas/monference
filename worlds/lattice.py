import numpy as np

EDGE_PROB = 0.65
GOAL_PROB = 0.2

def visualize(edges, goals, config):
    for i in range(config.size):
        for j in range(config.size-1):
            if (i, j) in goals:
                print "#",
            else:
                print "o",
            if ((i, j), (i, j+1)) in edges:
                print "-",
            else:
                print " ",
        if (i, config.size-1) in goals:
            print "#"
        else:
            print "o"

        for j in range(config.size):
            if ((i, j), (i+1, j)) in edges:
                print "|",
            else:
                print " ",
            print " ",
        print


class LatticeWorld(object):
    def __init__(self, config):
        self.config = config.world
        self.n_features = self.config.size * self.config.size * 6
        self.n_actions = 5

    def sample_scenario(self):
        config = self.config
        nodes = set((i, j) for i in range(config.size) for j in range(config.size))

        edges_h = [((i, j), (i, j+1)) 
                for i in range(config.size) for j in range(config.size - 1)]
        edges_v = [((i, j), (i+1, j))
                for i in range(config.size - 1) for j in range(config.size)]
        edges = edges_h + edges_v
        edges = [e for e in edges if np.random.random() < EDGE_PROB]
        edges = set(edges)

        goals = {n: np.random.random() for n in nodes 
                if np.random.random() < GOAL_PROB}

        init_pos = (np.random.randint(config.size), np.random.randint(config.size))

        return LatticeScenario(nodes, edges, goals, init_pos, config)

class LatticeScenario(object):
    def __init__(self, nodes, edges, goals, init_pos, config):
        self.nodes = nodes
        self.edges = edges
        self.goals = goals
        self.init_pos = init_pos
        self.config = config

        self.values = self.compute_values()
        self.features = self.compute_features()

    def compute_values(self):
        config = self.config
        err = np.inf
        values = np.zeros((config.size, config.size))
        states = [[LatticeState(self, (i, j), config) 
            for j in range(config.size)]
            for i in range(config.size)]
        while err >= 1e-3:
            new_values = np.zeros((config.size, config.size))
            for i in range(config.size):
                for j in range(config.size):
                    exit_val = states[i][j].step(4)[0]
                    neighbors = [states[i][j].step(d)[1].pos for d in range(4)]
                    neighbor_vals = [values[n] for n in neighbors]
                    val = max(exit_val, max(v * 0.9 for v in neighbor_vals))
                    new_values[i, j] = val
            err = np.square(new_values - values).max()
            values = new_values
        return values

    def compute_features(self):
        config = self.config
        features = np.zeros((5, config.size, config.size))
        for fr, to in self.edges:
            dr = to[0] - fr[0]
            dc = to[1] - fr[1]
            assert (dr == 1) ^ (dc == 1)
            if dr == 1:
                features[0, fr[0], fr[1]] = 1
                features[1, to[0], to[1]] = 1
            else:
                features[2, fr[0], fr[1]] = 1
                features[3, to[0], to[1]] = 1

        for goal, reward in self.goals.items():
            features[4, goal[0], goal[1]] = reward

        features = features.ravel()
        return features

    def init(self):
        return LatticeState(self, self.init_pos, self.config)

    def get_demonstration(self):
        config = self.config
        positions = [self.init_pos]
        actions = []
        states = [[LatticeState(self, (i, j), config) 
            for j in range(config.size)] 
            for i in range(config.size)]
        while True:
            pos = positions[-1]
            exit_val = self.goals[pos] if pos in self.goals else 0
            state = states[pos[0]][pos[1]]
            neighbors = [state.step(a)[1].pos for a in range(4)]
            neighbor_vals = [self.values[n] if n != pos else -np.inf for n in neighbors]
            vals = neighbor_vals + [exit_val]
            best_action = np.argmax(vals)
            if max(vals) == 0 or best_action == 4:
                actions.append(4)
                break
            else:
                actions.append(best_action)
                positions.append(neighbors[best_action])

        return actions

class LatticeState(object):
    def __init__(self, scenario, pos, config):
        self.scenario = scenario
        self.pos = pos
        self.config = config

    def features(self):
        config = self.config
        pos_features = np.zeros((config.size, config.size))
        pos_features[self.pos] = 1
        return np.concatenate((self.scenario.features, pos_features.ravel()))

    def step(self, action):
        x, y = self.pos

        reward = 0
        dx, dy = (0, 0)
        terminate = False

        if action == 0:
            dx, dy = (-1, 0)
        elif action == 1:
            dx, dy = (1, 0)
        elif action == 2:
            dx, dy = (0, -1)
        elif action == 3:
            dx, dy = (0, 1)
        elif action == 4:
            terminate = True
            if self.pos in self.scenario.goals:
                reward = self.scenario.goals[self.pos]

        nx, ny = (x + dx, y + dy)
        if ((x, y), (nx, ny)) not in self.scenario.edges and \
                ((nx, ny), (x, y)) not in self.scenario.edges:
            nx, ny = x, y

        return reward, LatticeState(self.scenario, (nx, ny), self.config), terminate
