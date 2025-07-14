import numpy as np
import matplotlib.pyplot as plt
import math
import random
from itertools import permutations
import Diadic

class MBDE:
    def __init__(self, T: int) -> None:
        self.T = T
        self.t = 1
        self.l = 0  # Episode counter
        self.c0 = 0.3
        self.t_start_replay = self.t

    def initialize_episode(self):
        self.l += 1
        print(f'Entering Episode {self.l}')
        self.block = 0
        self.m = 2  # Minimal depth

    def initialize_block(self):
        self.m += 1
        print(f'Entering Block {self.m}')
        self.starting_block = self.t
        self.ending_block = self.t + 8 ** self.m

        self.ScheduleReplays()
        self.t_start_replay = self.t
        # test

        self.tree = Diadic.Tree(self.m)
        self.tree.update_proba()
        self.B_MASTER = self.tree.active_depths[self.m]

    def ScheduleReplays(self):
        """
        Schedule replays ensuring that at most one replay is active at any time s.
        Preference is given to the deepest depth where a replay is triggered.
        """
        block_size = 8 ** self.m
        self.Replays = np.zeros((block_size, self.m - 2))  # depths 2 to m-1

        for s in range(2, block_size):
            selected_d_index = None
            for d in reversed(range(2, self.m)):  # reverse: prioritize deeper depths
                if s % 8 ** d == 0:
                    p = np.sqrt(8 ** d / (s + 1))
                    if np.random.rand() < p:
                        selected_d_index = d - 2  # adjust for zero-indexed column
                        break  # only one replay allowed per time s

            if selected_d_index is not None:
                self.Replays[s, selected_d_index] = 1

        self.get_mask()
        self.visualize_replays()


    def get_mask(self):
        block_size = 8 ** self.m
        self.active_mask = np.zeros_like(self.Replays)

        for t in range(2, block_size):
            for d_index in range(self.m - 2):
                d = d_index + 2
                length = 8 ** d
                s = t - (t % length)
                if s < block_size and self.Replays[s, d_index] == 1 and s <= t < s + length:
                    self.active_mask[t, :] = 0
                    self.active_mask[t, d_index] = 1
                    break  # Ensure exclusivity

    def get_starting_depths(self, t: int) -> set:
        return {
            d_index + 2
            for d_index in range(self.m - 2)
            if 0 <= t < 8 ** self.m and self.Replays[t, d_index] == 1
        }

    def get_ending_depths(self, t: int) -> set:
        result = set()
        for d_index in range(self.m - 2):
            d = d_index + 2
            s = t - 8 ** d
            if 0 <= s < 8 ** self.m and self.Replays[s, d_index] == 1:
                result.add(d)
        return result

    def active_depth_set(self) -> set:
        if not (0 <= self.t < self.active_mask.shape[0]):
            raise ValueError('Round t is out of bounds')
        return {self.m}.union(
            {d + 2 for d, active in enumerate(self.active_mask[self.t]) if active}
        )

    def check_if_replay(self):
        starting_depths = self.get_starting_depths(self.t)
        ending_depths = self.get_ending_depths(self.t)

        if ending_depths:
            d_ending = next(iter(ending_depths))
            print(f'Depth {d_ending} deactivated at t={self.t}')
            self.tree.de_activate_depth(d_ending)
            self.tree.visualize(t=self.t)

            if len(self.tree.active_depths) == 1: # donc si le seul active depth est m
                print('solo run at ', self.t)
                # alors cB_m(t) c'est exactement B_MASTER
                self.tree.active_depths[self.m] = [
                    self.tree.find_node(node.depth, node.index)
                    for node in self.B_MASTER
                    if self.tree.find_node(node.depth, node.index) is not None
                ]

        if starting_depths:
            self.t_start_replay = self.t
            d_starting = next(iter(starting_depths))
            print(f'Depth {d_starting} activated at t={self.t}')
            self.tree.activate_depth(d_starting)
            self.tree.activate_depth(self.m)
            self.tree.visualize(t=self.t)

    def choose_action(self):
        self.check_if_replay()

        min_depth = min(self.tree.active_depths)
        current = random.choices(
            self.tree.active_depths[min_depth],
            weights=[n.proba for n in self.tree.active_depths[min_depth]],
            k=1
        )[0]

        for d in sorted(self.tree.active_depths):
            if d <= current.depth:
                continue
            children = [
                node for node in self.tree.active_depths[d]
                if node.active and node.parent == current
            ]
            if not children:
                break
            current = random.choice(children)

        return current.sample()

    def update(self, x_t, y_t):
        self.tree.update_estimates(x_t, y_t)
        self.eviction_test()
        self.tree.update_proba()
        self.update_B_Master()
        self.t += 1
        self.check()

    def update_B_Master(self):
        B_MASTER = []
        intersection_nodes_index = set(self.tree.active_depths[self.m]) & set(self.B_MASTER)
        for node in intersection_nodes_index :
            node_in_tree = self.tree.find_node(node.depth, node.index)
            #B_MASTER.append(Diadic.Node(node.depth, node.index).clone_from(node_in_tree)) # la c'est ptet overkill mais je galere a faire marcher ca
            B_MASTER.append(node_in_tree)
        self.B_MASTER = B_MASTER

    def eviction_test(self):
        def threshold(s1, s2, d):
            return self.c0 * math.log(self.T) * math.sqrt((s2 - s1) * (2 ** d)) + (4 * (s2 - s1) / 2 ** d)

        def eviction_criteria(B1, B2, d):
            n = self.t - self.t_start_replay
            if n <= 1:
                return False
            diff = [B1.mean_estimates[i] - B2.mean_estimates[i] for i in range(n)]
            cumsum = 0
            for s2 in range(1, n):
                cumsum += diff[s2]
                if cumsum > threshold(0, s2, d):
                    return True
            return False

        flag = False
        for d, nodes in self.tree.active_depths.items():
            for B, Bp in permutations(nodes, 2):
                if eviction_criteria(B, Bp, d):
                    print(f'Node ({Bp.depth}, {Bp.index}) evicted at t={self.t}')
                    Bp.evict()
                    flag = True

        if flag:
            for d in self.tree.active_depths:
                self.tree.active_depths[d] = self.tree.collect_active_nodes(d)

    def check(self):
        if not self.B_MASTER:
            print(f'Shift at {self.t}')
            self.initialize_episode()
            self.initialize_block()
            
        elif self.t == self.ending_block:
            # Take the time to deactivate all depth (except m)
            for d in self.tree.active_depths.keys():
                if d != self.m :
                    self.tree.de_activate_depth(d)

            self.initialize_block()

    def visualize_replays(self):
        m = self.active_mask.shape[1] + 2
        T = self.active_mask.shape[0]
        plt.figure(figsize=(12, (m - 2) * 0.6))

        plt.imshow(
            self.active_mask.T,
            aspect='auto',
            cmap='YlOrBr',
            interpolation='nearest',
            extent=[self.starting_block, self.starting_block + T, 0, m - 2]
        )

        plt.ylabel("Depth")
        plt.xlabel("Time")
        plt.title("Active Intervals Over Time by Depth")

        plt.yticks(range(m - 2), [f"Depth {d}" for d in range(2, m)])
        step = max(1, T // 10)
        plt.xticks(
            [self.starting_block + t for t in range(0, T, step)],
            [str(self.starting_block + t) for t in range(0, T, step)]
        )

        for d_index in range(m - 2):
            for t in range(T):
                if self.Replays[t, d_index] == 1:
                    plt.plot(self.starting_block + t, d_index, 'ro', markersize=4)

        plt.tight_layout()
        plt.show()
