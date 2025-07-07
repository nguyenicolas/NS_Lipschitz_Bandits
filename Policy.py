import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import random
import math
import copy

import Diadic

class MBDE :
    def __init__(self, T) -> None:
        self.T = T
        self.t = 1
        self.l = 0 # current episode

        # Constants
        self.c0 = 0.5

    def initialize_episode(self):
        self.l+=1
        print(f'Entering Episode {self.l}')
        self.block = 0
        self.m = 2 # bon ca fait pas trop sens de considerer qu'une seule bin, on perd des rounds inutilement

    def initialize_block(self):
        
        self.m+=1
        print(f'Entering block {self.m}')
        self.ScheduleReplays()
        #self.tree.active_depths = self.active_depth_set() en fait on initialize direct a m au 1er round
        
        self.starting_block = copy.deepcopy(self.t)
        self.ending_block = self.starting_block + 8**self.m

        self.tree = Diadic.Tree(self.m)
        self.d_t = {self.m}
        self.tree.update_proba()

        self.B_MASTER = self.tree.active_depths[self.m]

    def ScheduleReplays(self):
        """ Schedule all replays for a given block of size 8^m, depths >= 2 """
        self.Replays = np.zeros((8**self.m, self.m - 2))  # depths from 2 to m-1 => m-2 columns

        for s in range(2, 8**self.m):  # 0-based indexing
            for d in range(2, self.m):  # depth d ∈ [2, m-1]
                if s % 8**d == 0:
                    p_s_d = np.sqrt(8**d / (s + 1))  # s+1 because s is 0-indexed now
                    R_s_d = int(np.random.random() < p_s_d)
                    self.Replays[s, d - 2] = R_s_d  # d-2 maps depth 2 → column 0, etc.

        self.get_mask()
        self.visualize_replays()


    def get_mask(self):
        self.active_mask = np.zeros_like(self.Replays)

        for d_index in reversed(range(self.m - 2)):  # corresponds to depths from 2 to m-1
            d = d_index + 2
            length = 8**d
            for s in range(2, 8**self.m):
                if self.Replays[s, d_index] == 1:
                    end = min(s + length, 8**self.m)
                    for t in range(s, end):
                        if not self.active_mask[t].any():  # only assign if no deeper depth active
                            self.active_mask[t, d_index] = 1



    def active_depth_set(self):
        """ 
        gives the set of active depths at current round
        """
        if self.t < 0 or self.t >= self.active_mask.shape[0]:
            raise ValueError('Round s is out of bounds')
        return {self.m}.union({d + 2 for d, active in enumerate(self.active_mask[self.t]) if active == 1})
    
    def check_if_replay(self):

        # Check if new replay begins
        old_set = set(self.tree.active_depths.keys())
        new_set = self.active_depth_set()
        new_depths = new_set - old_set
        removed_depths = old_set - new_set

        # la je fais dans le setting ou il n'y a que 1 seul replay à la fois
        
        if new_depths : # si on entre dans un replay
            self.t_start_replay = copy.deepcopy(self.t)
            
            for d in new_depths:
                print(f'We activate depth {d} at time {self.t}')
                self.tree.activate_depth(d)

            self.tree.activate_depth(self.m) # on restart aussi depth m
            self.tree.visualize(t=self.t)
                # Do something
                # ...
                # Je me rappelle plus si il faut reactiver toutes les bins ou juste celles à cette depth ?
        
        if removed_depths : # si on sort d'un replay
            self.tree.visualize(t=self.t)
            for d in removed_depths:
                print(f'We remove depth {d} at time {self.t}')
                self.tree.de_activate_depth(d)

            # now cB_t(m) = B_MASTER (je galere a faire cette partie)









            #copied_nodes = copy.deepcopy(self.B_MASTER)
            #self.tree.active_depths[self.m] = copied_nodes


            #for node in self.tree.get_all_nodes_at_depth(self.m):
            #    if node not in self.B_MASTER :
            #        node.active = False
            #        self.tree.active_depths[self.m]

                #self.tree.active_depths[self.m] = list(set(self.tree.active_depths[self.m]) & set(self.B_MASTER))
                #self.tree.node.active = True
                # Do something
                # ...


    def choose_action(self) :

        self.check_if_replay()

        min_depth = min(self.tree.active_depths)
        print('min depth = ', min_depth)
        candidates = self.tree.active_depths[min_depth]
        probs = [node.proba for node in candidates]
        print(probs)
        current = random.choices(candidates, weights=probs, k=1)[0]

        for depth in sorted(self.tree.active_depths) :
            if depth <= current.depth :
                continue

            children = [
                node for node in self.tree.active_depths[depth] 
                if node.active and node.parent == current
                ] # bon la ptet pas optimal
            
            if not children :
                break
            current = random.choice(children)
        x_t = current.sample()
        
        return x_t
    

    def update(self, x_t, y_t):
        self.tree.update_estimates(x_t, y_t)
        #if self.t >= 2 :
        print(f'active depths at {self.t} : ', self.active_depth_set())

        self.eviction_test()
        self.tree.update_proba()
        self.update_B_Master()

        self.t+=1
        self.check()

    def update_B_Master(self):
        B_MASTER = []
        intersection_nodes_index = set(self.tree.active_depths[self.m]) & set(self.B_MASTER)
        for node in intersection_nodes_index :
            node_in_tree = self.tree.find_node(node.depth, node.index)
            #B_MASTER.append(Diadic.Node(node.depth, node.index).clone_from(node_in_tree)) # la c'est ptet overkill mais je galere a faire marcher ca
            B_MASTER.append(node_in_tree)
        self.B_MASTER = B_MASTER

        #self.target_nodes = {Diadic.Node(n.depth, n.index) for n in intersection_nodes}
        #for new_node, original_node in zip(self.target_nodes, intersection_nodes):
        #    new_node.clone_from(original_node)
        #self.B_MASTER = list(self.target_nodes)

    def eviction_test(self):
        flag = False
        def treshhold(s1, s2, d):
            return self.c0 * math.log(self.T) * math.sqrt( (s2 - s1)*(2**d) ) + (4 * (s2 - s1) /2**d)
        
        def eviction_criteria(B1, B2, d) :
            """ Check if cumulative diff between B_1 and B_2 triggers positive test  """
            if d == self.m :
                s1 = self.starting_block
            else :
                s1 = self.t_start_replay
            n = self.t - s1
            
            if n <= 1:
                return False
            
            diff = [B1.mean_estimates[i] - B2.mean_estimates[i] for i in range(n)]
            #for s1 in range(n-1):
            #    cumsum = 0
            #    for s2 in range(s1+1, n):
            #        cumsum+= diff[s2]
            #        if cumsum > treshhold(s1, s2, d):
            #            print('true')
            #            return True

            cumsum = 0
            for s2 in range(s1+1, n):
                cumsum+= diff[s2]
                if cumsum > treshhold(s1, s2, d):
                    print('true')
                    return True

                    
        for d, active_nodes_d in self.tree.active_depths.items() :
            for B, B_prim in permutations(active_nodes_d, 2):
                if eviction_criteria(B, B_prim, d): 
                    print(f'{B_prim.index} evicted')
                    # then evict B
                    B_prim.evict()
                    flag = True
                    continue
        if flag :
            for d in self.tree.active_depths :
                active_nodes_d = self.tree.collect_active_nodes(d)
                self.tree.active_depths[d] = active_nodes_d


    def check(self):
        
        # If block ends because there is a shift
        if len(self.B_MASTER) == 0:
            print(f'Shift at {self.t}')
            self.initialize_episode()
            self.initialize_block()
            
        # If block ends naturally
        if self.t == self.ending_block : # if block ends naturally
            print(self.t)
            self.initialize_block()

    def visualize_replays(self):
        """
        Visualizes the active mask with active intervals and markers at actual replay start times.
        """
        m = self.active_mask.shape[1] + 2  # total number of depths including offset (depths 2 to m-1)
        T = self.active_mask.shape[0]
        
        plt.figure(figsize=(12, (m - 2) * 0.6))
        plt.imshow(self.active_mask.T, aspect='auto', cmap='YlOrBr', interpolation='nearest')
        plt.ylabel("Depth")
        plt.xlabel("Time")
        plt.title("Active Intervals Over Time by Depth")
        plt.yticks(
            ticks=range(m - 2),
            labels=[f"Depth {d}" for d in range(2, m)]
        )

        # Add red markers exactly where replays are triggered (from self.Replays)
        for d_index in range(m - 2):  # depth d = d_index + 2
            for t in range(T):
                if self.Replays[t, d_index] == 1:
                    plt.plot(t, d_index, marker='o', color='red', markersize=4)

        plt.tight_layout()
        plt.show()
