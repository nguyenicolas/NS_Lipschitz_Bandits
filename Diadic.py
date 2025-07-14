import numpy as np
from graphviz import Digraph
import random
import copy

class Node:
    def __init__(self, depth, index, parent=None) -> None:
        self.depth = depth
        self.index = index
        self.parent = parent
        self.left = None
        self.right = None
        self.active = False
        self.mean_estimates = []
        self.proba = 0.0
        self.size = 1/(2**(self.depth-1))

    def __eq__(self, other: object) -> bool:
        return self.depth == other.depth and self.index == other.index
    
    def __hash__(self) -> int:
        return hash((self.depth, self.index))

    def subdivide(self):
        self.left = Node(self.depth + 1, 2 * self.index - 1, parent=self)
        self.right = Node(self.depth + 1, 2 * self.index, parent=self)

    def evict(self):
        self.active = False
        if self.left :
            self.left.evict()
        if self.right:
            self.right.evict()

    def contains(self, x):
        left = (self.index-1) / (2**(self.depth-1))
        right = (self.index) / (2 ** (self.depth-1))
        return left <= x <= right
    
    def get_parent(self, target_depth):
        """ 
        Enumerate the list of children of a given node
        """
        if target_depth >= self.depth :
            return None
        current = self
        while current and current.depth > target_depth:
            current = current.parent
        return current

    def get_descendants_at_depth(self, target_depth):
        if target_depth <= self.depth :
            return []
        result = []

        def helper(node):
            if node is None :
                return
            if node.depth == target_depth :
                result.append(node)
            elif node.depth < target_depth :
                helper(node.left)
                helper(node.right)
        helper(self)
        return result
    
    def get_active_children(self, target_depth):
        if target_depth <= self.depth :
            return []
        
        result = []

        def dfs(node):
            if node is None :
                return
            if node.depth == target_depth and node.active :
                result.append(node)
            elif node.depth < target_depth :
                dfs(node.left)
                dfs(node.right)
                
        dfs(self)
        return result

    def sample(self):
        """ Sample one action (i.e. number) in node """
        left = (self.index-1) / (2**(self.depth-1))
        right = (self.index) / (2 ** (self.depth-1))
        return random.uniform(left, right)


    def __repr__(self) -> str:
        return f"({self.depth},{self.index})"

class Tree:

    def __init__(self, max_depth) -> None:
        self.max_depth = max_depth
        self.root = Node(1, 1)
        
        if max_depth is not None:
            self.build_full_tree(node=self.root, max_depth=max_depth)

        self.initialize_active_depth() 

    def build_full_tree(self, node, max_depth):
        """ Build tree of size max_depth recursively with node as root"""
        if node.depth >= max_depth:
            return
        node.subdivide()
        self.build_full_tree(node.left, max_depth)
        self.build_full_tree(node.right, max_depth)

    def initialize_active_depth(self):
        self.active_depths = {}
        active_nodes = self.collect_active_nodes(self.max_depth)
        self.active_depths[self.max_depth] = active_nodes
        self.activate_depth(self.max_depth)
        
    def clone_node(self, node):
        clone = Node(node.depth, node.index)
        clone.__dict__ = copy.deepcopy(node.__dict__)
        return clone
    
    def activate_depth(self, depth):
        """ 
        Make depth active, i.e. activate all its nodes and add this depth to the set of active depths
        Note that activating one node also actives all its children
        """
        
        # Activate nodes
        def activate_at_depth(node):
            if node is None :
                return
            if node.depth == depth :
                node.active = True
                #activate_children(node.left)
                #activate_children(node.right)
                
            elif node.depth < depth :
                activate_at_depth(node.left)
                activate_at_depth(node.right)

        activate_at_depth(self.root)
        nodes = self.collect_active_nodes(depth)
        self.active_depths[depth] = nodes
        self.update_proba() # en vrai jpense a chaque fois qu'on active un depth faut re-update les pb
    
    def de_activate_depth(self, depth):
        """ 
        Make depth not active, i.e. de-activate all its nodes and add this depth to the set of active depths
        """
        
        # Activate nodes
        def de_activate_at_depth(node):
            if node is None :
                return
            if node.depth == depth :
                node.active = False
                
            elif node.depth < depth :
                de_activate_at_depth(node.left)
                de_activate_at_depth(node.right)

        de_activate_at_depth(self.root)
        if depth in self.active_depths:
            del self.active_depths[depth]
        self.update_proba() # en vrai jpense a chaque fois qu'on active un depth faut re-update les pb
        
    def get_all_nodes_at_depth(self, depth):
        """
        Return a list of all nodes at the given depth (active or not).
        """
        result = []

        def dfs(node):
            if node is None:
                return
            if node.depth == depth:
                result.append(node)
            elif node.depth < depth:
                dfs(node.left)
                dfs(node.right)

        dfs(self.root)
        return result
    
    def find_node(self, depth, index):
        node = self.root
        for d in range(1, depth):
            if node is None :
                return None
            bit = ((index - 1) >> (depth - d - 1)) & 1
            node = node.right if bit else node.left
        return node if node and node.depth == depth and node.index == index else None
        
    def collect_active_nodes(self, depth):
        def helper_collect_active_nodes(node, depth):
        # base
            if node is None:
                return []
            if node.depth == depth and node.active :
                return [node]
            return helper_collect_active_nodes(node.left, depth) + helper_collect_active_nodes(node.right, depth) 
    
        return helper_collect_active_nodes(self.root, depth)
    
    def update_proba(self):
        """
        Update each node's probability according to the hierarchical sampling scheme.
        Only active nodes at active depths can receive nonzero probabilities.
        """
        # Reset all probabilities to 0
        def reset_probs(node):
            if node is None:
                return
            node.proba = 0.0
            reset_probs(node.left)
            reset_probs(node.right)

        reset_probs(self.root)

        if not self.active_depths:
            return

        active_depths = sorted(self.active_depths.keys())
        min_depth = active_depths[0]
        min_depth_nodes = [n for n in self.active_depths[min_depth] if n.active]

        if not min_depth_nodes:
            return

        base_prob = 1.0 / len(min_depth_nodes)

        # Recursively assign probability only to active children at active depths
        def recurse(node, depth_idx, prob):
            if depth_idx >= len(active_depths):
                return

            current_depth = active_depths[depth_idx]

            if node.depth != current_depth or not node.active:
                return

            node.proba = prob

            # Recurse to next active depth
            if depth_idx + 1 >= len(active_depths):
                return

            next_depth = active_depths[depth_idx + 1]

            # Get all active children of `node` at the next active depth
            children_at_next = node.get_descendants_at_depth(next_depth)
            active_children = [child for child in children_at_next if child.active]

            if not active_children:
                return

            child_prob = prob / len(active_children)

            for child in active_children:
                recurse(child, depth_idx + 1, child_prob)

        for node in min_depth_nodes:
            recurse(node, 0, base_prob)

        
    def update_estimates(self, x_t, y_t):

        def update_node(node):
            if node is None:
                return      
            
            if not node.contains(x_t) or not node.active:
                node.mean_estimates.append(0.0)
            
            else :
                p = node.proba
                mean_estimate_t = (y_t/p) 
                node.mean_estimates.append(mean_estimate_t)

            update_node(node.left)
            update_node(node.right)
            
        update_node(self.root)

    def set_to_list(self, list_nodes):
        for node in list_nodes :
            current_node = self.find_node(node.depth, node.index)
            current_node.__dict__.update(node.__dict__)

    def visualize(self, t=None):
            filename=f"Trees/dyadic_tree{t}"
            
            dot = Digraph()

            def add_nodes_edges(node):
                if node is None:
                    return
                # Color node based on activity
                color = "green" if node.active else "red"
                label = f"p={np.round(node.proba, 3)}, {np.round(np.mean(node.mean_estimates), 3) if node.mean_estimates != [] else 0.0}"
                dot.node(name=str(id(node)), label=label, style="filled", fillcolor=color)

                for child in [node.left, node.right]:
                    if child:
                        dot.edge(str(id(node)), str(id(child)))
                        add_nodes_edges(child)

            add_nodes_edges(self.root)
            dot.render(filename, format="png", cleanup=True)
            #print(f"Tree image saved as {filename}.png")