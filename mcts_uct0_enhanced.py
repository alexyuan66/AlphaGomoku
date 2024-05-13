import time
import collections
import concurrent.futures
from math import log, sqrt
from random import choice

class Node:
    """Class for each node in the monte carlo tree"""
    
    def __init__(self, state, parent = None):
        """Contains variables for explicit tree"""

        # Fundamentals
        self.state = state
        self.value = 0
        self.visits = 0

        # Maps action to next state
        self.children = {}

        # For backpropogating
        self.parent = parent

        # For calculating UCB
        self.player = state.actor()

        # For getting leaf from UCB -> isFullyExpanded means all children in children
        self.isFullyExpanded = self.state.is_terminal()

        # Store actions
        self._actions = None

        # If the node is not terminal, get and cache the actions
        if not self.isFullyExpanded:
            self._actions = self.state.get_actions()
    
    @property
    def actions(self):
        # This property will return the cached actions
        return self._actions

    def calculate_avg(self):
        """For choosing the best action at the end"""

        return self.value / self.visits

    def __str__(self):
        """For debugging purposes"""

        return str(self.visits) + " | " +str(self.value) + " | " + str(self.isFullyExpanded)
    
class MonteCarloTree:
    """Class for explicit Monte Carlo Tree -> benefits = faster performance so more iterations"""

    def __init__(self, root, time_limit):
        """Initialize root of tree and time limit given as parameters"""
        self.root = root
        self.time_limit = time_limit
        self.amaf = collections.defaultdict(set)


    def calculate_action(self, node):
        """
        Gets the most promising action from the UCB 
        Player 0 -> Maximize and + for UCB calculation
        Player 1 -> Minimize and - for UCB calculation
        Assumptions: children already in tree -> always true because we only add children to node when it is in tree
        """

        # Use array because may be multiple actions with same UCB, want to generate random
        curr_best_state = []

        # Player 0 wants to maximize and player 1 wants to minimize
        curr_best_ucb = float('-inf') if node.player == 0 else float('inf')
        
        # Traverse children values that have been seen
        for next_state in node.children.values():
            if node.player == 0:
                ucb_val = (next_state.value / next_state.visits) + sqrt((2 * log(node.visits)) / next_state.visits)
            
                # Maximize
                if ucb_val == curr_best_ucb:
                    curr_best_state.append(next_state)
                elif ucb_val > curr_best_ucb:
                    curr_best_state = [next_state]
                    curr_best_ucb = ucb_val
            else:
                ucb_val = (next_state.value / next_state.visits) - sqrt((2 * log(node.visits)) / next_state.visits)
            
                # Minimize
                if ucb_val == curr_best_ucb:
                    curr_best_state.append(next_state)
                elif ucb_val < curr_best_ucb:
                    curr_best_state = [next_state]
                    curr_best_ucb = ucb_val

        # Return random state from best states
        return choice(curr_best_state)
    
    def get_leaf(self, node):
        """Gets a leaf to either rollout and expand from"""

        # Keep searching until you either hit a terminal or a node that hasn't been fully expanded
        while not node.state.is_terminal() and node.isFullyExpanded:
            # Calculate UCB and get best UCB
            next_node = self.calculate_action(node)

            # Traverse next node
            node = next_node
        
        # Return leaf node
        return node

    def expand(self, node):
        """
        Used for expanding out the chosen leaf node from the UCB selection
        """

        # Get all available actions
        legal_actions = node.actions

        # Get action that hasn't been seen before -> len of children already seen optimization
        # Inserts one child at a time rather than all at once because shows performance benefits for low time games
        index = len(node.children)
        child = Node(node.state.successor(legal_actions[index]), parent=node)
        node.children[legal_actions[index]] = child
        self.amaf[legal_actions[index]].add(child)

        # If searched all children then isFullyExpanded is true
        if len(node.children) == len(legal_actions):
            node.isFullyExpanded = True

        # Return nonsearched child
        return child

    def rollout(self, pos):
        """
        Used for rolling out to find a terminal value -> later to be propagated upwards
        """
        # Maintain a list of actions for AMAF
        amaf_actions = []

        # Keep searching until you hit a terminal
        while True:
            # Found terminal so return payoff value
            if pos.is_terminal():
                return pos.payoff(), amaf_actions
            
            # Choose random action from available action until terminal
            available_actions = pos.get_actions()
            action = choice(available_actions)

            # Check if action in tree, if so then add
            if action in self.amaf:
                amaf_actions.append(action)

            pos = pos.successor(action)
    
    def backpropagate(self, node, val, rollouts):
        """
        Used for propagating the value achieved form the terminal node back to root
        """

        # Keep going until you hit root
        while node:
            node.value += val
            node.visits += rollouts

            # Use stored parent nodes to propogate up
            node = node.parent

    def get_best_move(self):
        """
        Gets the most promising action from the average after time limit
        Player 0 -> Maximize
        Player 1 -> Minimize
        """

        # Use array because may be multiple actions with same UCB, want to generate random
        curr_best_move = []

        # Player 0 = maximize, player 1 = minimize
        curr_best_avg = float('-inf') if self.root.player == 0 else float('inf')
        
        # Traverse children that have been traversed -> none will have visit of 0 from our algorithm
        for action, next_state in self.root.children.items():
            # Calculate average
            avg_val = next_state.calculate_avg()

            # If equals, add to best moves list
            if avg_val == curr_best_avg:
                curr_best_move.append(action)
            elif self.root.player == 0 and avg_val > curr_best_avg:
                # Found a better maximizing action for player 0
                curr_best_move = [action]
                curr_best_avg = avg_val
            elif self.root.player == 1 and avg_val < curr_best_avg:
                # Found a better minimizing action for player 1
                curr_best_move = [action]
                curr_best_avg = avg_val

        # Return random action from best move list
        return choice(curr_best_move)
    
    def find_move(self):
        """Function that returns the best move from the MCTS"""

        # Calculate end time
        end_time = time.time() + self.time_limit

        # Traverse until that time is up
        while time.time() < end_time:
            # Traverse: Find leaf node
            current = self.get_leaf(self.root)
            
            # Expand: Only if leaf is not terminal
            if not current.state.is_terminal():
                current = self.expand(current)

            # Perform 4 parallel rollouts
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.rollout, current.state) for _ in range(4)]

                # Wait for all rollouts to complete and collect the results
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

            # Update AMAF nodes
            total_value = 0
            for value, amaf_actions in results:
                total_value += value
                for action in amaf_actions:
                    for node in self.amaf[action]:
                        node.value += value
                        node.visits += 1
            
            # Update: Propagate back up to the root and update values and visits
            self.backpropagate(current, total_value, rollouts=1)
        
        # After time is up, find move that gets best average
        return self.get_best_move()

def mcts_policy(time_limit):
    """
    Returns a function that takes a position and returns the move suggested by running MCTS for 
    # that amount of time starting with that position -> enhanced bt AMAF and leaf parallelism
    """

    def policy(position):
        """Returned function"""

        # First create Monte Carlo Tree with the root node
        mcts_obj = MonteCarloTree(Node(position), time_limit)

        # Return the best move form the MCTS
        return mcts_obj.find_move()
    
    # Return the policy function
    return policy