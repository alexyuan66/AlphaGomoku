import time
import math
import random
import collections
import concurrent.futures

# Weight for UCB heuristic
UCB_EXPLORE_WEIGHT = 2

def start_payoff(state, start_actor):
    """Maintains minimization or maximization for p0 or p1"""

    # If player 0
    if start_actor == 0:
        return state.payoff()
    
    # If player 1
    return -1 * state.payoff()

def explore_calculate(total_plays, state_plays):
    """Calculates explore term for UCB heuristic"""

    # Ensures no divide by 0
    if total_plays == 0 or state_plays == 0:
        return 0

    # Calculate and return explore term
    ans = math.sqrt(UCB_EXPLORE_WEIGHT * math.log(total_plays) / state_plays)
    return ans
    

def ucb(parent, succ, total_plays, states, actor, start_actor):
    """Calculates the UCB heuristic for a node"""

    # Check if it hasn't been visited from the parent node before
    if succ not in states or states[parent][2][succ] == 0 or states[succ][1] == 0 or total_plays == 0:
        return None
    
    # If terminal then return some payoff
    if succ.is_terminal():
        return 10 * start_payoff(succ, start_actor)
    
    # Otherwise, calculate exploit and explore terms for UCB heuristics
    exploit = (states[succ][0] / states[succ][1])
    explore = explore_calculate(total_plays, states[parent][2][succ])

    # Get respective heuristic given p0 or p1
    if actor == start_actor:
        return explore + exploit
    else:
        return explore - exploit
    

def max_ucb(state, states, start_actor):
    """Gets the best node from UCB heuristics"""

    # Initialize best values
    best_ucb = None
    best_succ = None
    best_action = None
    actor = state.actor()
    total_plays = states[state][1]

    # Go through actions and get UCB value
    for action in state.get_actions():
        succ = state.successor(action)

        # Get UCB value
        ucb_val = ucb(state, succ, total_plays, states, actor, start_actor)

        # If none, then hasn't been visited to select this
        if ucb_val is None:
            return action, succ, None

        # Otherwise, has been traversed
        if best_ucb is None or ucb_val > best_ucb:
            best_ucb = ucb_val
            best_succ = succ
            best_action = action
    
    # Return best node and action for AMAF
    return best_action, best_succ, best_ucb

def create_entry(state, states):
    """Function to create an entry in the implicit tree"""

    # Check if not already inside implicit tree
    if state not in states:

        # Store value, visits, and map from succesive action to visits (maintains edges)
        states[state] = [0, 0, dict()]
        for action in state.get_actions():
            states[state][2][state.successor(action)] = 0

def simulate(state, start_actor, amaf_memo):
    """Gets the terminal value from random actions"""

    # Maintain a list of actions for AMAF
    amaf_actions = []

    # Continue until terminal
    while not state.is_terminal():
        # Choose random action
        action = random.choice(state.get_actions())
        state = state.successor(action)

        # Check if action in existing tree, if so then add value
        if action in amaf_memo:
            amaf_actions.append(action)

    # Return terminal value
    return start_payoff(state, start_actor), amaf_actions

def visit(start_state, states, start_actor, amaf_memo):
    """Traverses the Monte Carlo Tree to gather statistics"""

    # Traverse from current state
    curr_state = start_state

    # Record previous states for backpropagation
    past_states = [start_state]

    # Expand: Only if leaf is not terminal -> using ucb heuristic
    while not curr_state.is_terminal():
        action, curr_state, action_ucb = max_ucb(curr_state, states, start_actor)
        if action_ucb is None:
            break
        past_states.append(curr_state)
    
    # Rollout/simulate: Play from leaf node
    if curr_state.is_terminal():
        # Terminal state
        value = start_payoff(curr_state, start_actor)
    else:
        # Add move to amaf memo
        amaf_memo[action].add(curr_state)

        # Child without an entry yet
        create_entry(curr_state, states)
        past_states.append(curr_state)

        # Perform 4 parallel rollouts
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(simulate, curr_state, start_actor, amaf_memo) for _ in range(4)]

            # Wait for all rollouts to complete and collect the results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Update AMAF nodes
        value = 0
        for v, amaf_actions in results:
            value += v
            for action in amaf_actions:
                for stat in amaf_memo[action]:
                    states[stat][0] += v
                    states[stat][1] += 1

    # Backpropagate
    for i in range(len(past_states)):
        # Add visit count along edges for UCT2 implementation
        if i > 0:
            states[past_states[i - 1]][2][past_states[i]] += 1
        
        # Add visit and value to node to calculate average value
        states[past_states[i]][0] += value
        states[past_states[i]][1] += 1

def mcts(start_state, time_limit, states, amaf_memo):
    """Returns the best move from MCTS"""

    # Get start time
    start_time = time.process_time()

    # Create root entry and player
    create_entry(start_state, states)
    start_actor = start_state.actor()

    # While time limit is not up continue to traverse
    while (time.process_time() - start_time < time_limit):
        visit(start_state, states, start_actor, amaf_memo)
    
    # Get best move from MCTS statistics
    ans = None
    ans_val = None

    # Traverse actions
    for action in start_state.get_actions():

        # Get statistics
        succ = start_state.successor(action)
        if succ.is_terminal():
            # If terminal then return value
            val = start_payoff(succ, start_actor)

        elif succ in states and states[succ][1] != 0:
            # If visited then return average value
            val = states[succ][0] / states[succ][1]
        else:
            val = None
        
        # Check if statistics are better than prior moves
        if ans_val is None or (val is not None and val > ans_val):
            ans_val = val
            ans = action
    
    # Return best action
    return ans


def mcts_policy(time_limit):
    """Returned mcts policy"""

    # Dictionary to store implicit tree
    states_memo = dict()

    # Dictionary to store moves for all moves as first
    amaf_memo = collections.defaultdict(set)

    # Return best move from MCTS UCT2 implementation with AMAF and leaf parallelization
    return lambda pos : mcts(pos, time_limit, states_memo, amaf_memo)