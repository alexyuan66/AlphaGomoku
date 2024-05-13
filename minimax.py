import time

START_ALPHA = float("-inf")
START_BETA = float("inf")

WEIGHT_WIN = 10 ** 10
# weights for 1, 2, 3, or 4 in a row
WEIGHT_ROWS = [1, 10, 100, 1000] + [WEIGHT_WIN for i in range(7)]
# weights for room to make 5, 6, 7, 8, 9, 10, or 11
WEIGHT_SPACE = [(10 + i) / 10 for i in range(7)]
# hard coded for board size of 11 and goal of 5+ in a row, but can change
BOARD_SIZE = 11

# WEIGHT_ROWS, WEIGHT_SPACE = [[1, 5.380144337410294, 46.81350414753168, 3928.1614264827967], [1, 1.6266102619854206, 2.0229625144202052, 2.9902217791604953, 2.3881491832469295, 1.5511375752385905, 1.964499139404]]

# iterators to give every row, column, or diagonal that can be used as a straight line to make a win
def iterate_horiz(board, row):
    for i in range(BOARD_SIZE):
        yield board[row][i]

def iterate_vert(board, col):
    for i in range(BOARD_SIZE):
        yield board[i][col]

def iterate_diag_down(board, y_start):
    x_curr, y_curr = 0, y_start
    while x_curr < BOARD_SIZE and y_curr < BOARD_SIZE:
        yield board[x_curr][y_curr]
        x_curr += 1
        y_curr += 1

def iterate_diag_up(board, y_start):
    x_curr, y_curr = 0, y_start
    while x_curr < BOARD_SIZE and y_curr >= 0:
        yield board[x_curr][y_curr]
        x_curr += 1
        y_curr -= 1

def iterate_diag_left(board, y_start):
    x_curr, y_curr = BOARD_SIZE - 1, y_start
    while x_curr >= 0 and y_curr >= 0:
        yield board[x_curr][y_curr]
        x_curr -= 1
        y_curr -= 1

def iterate_diag_right(board, y_start):
    x_curr, y_curr = BOARD_SIZE - 1, y_start
    while x_curr >= 0 and y_curr < BOARD_SIZE:
        yield board[x_curr][y_curr]
        x_curr -= 1
        y_curr += 1

DIRECTIONS = [[iterate_horiz, 0, BOARD_SIZE], [iterate_vert, 0, BOARD_SIZE], 
              [iterate_diag_down, 0, BOARD_SIZE - 4], [iterate_diag_up, 4, BOARD_SIZE], 
              [iterate_diag_left, 4, BOARD_SIZE - 1], [iterate_diag_right, 1, BOARD_SIZE - 4]]

# returns value for player 0
def evaluate_direction(board, range_start, range_end, iterator):
    total = 0
    for i in range(range_start, range_end):
        prev = [[-1, 0, 0, 0]]
        for space in iterator(board, i):
            if space == -1:
                prev[-1][3] += 1
            elif space == prev[-1][0] and prev[-1][3] == 0:
                prev[-1][2] += 1
            else:
                prev.append([space, prev[-1][3], 1, 0])
        for player, blanks_before, count, blanks_after in prev:
            if player == -1 or blanks_before + count + blanks_after < 5:
                continue
            factor = 1 if player == 0 else -1
            total += factor * WEIGHT_ROWS[count - 1] * WEIGHT_SPACE[blanks_before + count + blanks_after - 5]
    return total

# returns estimate of state value for player 0
def heuristic(state):
    if state.is_terminal():
        return state.payoff() * WEIGHT_WIN
    total = 0
    board = state.board()
    for iterator, start, end in DIRECTIONS:
        total += evaluate_direction(board, start, end, iterator)
    return total

# chooses action that leads to the state with greatest heuristic value
def greedy(state):
    factor = 1 if state.actor() == 0 else -1
    best_val = None
    best_action = None
    actions = state.get_actions()
    # random.shuffle(actions)
    for action in actions:
        val = heuristic(state.successor(action))
        if best_val is None or val * factor > best_val:
            best_val = val * factor
            best_action = action
    return best_action


def greedy_policy():
    return lambda pos : greedy(pos)
    
# uses iterative deepning and alpha beta pruning to find the best move, depth is always at least 2 and increases if time allows
def minimax(state, max_time):
    factor = 1 if state.actor() == 0 else -1
    depth = 1
    start_time = time.time()
    while (time.time() - start_time < max_time):
        best_action = None
        best_val = None
        actions = state.get_actions()
        # random.shuffle(actions)
        for action in actions:
            succ = state.successor(action)
            val = alphabeta(succ, depth, START_ALPHA, START_BETA)
            if best_val is None or val * factor > best_val:
                best_val = val * factor
                best_action = action
        depth += 1
    return best_action
            
def alphabeta(state, depth, alpha, beta):
    if state.is_terminal():
        return WEIGHT_WIN * state.payoff()
    if depth == 0:
        return heuristic(state)
    curr_actor = state.actor()
    if curr_actor == 0:
        a = START_ALPHA
        for action in state.get_actions():
            if alpha >= beta:
                break
            successor = state.successor(action)
            a = max(a, alphabeta(successor, depth - 1, alpha, beta))
            alpha = max(alpha, a)
        return a
    else:
        b = START_BETA
        for action in state.get_actions():
            if alpha >= beta:
                break
            successor = state.successor(action)
            b = min(b, alphabeta(successor, depth - 1, alpha, beta))
            beta = min(beta, b)
        return b


def minimax_policy(max_time):
    return lambda pos : minimax(pos, max_time)