import time

START_ALPHA = float("-inf")
START_BETA = float("inf")

WEIGHT_WIN = 10 ** 10
# hard coded for board size of 11 and goal of 5+ in a row, but can change
BOARD_SIZE = 11

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
def evaluate_direction(board, range_start, range_end, iterator, weight_rows, weight_space):
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
            total += factor * weight_rows[count - 1] * weight_space[blanks_before + count + blanks_after - 5]
    return total

# returns values for player 0
def heuristic(state, weight_rows, weight_space):
    if state.is_terminal():
        return state.payoff() * WEIGHT_WIN
    total = 0
    board = state.board()
    for iterator, start, end in DIRECTIONS:
        total += evaluate_direction(board, start, end, iterator, weight_rows, weight_space)
    return total

def greedy(state, weight_rows, weight_space):
    factor = 1 if state.actor() == 0 else -1
    best_val = None
    best_action = None
    actions = state.get_actions()
    # random.shuffle(actions)
    for action in actions:
        val = heuristic(state.successor(action), weight_rows, weight_space)
        if best_val is None or val * factor > best_val:
            best_val = val * factor
            best_action = action
    return best_action


def greedy_policy(weight_rows, weight_space):
    return lambda pos : greedy(pos, weight_rows, weight_space)

def minimax(state, max_time, weight_rows, weight_space):
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
            val = alphabeta(succ, depth, START_ALPHA, START_BETA, weight_rows, weight_space)
            if best_val is None or val * factor > best_val:
                best_val = val * factor
                best_action = action
        depth += 1
    return best_action
            


def alphabeta(state, depth, alpha, beta, weight_rows, weight_space):
    if state.is_terminal():
        return WEIGHT_WIN * state.payoff()
    if depth == 0:
        return heuristic(state, weight_rows, weight_space)
    curr_actor = state.actor()
    if curr_actor == 0:
        a = START_ALPHA
        for action in state.get_actions():
            if alpha >= beta:
                break
            successor = state.successor(action)
            a = max(a, alphabeta(successor, depth - 1, alpha, beta, weight_rows, weight_space))
            alpha = max(alpha, a)
        return a
    else:
        b = START_BETA
        for action in state.get_actions():
            if alpha >= beta:
                break
            successor = state.successor(action)
            b = min(b, alphabeta(successor, depth - 1, alpha, beta, weight_rows, weight_space))
            beta = min(beta, b)
        return b


def minimax_policy(max_time, weight_rows, weight_space):
    return lambda pos : minimax(pos, max_time, weight_rows, weight_space)