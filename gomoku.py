from game import Game, State
import copy

class Gomoku(Game):
    def __init__(self, board_size):
        ''' Creates a board_size x board_size board for gomoku
        '''
        if board_size < 0:
            raise ValueError('Board size must be positive: %d' % board_size)
        
        self.board_size = board_size
        # -1 for empty spaces, 0/1 for each player's pieces
        self.board = [[-1 for i in range(board_size)] for i in range(board_size)]
        self.board[board_size // 2][board_size // 2] = 0
        self.stones_played = 1
            
    def initial_state(self):
        ''' Creates the initial state for this board.
        '''
        return Gomoku.State(self.board, 1, self.stones_played)

    
    class State(State):
        def __init__(self, board, turn, stones_played):
            if board is None:
                raise ValueError('board cannot be None')
            if turn != 0 and turn != 1:
                raise ValueError('invalid turn %d' % turn)
            
            self._board = copy.deepcopy(board)
            self._stones_played = stones_played
            self._turn = turn
            self._board_size = len(board)
            self.winner = None

            self._compute_hash()

            
        def is_initial(self):
            ''' Determines if this state is the initial state.
            '''
            return self._stones_played == 1

            
        def actor(self):
            ''' Returns the index of the player who makes the next move from
                this state.  The index will be 0 or 1.
            '''
            return self._turn
        
        def neighbors(self, x, y):
            if x > 0 and y > 0:
                yield x - 1, y - 1
            if x > 0 and y < self._board_size - 1:
                yield x - 1, y + 1
            if x < self._board_size - 1 and y > 0:
                yield x + 1, y - 1
            if x < self._board_size - 1 and y < self._board_size - 1:
                yield x + 1, y + 1
            if x > 0:
                yield x - 1, y
            if x < self._board_size - 1:
                yield x + 1, y
            if y > 0:
                yield x, y - 1
            if y < self._board_size - 1:
                yield x, y + 1

        
        def is_legal(self, action):
            '''
            Determines if a given move is legal, given an x, y tuple of an action
            '''
            x, y = action
            if x < 0 or y < 0 or x >= self._board_size or y >= self._board_size:
                raise ValueError('Move out of bounds: ', str(action))
            if self._board[x][y] != -1:
                return False
            # only allow moves adjacent or diagonal to already placed stones
            has_neighbor = False
            for n_x, n_y in self.neighbors(x, y):
                # print(action, n_x, n_y)
                if self._board[n_x][n_y] != -1:
                    has_neighbor = True
                    break
            return has_neighbor

        
        def get_actions(self):
            ''' Returns a list of legal moves from this state.
                The list of moves is given as a list of pits to sow from.
                Pits are indexed clockwise starting with 0 for player 0's
                first pit.
            '''
            moves = []
            for x in range(self._board_size):
                for y in range(self._board_size):
                    tup = (x, y)
                    if self.is_legal(tup):
                        moves.append(tup)
            return moves

        def make_move(self, action, player):
            x, y = action
            self._board[x][y] = player

        def check_win(self, last_action):
            x_start, y_start = last_action
            player = self._board[x_start][y_start]
            for x_diff, y_diff in ((1, 1), (1, 0), (0, 1), (1, -1)):
                total = 1
                for direction in (-1, 1):
                    x_curr, y_curr = x_start, y_start
                    for i in range(self._board_size):
                        x_curr += x_diff * direction
                        y_curr += y_diff * direction
                        if x_curr >= 0 and y_curr >= 0 and x_curr < self._board_size and y_curr < self._board_size and self._board[x_curr][y_curr] == player:
                            total += 1
                        else:
                            break
                if total >= 5:
                    return True
            return False



            
        def successor(self, action):
            ''' Returns the state that results from the given action
            '''
            if not self.is_legal(action):
                raise ValueError('Illegal move: ', str(action))

            succ = Gomoku.State(self._board, 1 - self._turn, self._stones_played + 1)
            succ.make_move(action, self._turn)

            # if game is over, update winner
            if succ.check_win(action):
                succ.winner = self._turn
            elif not succ.get_actions():
                succ.winner = 2

            return succ

        
        def is_terminal(self):
            ''' Determines if this state is terminal -- whether the game is over having
                reached this state.
            '''
            return self.winner is not None

        
        def payoff(self):
            ''' Returns the payoff to player 0 at this state: 1 for a win, 0 for a draw, -1 for
                a loss.  The return value is None if this state is not terminal.
            '''
            if not self.is_terminal():
                return None
            elif self.winner == 2:
                return 0
            else:
                return 1 if self.winner == 0 else -1
            
        def __str__(self) -> str:
            ans = f"{self._turn}, {self._stones_played}\n"
            for row in self._board:
                row_str = []
                for space in row:
                    if space == -1:
                        row_str.append("\033[1;30mE")
                    elif space == 0:
                        row_str.append("\033[0;34m0")
                    else:
                        row_str.append("\033[1;33m1")
                row_str.append("\033[0m")
                ans += "".join(row_str) + "\n"
            return ans
        
        def board(self):
            return self._board

        
        def _compute_hash(self):
            self.hash = hash(str(self._board)) * 2 + self._turn
            
        def __hash__(self):
            return self.hash
        
        def __eq__(self, other):
            return isinstance(other, self.__class__) and self._turn == other._turn and self._board == other._board


if __name__ == '__main__':
    board = Gomoku(11)
    pos = board.initial_state()
    print(pos._board)
    print(pos.get_actions())
    pos = pos.successor((5, 6))
    pos = pos.successor((6, 7))
    print(pos._board)
    for i in pos.get_actions():
        print(i)

    
