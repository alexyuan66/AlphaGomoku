import random
import sys
import gomoku
import argparse
import time
import mcts_uct0
import mcts_uct2
import mcts_uct0_enhanced
import mcts_uct2_enhanced
import minimax
import minimax_genetic
import genetic


from gomoku import Gomoku

# random.seed(11)

class MCTSTestError(Exception):
    pass
        

def random_choice(position):
    moves = position.get_actions()
    return random.choice(moves)


def compare_policies(game, p1, p2, games, prob, time_limit_1, time_limit_2, print_final):
    p1_wins = 0
    p2_wins = 0
    p1_score = 0
    p1_time = 0.0
    p2_time = 0.0

    for i in range(games):
        # start with fresh copies of the policy functions
        p1_policy = p1()
        p2_policy = p2()
        position = game.initial_state()
        copy = position
        if print_final:
            print("---------------------NEW GAME---------------------")
        while not position.is_terminal():
            # DEBUG
            # print(position)
            if random.random() < prob:
                if position.actor() == i % 2:
                    start = time.time()
                    move = p1_policy(position)
                    p1_time = max(p1_time, time.time() - start)
                else:
                    start = time.time()
                    move = p2_policy(position)
                    p2_time = max(p2_time, time.time() - start)
            else:
                move = random_choice(position)
            position = position.successor(move)
        if print_final:
            print(position)

        #checking that minimax is working correctly by testing on pegging
        # and ensuring that MCTS never beats minimax with depth 14, which can search the entire
        # tree and so is optimal
        #while not copy.is_terminal():
        #    move = p2_policy(copy)
        #    copy = copy.successor(move)
        #if (i % 2 == 0 and position.payoff() > copy.payoff()) or (i % 2 == 1 and position.payoff() < copy.payoff()):
        #    print("COPY: " + str(copy))
            
        # to see final position, which for pegging includes the
        # complete sequence of cards played
        # print(position)

        p1_score += position.payoff() * (1 if i % 2 == 0 else -1)
        if position.payoff() == 0:
            p1_wins += 0.5
            p2_wins += 0.5
        elif (position.payoff() > 0 and i % 2 == 0) or (position.payoff() < 0 and i % 2 == 1):
            p1_wins += 1
        else:
            p2_wins += 1

    return p1_score / games, p1_wins / games


def test_game(game, count, p_random, p1_policy_fxn, p2_policy_fxn, time_limit_1, time_limit_2, print_final):
    ''' Tests a search policy through a series of complete games of Kalah.
        The test passes if the search wins at least the given percentage of
        games and calls its heuristic function at most the given proportion of times
        relative to Minimax.  Writes the winning percentage of the second
        policy to standard output.

        game -- a game
        count -- a positive integer
        p_random -- the probability of making a random move instead of the suggested move
        p1_policy_fxn -- a function that takes no arguments and returns
                         a function that takes a position and returns the
                       suggested move
        p2_policy_fxn -- a function that takes no arguments and returns
                         a function that takes a position and returns the
                       suggested move
                      
    '''
    margin, wins = compare_policies(game, p1_policy_fxn, p2_policy_fxn, count, 1.0 - p_random, time_limit_1, time_limit_2, print_final)

    print("NET: ", margin, "; WINS: ", wins, sep="")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TestGomoku script")
    parser.add_argument("game_count", type=int, help="Number of games to play")
    parser.add_argument("p1_time", type=float, nargs='?', help="Maximum time for p1 mode", default=float('inf'))
    parser.add_argument("p2_time", type=float, nargs='?', help="Maximum time for p2 mode", default=float('inf'))
    parser.add_argument("random_prob", type=float, help="Probability of random move")
    parser.add_argument("p1_mode", help="Player 1 mode", choices=['uct0', 'uct0_enhanced', 'uct2', 'uct2_enhanced', 'greedy', 'minimax', 'greedy_genetic', 'random'])
    parser.add_argument("p2_mode", help="Player 2 mode", choices=['uct0', 'uct0_enhanced', 'uct2', 'uct2_enhanced', 'greedy', 'minimax', 'greedy_genetic', 'random'])
    parser.add_argument("--print_final", action='store_true', help="Print final board")
    args = parser.parse_args()

    if args.p1_mode == 'uct0' or args.p1_mode == 'uct0_enhanced' or args.p1_mode == 'uct2' or args.p1_mode == 'uct2_enhanced' or args.p1_mode == 'greedy_genetic':
        if args.p1_time == float('inf'):
            raise ValueError("p1_time is required for p1 mode")

    if args.p2_mode == 'uct0' or args.p2_mode == 'uct0_enhanced' or args.p2_mode == 'uct2' or args.p2_mode == 'uct2_enhanced' or args.p2_mode == 'greedy_genetic':
        if (args.p1_mode == 'greedy' or args.p1_mode == 'minimax' or args.p1_mode == 'random') and args.p2_time == float('inf'):
            if args.p1_time == float('inf'):
                raise ValueError("p2_time is required for p2 mode")
            else:
                args.p2_time = args.p1_time
        elif args.p2_time == float('inf'):
            raise ValueError("p2_time is required for p2 mode")

    try: 
        count = args.game_count
        p1_time = args.p1_time
        p2_time = args.p2_time
        random_prob = args.random_prob
        policies = [None, None]
        for i in range(2):
            if i == 0:
                max_time = p1_time
                policy_name = args.p1_mode
            else:
                max_time = p2_time
                policy_name = args.p2_mode
            
            if policy_name == "random":
                policies[i] = lambda: random_choice
            elif policy_name == "uct0":
                policies[i] = lambda max_time=max_time: mcts_uct0.mcts_policy(max_time)
            elif policy_name == "uct0_enhanced":
                policies[i] = lambda max_time=max_time: mcts_uct0_enhanced.mcts_policy(max_time)
            elif policy_name == "uct2":
                policies[i] = lambda max_time=max_time: mcts_uct2.mcts_policy(max_time)
            elif policy_name == "uct2_enhanced":
                policies[i] = lambda max_time=max_time: mcts_uct2_enhanced.mcts_policy(max_time)
            elif policy_name == "greedy":
                policies[i] = lambda: minimax.greedy_policy()
            elif policy_name == "minimax":
                # 0.05 to always search to depth 2
                policies[i] = lambda: minimax.minimax_policy(0.05)
            elif policy_name == "greedy_genetic":
                weights = genetic.generate_weights(max_time)
                print(f"Best Weights: {weights}")
                policies[i] = lambda: minimax_genetic.greedy_policy(weights[0], weights[1])
            else:
                raise ValueError
        print_final = args.print_final
    except ValueError:
        print("Invalid argument type")
        sys.exit(1)
    try:
        game = Gomoku(11)
        test_game(game, count, random_prob, policies[0], policies[1], max_time, float("inf"), print_final)
        sys.exit(0)
    except MCTSTestError as err:
        print(sys.argv[0] + ":", str(err))
        sys.exit(1)
    
