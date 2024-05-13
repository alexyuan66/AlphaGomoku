---------------------------------------------------------------------------
MEMBERS: Aaron Yu (amy23) and Alex Yuan (amy24)

---------------------------------------------------------------------------
Required packages: concurrent.futures, argparse

---------------------------------------------------------------------------
To run TestGomoku:
./TestGomoku game_count [p1_time] [p2_time] random_prob p1_mode p2_mode [--print_final]

game_count: Number of games to play.
p1_time: Time to train p1_mode in seconds -> only needed if p1_mode is uct0, uct0_enhanced, uct2, uct2_enhanced, greedy_genetic
p2_time: Time to train p2_mode in seconds -> only needed if p2_mode is uct0, uct0_enhanced, uct2, uct2_enhanced, greedy_genetic
random_prob: Probability of either player making a random move on a turn.
p1_mode, p2_mode: One of the following:
    random: Random player.
    uct0: MCTS player with UCT0 MCTS implementation.
    uct0_enhanced: MCTS player with UCT0 MCTS implementation and AMAF + Leaf Parallelism enhancements.
    uct2: MCTS player with UCT2 MCTS implementation.
    uct2_enhanced: MCTS player with UCT2 MCTS implementation and AMAF + Leaf Parallelism enhancements.
    greedy: Greedy player (minimax depth 1) that moves to maximize a heuristic.
    minimax: Minimax player that uses iterative deepening, alpha-beta pruning, and a heuristic to choose the best move. Always searches to a depth of at least 2.
    greedy_genetic: Greedy player with heuristics tuned by genetic algorithm, trained for the amount of time passed in (1200 seconds of training is around 4 generations, but genetic.py already starts with the results of 30 generations of training).
--print_final: If --print_final is provided as a final argument, the final board state of each game tested will be printed. Otherwise, nothing will be printed.

Examples:
./TestGomoku 1000 0.1 greedy random
./TestGomoku 100 1.0 0.1 uct0 random
./TestGomoku 100 1.0 0.1 random uct0 --print_final
./TestGomoku 10 1200 5 0.1 greedy_genetic uct0_enhanced --print_final

***Note that some agents may run fairly slowly. MCTS generally needs at least one second per move, preferably more, to visit enough states to avoid making random moves too much of the time. Minimax always searches at a depth of at least 2, so it may take up to 15 seconds per turn. Gomoku games can also be up to 121 moves long, which can also extend the time needed to run tests. Also, training the genetic algorithm takes around 5 minutes per generation, in addition to another 5 minutes to find the best heuristic in the last generation, on top of the time taken to actually test the resulting agent. As a result, training will most likely take at least 10 minutes + the time needed to test the number of games passed in. ***

---------------------------------------------------------------------------
GAME DESCRIPTION
Gomoku (5-in-a-row) is a 2 player game with perfect information, played with the same board and pieces as Go. The game is played on a 15 by 15 grid, where one player plays the black stones, and one player plays the white stones, with black always playing first. In each turn, a player places a stone of their color on any of the empty intersections on the board, and unlike in Go, stones are never removed from the board once they are placed. The goal of the game is for a player to have 5 or more stones of their color in a row, either horizontally, vertically, or diagonally. If the board is filled and neither player has won, the game ends in a draw.

---------------------------------------------------------------------------
CODE DESCRIPTION
We have implemented several files for our game and agents. Our code implements the game Gomoku as well as several agents for playing the game: random, MCTS (with and without AMAF, leaf parallelism, and UCT enhancements), minimax, greedy, and a genetically tuned greedy algorithm. The rundown of the files that correlate to each part of our implementation are as follows: 

Game Implementation: game.py, gomoku.py
Agents: genetic.py, mcts_uct0.py, mcts_uct0_enhanced.py, mcts_uct2.py, mcts_uct2_enhanced.py, minimax_genetic.py, minimax.py
Test Driver: test_gomoku.py

---------------------------------------------------------------------------
REPORT
The first algorithms we wrote were the MCTS algorithm, the greedy heuristic algorithm, and the random algorithm. Initially, the MCTS algorithm (UCT0) performed better than the random algorithm, winning 77.8% of its games when given 0.05 seconds per move over 10000 games. The greedy agent consistently outperformed the random agent, winning roughly 99.85% of its 10000 games. When comparing the greedy agent and the MCTS agent, the greedy agent won around 90% of its 1000 games when testing the MCTS agent with 0.5 seconds per move, and the greedy agent won 80% over 1000 games when giving MCTS 1 second per move. The reason the greedy agent performed better is most likely because the greedy agent's heuristic favored building longer chains of pieces and took winning moves whenever it had the opportunity, meaning it played with better short-term strategy, while the large state space of gomoku prevented the MCTS agent from finding a strategy that would perform well in the long term.

Then, we improved the MCTS algorithm by using All Moves as First, leaf parallelism, and experimentation with UCT2. We experimented with combinations of these against the random agent. The results are as follows. The agent using UCT2 won roughly 92.8% of its 10000 games when given 0.05 seconds per move. The agent using AMAF, parallelism, and UCT0 won 98.1% of its 10000 games, and the agent with AMAF, parallelism, and UCT2 won 98% of its 10000 games. Given these results, the enhanced MCTS agents performed better than the baseline MCTS agent. However, although the enhanced MCTS agents performed better than the baseline MCTS agent against the random agent, the enhanced agents won less than 50% of their games when playing 1000 games against the baseline MCTS agent (with 1 second each). However, as the amount of time given to both MCTS agents was decreased, the enhanced agent's winrate began increasing. Over 1000 games each, we obtained the following winrates for the baseline UCT0 agent against the enhanced UCT0 agent when giving each agent the following times:

* 0.01s: 55.1% (close to 50% since both agents are close to random)
* 0.05s: 24.7%
* 0.1s: 44.7%
* 0.25s: 70.6%

Similar results were obtained when playing the baseline UCT0 agent against the UCT2 agent and the enhanced UCT2 agent. This is likely because the large state space of gomoku means that the number of branches of the game tree explored has a larger impact on the performance of MCTS than being able to better estimate the value of the states visited, which was further evidenced by later testing against minimax agents. The improved performance of the enhanced agent likely resulted from making better moves in states that it had the chance to visit because of its higher-quality estimates of each action's value. Therefore, since the added enhancements increased the time needed to play a single episode, they likely resulted in fewer possible states being explored compared to the baseline MCTS.

Next, we wrote a minimax agent using iterative deepening, alpha-beta pruning, and the same heuristic used for the greedy agent. Although this algorithm ran much more slowly, it performed better than all of the other agents used, winning 100% of its games against the random agent, 94.2% of games against the enhanced MCTS agent with 5 seconds per move, 96% of games against the enhanced MCTS agent with 20 seconds per move, and 87% of games against the greedy agent, where each agent played 1000 games with the minimax agent, except the 20-second MCTS agent, which only played 100 games.

Lastly, we used a genetic algorithm to fine-tune the parameters of the heuristic used by the greedy and minimax agents. Because of the slow runtime of the minimax agent, the algorithm was trained and tested on the greedy agent, although the same algorithm could be applied to the minimax agent, if given more time. Over 100 games, the greedy agent initially won only 48% of its games against the enhanced MCTS agent when the MCTS agent was given 5 seconds per move. Then, the greedy agent was trained against the MCTS agent with 1 second per move over 30 generations of 8 heuristics each. The performance began improving across generations, and after testing against the MCTS agent with 5 seconds per move, the greedy agent was able to win 84% out of 100 games. When running the agent with the greedy_genetic argument, the starting population can either be randomly generated by leaving population = [] in line 80 of genetic.py, or the previous generation can be used as a starting population by setting population equal to the previous generation in list form. However, the genetic algorithm worsened the greedy agent's performance against the MCTS agent with no enhancements, and using the heuristic generated by it for the minimax agent worsened its performance against all agents. This was most likely because the genetic algorithm was only trained to optimize the greedy agent's performance against the enhanced MCTS agent, so the same heuristic did not generalize well to other algorithms or opponents. Despite this, using the same genetic algorithm, the greedy or minimax agents could still be trained against other opponents in the same way.

***Note that for all tests, every agent was tested with a 0.1 probability of making a random move.***

---------------------------------------------------------------------------
CONCLUSION
This project aimed to answer several questions about using computational intelligence to play gomoku. First, we wanted to know how tree-search and heuristic-based algorithms would perform relative to each other and compared to a random agent. We initially thought that heuristic-based algorithms would perform poorly, but because of the large state space, Monte Carlo methods did not initially perform as well as we expected against the greedy agent, although both performed well against the random agent. Second, we wanted to find out how MCTS enhancements and minimax algorithms would improve the performance of the MCTS and heuristic-based methods. We found that leaf parallelism improved the performance of MCTS against random algorithms, but caused it to win less than 50% of its games against the MCTS agent with no enhancements when each agent was given 1 second per move. This most likely indicates that parallelism provides a noticeably better estimate of the values of game states because of the large number of possible sequences of moves that could follow each state, but the additional overhead from AMAF and parallelism resulted in the enhanced MCTS running less iterations than the MCTS agent with no enhancements, causing the enhanced MCTS to perform worse when playing against the baseline MCTS agent. This appears to indicate that because of the high likelihood of reaching states that were not visited in MCTS exploration, the number of iterations run in gomoku for MCTS algorithms is a very important factor in performance, which was also evidenced by the improvements in MCTS performance when giving more time to the algorithms. However, when decreasing the time given to both agents, the number of states visited by both agents was decreased, causing the enhanced agent to perform better because it was able to make make better estimates of the value of each possible action it explored. Third, we wanted to determine how much a minimax agent could be improved by using a genetic algorithm. Because the minimax agent with depth 2 was already winning most of its games and ran too slowly to train in a reasonable amount of time, we chose to use the greedy agent with depth 1 to test the genetic algorithm, since it initially won less than 50% of its games against the MCTS agent. The resulting optimized heuristic had much better performance against the MCTS agent, but worse performance against other opponents and when used for minimax with depth 2, most likely since the genetic algorithm overfit to play against a single opponent, and was not designed to generalize.
