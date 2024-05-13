import random
import time
import sys
from test_gomoku import compare_policies
from gomoku import Gomoku
import minimax_genetic
import mcts_uct0_enhanced

POP_SIZE = 8
MUTATION_PROB = 0.1
CROSSOVER_POINTS = []
for i in range(10):
    for j in range(i + 1, 10):
        CROSSOVER_POINTS.append((i, j))
def generate_weights(time_limit):
    game = Gomoku(11)

    num_gens = 0
    start_time = time.time()
    while time.time() - start_time < time_limit:
        # generate initial population
        population = generate_population()
        # fitness
        results = []
        for policy in population:
            # time doesn't really matter since it'll end up being depth 1 regardless
            results.append([policy, compare_policies(game, lambda: minimax_genetic.greedy_policy([1] + policy[0], [1] + policy[1]), lambda: mcts_uct0_enhanced.mcts_policy(1), 10, 1, float("inf"), float("inf"), False)])
        results.sort(key = lambda x : x[1])
        # select for crossover
        parents = []
        # assumes POP_SIZE is a multiple of 4
        block_size = POP_SIZE // 4
        for i in range(4):
            for j in range(i * block_size, (i + 1) * block_size):
                for rep in range(i + 1):
                    parents.append(results[j][0])
        # crossover
        children = []
        for i in range(POP_SIZE // 2):
            parent1, parent2 = random.choice(parents), random.choice(parents)
            parent_arr = [parent1[0] + parent1[1], parent2[0] + parent2[1]]
            i, j = random.choice(CROSSOVER_POINTS)
            for c in range(2):
                child_arr = parent_arr[c][:i] + parent_arr[1 - c][i:j] + parent_arr[c][j:]
                children.append([child_arr[:3], child_arr[3:]])
        # no selecting for survival since it would make the time taken a lot longer
        # mutate
        for child in children:
            for i in range(3):
                if random.random() < MUTATION_PROB:
                    # multiply by a random number between 0.8 and 1.2
                    child[0][i] *= 0.8 + (0.4 * random.random())
            for i in range(6):
                if random.random() < MUTATION_PROB:
                    # multiply by a random number between 0.8 and 1.2
                    child[1][i] *= 0.8 + (0.4 * random.random())    
        population = children
        num_gens += 1
        # for p in population:
        #     print(p)
    # write results to output
    outfile = open("genetic_output", "a")
    outfile.write(str(population) + "\n\n")
    for p in population:
        to_write = [[1] + p[0], [1] + p[1]]
        outfile.write(str(to_write) + "\n")
    print(f"Ran for {num_gens} generation(s)")
    final_values = []
    for policy in population:
        weights = [[1] + policy[0], [1] + policy[1]]
        final_values.append([weights, compare_policies(game, lambda: minimax_genetic.greedy_policy(weights[0], weights[1]), lambda: mcts_uct0_enhanced.mcts_policy(1), 10, 1, float("inf"), float("inf"), False)])
    final_values.sort(key=lambda x : x[1])
    outfile.write(f"Best weights: {final_values[-1][0]}" + "\n\n\n")
    outfile.close()
    # print(population)
    return final_values[-1][0]

# return population if a preexisting population is used, otherwise, randomly generates a new one
def generate_population():
    population = [[[5.109256903650978, 427.64643184117085, 3928.1614264827967], [1.6798841625632652, 2.0229625144202052, 2.9902217791604953, 2.3881491832469295, 2.0310098433316046, 1.964499139404]], [[36.66090013736813, 372.4481555857895, 3928.1614264827967], [1.6749188341284955, 2.0229625144202052, 2.7374820896244065, 1.3979692209298764, 1.5511375752385905, 2.9664989929183507]], [[36.66809926372123, 372.4481555857895, 3928.1614264827967], [1.6749188341284955, 2.0229625144202052, 2.7374820896244065, 1.3979692209298764, 2.0310098433316046, 2.9664989929183507]], [[4.655409828331241, 427.64643184117085, 3928.1614264827967], [1.6749188341284955, 2.0229625144202052, 2.7374820896244065, 1.3979692209298764, 1.5511375752385905, 1.964499139404]], [[40.6535559613664, 427.64643184117085, 3928.1614264827967], [1.5778728720608104, 2.0229625144202052, 3.5824050003413013, 2.3881491832469295, 5.133652652443427, 3.5659344983720533]], [[36.66809926372123, 372.4481555857895, 3928.1614264827967], [1.6749188341284955, 2.0229625144202052, 2.7374820896244065, 1.3979692209298764, 2.0310098433316046, 2.9664989929183507]], [[5.109256903650978, 427.64643184117085, 3928.1614264827967], [1.360570302414582, 2.0229625144202052, 2.56663676888618, 2.3881491832469295, 1.5511375752385905, 1.964499139404]], [[5.380144337410294, 46.81350414753168, 3928.1614264827967], [1.6266102619854206, 2.0229625144202052, 2.9902217791604953, 2.3881491832469295, 1.5511375752385905, 1.964499139404]]]
    # population = []
    if not population:
        for i in range(POP_SIZE):
            row_weights = []
            for j in range(3):
                row_weights.append(10 ** (4 * random.random()))
            space_weights = []
            for j in range(6):
                space_weights.append(10 ** random.random())
            population.append([row_weights, space_weights])
    return population
        
        
