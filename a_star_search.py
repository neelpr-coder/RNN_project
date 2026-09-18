import figure2_generation as f2g
import math
import random
import os
import torch 
import numpy as np
import small_model
import copy
import heapq

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device("cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def rand_gen_start_goal(dictionary):
    all_keys = set()
    all_possible_starts = set()
    for key1 in dictionary:
        all_keys.add(key1)
        if dictionary[key1]: 
            all_possible_starts.add(key1)
        for key2 in dictionary[key1]:
            all_keys.add(key2)
    start = random.choice(list(all_possible_starts))
    goal = random.choice(list(all_keys))
    return start, goal

def convert_probability_to_cost(prob):
    if prob <= 0:
        return float('inf')
    return -math.log(prob)

def get_neighbors(prob_dictionary, key1):
    return [(convert_probability_to_cost(probability), next_node) for next_node, probability in prob_dictionary[key1].items()] if key1 in prob_dictionary else []

def heuristic():
    """define heuristic later, for now just return 0"""
    return 0

def a_star_search(dictionary, start, goal):
    # cost eval by -log(probability) where higher probability edges have lower cost and vice versa
    prob_dictionary = f2g.convert_count_to_probability(dictionary)

    if start == goal:
        print("Start is the same as goal. Total Steps Taken: 1")
        return 1, [start]

    came_from = {start: None} # parent mapping for path reconstruction
    open_list = []
    closed_list = set() # store nodes only

    node = (0, start) # (cost, node) tuple

    g_scores = {start: 0}  # Dictionary to store g scores for each node
    heapq.heappush(open_list, node)
    cur_g = g_scores[start]

    while open_list:
        # Get the node with the lowest cost
        cheapest_node = heapq.heappop(open_list)  # pop the cheapest node (f, vector_key) from the open list
        if cheapest_node[1] in closed_list:
            continue  # Skip if the cheapest node is already evaluated
        cur_g = g_scores[cheapest_node[1]] # get the cheapest node's current g score from the g_score dictionary 

        if cheapest_node[1] == goal:
            # Reconstruct the path
            total_path = []
            current = goal
            while current is not None:
                total_path.append(current)
                current = came_from[current]
            total_path.reverse()
            print("Path found:", total_path)
            print("Total Steps Taken:", len(total_path))
            steps = len(total_path) - 1
            return steps, total_path
        # get next nodes from cheapest nodes
        closed_list.add(cheapest_node[1])  # Add the cheapest node to the closed list
        neighbors = get_neighbors(prob_dictionary, cheapest_node[1]) # gives list of sucessor edge cost and keys from the cheapest node
        for next_node in neighbors:
            if next_node[1] in closed_list:
                continue # Skip if the neighbor vector_key is already evaluated
            temp_g = cur_g + next_node[0]  # Calculate the temp cur g score for the neighbor
            if next_node[1] not in g_scores.keys() or temp_g < g_scores[next_node[1]]:
                # Have I seen this node before or is this new g_score cheaper than the existing g_score? 
                # if yes update the g_score and add to open list
                f = temp_g + heuristic()  # Calculate f score for the neighbor
                heapq.heappush(open_list, (f, next_node[1]))
                came_from[next_node[1]] = cheapest_node[1]  # Track the path
                g_scores[next_node[1]] = temp_g  # Update g score for the neighbor
    print("no path found")
    return 0, []

def find_highest_probability_node(prob_dictionary, key1):
    highest_prob_transition = -math.inf
    highest_prob_node = None
    for key2, prob in prob_dictionary[key1].items():
        if prob > highest_prob_transition:
            highest_prob_transition = prob
            highest_prob_node = key2
    return highest_prob_transition, highest_prob_node

def forced_perturbation_search(dictionary, a_star_route):
    '''at each time step, if the naturally outgoing node is not the optimal node in the A* route, force a perturbation to the optimal node.
    Then update the original transition dictionary to reflect the forced perturbation. 
    This maintains the dynamic nauture of the transition dictionary while allowing us to track the forced perturbations'''

    dynamic_dictionary = copy.deepcopy(dictionary)
    prob_dictionary = f2g.convert_count_to_probability(dynamic_dictionary)
    start = a_star_route[0] if a_star_route else None
    if start is None:
        print("[Log] A* route is empty. Cannot proceed with forced perturbation search.")
        return [], 0, [], [], None
    route = [start]
    forced_perturbations = []
    cost = [] # difference in cost between optimal and natural transitions at each step i
    for i in range(len(a_star_route)-1):
        optimal_next_node = a_star_route[i+1]
        cur_node = a_star_route[i]
        if cur_node not in prob_dictionary.keys() or not prob_dictionary[cur_node].keys():
            print(f"[Log] Step {optimal_next_node} not in probability dictionary. Cannot proceed with forced perturbation search.")
            return [], 0, [], [], None

        try: 
            natural_transition_prob, natural_next_node = find_highest_probability_node(prob_dictionary, cur_node)
            natural_cost = convert_probability_to_cost(natural_transition_prob)
            optimal_transition_prob = prob_dictionary[cur_node].get(optimal_next_node, 0)
            optimal_cost = convert_probability_to_cost(optimal_transition_prob)
            cost.append((i, math.fabs(optimal_cost - natural_cost)))
        except KeyError:
            print(f"[Log] Step {optimal_next_node} not in probability dictionary. Cannot proceed with forced perturbation search.")
            return [], 0, [], [], None

        if natural_next_node == optimal_next_node:
            print(f"[Log] Highest probability node {natural_next_node} is the same as the current step {optimal_next_node}. No perturbation needed.")
            route.append(natural_next_node)
            continue
        else:
            # force perturbation 
            forced_perturbations.append((i, cur_node, natural_next_node, optimal_next_node))
            route.append(optimal_next_node)
            # Update the transition dictionary to reflect the forced perturbation
            dynamic_dictionary[cur_node][optimal_next_node] += 1
            prob_dictionary =f2g.convert_count_to_probability(dynamic_dictionary)
            print(f"[Log] Forced perturbation from {natural_next_node} to {optimal_next_node} at step {i}")
    steps = len(route) - 1
    return route, steps, forced_perturbations, cost, dynamic_dictionary


if __name__ == "__main__":
    SEED = 42
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    model = small_model.RNN().to(device)
    model.eval()
    
    BASELINE_MODEL_PATH = os.path.join(SCRIPT_DIR, "post_stage1_model_sd42.pt")
    
    torch.save(model.state_dict(), BASELINE_MODEL_PATH)
    
    print(f"[Log] Baseline model saved to {BASELINE_MODEL_PATH}")

    pair_transition_dict, behavioral_transition_dict, neural_transition_dict, all_visit_count_b_dict, all_visit_count_n_dict = f2g.generate_dicts(model)

    sum_steps = 0
    num_successful_searches = 0
    for _ in range(10):
        start, stop = rand_gen_start_goal(neural_transition_dict)
        result_a_star = a_star_search(neural_transition_dict, start, stop)
        if result_a_star[0] != 0:
            sum_steps += result_a_star[0]
            num_successful_searches += 1
        else:
            continue  # Skip if no path found

    print(f"[Log] Average steps: {sum_steps / num_successful_searches if num_successful_searches > 0 else 'No successful searches'}")