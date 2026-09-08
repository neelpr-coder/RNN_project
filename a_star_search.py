import figure2_generation as f2g
import math
import random
import os
import torch 
import numpy as np
import small_model
import shutil

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device("cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FORCED_CACHE_DIR = os.path.join(SCRIPT_DIR, "forced_cache")
FORCED_CACHE_FOLDER = os.path.join(FORCED_CACHE_DIR, "forced_cache_folder")
n_path_copy = os.path.join(FORCED_CACHE_FOLDER, "n_state_min100_max151_sd42_bin_size0.3.npz")

# Create the forced cache folder if it doesn't exist
os.makedirs(FORCED_CACHE_FOLDER, exist_ok=True)
if not os.path.exists(n_path_copy):
    shutil.copy("RNN_cache/n_state/n_state_min100_max151_sd42_bin_size0.3.npz", n_path_copy)

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

def get_lowest_cost_node(open_list):
    return open_list[0] if open_list else None

def convert_probability_to_cost(prob):
    if prob <= 0:
        return float('inf')
    return -math.log(prob)

def get_neighbors(prob_dictionary, key1):
    return [(convert_probability_to_cost(probability), next_node) for next_node, probability in prob_dictionary[key1].items()]


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
    open_list.append(node) 
    cur_g = g_scores[start]

    while open_list:
        # Get the node with the lowest cost
        cheapest_node = get_lowest_cost_node(open_list)
        open_list.remove(cheapest_node)
        cur_g = g_scores[cheapest_node[1]]
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
            return len(total_path), total_path
        # get next nodes from cheapest nodes
        neighbors = get_neighbors(prob_dictionary, cheapest_node[1]) # gives list of sucessor cost and keys from the cheapest node
        for next_node in neighbors:
            temp_g = cur_g + next_node[0]  # Calculate the temp cur g score for the neighbor
            h = 0  # Heuristic temp set to 0 for uniform cost search
            f = temp_g + h
            if next_node[1] in closed_list:
                continue  # Skip if the neighbor is already evaluated
            if next_node[1] not in [n[1] for n in open_list]:
                # If the neighbor is not in the open list, add it
                open_list.append((f, next_node[1]))
                came_from[next_node[1]] = cheapest_node[1]  # Track the path
                g_scores[next_node[1]] = temp_g  # Update g score for the neighbor
            else:
                # If neighbor in open list, check if this path is better
                existing_node = next(n for n in open_list if n[1] == next_node[1])
                if temp_g < g_scores[next_node[1]]:
                    # Update the g score and path if this path is better
                    g_scores[next_node[1]] = temp_g
                    came_from[next_node[1]] = cheapest_node[1]
                    # Update the cost in the open list
                    open_list.remove(existing_node)
                    open_list.append((f, next_node[1]))

        closed_list.add(cheapest_node[1])
        open_list.sort(key=lambda x: x[0]) # sort by cost
    print("no path found")
    return 0, []

def find_highest_probability_node(prob_dictionary, key1):
    highest_prob_transition = -math.inf
    highest_prob_node = None
    for key2, prob in prob_dictionary[key1].itmems():
        if prob > highest_prob_transition:
            highest_prob_transition = prob
            highest_prob_node = key2
    return highest_prob_transition, highest_prob_node

def forced_perturbation_search(dictionary, start, goal, a_star_route):
    # find chepaest node at each time step out of all the nodes, compare chepaest node key to key in a_star route
    # perturb the cheapest key into the key in the a_star route (change the cheapest transition into the optimal key from the a_star route)
    # (track perturbation?) create a separate list corresponding to forced perturbations
    # update the transition dictionary and cache at the end.   

    '''at each time step, if the cheapest node is not the optimal node in the A* route, force a perturbation to the optimal node.
    Then update the original transition dictionary to reflect the forced perturbation. 
    This maintains the dynamic nauture of the transition dictionary while allowing us to track the forced perturbations'''
    prob_dictionary = f2g.convert_count_to_probability(dictionary)
    route = []
    forced_perturbations = []
    for i in range(len(a_star_route)):
        step = a_star_route[i+1] if i + 1 < len(a_star_route) else goal
        if step not in prob_dictionary:
            print(f"[Log] Step {step} not in probability dictionary. Cannot proceed with forced perturbation search.")
            return 0, []

        try: 
            highest_transition_prob, highest_prob_node = find_highest_probability_node(prob_dictionary, step)
        except KeyError:
            print(f"[Log] Step {step} not in probability dictionary. Cannot proceed with forced perturbation search.")
            return 0, []

        if highest_prob_node == step:
            print(f"[Log] Highest probability node {highest_prob_node} is the same as the current step {step}. No perturbation needed.")
            continue
        else:
            # force perturbation 
            cheapest_node = (convert_probability_to_cost(highest_transition_prob), highest_prob_node)
            optimal_state_key = step
            forced_perturbed_node = cheapest_node
            forced_perturbed_node[1] = optimal_state_key  # Change the cheapest node to the optimal node
            forced_perturbations.append((cheapest_node[1], optimal_state_key))
            route.append(forced_perturbed_node)
            # Update the transition dictionary to reflect the forced perturbation
            dictionary[cheapest_node[1]][optimal_state_key] += 1  # Increment the count for the forced perturbation
            prob_dictionary = f2g.convert_count_to_probability(dictionary)  # Update the probability dictionary
            print(f"[Log] Forced perturbation from {cheapest_node[1]} to {optimal_state_key} at step {step}")
    
    return route, len(route), forced_perturbations
            


def first_idea(start, goal, dictionary, a_star_route):
    if start == goal:
        print("Start is the same as goal. Total Steps Taken: 1")
        return 1, [start]

    open_list = []
    closed_list = set() 
    forced_perturbations = []  # List to track forced perturbations
    came_from = {start: None}  # parent mapping for path reconstruction
    g_scores = {start: 0}  # g scores for each node
    # differences = []

    start_node = (0, start) 
    g_scores[start] = 0
    open_list.append(start_node)
    time_step = 0
    while open_list:
        cheapest_node = get_lowest_cost_node(open_list)
        open_list.remove(cheapest_node)
        cur_g = g_scores[cheapest_node[1]]
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
            return len(total_path), total_path
        # get next nodes from cheapest nodes
        neighbors = get_neighbors(prob_dictionary, cheapest_node[1])

        # If the cheapest node is not the optimal node, we can force a perturbation
        if cheapest_node[1] == a_star_route[time_step]: 
            time_step += 1
            neighbors = get_neighbors(prob_dictionary, cheapest_node[1])  # Update neighbors after moving to the next time step
            for next_node in neighbors:
                temp_g = cur_g + next_node[0]  # Calculate the temp cur g score for the neighbor
                h = 0  # Heuristic temp set to 0 for uniform cost search
                f = temp_g + h
                if next_node[1] in closed_list:
                    continue  # Skip if the neighbor is already evaluated
                if next_node[1] not in [n[1] for n in open_list]:
                    # If the neighbor is not in the open list, add it
                    open_list.append((f, next_node[1]))
                    came_from[next_node[1]] = cheapest_node[1]  # Track the path
                    g_scores[next_node[1]] = temp_g  # Update g score for the neighbor
                else:
                    # If neighbor in open list, check if this path is better
                    existing_node = next(n for n in open_list if n[1] == next_node[1])
                    if temp_g < g_scores[next_node[1]]:
                        # Update the g score and path if this path is better
                        g_scores[next_node[1]] = temp_g
                        came_from[next_node[1]] = cheapest_node[1]
                        # Update the cost in the open list
                        open_list.remove(existing_node)
                        open_list.append((f, next_node[1]))
        else:
            # Force a perturbation to the optimal node in the A* route
            optimal_node = a_star_route[time_step]
            if optimal_node != cheapest_node[1] and optimal_node in prob_dictionary[cheapest_node[1]]:
                # Update the transition dictionary to force the perturbation
                dictionary[cheapest_node[1]][optimal_node] += 1  
                prob_dictionary = f2g.convert_count_to_probability(dictionary)  # Update the probability dictionary
                came_from[optimal_node] = cheapest_node[1]  # Track the path
                g_scores[optimal_node] = cur_g + convert_probability_to_cost(prob_dictionary[cheapest_node[1]][optimal_node])  # Update g score for the optimal node
                h = 0 # change heurisitc later to something more meaningful. unifrom cost search effective for now
                f = g_scores[optimal_node] + h
                old_cheapest_node = (f, cheapest_node[1])  # Create a new cheapest node with the updated cost
                new_cheapest_node = optimal_node  # Update the cheapest node to the optimal node
                open_list.append((f, new_cheapest_node))  # Add the optimal node to the open list
                forced_perturbations.append((new_cheapest_node, optimal_node))  # Track the forced perturbation
                print(f"Forced perturbation from {old_cheapest_node[1]} to {optimal_node} at time step {time_step}")
                # add the neighbors of the new_cheapest_node aka optimal node to the open list
                neighbors = get_neighbors(prob_dictionary, optimal_node)
                for next_node in neighbors:
                    temp_g = g_scores[optimal_node] + next_node[0]  # Calculate the temp cur g score for the neighbor
                    h = 0  # Heuristic temp set to 0 for uniform cost search
                    f = temp_g + h
                    if next_node[1] in closed_list:
                        continue  # Skip if the neighbor is already evaluated
                    if next_node[1] not in [n[1] for n in open_list]:
                        # If the neighbor is not in the open list, add it
                        open_list.append((f, next_node[1]))
                        came_from[next_node[1]] = optimal_node  # Track the path
                        g_scores[next_node[1]] = temp_g  # Update g score for the neighbor
                    else:
                        # If neighbor in open list, check if this path is better
                        existing_node = next(n for n in open_list if n[1] == next_node[1])
                        if temp_g < g_scores[next_node[1]]:
                            # Update the g score and path if this path is better
                            g_scores[next_node[1]] = temp_g
                            came_from[next_node[1]] = optimal_node
                            # Update the cost in the open list
                            open_list.remove(existing_node)
                            open_list.append((f, next_node[1]))
            else:
                print(f"Optimal node {optimal_node} not reachable from {cheapest_node[1]} at time step {time_step}")
            time_step += 1
        closed_list.add(cheapest_node[1])
        open_list.sort(key=lambda x: x[0])  # sort by cost
    print("no path found")
    return 0, []


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

    start, stop = rand_gen_start_goal(neural_transition_dict)
    print(f"[Log] Start: {start}\n[Log] Stop: {stop}")
    result_a_star = a_star_search(neural_transition_dict, start, stop)

