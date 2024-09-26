import networkx as nx 
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product
from itertools import combinations
from matplotlib.patches import Rectangle
import random
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
from itertools import permutations
import numpy as np
import pickle
import sys
import config_and_execute

global arguments #Arguments
global attacks #attacks
global ranks #ranking
global AG #AF
global RG #ranking graph
global CRG #ranking condensed graph

#Variables required for function definition
global scc_dict
global all_pairs
global unattacked
global node_defenders
global all_cycles
global odd_all_cycles
global flat_odd_all_cycles
global even_all_cycles
global flat_even_all_cycles

#Extensions
global conflicts 
global naives
global admissibles 
global completes 
global prefereds 
global groundeds 
global stables 

#normal, plus, minus
global n_gr
global p_gr
global m_gr

global adding

#all properties
variable_names = ["VP", "SC","CP","QP","DP", "CT","SCT","NaE", "AvsFD","DDP","OE","AE","PR","MVP","MDP","MCP","MQP","MDDP","COM21","naco","admco","grco","stco","naweak","admweak","coweak","grweak","stweak","prweak","admstrong","costrong","SplusDB","plusDB","IDB","IAB","plusAB"]
lpr = len(variable_names)

#left-hand side number
global num_to_one
num_to_one = config_and_execute.num_to_one

global num_check
num_check = config_and_execute.num_check

global num_finish
num_finish = config_and_execute.num_finish

global list_len_property
list_len_property = list(range(lpr))

#make a box to store
def generate_multi_dimensional_list(dimensions, size):
    if dimensions == 0:
        return [0] * size
    else:
        return [generate_multi_dimensional_list(dimensions - 1, size) for _ in range(size)]

def make_Mat(dimensions, size):
    mM = []
    for i in range(dimensions):
        mM.append(generate_multi_dimensional_list(i+1, size))
    return mM

def generate_permutations(elements, n):
    all_permutations = []
    for perm in permutations(elements, n):
        all_permutations.append(list(perm))
    return all_permutations

def gene_com(elements,n):
    all_permutations = []
    alll = []
    for perm in combinations(elements, n-1):
        all_permutations.append(list(perm))
    for g in all_permutations:
        for h in range(lpr):
            pre = g.copy()
            pre.extend([elements[h]])
            alll.append(pre)
    return alll

#Store whether a dependency has emerged
global Matrix_A 

Matrix_A = make_Mat(num_to_one, lpr)

#Store counterexamples
global Graph_Matrix_A

Graph_Matrix_A = make_Mat(num_to_one, lpr)

global Dep
Dep = []
for i in range(num_to_one):
    Dep_k = gene_com(list_len_property, i+2) 
    Dep.append(Dep_k)

#print(Dep)

#Stored List
global Table_A
Table_A = []

global UntilT
UntilT = 0

global CheckOut_A
CheckOut_A = []

global Previous_num
Previous_num = 0

#parameter : graph's node
global num_node
num_node = config_and_execute.num_node

#parameter : graph's edge per
global per_edge
per_edge = config_and_execute.per_edge

#filtering incompatibility
global checkout_c_2
checkout_c_2 = config_and_execute.checkout_c_2

#for check only AF which certain property satisfy
global check_property_sign, check_property_num
check_property_sign = config_and_execute.check_property_sign
check_property_num = config_and_execute.check_property_num

#for fix AF
global fix_sign, fix_situation, fix_arguments, fix_attacks, fix_rank_sign, fix_rank
fix_sign = config_and_execute.fix_sign
fix_situation = config_and_execute.fix_situation
fix_arguments = config_and_execute.fix_arguments
fix_attacks = config_and_execute.fix_attacks
fix_rank_sign = config_and_execute.fix_rank_sign
fix_rank = config_and_execute.fix_rank

if fix_sign:
    print("Now, Fix the graph")
    if fix_rank_sign:
        print("Now, also fix the rank")

if not fix_sign and fix_rank_sign:
    print("If the graph is not fixed, rank fix will not apply. The investigation will proceed without any fix.")

try: 
    with open('Matrix_A.pickle', mode='br') as fi:
        Matrix_A_p = pickle.load(fi) 
        if len(Matrix_A_p) != len(Matrix_A) : #Handle when extended
            print(len(Matrix_A_p), len(Matrix_A))
            for i in range(min(len(Matrix_A_p), len(Matrix_A))):
                Matrix_A[i] = Matrix_A_p[i]
        else:
            Matrix_A = Matrix_A_p 

    print("Loaded : dependency box")
except:
    print("Firsttime : dependency box")

try: 
    with open('Graph_Matrix_A.pickle', mode='br') as fi:
        Graph_Matrix_A = pickle.load(fi)
    print("Loaded : counterexample box")
except:
    print("Firsttime : counterexample box")

try: 
    with open('Table_A.pickle', mode='br') as fi:
        Table_A = pickle.load(fi)
    print("Loaded : Boolean List")
except:
    print("First : Boolean List")

try: 
    with open('UntilT_A.pickle', mode='br') as fi:
        UntilT = pickle.load(fi)
    print("Loaded : Number of Boolean lists already investigated", UntilT)
except:
    print("First : Number of Boolean lists already investigated")

try: 
    with open('Previous_num_A.pickle', mode='br') as fi:
        Previous_num = pickle.load(fi)
    print("Loaded : Last time's left-hand side number", Previous_num)
except:
    print("First : Last time's left-hand side number")

#new all combinations
try: 
    with open('Dep.pickle', mode='br') as fi:
        Dep = pickle.load(fi)
    print("Loaded : Remain combinations")
except:
    print("First : Remain combinations")


if num_to_one > len(Matrix_A): 
    print("Error. The size of Matrix_A is not enough. ")
    sys.exit()


#Generate graph
def generate_random_graph():
    #Number of nodes
    n = random.randint(num_node[0], num_node[1]) 

    #Percentage of edge
    p = random.uniform(per_edge[0], per_edge[1])

    #Generate alphabetical labels.
    alphabet_labels = [chr(ord('a') + i) for i in range(n)]

    #Generate a directed graph and assign alphabetical labels to the nodes
    random_graph = nx.fast_gnp_random_graph(n, p, directed=True)
    
    random_graph = nx.relabel_nodes(random_graph, dict(zip(range(n), alphabet_labels)))

    return random_graph


def generate_multi_dimensional_list(dimensions, size):
    if dimensions == 1:
        return [0] * size
    else:
        return [generate_multi_dimensional_list(dimensions - 1, size) for _ in range(size)]
    
def make_Mat(dimensions, size):
    mM = []
    for i in range(dimensions-1):
        mM.append(generate_multi_dimensional_list(i+2, size))
    return mM

#Add self-roop
def add_self_loops_random(graph, probability):
    for node in graph.nodes():
        if random.random() < probability and not graph.has_edge(node, node):
            graph.add_edge(node, node)

    return graph

def are_sublists_unique(lst_of_lst):
    return len(lst_of_lst) == len(set(map(tuple, lst_of_lst)))


#forPR
def add_even_length_path_PR(graph, target_node_pr):
    
    alpha_list = (graph.nodes()) 

    max_node = max_alphabet_node(alpha_list) 

    new_node = chr(ord(max_node) + 1) 

    graph.add_node(new_node) 
    graph.add_edge(new_node, target_node_pr) 

    path_length_pr = random.randint(1, 2) * 2  
    current_node = new_node
    for _ in range(path_length_pr - 1):
        
        new_node = chr(ord(new_node) + 1) 
        graph.add_node(new_node)
    
        graph.add_edge(new_node, current_node)
        current_node = new_node

def max_alphabet_node(nodes):
    if not nodes:
        return None  

    max_node = max(nodes, key=lambda x: ord(x)) 
    return max_node

def create_PR_graph(target_node_pr):
    G_pr = nx.DiGraph()
    G_pr.add_node(target_node_pr)

    num_paths_pr = random.randint(1, 2)

    for _ in range(num_paths_pr):
        add_even_length_path_PR(G_pr, target_node_pr)     

    return G_pr


#make total_order
def create_combinations_total(lst):
    shuffled_list = lst.copy()
    random.shuffle(shuffled_list)
    combinations = [shuffled_list[i:i+2] for i in range(len(shuffled_list)-1)]
    return combinations

#Select n elements randomly, invert them, and add them back to the original list
def add_reverse_to_original(list_of_lists):
    n = random.randint(0, len(list_of_lists))

    selected_lists = random.sample(list_of_lists, n)
    
    for lst in selected_lists:
        list_of_lists.append(lst[::-1])  


"""
create graphs for adding branches cases 
"""


#Duplicate AF
def duplicate_and_rename(graph, suffix):
    global copy_mapping 

    new_graph = graph.copy()
    mapping = {node: f"{node}{suffix}" for node in graph.nodes}
    new_graph = nx.relabel_nodes(new_graph, mapping)
    copy_mapping = mapping
    return new_graph

#Add a path of even length. 1, 2, … 
def add_even_length_path(graph, target_node):
    global path_length 
    new_node = 1
    graph.add_node(str(new_node)) 

    #Add an edge connecting a new node to an existing node
    graph.add_edge(str(new_node), target_node) 

    #even_length_branch
    path_length = random.randint(1, 1) * 2  
    current_node = new_node
    for _ in range(path_length - 1):
        new_node += 1 
        graph.add_node(str(new_node)) 
        graph.add_edge(str(new_node), str(current_node))
        current_node = new_node

        
#odd length
def add_odd_length_path(graph, target_node):
    global path_length 
    new_node = 1
    graph.add_node(str(new_node)) 

    graph.add_edge(str(new_node), target_node) 

    path_length = random.randint(1, 1) * 2 - 1 
    current_node = new_node
    for _ in range(path_length - 1):
        new_node += 1 
        graph.add_node(str(new_node)) 
        graph.add_edge(str(new_node), str(current_node))
        current_node = new_node

def union_preserve_names(graph_a, graph_b):
    union_graph = nx.DiGraph()
    union_graph.add_nodes_from(graph_a.nodes(data=True))
    union_graph.add_nodes_from(graph_b.nodes(data=True))
    union_graph.add_edges_from(graph_a.edges())
    union_graph.add_edges_from(graph_b.edges())
    return union_graph


#Create a ranking graph.

#condensed graph
def strongly_connected_components(graph):
    return list(nx.strongly_connected_components(graph))

def condensed_graph(original_graph, scc_list): 
    condensed_graph = nx.DiGraph()

    for i, scc in enumerate(scc_list):
        condensed_graph.add_node(i, nodes=scc)

    for i, scc in enumerate(scc_list):
        for node in scc:
            for neighbor in original_graph.successors(node):
                j = find_scc_index(scc_list, neighbor)
                condensed_graph.add_edge(i, j)

    return condensed_graph

def find_scc_index(scc_list, node):
    for i, scc in enumerate(scc_list):
        if node in scc:
            return i
    return None


"""
From here on, we'll begin implementing each property
To start, we'll prepare the necessary groundwork for creating the properties
"""

#Acyclic or not
def is_dag(graph):
    if nx.is_directed_acyclic_graph(graph):
        return True
    else:
        return False

#make all combinations
def nC2(input_list):
    result = [list(comb) for comb in combinations(input_list, 2)]
    return result

#get attacker
def get_attackers(node):
    global attacks

    attackers = [attacker for attacker, attacked in attacks if attacked == node]
    return attackers

#find unattacked nodes
def find_unattacked_nodes(arguments, attacks):
    attacked_nodes = set()
    for attack in attacks:
        attacked_nodes.add(attack[1])  
    unattacked_nodes = [node for node in arguments if node not in attacked_nodes]
    return unattacked_nodes

#Detecting nodes attacking themselves
def find_selfattacking_nodes(attacks):
    self_attacking_nodes = {attacker for attacker, attacked in attacks if attacker == attacked}
    return list(self_attacking_nodes)


#find defender
def find_node_defenders(attacks):
    global arguments

    attackers = {attacked: [] for attacker, attacked in attacks}
    for attacker, attacked in attacks:
        attackers[attacked].append(attacker)

    defenders = {key : [] for key in arguments}
    for attacked, attacker_list in attackers.items():
        for attacker in attacker_list:
        
            if attacker in attackers:
                defenders[attacked].extend(attackers[attacker])
                defenders[attacked] = list(set(defenders[attacked]))

    return defenders


def find_key(dictionary, value):
    for key, val_list in dictionary.items():
        if value in val_list:
            return key 
    return None

def find_keys_for_values(dictionary, target_values):
    keys_list = {target: [key for key, value_list in dictionary.items() if target in value_list] for target in target_values}
    result_list_1 = list(keys_list.values())
    result_list_1 = [element for sublist in result_list_1 for element in sublist] 
    return result_list_1

#Group comparison
def group_comparison(list1, list2): 

    which_strong = 0 #which is strong

    if group_saiki(list1,list2) and group_saiki(list2,list1):# = 
        which_strong = 0
    elif group_saiki(list1,list2):# < 
        which_strong = 1
    elif group_saiki(list2,list1):# >
        which_strong = -1
    else:
        which_strong = 10 # can't compare

    return which_strong 

# to use in group_comparison
def group_saiki(list1, list2):
    if len(list1) == 0: 
        return True
    else:
        x = list1[0] 
        for i in range(len(list2)):
            if nx.has_path(RG, x, list2[i]):
                    
                if group_saiki(list1[1:], list2[:i] + list2[i+1:]):
                    return True 
            else: 
                pass 
        return False
    
#find root
def check_paths(source_node, target_node):
    global AG
    global flat_odd_all_cycles

    m = 0

    all_paths = list(nx.all_simple_paths(AG, source_node, target_node))

    flat_all_paths = [item for pa in all_paths for item in pa] 

    for z in range(len(even_all_cycles)):
        for s in range(len(even_all_cycles)):
            if not set(even_all_cycles[s]).isdisjoint(set(flat_all_paths)):
                flat_all_paths.extend(even_all_cycles[s])
    
    if not set(flat_all_paths).isdisjoint(set(flat_odd_all_cycles)):
            m = 1

    has_odd_paths = any(len(path) % 2 == 0 for path in all_paths)
    has_even_paths = any(len(path) % 2 != 0 for path in all_paths)

    if m == 1: 
        has_odd_paths = True
        has_even_paths = True

    if has_odd_paths and has_even_paths:
        return 0  # There are both paths of odd and even lengths.
    elif has_odd_paths:
        return 1  # only odd paths
    elif has_even_paths:
        return 2  # only even paths
    else:
        return -1  # no path

#find root
def find_root(target):
    global unattacked #unattacked nodes

    odd_list = []
    even_list = []

    for node in unattacked:
        if check_paths(node, target) == 0:
            odd_list.append(node)
            even_list.append(node) 
        elif check_paths(node, target) == 1:
            odd_list.append(node) 
        elif check_paths(node, target) == 2:
            even_list.append(node) 
        elif check_paths(node, target) == -1:
            pass
    
    return (odd_list, even_list)

#check simple
def simple(node):
    global attacks
    global node_defenders

    j = 0 

    defenders_s = node_defenders[node] 
    attackers_s = get_attackers(node) 

    if len(defenders_s) == 0:
        return False
    else:
        for defender_s in defenders_s:
            t = 0 
            for attacker_s in attackers_s:
                if [defender_s, attacker_s] in attacks: 
                    t = t + 1
            if t == 1:
                j = j + 1
        if j == len(defenders_s):
            return True
        else:
            return False

#check distributed
def distributed(node):
    j = 0 

    attackers_d = get_attackers(node) 

    for attacker_d in attackers_d:
        list1 = get_attackers(attacker_d) 
        if len(list1) <= 1: 
            j = j + 1
    if j == len(attackers_d):
        return True
    else:
        return False
    

#For OE
def OE_saiki(list1, list2):
    global RG

    if len(list1) == len(list2) == 0: 
        return True
    else:
        x = list1[0] 
        for i in range(len(list2)):
            if nx.has_path(RG, x, list2[i]) and nx.has_path(RG, list2[i], x):
                if OE_saiki(list1[1:], list2[:i] + list2[i+1:]):
                    return True 
            else: 
                pass 
        return False
    
#For AE
def reachable_nodes_to_target(graph,target_node):
    
    reachable = set()

    def dfs(node):
        # DFS
        reachable.add(node) 
        for predecessor in graph.predecessors(node):
            if predecessor not in reachable:
                dfs(predecessor)

    dfs(target_node)
    return list(reachable)

#For PR
def get_attacked_nodes(node):
    global attacks

    attacked_nodes_pr = [target for attacker, target in attacks if attacker == node]
    return attacked_nodes_pr

#For PR
def unique_path_exists(graph, source_node, target_node):
    if nx.has_path(graph, source_node, target_node):
        # unique path or not
        all_paths_pr = list(nx.all_simple_paths(graph, source=source_node, target=target_node))
        return len(all_paths_pr) == 1
    else:
        return False

#conflict-free
def is_new_conflict_free(node_list):
    global attacks
    if node_list == []:
        return True

    for node in node_list:
        for attack in attacks:
            if node == attack[0] and attack[1] in node_list:
                return False  
    return True

def conflict_unique_list_of_lists(list_of_lists):
    return list(set(item for sublist in list_of_lists for item in sublist))

#conflict-free less time
def newconflict(nlist, arg1, conf, j):
    global attacks

    h = [] 
    i = []
    result = []

    if len(nlist) == 0 and j != 0:
        return conf
    else:
        j = j + 1
        for list1 in nlist: 
            for arg in arg1: 
                list2 = list1.copy() 
                list2.append(arg) 
                if is_new_conflict_free(list2) and len(list2) == len(set(list2)) and sorted(list2) not in conf: 
                    h.append(sorted(list2))
                    conf.append(sorted(list2))
        i = conflict_unique_list_of_lists(h)
        result = newconflict(h, i, conf, j) 
        return result
        
#naive extension
def fnaive():#Extracting the maximal set from conflict-free
    global conflicts

    naives = []
    for l in conflicts:
        is_maximal = True
        for m in conflicts:
            if l != m and set(l).issubset(set(m)): 
                is_maximal = False 
                break
        if is_maximal:
            naives.append(l)
    return naives

#admissible extension
def fadmissible():
    global attacks
    global conflicts
    admissibles = []
    list1 = []
    list2 = []
    
    for i in range(len(conflicts)):
            b = 0
            for j in range(len(conflicts[i])): 
                list1 = []
                for k in range(len(attacks)):
                    if attacks[k][1] == conflicts[i][j]: 
                        list1.append(attacks[k][0]) 
                a = 0
                
                for l in range(len(list1)):
                    list2 = [] 
                    for m in range(len(attacks)):
                        if attacks[m][1] == list1[l]: 
                            list2.append(attacks[m][0])
                    if not set(list2).isdisjoint(set(conflicts[i])):
                        a = a + 1
                if a == len(list1): 
                    b = b + 1
            if b == len(conflicts[i]):
                admissibles.append(conflicts[i]) 
    return admissibles

#complete extensions
def fcomplete():
    global arguments
    global attacks
    global conflicts
    global admissibles
    completes = []
    list1 = []
    list2 = []

    for i in range(len(admissibles)):
            b = 0
            newad = list(set(arguments)-set(admissibles[i])) 
            for j in range(len(newad)): 
                list1 = []
                for k in range(len(attacks)):
                    if attacks[k][1] == newad[j]:
                        list1.append(attacks[k][0])  
                a = 0
                for l in range(len(list1)):
                    list2 = [] 
                    for m in range(len(attacks)):
                        if attacks[m][1] == list1[l]: 
                            list2.append(attacks[m][0]) 
                    if not set(list2).isdisjoint(set(admissibles[i])): 
                        a = a + 1
                if a != len(list1): 
                    b = b + 1
            if b == len(newad): 
                completes.append(admissibles[i]) 
    return completes

#preferred extension
def fprefered(): #Extracting the maximal set
    global completes

    prefereds = []
    for l in completes:
        is_maximal = True
        for m in completes:
            if l != m and set(l).issubset(set(m)): 
                is_maximal = False 
                break
        if is_maximal:
            prefereds.append(l)
    return prefereds


#stable extension
def fstable():
    global prefereds
    global arguments
    global attacks

    stables = []

    for l in prefereds: 
        others = list(set(arguments) - set(l)) 
        h = 0
        for m in others: 
            z = 0 
            for at in attacks:
                
                if at[1] == m and at[0] in l: 
                    z = z + 1

            if z > 0: 
                h = h + 1
            
        if h == len(others): 
            stables.append(l)
        
    return stables

#grounded extension
def fgrounded():#Extracting the minimal set
    global completes

    groundeds = []
    for l in completes:
        is_minimal = True
        for m in completes:
            if l != m and set(m).issubset(set(l)):
                is_minimal = False
                break
        if is_minimal:
            groundeds.append(l)
    return groundeds


def fgrounded2():
    global arguments
    global attacks
    grounded = []
    
    for arg in arguments:
        is_ground = True
        for attack in attacks:
            if attack[1] == arg and attack[0] in grounded:
                is_ground = False
                break
        if is_ground:
            grounded.append(arg)
    
    return [grounded]

def get_value_from_listZ(T, ListZ, index=0):
    if index == len(T):
        return ListZ
    return get_value_from_listZ(T, ListZ[T[index]], index + 1)

def assign_value_to_listZ(T, ListZ, value, index=0):
    if index == len(T) - 1:
        ListZ[T[index]] = value
        return
    assign_value_to_listZ(T, ListZ[T[index]], value, index + 1)

def is_sublist_A(list1, list2):
    # Check if list1 is a sublist of list2
    if len(list1) > len(list2):
        return False
    for i in range(len(list2) - len(list1) + 1):
        if list2[i:i+len(list1)] == list1:
            return True
    return False

def is_sublist_in_order(list1, list2):
    # Check if list1 is a sublist of list2 (same order)
    n = len(list1)
    m = len(list2)
    
    i = 0  
    j = 0 
    while i < n and j < m:
        if list1[i] == list2[j]:
            i += 1
        j += 1
    
    return i == n


"""
Up to this point has been the preparation for property
Now, we will proceed with implementing each property
For an overview of the structure of the properties, it is helpful to refer to VP.
Note that the return value Boolean is given as either true, none, or false. 
However, note that when considering dependencies, none can be treated the same as true.
"""

#VP
def VP():

    j = 0 #Condition for returning True
    non = 0 #Condition for returning None (None means no pair satisfies precondition. in VP case, no argument is unattacked)
    """
    #in this dependency investigation, None can ultimately be treated the same as True.
    """
    #survey all pairs
    for pair in all_pairs: 
        if len(get_attackers(pair[0])) > 0 and len(get_attackers(pair[1])) == 0: #only pair[0] has attacker
            x = find_key(scc_dict, pair[0]) #using condensed ranking graph. and check ranking
            y = find_key(scc_dict, pair[1])
            if x == y: #same ranking
                return False
            else:
                if nx.has_path(CRG, x, y): #if a path from x to y exists, then x < y in ranking
                    non = non + 1
                    j = j + 1
                else: #don't satisfy the condition
                    return False
        elif len(get_attackers(pair[1])) > 0 and len(get_attackers(pair[0])) == 0: #only pair[1] has attacker
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else:
                if nx.has_path(CRG, y, x): #y < x in ranking
                    non = non + 1
                    j = j + 1
                else: 
                    return False
        else:#doesn't meet the conditions necessary to investigate VP in the first place
            j = j + 1
    #after loop check j
    if j == len(all_pairs) : 
        if non > 0: 
            return True
        else: #all None
            return None


#SC
def SC():

    j = 0 
    non = 0

    for pair in all_pairs: 
        if pair[0] in get_attackers(pair[0]) and pair[1] not in get_attackers(pair[1]) : #pair[0] is self-attacking and pair[1] is not.
            x = find_key(scc_dict, pair[0])
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, x, y): 
                    non = non + 1
                    j = j + 1
                else:
                    return False
        elif pair[0] not in get_attackers(pair[0]) and pair[1] in get_attackers(pair[1]) : 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, y, x):
                    non = non + 1
                    j = j + 1
                else:
                    return False
        else : 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0:
            return True
        else:
            return None


#CP
def CP():

    j = 0 
    non = 0

    for pair in all_pairs: 
        if len(get_attackers(pair[0])) > len(get_attackers(pair[1])): #a's attacker > c's attacker (number)
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, x, y): 
                    non = non + 1
                    j = j + 1
                else:
                    return False
                
        elif len(get_attackers(pair[0])) < len(get_attackers(pair[1])): 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y:
                return False
            else :
                if nx.has_path(CRG, y, x): 
                    non = non + 1
                    j = j + 1
                else:
                    return False
        else : 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None


#QP
def QP():

    j = 0 
    non = 0

    for pair in all_pairs: 
        left = 0
        right = 0
        #get attackers
        list1 = get_attackers(pair[0]) 
        list2 = get_attackers(pair[1]) 

        c_list1 = find_keys_for_values(scc_dict, list1) 
        c_list2 = find_keys_for_values(scc_dict, list2) 
        
        for s in range(len(c_list2)):
            q = 0 
            for t in range(len(c_list1)):
                if nx.has_path(CRG, c_list1[t], c_list2[s]) and c_list1[t] != c_list2[s]:
                    q = q + 1
            if q == len(c_list1): 
                left = 1
        for t in range(len(c_list1)):
            q = 0 
            for s in range(len(c_list2)):
                if nx.has_path(CRG, c_list2[s], c_list1[t]) and c_list1[t] != c_list2[s]:
                    q = q + 1
            if q == len(c_list2): 
                right = 1
        
        if left > 0 and right > 0 :
            return "error" 
        elif left > 0: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, y, x): 
                    non = non + 1
                    j = j + 1
                else:
                    return False
        elif right > 0: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, x, y): 
                    non = non + 1
                    j = j + 1
                else:
                    return False
        else: 
            j = j+1  

    if j == len(all_pairs) : 
        if non > 0:
            return True
        else:
            return None


#DP
def DP():

    j = 0 
    non = 0

    for pair in all_pairs:
        list1 = get_attackers(pair[0])
        list2 = get_attackers(pair[1])

        if len(list1) == len(list2) : #the number of attacker is same
            list3 = [] 
            list4 = [] 
            for s in list1:
                list3.extend(get_attackers(s))
            for t in list2: 
                list4.extend(get_attackers(t))

            if len(list3) == 0 and len(list4) > 0: #only c has defender
                x = find_key(scc_dict, pair[0]) 
                y = find_key(scc_dict, pair[1])
                if x == y: 
                    return False
                else:
                    if nx.has_path(CRG, x, y): 
                        non = non + 1
                        j = j + 1
                    else: 
                        return False
            elif len(list3) > 0 and len(list4) == 0: 
                x = find_key(scc_dict, pair[0]) 
                y = find_key(scc_dict, pair[1])
                if x == y: 
                    return False
                else:
                    if nx.has_path(CRG, y, x):
                        non = non + 1
                        j = j + 1
                    else: 
                        return False
            else:
                j = j + 1
        else : 
            j = j + 1

    if j == len(all_pairs) : 
        if non > 0:
            return True
        else:
            return None


#CT
#use group comparison
def CT():

    j = 0 
    non = 0

    for pair in all_pairs: 
        list1 = get_attackers(pair[0])
        list2 = get_attackers(pair[1])

        g = group_comparison(list1,list2) 
        if  g == 10:
            j = j + 1
        else:
            if g == 0:
                # exist path to each other
                if nx.has_path(RG, pair[0], pair[1]) and nx.has_path(RG, pair[1], pair[0]):
                    non = non + 1
                    j = j + 1
                else:
                    return False
            elif g <= 0: # list1 is as strong as or stronger than list2
                if nx.has_path(RG, pair[0], pair[1]): 
                            non = non + 1
                            j = j + 1
                else:
                    return False
            elif g >= 0 : 
                if nx.has_path(RG, pair[1], pair[0]): 
                            non = non + 1
                            j = j + 1
                else:
                    return False
    if j == len(all_pairs) : 
        if non > 0:
            return True
        else:
            return None


#SCT
def SCT():

    j = 0 
    non = 0

    for pair in all_pairs: 
        list1 = get_attackers(pair[0])
        list2 = get_attackers(pair[1])

        g = group_comparison(list1,list2) 
        
        if  g == 10 or g == 0:
            j = j + 1
        else:
            if g < 0: #list1 is stronger than list2
                x = find_key(scc_dict, pair[0]) 
                y = find_key(scc_dict, pair[1])
                if x == y: 
                    return False
                else:
                    if nx.has_path(CRG, x, y): 
                        non = non + 1
                        j = j + 1
                    else: 
                        return False
            elif g > 0 : 
                x = find_key(scc_dict, pair[0]) 
                y = find_key(scc_dict, pair[1])
                if x == y: 
                    return False
                else:
                    if nx.has_path(CRG, y, x): 
                        non = non + 1
                        j = j + 1
                    else: 
                        return False
    if j == len(all_pairs) : 
        if non > 0:
            return True
        else:
            return None


#Tot
def Tot():

    j = 0 

    for pair in all_pairs: 
        if nx.has_path(RG, pair[0], pair[1]) or nx.has_path(RG, pair[1], pair[0]):
            j = j + 1

    if j == len(all_pairs) : 
        return True
    else : 
        return False


#NaE
def NaE():

    j = 0 
    non = 0

    for pair in all_pairs: 
        
        list1 = get_attackers(pair[0])
        list2 = get_attackers(pair[1])
        
        if len(list1) == len(list2) == 0: #0 and 0 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                non = non + 1
                j = j + 1
            else: 
                return False
        else: 
            j = j + 1
    
    if j == len(all_pairs) :
        if non > 0 :
            return True
        else:
            return None


#AvsFD
def AvsFD():

    j = 0 
    non = 0

    #acyclic or not
    if is_dag(AG):
        pass
    else:
        return None

    for pair in all_pairs: 
        
        if len(get_attackers(pair[0])) == 1 and (get_attackers(pair[0])[0] in unattacked) and len(find_root(pair[1])[0]) == 0:
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else:
                if nx.has_path(CRG, x, y): 
                    non = non + 1
                    j = j + 1
                else: 
                    return False
        elif len(get_attackers(pair[1])) == 1 and (get_attackers(pair[1])[0] in unattacked) and len(find_root(pair[0])[0]) == 0:
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y:
                return False
            else:
                if nx.has_path(CRG, y, x): 
                    non = non + 1
                    j = j + 1
                else: 
                    return False
        else:
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None


#DDP
def DDP():

    j = 0 
    non = 0

    for pair in all_pairs: 
        if len(get_attackers(pair[0])) == len(get_attackers(pair[1])) and len(node_defenders[pair[0]]) == len(node_defenders[pair[1]]):
            
            if (simple(pair[0]) and distributed(pair[0])) and (simple(pair[1]) and not distributed(pair[1])):
                x = find_key(scc_dict, pair[0]) 
                y = find_key(scc_dict, pair[1])
                if x == y: 
                    return False
                else:
                    if nx.has_path(CRG, y, x): 
                        non = non + 1
                        j = j + 1
                    else: 
                        return False
            elif (simple(pair[1]) and distributed(pair[1])) and (simple(pair[0]) and not distributed(pair[0])):
                
                x = find_key(scc_dict, pair[0]) 
                y = find_key(scc_dict, pair[1])
                if x == y: 
                    return False
                else:
                    if nx.has_path(CRG, x, y): 
                        non = non + 1
                        j = j + 1
                    else: 
                        return False
            else:
                j = j + 1
        else:
            j = j + 1
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None


#OE
def OE():

    j = 0 
    non = 0

    for pair in all_pairs: 
        if len(get_attackers(pair[0])) == len(get_attackers(pair[1])) :
            if OE_saiki(get_attackers(pair[0]),get_attackers(pair[1])): 
                x = find_key(scc_dict, pair[0]) 
                y = find_key(scc_dict, pair[1])
                if x == y: 
                    non = non + 1
                    j = j + 1
                else: 
                    return False
            else: 
                j = j + 1
        else: 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None


#AE
def AE():

    j = 0 
    non = 0

    for pair in all_pairs: 
        t = 0 #For verifying all isomorphisms.
        list1 = reachable_nodes_to_target(AG, pair[0]) 
        list2 = reachable_nodes_to_target(AG, pair[1]) 
        #cc graphs
        subAG1 = AG.subgraph(list1) 
        subAG2 = AG.subgraph(list2) 
        
        gm = nx.isomorphism.GraphMatcher(subAG1, subAG2)

        if gm.is_isomorphic(): 
            all_mappings = list(gm.isomorphisms_iter())
            for mapping in all_mappings:
                if mapping[pair[0]] == pair[1]:
                    t = t + 1
            if t > 0:
                x = find_key(scc_dict, pair[0]) 
                y = find_key(scc_dict, pair[1])
                if x == y: 
                    non = non + 1
                    j = j + 1
                else: 
                    return False
            else: 
                j = j + 1
        else: 
            j = j + 1
    if j == len(all_pairs) : 
        if non > 0:
            return True
        else:
            return None


#PR
#For each node, verify if it is the PR of 𝑥
#If so, check if it is stronger than the non-attacked nodes.
def PR():
    nonn = 0

    if is_dag(AG):
        pass
    else:
        return None

    for node in arguments:
        p = 0 #For verifying the existence of a path from every node.
        q = 0 #For verifying that there is exactly one path from every node.
        if len(find_root(node)[0]) == 0:
            
            if len(find_root(node)[1]) > 0:
                
                if len(get_attacked_nodes(node)) == 0:
                    
                    for x in arguments:
                        if nx.has_path(AG, x, node): 
                            p = p + 1
                    if p == len(arguments):
                        for x in arguments:
                            if unique_path_exists(AG, x, node): 
                                q = q + 1
                        if q == (len(arguments) - 1): 
                            for no in unattacked:
                                x = find_key(scc_dict, node) 
                                y = find_key(scc_dict, no)
                                if x == y: 
                                    return False
                                else:
                                    if nx.has_path(CRG, y, x): 
                                        pass
                                    else: 
                                        return False
                            nonn = nonn + 1

                        else: 
                            pass
                    else: 
                        pass
                else: 
                    pass
            else: 
                pass
        else: 
            pass
    if nonn > 0:
        return True
    else:
        return None


    
#MVP almost as same as VP
def MVP():


    j = 0 
    non = 0 

    for pair in all_pairs: 
        if len(get_attackers(pair[0])) > 0 and len(get_attackers(pair[1])) == 0: 
            if nx.has_path(RG, pair[0], pair[1]): 
                            non = non + 1
                            j = j + 1
            else:
                return False
        elif len(get_attackers(pair[1])) > 0 and len(get_attackers(pair[0])) == 0: 
            if nx.has_path(RG, pair[1], pair[0]): 
                            non = non + 1
                            j = j + 1
            else:
                return False
        else:
            j = j + 1
    if j == len(all_pairs) : 
        if non > 0: 
            return True
        else:
            return None

        
#MDP
def MDP():

    j = 0 
    non = 0

    for pair in all_pairs: 
        list1 = get_attackers(pair[0])
        list2 = get_attackers(pair[1])
        if len(list1) == len(list2) : 
            list3 = [] 
            list4 = [] 
            for s in list1:
                list3.extend(get_attackers(s))
            for t in list2: 
                list4.extend(get_attackers(t))
            
            if len(list3) == 0 and len(list4) > 0: 
                
                if nx.has_path(RG, pair[0], pair[1]): 
                    non = non + 1
                    j = j + 1
                else: 
                    return False
            elif len(list3) > 0 and len(list4) == 0:
                if nx.has_path(RG, pair[1], pair[0]): 
                    non = non + 1
                    j = j + 1
                else: 
                    return False
            else:
                j = j + 1
        else : 
            j = j + 1

    if j == len(all_pairs) : 
        if non > 0:
            return True
        else:
            return None

        
#MCP
def MCP():

    j = 0 
    non = 0

    for pair in all_pairs: 
        if len(get_attackers(pair[0])) > len(get_attackers(pair[1])): 
            
            if nx.has_path(RG, pair[0], pair[1]): 
                non = non + 1
                j = j + 1
            else: 
                return False
                
        elif len(get_attackers(pair[0])) < len(get_attackers(pair[1])): 
            if nx.has_path(RG, pair[1], pair[0]): 
                non = non + 1
                j = j + 1
            else: 
                return False
        else : 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None
    
        
#MQP
def MQP():

    j = 0 
    non = 0

    for pair in all_pairs: 
        left = 0
        right = 0
        list1 = get_attackers(pair[0]) 
        list2 = get_attackers(pair[1]) 
    
        c_list1 = find_keys_for_values(scc_dict, list1) 
        c_list2 = find_keys_for_values(scc_dict, list2) 
        
        for s in range(len(c_list2)):
            q = 0 
            for t in range(len(c_list1)):
                if nx.has_path(CRG, c_list1[t], c_list2[s]) and c_list1[t] != c_list2[s]:
                    q = q + 1
            if q == len(c_list1): 
                left = 1
        for t in range(len(c_list1)):
            q = 0 
            for s in range(len(c_list2)):
                if nx.has_path(CRG, c_list2[s], c_list1[t]) and c_list1[t] != c_list2[s]:
                    q = q + 1
            if q == len(c_list2): 
                right = 1
        
        if left > 0 and right > 0 :
            return "error"
        elif left > 0: 
            if nx.has_path(RG, pair[1], pair[0]): 
                non = non + 1
                j = j + 1
            else: 
                return False
        elif right > 0: 
            if nx.has_path(RG, pair[0], pair[1]): 
                non = non + 1
                j = j + 1
            else: 
                return False
        else: 
            j = j + 1  

    if j == len(all_pairs) : 
        if non > 0:
            return True
        else:
            return None
        
#MDDP
def MDDP():

    j = 0 
    non = 0

    for pair in all_pairs: 
        if len(get_attackers(pair[0])) == len(get_attackers(pair[1])) and len(node_defenders[pair[0]]) == len(node_defenders[pair[1]]):
            if (simple(pair[0]) and distributed(pair[0])) and (simple(pair[1]) and not distributed(pair[1])):
            
                if nx.has_path(RG, pair[1], pair[0]):
                    non = non + 1
                    j = j + 1
                else: 
                    return False
            elif (simple(pair[1]) and distributed(pair[1])) and (simple(pair[0]) and not distributed(pair[0])):
                if nx.has_path(RG, pair[0], pair[1]): 
                    non = non + 1
                    j = j + 1
                else: 
                    return False
            else:
                j = j + 1
        
        else:
            j = j + 1
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None


#COM21
def COM21():

    j = 0 
    non = 0

    for pair in all_pairs: 
        
        list1 = get_attackers(pair[0])
        list2 = get_attackers(pair[1])
        if len(list1) == 1 and len(list2) == 2:
            
            list3 = get_attackers(list2[0])
            list4 = get_attackers(list2[1])
            
            if len(list3) == 1 and len(list4) == 1 and list3[0] in unattacked and list4[0] in unattacked and list1[0] in unattacked:
                
                x = find_key(scc_dict, pair[0]) 
                y = find_key(scc_dict, pair[1])
                if x == y:
                    non = non + 1
                    j = j + 1
                else: 
                    return False
            else: 
                j = j + 1
        elif len(list2) == 1 and len(list1) == 2:
            
            list3 = get_attackers(list1[0])
            list4 = get_attackers(list1[1])
            if len(list3) == 1 and len(list4) == 1 and list3[0] in unattacked and list4[0] in unattacked and list2[0] in unattacked:
                x = find_key(scc_dict, pair[0]) 
                y = find_key(scc_dict, pair[1])
                if x == y: 
                    non = non + 1
                    j = j + 1
                else: 
                    return False
            else:
                j = j + 1
        else : 
            j = j + 1
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None


#naive-co
def naco():
    global naives

    j = 0 
    non = 0

    flat_na_list = [item for sublist in naives for item in sublist]
    
    for pair in all_pairs: 
        
        if pair[0] in flat_na_list and pair[1] not in flat_na_list:
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, y, x): 
                    non = non + 1
                    j = j + 1
                else:
                    return False         
                
        elif pair[0] not in flat_na_list and pair[1] in flat_na_list: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, x, y): 
                    non = non + 1
                    j = j + 1
                else:
                    return False
        else : 
            j = j + 1
    
    if j == len(all_pairs) :
        if non > 0 :
            return True
        else:
            return None

#adm-co
def admco():
    global admissibles

    j = 0 
    non = 0

    flat_adm_list = [item for sublist in admissibles for item in sublist]

    for pair in all_pairs: 
        if pair[0] in flat_adm_list and pair[1] not in flat_adm_list:
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, y, x): 
                    non = non + 1
                    j = j + 1
                else:
                    return False         
                
        elif pair[0] not in flat_adm_list and pair[1] in flat_adm_list: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, x, y): 
                    non = non + 1
                    j = j + 1
                else:
                    return False
        else : 
            j = j + 1
    
    if j == len(all_pairs) :
        if non > 0 :
            return True
        else:
            return None

#complete-co
def coco():
    global completes

    j = 0 
    non = 0

    flat_co_list = [item for sublist in completes for item in sublist]

    for pair in all_pairs: 
        if pair[0] in flat_co_list and pair[1] not in flat_co_list:
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, y, x): 
                    non = non + 1
                    j = j + 1
                else:
                    return False         
                
        elif pair[0] not in flat_co_list and pair[1] in flat_co_list: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, x, y): 
                    non = non + 1
                    j = j + 1
                else:
                    return False
        else : 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None

#grounded-co
def grco():
    global groundeds

    j = 0 
    non = 0

    flat_gr_list = [item for sublist in groundeds for item in sublist]

    for pair in all_pairs: 
        
        if pair[0] in flat_gr_list and pair[1] not in flat_gr_list:
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, y, x): 
                    non = non + 1
                    j = j + 1
                else:
                    return False         
                
        elif pair[0] not in flat_gr_list and pair[1] in flat_gr_list: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, x, y): 
                    j = j + 1
                else:
                    return False
        else :
            j = j + 1

    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None

#stable-co
def stco():
    global stables

    j = 0 
    non = 0

    flat_st_list = [item for sublist in stables for item in sublist]

    for pair in all_pairs: 
        if pair[0] in flat_st_list and pair[1] not in flat_st_list:
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, y, x): 
                    non = non + 1
                    j = j + 1
                else:
                    return False         
                
        elif pair[0] not in flat_st_list and pair[1] in flat_st_list: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, x, y): 
                    non = non + 1
                    j = j + 1
                else:
                    return False
        else : 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None

#preferred-co
def prco():
    global prefereds

    j = 0 
    non = 0

    flat_pr_list = [item for sublist in prefereds for item in sublist]

    for pair in all_pairs: 
        if pair[0] in flat_pr_list and pair[1] not in flat_pr_list:
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, y, x): 
                    non = non + 1
                    j = j + 1
                else:
                    return False         
                
        elif pair[0] not in flat_pr_list and pair[1] in flat_pr_list: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else :
                if nx.has_path(CRG, x, y): 
                    non = non + 1
                    j = j + 1
                else:
                    return False
        else : 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None

#weakly naive support
def naweak():
    global naives

    j = 0 
    non = 0

    for pair in all_pairs:
        list1 = [] 
        list2 = [] 

        for naive_set in naives:
            if pair[0] in naive_set and pair[1] in naive_set:
                list1.append(naive_set)
                list2.append(naive_set)
            elif pair[0] in naive_set and pair[1] not in naive_set:
                list1.append(naive_set)
            elif pair[0] not in naive_set and pair[1] in naive_set:
                list2.append(naive_set)
            else:
                pass
        
        ainc = all(sublist in list2 for sublist in list1) 
        cina = all(sublist in list1 for sublist in list2)

        if list1 == []:
            ainc = False
        if list2 == []:
            cina = False
        
        if ainc and cina: 
            
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                non = non + 1
                j = j + 1
            else :
                return False         
                
        elif ainc and not cina: 
            
            if nx.has_path(RG, pair[0], pair[1]): 
                non = non + 1
                j = j + 1
            else: 
                return False
            
        elif not ainc and cina: 
            
            if nx.has_path(RG, pair[1], pair[0]): 
                non = non + 1
                j = j + 1
            else: 
                return False
            
        else: 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None

#weakly admissible support
def admweak():
    global admissibles

    j = 0 
    non = 0

    for pair in all_pairs: 
        list1 = [] 
        list2 = [] 

        for adm_set in admissibles:
            if pair[0] in adm_set and pair[1] in adm_set:
                list1.append(adm_set)
                list2.append(adm_set)
            elif pair[0] in adm_set and pair[1] not in adm_set:
                list1.append(adm_set)
            elif pair[0] not in adm_set and pair[1] in adm_set:
                list2.append(adm_set)
            else:
                pass
        
        ainc = all(sublist in list2 for sublist in list1) 
        cina = all(sublist in list1 for sublist in list2)

        if list1 == []:
            ainc = False
        if list2 == []:
            cina = False
        
        if ainc and cina: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                non = non + 1
                j = j + 1
            else :
                return False         
                
        elif ainc and not cina:
            
            if nx.has_path(RG, pair[0], pair[1]): 
                non = non + 1
                j = j + 1
            else: 
                return False
            
        elif not ainc and cina: 
            
            if nx.has_path(RG, pair[1], pair[0]): 
                non = non + 1
                j = j + 1
            else: 
                return False
            
        else: 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None

#weakly complete support
def coweak():
    global completes

    j = 0 
    non = 0

    for pair in all_pairs: 
        list1 = [] 
        list2 = [] 

        for co_set in completes:
            if pair[0] in co_set and pair[1] in co_set:
                list1.append(co_set)
                list2.append(co_set)
            elif pair[0] in co_set and pair[1] not in co_set:
                list1.append(co_set)
            elif pair[0] not in co_set and pair[1] in co_set:
                list2.append(co_set)
            else:
                pass
        
        ainc = all(sublist in list2 for sublist in list1) 
        cina = all(sublist in list1 for sublist in list2)

        if list1 == []:
            ainc = False
        if list2 == []:
            cina = False
        
        if ainc and cina:
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                non = non + 1
                j = j + 1
            else :
                return False         
                
        elif ainc and not cina: 
            
            if nx.has_path(RG, pair[0], pair[1]): 
                non = non + 1
                j = j + 1
            else: 
                return False
            
        elif not ainc and cina: 
            
            if nx.has_path(RG, pair[1], pair[0]): 
                non = non + 1
                j = j + 1
            else: 
                return False
            
        else: 
            j = j + 1
    
    if j == len(all_pairs) :
        if non > 0 :
            return True
        else:
            return None

#weakly grounded support
def grweak():
    global groundeds

    j = 0 
    non = 0

    for pair in all_pairs: 
        list1 = [] 
        list2 = [] 

        for gr_set in groundeds:
            if pair[0] in gr_set and pair[1] in gr_set:
                list1.append(gr_set)
                list2.append(gr_set)
            elif pair[0] in gr_set and pair[1] not in gr_set:
                list1.append(gr_set)
            elif pair[0] not in gr_set and pair[1] in gr_set:
                list2.append(gr_set)
            else:
                pass
        
        ainc = all(sublist in list2 for sublist in list1) 
        cina = all(sublist in list1 for sublist in list2)

        if list1 == []:
            ainc = False
        if list2 == []:
            cina = False
        
        if ainc and cina: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                non = non + 1
                j = j + 1
            else :
                return False         
                
        elif ainc and not cina:
            
            if nx.has_path(RG, pair[0], pair[1]): 
                non = non + 1
                j = j + 1
            else: 
                return False
            
        elif not ainc and cina: 
            
            if nx.has_path(RG, pair[1], pair[0]): 
                non = non + 1
                j = j + 1
            else: 
                return False
            
        else: 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None

#weakly stable support
def stweak():
    global stables

    j = 0 
    non = 0

    for pair in all_pairs: 
        list1 = [] 
        list2 = [] 

        for st_set in stables:
            if pair[0] in st_set and pair[1] in st_set:
                list1.append(st_set)
                list2.append(st_set)
            elif pair[0] in st_set and pair[1] not in st_set:
                list1.append(st_set)
            elif pair[0] not in st_set and pair[1] in st_set:
                list2.append(st_set)
            else:
                pass

        ainc = all(sublist in list2 for sublist in list1)
        cina = all(sublist in list1 for sublist in list2)

        if list1 == []:
            ainc = False
        if list2 == []:
            cina = False
        
        if ainc and cina: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                non = non + 1
                j = j + 1
            else :
                return False         
                
        elif ainc and not cina: 
            
            if nx.has_path(RG, pair[0], pair[1]): 
                non = non + 1
                j = j + 1
            else: 
                return False
            
        elif not ainc and cina: 
            
            if nx.has_path(RG, pair[1], pair[0]): 
                non = non + 1
                j = j + 1
            else: 
                return False
            
        else: 
            j = j + 1

    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None

#weakly preferred support
def prweak():
    global prefereds

    j = 0 
    non = 0

    for pair in all_pairs: 
        list1 = [] 
        list2 = [] 

        for pr_set in prefereds:
            if pair[0] in pr_set and pair[1] in pr_set:
                list1.append(pr_set)
                list2.append(pr_set)
            elif pair[0] in pr_set and pair[1] not in pr_set:
                list1.append(pr_set)
            elif pair[0] not in pr_set and pair[1] in pr_set:
                list2.append(pr_set)
            else:
                pass
        
        ainc = all(sublist in list2 for sublist in list1) 
        cina = all(sublist in list1 for sublist in list2)

        if list1 == []:
            ainc = False
        if list2 == []:
            cina = False

        
        if ainc and cina: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                non = non + 1
                j = j + 1
            else :
                return False         
                
        elif ainc and not cina: 
            
            if nx.has_path(RG, pair[0], pair[1]): 
                non = non + 1
                j = j + 1
            else: 
                return False
            
        elif not ainc and cina: 
            
            if nx.has_path(RG, pair[1], pair[0]): 
                non = non + 1
                j = j + 1
            else: 
                return False
            
        else: 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None

#strongly naive support
def nastrong():
    global naives

    j = 0 
    non = 0

    
    for pair in all_pairs: 
        list1 = [] 
        list2 = [] 

        for naive_set in naives:
            if pair[0] in naive_set and pair[1] in naive_set:
                list1.append(naive_set)
                list2.append(naive_set)
            elif pair[0] in naive_set and pair[1] not in naive_set:
                list1.append(naive_set)
            elif pair[0] not in naive_set and pair[1] in naive_set:
                list2.append(naive_set)
            else:
                pass

        list3 = [[elem for elem in sublist if elem != pair[0]] for sublist in list1]
        list4 = [[elem for elem in sublist if elem != pair[1]] for sublist in list2]
        
        csupa = True 
        for a in list3 :
            t = 0
            for c in list2:
                if set(c) <= set(a) :
                    break 
                else: 
                    t = t + 1
            if t == len(list2):
                csupa = False 
                break 
        if list3 == []:
            csupa = False
            
        asupc = True 
        for c in list4 :
            t = 0
            for a in list1: 
                if set(a) <= set(c) :
                    break 
                else: 
                    t = t + 1
            if t == len(list1):
                asupc = False 
                break 
        if list4 == []:
            asupc = False
        
        if asupc: 
            if csupa : 
                return "error"
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else:
                if nx.has_path(CRG, y, x): 
                    non = non + 1
                    j = j + 1
                else: 
                    return False   
                
        elif csupa: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else:
                if nx.has_path(CRG, x, y): 
                    non = non + 1
                    j = j + 1
                else: 
                    return False    
        else: 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None

#strongly admissible support
def admstrong():
    global admissibles

    j = 0 
    non = 0

    for pair in all_pairs: 
        
        list1 = [] 
        list2 = [] 

        for adm_set in admissibles:
            if pair[0] in adm_set and pair[1] in adm_set:
                list1.append(adm_set)
                list2.append(adm_set)
            elif pair[0] in adm_set and pair[1] not in adm_set:
                list1.append(adm_set)
            elif pair[0] not in adm_set and pair[1] in adm_set:
                list2.append(adm_set)
            else:
                pass

        list3 = [[elem for elem in sublist if elem != pair[0]] for sublist in list1]
        list4 = [[elem for elem in sublist if elem != pair[1]] for sublist in list2]
        
        csupa = True 
        for a in list3 :
            t = 0
            for c in list2:
                if set(c) <= set(a) :
                    break 
                else: 
                    t = t + 1
            if t == len(list2):
                csupa = False 
                break 
        if list3 == []:
            csupa = False
            
        asupc = True 
        for c in list4 :
            t = 0
            for a in list1: 
                if set(a) <= set(c) :
                    break 
                else: 
                    t = t + 1
            if t == len(list1):
                asupc = False 
                break 
        if list4 == []:
            asupc = False
        
        if asupc: 
            if csupa : 
                return "error"
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else:
                if nx.has_path(CRG, y, x): 
                    non = non + 1
                    j = j + 1
                else:
                    return False   
                
        elif csupa: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else:
                if nx.has_path(CRG, x, y): 
                    non = non + 1
                    j = j + 1
                else: 
                    return False    
        else: 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None

#strongly complete support
def costrong():
    global completes

    j = 0 
    non = 0

    for pair in all_pairs: 
        list1 = [] 
        list2 = [] 

        for co_set in completes:
            if pair[0] in co_set and pair[1] in co_set:
                list1.append(co_set)
                list2.append(co_set)
            elif pair[0] in co_set and pair[1] not in co_set:
                list1.append(co_set)
            elif pair[0] not in co_set and pair[1] in co_set:
                list2.append(co_set)
            else:
                pass

        list3 = [[elem for elem in sublist if elem != pair[0]] for sublist in list1]
        list4 = [[elem for elem in sublist if elem != pair[1]] for sublist in list2]
        
        csupa = True 
        for a in list3 :
            t = 0
            for c in list2:
                if set(c) <= set(a) :
                    break 
                else: 
                    t = t + 1
            if t == len(list2):
                csupa = False 
                break 
        if list3 == []:
            csupa = False
            
        asupc = True 
        for c in list4 :
            t = 0
            for a in list1: 
                if set(a) <= set(c) :
                    break
                else: 
                    t = t + 1
            if t == len(list1):
                asupc = False 
                break 
        if list4 == []:
            asupc = False
        
        if asupc: 
            if csupa :
                return "error"
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else:
                if nx.has_path(CRG, y, x): 
                    non = non + 1
                    j = j + 1
                else: 
                    return False   
                
        elif csupa: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else:
                if nx.has_path(CRG, x, y): 
                    non = non + 1
                    j = j + 1
                else: 
                    return False    
        else: 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None

#strongly grounded support
def grstrong():
    global groundeds

    j = 0 
    non = 0

    for pair in all_pairs: 
        list1 = [] 
        list2 = []

        for gr_set in groundeds:
            if pair[0] in gr_set and pair[1] in gr_set:
                list1.append(gr_set)
                list2.append(gr_set)
            elif pair[0] in gr_set and pair[1] not in gr_set:
                list1.append(gr_set)
            elif pair[0] not in gr_set and pair[1] in gr_set:
                list2.append(gr_set)
            else:
                pass

        list3 = [[elem for elem in sublist if elem != pair[0]] for sublist in list1]
        list4 = [[elem for elem in sublist if elem != pair[1]] for sublist in list2]
        
        csupa = True 
        for a in list3 :
            t = 0
            for c in list2:
                if set(c) <= set(a) :
                    break 
                else: 
                    t = t + 1
            if t == len(list2):
                csupa = False 
                break 

        if list3 == []:
            csupa = False
            
        asupc = True 
        for c in list4 :
            t = 0
            for a in list1: 
                if set(a) <= set(c) :
                    break 
                else: 
                    t = t + 1
            if t == len(list1):
                asupc = False 
                break 
        if list4 == []:
            asupc = False

        if asupc: 
            if csupa : 
                return "error"
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else:
                if nx.has_path(CRG, y, x): 
                    non = non + 1
                    j = j + 1
                else: 
                    return False   
                
        elif csupa: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else:
                if nx.has_path(CRG, x, y): 
                    non = non + 1
                    j = j + 1
                else: 
                    return False    
        else: 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None

#strongly stable support
def ststrong():
    global stables

    j = 0 
    non = 0

    for pair in all_pairs: 
        list1 = [] 
        list2 = [] 

        for st_set in stables:
            if pair[0] in st_set and pair[1] in st_set:
                list1.append(st_set)
                list2.append(st_set)
            elif pair[0] in st_set and pair[1] not in st_set:
                list1.append(st_set)
            elif pair[0] not in st_set and pair[1] in st_set:
                list2.append(st_set)
            else:
                pass

        list3 = [[elem for elem in sublist if elem != pair[0]] for sublist in list1]
        list4 = [[elem for elem in sublist if elem != pair[1]] for sublist in list2]
        
        csupa = True 
        for a in list3 :
            t = 0
            for c in list2:
                if set(c) <= set(a) :
                    break 
                else: 
                    t = t + 1
            if t == len(list2):
                csupa = False 
                break 
        if list3 == []:
            csupa = False
            
        asupc = True 
        for c in list4 :
            t = 0
            for a in list1: 
                if set(a) <= set(c) :
                    break 
                else: 
                    t = t + 1
            if t == len(list1):
                asupc = False 
                break 
        if list4 == []:
            asupc = False
        
        if asupc: 
            if csupa : 
                return "error"
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y:
                return False
            else:
                if nx.has_path(CRG, y, x): 
                    non = non + 1
                    j = j + 1
                else: 
                    return False   
                
        elif csupa: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else:
                if nx.has_path(CRG, x, y): 
                    non = non + 1
                    j = j + 1
                else: 
                    return False    
        else: 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None

#strongly preferred support
def prstrong():
    global prefereds

    j = 0 
    non = 0

    for pair in all_pairs: 
        list1 = [] 
        list2 = [] 

        for pr_set in prefereds:
            if pair[0] in pr_set and pair[1] in pr_set:
                list1.append(pr_set)
                list2.append(pr_set)
            elif pair[0] in pr_set and pair[1] not in pr_set:
                list1.append(pr_set)
            elif pair[0] not in pr_set and pair[1] in pr_set:
                list2.append(pr_set)
            else:
                pass

        list3 = [[elem for elem in sublist if elem != pair[0]] for sublist in list1]
        list4 = [[elem for elem in sublist if elem != pair[1]] for sublist in list2]
        
        csupa = True 
        for a in list3 :
            t = 0
            for c in list2:
                if set(c) <= set(a) :
                    break 
                else: 
                    t = t + 1
            if t == len(list2):
                csupa = False 
                break 
        if list3 == []:
            csupa = False
            
        asupc = True 
        for c in list4 :
            t = 0
            for a in list1:
                if set(a) <= set(c) :
                    break
                else: 
                    t = t + 1
            if t == len(list1):
                asupc = False 
                break 
        if list4 == []:
            asupc = False
        
        if asupc: 
            if csupa : 
                return "error"
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else:
                if nx.has_path(CRG, y, x): 
                    non = non + 1
                    j = j + 1
                else: 
                    return False   
                
        elif csupa: 
            x = find_key(scc_dict, pair[0]) 
            y = find_key(scc_dict, pair[1])
            if x == y: 
                return False
            else:
                if nx.has_path(CRG, x, y): 
                    non = non + 1
                    j = j + 1
                else: 
                    return False    
        else: 
            j = j + 1
    
    if j == len(all_pairs) : 
        if non > 0 :
            return True
        else:
            return None

#++DB
def SplusDB():

    base_target = find_key(copy_mapping, plus_target_node)

    x = find_key(scc_dict, base_target)
    y = find_key(scc_dict, plus_target_node) 

    if x == y: 
        return False
    else:
        if nx.has_path(CRG, x, y):
            return True
        else:
            return False

#+DB
def plusDB():
    
    base_target = find_key(copy_mapping, plus_target_node)

    if base_target in unattacked:
        return None
    else:
        x = find_key(scc_dict, base_target)
        y = find_key(scc_dict, plus_target_node) 

        if x == y: 
            return False
        else:
            if nx.has_path(CRG, x, y):
                return True
            else:
                return False
            
#↑DB
def IDB():

    j = 0
    
    base_target = find_key(copy_mapping, plus_target_node)
    list1 = []

    for argument in base_arguments : 
        if base_target in find_root(argument)[1] and base_target not in find_root(argument)[0]:
            
            list1.append(argument) 

    for argument in list1:
        x = find_key(scc_dict, argument)
        y = find_key(scc_dict, copy_mapping[argument]) 
        
        if  x == y: 
            return False
        else:
            if nx.has_path(CRG, y, x):
                j = j + 1
            else:
                return False
    if j > 0 :
        return True
    else:
        return None


#↑AB
def IAB():

    j = 0
    
    base_target = find_key(copy_mapping, plus_target_node)
    list1 = []

    for argument in base_arguments : 
        if base_target in find_root(argument)[0] and base_target not in find_root(argument)[1]:
            list1.append(argument) 

    for argument in list1:
        x = find_key(scc_dict, argument)
        y = find_key(scc_dict, copy_mapping[argument])
        
        if  x == y: 
            return False
        else:
            if nx.has_path(CRG, x, y):
                j = j + 1
            else:
                return False
    if j > 0 :
        return True
    else:
        return None

#+AB
def plusAB():
    base_target = find_key(copy_mapping, minus_target_node)

    x = find_key(scc_dict, base_target)
    y = find_key(scc_dict, minus_target_node) 

    if x == y: 
        return False
    else:
        if nx.has_path(CRG, y, x):
            return True
        else:
            return False


#for check one property
global check_property_list
check_property_list = [VP, SC, CP, QP, DP, CT, SCT, NaE, AvsFD, DDP, OE, AE, PR, MVP, MDP, MCP, MQP, MDDP, COM21, naco, admco, grco, stco, naweak, admweak, coweak, grweak, stweak, prweak, admstrong, costrong, SplusDB, plusDB, IDB, IAB, plusAB]


"""
Up to this point is the definition of the properties. 
Survey function is a function that uses these functions to apply the properties to AF and retains their outputs.
"""



def Survey():

    global arguments
    global attacks
    global ranks
    global AG
    global RG
    global CRG
    global scc_dict
    global all_pairs
    global unattacked
    global node_defenders

    global all_cycles
    global odd_all_cycles
    global flat_odd_all_cycles
    global even_all_cycles
    global flat_even_all_cycles

    global copy_mapping
    global path_length
    global baseAG
    global base_arguments
    global base_attacks
    global plus_target_node
    global minus_target_node

    global conflicts 
    global naives
    global admissibles 
    global completes 
    global prefereds 
    global groundeds 
    global stables 

    global n_gr
    global p_gr
    global m_gr
    global adding

    global num_to_one
    global Table_A

    #make random graph
    random_graph_a = generate_random_graph()
    random_graph_a = add_self_loops_random(random_graph_a, 0.1)


    #extract arguments and attacks
    arguments = list(random_graph_a.nodes())
    attacks = list(random_graph_a.edges)
    attacks = [list(edge) for edge in attacks]

    #make Argumentation framework
    AG = random_graph_a

    if n_gr == 1: #normal graph

        #fix graph
        if fix_sign:
            if fix_situation == 0:
                arguments = fix_arguments
                attacks = fix_attacks

                # Check if all elements in attacks are in arguments
                for attacky in attacks:
                    for argy in attacky:
                        if argy not in arguments:
                            print(f"Error: Argument '{argy}' in attacks is not in the list of arguments.")
                            sys.exit(1) 

                random_graph_a = nx.DiGraph()
                random_graph_a.add_nodes_from(arguments)
                for attack in attacks:
                    random_graph_a.add_edge(attack[0], attack[1])
                AG = random_graph_a
            else:
                return

        total_ranks = create_combinations_total(arguments)
        add_reverse_to_original(total_ranks)
        ranks = total_ranks

        #fix also rank
        if fix_sign:
            if fix_situation == 0:
                if fix_rank_sign:
                    ranks = fix_rank

                    #Check if the sets of arguments in 'arguments' and 'ranks' match
                    ranked_arguments_set = set(arg for pair in ranks for arg in pair)
                    arguments_set = set(arguments)
                    if ranked_arguments_set != arguments_set:
                        print("Error: The sets of arguments in 'arguments' and 'rank' do not match.")
                        print(f"'arguments' set: {arguments_set}")
                        print(f"'rank' set: {ranked_arguments_set}")
                        sys.exit(1)

        adding = 0

    elif p_gr == 1: #plus graph

        if fix_sign:
            if fix_situation == 1:
                arguments = fix_arguments
                attacks = fix_attacks

                # Check if all elements in attacks are in arguments
                for attacky in attacks:
                    for argy in attacky:
                        if argy not in arguments:
                            print(f"Error: Argument '{argy}' in attacks is not in the list of arguments.")
                            sys.exit(1) 

                random_graph_a = nx.DiGraph()
                random_graph_a.add_nodes_from(arguments)
                for attack in attacks:
                    random_graph_a.add_edge(attack[0], attack[1])
                AG = random_graph_a
            else:
                return
        
        baseAG = AG
        base_arguments = arguments
        base_attacks = attacks

        #defense branch
        new_plus_AG = duplicate_and_rename(AG, 1) 

        path_length = 0

        plus_target_node = random.choice(list(new_plus_AG.nodes))

        add_even_length_path(new_plus_AG, plus_target_node) 

        AGplus = union_preserve_names(AG, new_plus_AG)

        plus_arguments = list(AGplus.nodes())
        plus_attacks = list(AGplus.edges)
        plus_attacks = [list(edge) for edge in plus_attacks]

        #total order
        total_ranks = create_combinations_total(plus_arguments)
        add_reverse_to_original(total_ranks)
        plus_ranks = total_ranks

        AG = AGplus
        arguments = plus_arguments
        attacks = plus_attacks
        ranks = plus_ranks

        #fix also rank
        if fix_sign:
            if fix_situation == 1:
                if fix_rank_sign:
                    ranks = fix_rank

                    #Check if the sets of arguments in 'arguments' and 'ranks' match
                    ranked_arguments_set = set(arg for pair in ranks for arg in pair)
                    arguments_set = set(arguments)
                    if ranked_arguments_set != arguments_set:
                        print("Error: The sets of arguments in 'arguments' and 'rank' do not match.")
                        print(f"'arguments' set: {arguments_set}")
                        print(f"'rank' set: {ranked_arguments_set}")
                        sys.exit(1)
                    
        adding = 1

    elif m_gr == 1:

        if fix_sign:
            if fix_situation == 2:
                arguments = fix_arguments
                attacks = fix_attacks

                # Check if all elements in attacks are in arguments
                for attacky in attacks:
                    for argy in attacky:
                        if argy not in arguments:
                            print(f"Error: Argument '{argy}' in attacks is not in the list of arguments.")
                            sys.exit(1) 

                random_graph_a = nx.DiGraph()
                random_graph_a.add_nodes_from(arguments)
                for attack in attacks:
                    random_graph_a.add_edge(attack[0], attack[1])
                AG = random_graph_a
            else:
                return

        baseAG = AG
        base_arguments = arguments
        base_attacks = attacks

        new_minus_AG = duplicate_and_rename(AG, 1) 

        path_length = 0

        minus_target_node = random.choice(list(new_minus_AG.nodes))

        add_odd_length_path(new_minus_AG, minus_target_node) 

        AGminus = union_preserve_names(AG, new_minus_AG)

        minus_arguments = list(AGminus.nodes())
        minus_attacks = list(AGminus.edges)
        minus_attacks = [list(edge) for edge in minus_attacks]

        total_ranks = create_combinations_total(minus_arguments)
        add_reverse_to_original(total_ranks)
        minus_ranks = total_ranks

        AG = AGminus
        arguments = minus_arguments
        attacks = minus_attacks
        ranks = minus_ranks

        #fix also rank
        if fix_sign:
            if fix_situation == 2:
                if fix_rank_sign:
                    ranks = fix_rank

                    #Check if the sets of arguments in 'arguments' and 'ranks' match
                    ranked_arguments_set = set(arg for pair in ranks for arg in pair)
                    arguments_set = set(arguments)
                    if ranked_arguments_set != arguments_set:
                        print("Error: The sets of arguments in 'arguments' and 'rank' do not match.")
                        print(f"'arguments' set: {arguments_set}")
                        print(f"'rank' set: {ranked_arguments_set}")
                        sys.exit(1)

        adding = 2

    else:
        print("error")

    #ranking graph
    RG = nx.DiGraph()
    RG.add_nodes_from(arguments)
    for rank in ranks:
        RG.add_edge(rank[0], rank[1])#

    # Strongly connected component decomposition
    scc_list = strongly_connected_components(RG) 

    # condensed ranking graph
    CRG = condensed_graph(RG, scc_list)

    all_pairs = nC2(arguments)

    unattacked = find_unattacked_nodes(arguments, attacks)

    node_defenders = find_node_defenders(attacks)

    scc_dict = {i: list(scc) for i, scc in enumerate(scc_list)}

    all_cycles = list(nx.simple_cycles(AG)) 
    odd_all_cycles = [cyc for cyc in all_cycles if len(cyc) % 2 == 1] 
    even_all_cycles = [cyc for cyc in all_cycles if len(cyc) % 2 == 0] 
    flat_odd_all_cycles = [item for odd_cyc in odd_all_cycles for item in odd_cyc] 


    if Tot() == False:
        print("total_error")
        return
    
    #efficiently check AF which certain property satisfy
    if check_property_sign:
        if check_property_num <= 18:
            if check_property_list[check_property_num]() == False:
                return      
        elif 31 <= check_property_num <= 34:
            if adding != 1:
                return
            else:
                if check_property_list[check_property_num]() == False:
                    return      
        elif check_property_num == 35:
            if adding != 2:
                return
            else:
                if check_property_list[check_property_num]() == False:
                    return  
    
    conflicts = sorted(newconflict([[]], arguments,[[]], 0))
    naives = fnaive()
    admissibles = fadmissible()
    completes = fcomplete()
    prefereds = fprefered()
    groundeds = fgrounded()
    stables = fstable()

    #efficiently
    if check_property_sign:
        if 19 <= check_property_num <= 30:
            if check_property_list[check_property_num]() == False:
                return 

    if adding == 0: #normal graph, +DB,  … are not defined.
        table_list = [VP(), SC(),CP(),QP(),DP(), CT(),SCT(),NaE(), AvsFD(),DDP(),OE(),AE(),PR(),MVP(),MDP(),MCP(),MQP(),MDDP(),COM21(),naco(),admco(),grco(),stco(),naweak(),admweak(),coweak(),grweak(),stweak(),prweak(),admstrong(),costrong(),"undefined","undefined","undefined","undefined","undefined"]
    elif adding == 1: #plus graph, +AB is not defined.
        table_list = [VP(), SC(),CP(),QP(),DP(), CT(),SCT(),NaE(), AvsFD(),DDP(),OE(),AE(),PR(),MVP(),MDP(),MCP(),MQP(),MDDP(),COM21(),naco(),admco(),grco(),stco(),naweak(),admweak(),coweak(),grweak(),stweak(),prweak(),admstrong(),costrong(),SplusDB(),plusDB(),IDB(),IAB(),"undefined"]
    elif adding == 2: #minus graph, +DB,  … are not defined.
        table_list = [VP(), SC(),CP(),QP(),DP(), CT(),SCT(),NaE(), AvsFD(),DDP(),OE(),AE(),PR(),MVP(),MDP(),MCP(),MQP(),MDDP(),COM21(),naco(),admco(),grco(),stco(),naweak(),admweak(),coweak(),grweak(),stweak(),prweak(),admstrong(),costrong(),"undefined","undefined","undefined","undefined",plusAB()]
    else:
        print("adding_error")
        
    # if list is new, store it
    if table_list in [entry[0] for entry in Table_A]:
        return
    else:
        Table_A.append([table_list, [arguments, attacks, ranks]])
    

#main
def main(): 
    global n_gr
    global m_gr
    global p_gr
    global Previous_num
    global Matrix_A
    global Graph_Matrix_A
    global UntilT
    global Table_A
    # from config
    print("num_to_one:", num_to_one)
    print("num_node:", num_node)
    print("per_edge:", per_edge)
    print("num_check:", num_check)
    print("num_finish:", num_finish)

    for s in range(num_finish): #all steps

        if s % 10000 == 0: 
            print(s)

        if s % num_check == 0 and s > 0 : #period steps

            CheckOut_A = [] #Store the dependencies at that point.
            
            for i in range(num_to_one): 
                
                if Previous_num != num_to_one :
                    #for tab in Table_A: #If the number on the left-hand side is changed, recheck everything.
                    #    Comb = gene_com(tab[0], i+2)
                    #    Comb_n = gene_com(list_len_property, i+2) 
                    #    for z in range(len(Comb)):
                    #        if False not in Comb[z][:i+1] and ("undefined" not in Comb[z][:i+1]) and Comb[z][i+1] == False: #True(None),True(None)… → False
                    #            if get_value_from_listZ(Comb_n[z], Matrix_A[i]) == 0:
                    #                assign_value_to_listZ(Comb_n[z], Matrix_A[i], 1) 
                    #                assign_value_to_listZ(Comb_n[z], Graph_Matrix_A[i], tab[1]) 

                    for tab in Table_A: #If the number on the left-hand side is changed, recheck everything.
                        #Comb = gene_com(tab[0], i+2)
                        #Comb_n = gene_com(list_len_property, i+2) 
                        for com in Dep[i]:
                            Ta = [tab[0][r] for r in com]
                            if False not in Ta[:i+1] and ("undefined" not in Ta[:i+1]) and Ta[i+1] == False: #True(None),True(None)… → False
                                if get_value_from_listZ(com, Matrix_A[i]) == 0:
                                    assign_value_to_listZ(com, Matrix_A[i], 1) 
                                    assign_value_to_listZ(com, Graph_Matrix_A[i], tab[1]) 
                                    Dep[i].remove(com)

                else :
                    #for tab in Table_A[UntilT:]: #only new stored lists
                    #    Comb = gene_com(tab[0], i+2)
                    #    Comb_n = gene_com(list_len_property, i+2) 
                    #    for z in range(len(Comb)):
                    #        if False not in Comb[z][:i+1] and ("undefined" not in Comb[z][:i+1]) and Comb[z][i+1] == False: 
                    #            if get_value_from_listZ(Comb_n[z], Matrix_A[i]) == 0:
                    #                assign_value_to_listZ(Comb_n[z], Matrix_A[i], 1) 
                    #                assign_value_to_listZ(Comb_n[z], Graph_Matrix_A[i], tab[1]) 

                    for tab in Table_A[UntilT:]: #If the number on the left-hand side is changed, recheck everything.
                        #Comb = gene_com(tab[0], i+2)
                        #Comb_n = gene_com(list_len_property, i+2) 
                        for com in Dep[i]:
                            Ta = [tab[0][r] for r in com]
                            if False not in Ta[:i+1] and ("undefined" not in Ta[:i+1]) and Ta[i+1] == False: #True(None),True(None)… → False
                                if get_value_from_listZ(com, Matrix_A[i]) == 0:
                                    assign_value_to_listZ(com, Matrix_A[i], 1) 
                                    assign_value_to_listZ(com, Graph_Matrix_A[i], tab[1]) 
                                    Dep[i].remove(com)

            UntilT = len(Table_A) #Record up to which list has been checked.
            print("Length_of_Table", UntilT)
            Previous_num = num_to_one
            print("remains", len(Dep[0]) + len(Dep[1]))

            for i in range(num_to_one):
                Comb_n = gene_com(list_len_property, i+2)
                for z in Comb_n:
                    if get_value_from_listZ(z, Matrix_A[i]) == 0 and len(set(z)) == len(z): 
                        fil = 0
                        for t in range(len(CheckOut_A)):
                            if is_sublist_in_order(CheckOut_A[t], z) or ((set(z[:-1]) == set(CheckOut_A[t][:-1])) and (z[-1] == CheckOut_A[t][-1])): #filtering
                                fil = 1
                        for c in checkout_c_2:
                            if set(c) <= set(z):
                                fil = 1
                        if fil == 0:
                            CheckOut_A.append(z)
            
            print(CheckOut_A)

            for t in CheckOut_A: 
                for i, var_index in enumerate(t): 
                    print(variable_names[var_index], end=" ")
                    if i < len(t) - 2:
                        print("and", end=" ")
                    if i == len(t) - 2:
                        break
                print("→", variable_names[t[-1]])


            #store the survey results
            with open('Matrix_A.pickle', mode='wb') as fo:
                pickle.dump(Matrix_A, fo)
            with open('Graph_Matrix_A.pickle', mode='wb') as fo:
                pickle.dump(Graph_Matrix_A, fo)
            with open('Table_A.pickle', mode='wb') as fo:
                pickle.dump(Table_A, fo)
            with open('UntilT_A.pickle', mode='wb') as fo:
                pickle.dump(UntilT, fo)
            with open('Previous_num_A.pickle', mode='wb') as fo:
                pickle.dump(num_to_one, fo)
            with open('Dep.pickle', mode='wb') as fo:
                pickle.dump(Dep, fo)
            
        #Which graph to investigate. normal, adding defense branch, or add attack branch
        
        if s % 2 == 1:
            n_gr = 0
            p_gr = 1
            m_gr = 0
        elif s % 30 == 0:
            n_gr = 0
            p_gr = 0
            m_gr = 1
        else:
            n_gr = 1
            p_gr = 0
            m_gr = 0

        Survey()


if __name__ == "__main__":
    main()

#memo