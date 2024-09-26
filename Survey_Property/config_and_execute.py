# input_and_execute.py
import subprocess
import sys

"""
Provide the parameters as follows.
"""

#The number of properties on the left-hand side
num_to_one = 2
#The range of the number of nodes in the random graph
num_node = [3, 4]
#The range of the node generation probability in the random graph
per_edge = [0.1, 0.4]

#How often to save and output the results
num_check = 100000
#The number of steps for stopping
num_finish = 10000000

# VP = 0, SC = 1, … 
"""
property_list = ["VP", "SC","CP","QP","DP", "CT","SCT","NaE", "AvsFD","DDP","OE","AE","PR","MVP","MDP", 
                "MCP","MQP","MDDP","COM21","naco","admco","grco","stco","naweak","admweak","compweak","grweak", 
                "stweak","prweak","admstrong","compstrong","SplusDB","plusDB","IDB","IAB","plusAB"]
"""
#filtering
checkout_c_2 = [[0,26],[2,26],[3,26],[6,26], [0,25],[2,25],[3,25],[6,25], [0,28],[2,28],[3,28],[6,28],#like, VP and grweak
                [12,31],[12,32],[12,33],[12,34],[12,35],#+- and PR
                [31,35],[32,35],[33,35],[34,35],#+-
                [2,31],[2,32],#like, CP and plusDB
                [2,18],#like, CP and COM21
                [3,15],[2,3],#like, QP and MCP
                [2,16],#like, CP and MQP
                [0,31],[2,31],[3,31],[5,31],[6,31],[13,31],[15,31],[16,31],#like, MVP and splusDB
                [8,18],#like, AvsFD and COM21
                [26,31],[25,31],[28,31],#like, grweak and splusDB
                [26,29],[25,29],[28,29],#like, grweak and admstrong
                [15,20],[15,21],[2,20],[2,21],#like, MCP and grco,admco
                [12,26],[12,25],[12,28],[12,27],#like, PR and grweak,stweak
                [2,8],#like, CP and AvsFD
                [0,27],[2,27],[3,27],[6,27]#like, VP and stweak
                ]

"""
Check only AF that certain property satisfy
"""
#check only AF which certain property satisfy
#If you don't use this functionality, set check_property_sign = False
check_property_sign = False
check_property_num = 35

"""
Fix AF
"""
#If you don't use this functionality, set fix_sign = False
#If you also fix rank, set fix_rank_sign = True
#If Tot is not satisfied, print "total_error"
fix_sign = False
fix_situation = 0 # if normal graph; 0, defense branch added; 1, attack branch added; 2.
fix_arguments = ['a', 'b', 'c']
fix_attacks = [['b', 'a'], ['a', 'c']]

fix_rank_sign = False
fix_rank = [['b', 'a'], ['a', 'c']]


"""
Execute by the following code.
"""

if __name__ == "__main__":
    import subprocess
    subprocess.run(["python", "main.py"])