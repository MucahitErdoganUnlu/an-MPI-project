from mpi4py import MPI
import sys
import numpy as np

file_path = sys.argv[1]  # input file

number_of_machines = 0
number_of_cycles = 0
#wearout factors
wf = np.array([], dtype=int)
threshold = 0
#numpy array of machine ids
machine_ids = np.array([], dtype=int)
#the machines which are parents
parent_ids = []
#nunmpy array of initial products. even if its type is set to be int, strings still work
initial_products = np.array([], dtype=int)
#the mchines which are not parents of any machine
leaf_machine_ids = np.array([], dtype=int)
#dictionary with id of the parent as key, list of the children's ids as value
children_ranks = {}
#dictionary with rank as key, [id, parent_id, initial_operation] as value
rank_id_parent_op = {}
#dictionary with id of the child as key, id of the parent as value
parents_dict = {}
#dictionary with id of a machine as key, rank of the machine as value
ids_ranks = {}

with open(file_path, 'r') as file:
    #Read the number of machines
    line = file.readline().strip()
    number_of_machines = int(line)

    #spawn the machines
    sub_comm = MPI.COMM_SELF.Spawn(sys.executable, args=['p2.py'], maxprocs=number_of_machines)
    #merge the spawns with this communication world. the correspondent in the other file is set as True
    common_comm=sub_comm.Merge(False)
    #send the id of the terminal machine
    common_comm.send(1, dest = 1)
    #set the rank of the terminal machine as one
    ids_ranks[1] = 1
    #add the terminal machine's id
    machine_ids = np.append(machine_ids, 1)

    # Read number of production cycles
    line = file.readline().strip()
    number_of_cycles = int(line)
    # Read the wear factors
    line = file.readline().strip().split()
    wf = np.array([int(element) for element in line])
    # Read the threshold
    line = file.readline().strip()
    threshold = int(line)

    for i in range(2, number_of_machines+1):
        # read the machine id, parent id, initial operation
        line = file.readline().strip().split()
        # store them in the rank_id_parent_op dictionary
        rank_id_parent_op[i] = [int(line[0]), int(line[1]), line[2]]
        # store the ranks of the children of the current machine's id. It is important to use while sending messages
        if(int(line[1]) in children_ranks):
            children_ranks[int(line[1])].append(i)
        else:
            children_ranks[int(line[1])] = [i]
        #store the machine ids
        machine_ids = np.append(machine_ids, int(line[0]))
        #store machines' ids as keys, the ranks of the machines as values
        ids_ranks[int(line[0])] = i
        #store the machines' ids as keys, parent's ids as values
        parents_dict[int(line[0])] = int(line[1])
        #store the machines' ids which are parents
        parent_ids.append(int(line[1]))

    #if a machine id does not appear in the parent_ids, then it is a leaf machine
    for i in range(1, number_of_machines+1):
        if(i not in parent_ids):
            leaf_machine_ids = np.append(leaf_machine_ids, i)
    #read the initial products
    for line in file:
        initial_products = np.append(initial_products, line.strip())

for i in range(2, number_of_machines+1):
    # send the machine id
    common_comm.send(rank_id_parent_op[i][0], dest = i)
    # send the parent id
    common_comm.send(rank_id_parent_op[i][1], dest = i)
    # send the initial operation
    common_comm.send(rank_id_parent_op[i][2], dest = i)

# sort the ids
machine_ids.sort()

for i in machine_ids:
    #retrieve the rank of the machine
    rank = ids_ranks[i]
    if(i in leaf_machine_ids):
        # leaf nodes have no child, send an empty list
        common_comm.send([], dest = rank)
    else:
        # send the children list
        common_comm.send(children_ranks[i], dest = rank)

#sorted machine ids are useful for sending the initial products to the machine ids in ascending order,
#and not in the order they are appearing in the adjancency list
for _ in range(number_of_cycles):
    ix = 0
    for i in machine_ids:
        if i in leaf_machine_ids:
            #retrieve the rank of the machine
            rank = ids_ranks[i]
            # send the next product to the leaf machine that has the lowest id
            common_comm.send(initial_products[ix], dest = rank)
            ix += 1
#wearout factors are common for all processes. therefore, broadcast is useful
common_comm.Bcast([wf, MPI.INT], root=0)
#threshold is common for all processes. therefore, broadcast is useful
common_comm.bcast(threshold, root=0)
#number of cycles is common for all processes. therefore, broadcast is useful
common_comm.bcast(number_of_cycles, root=0)
#rank_id_parent_op dictionary is useful for determining the id associated with the rank
common_comm.bcast(rank_id_parent_op, root=0)
#wearout factors are common for all processes. therefore, broadcast is useful
common_comm.bcast(ids_ranks, root=0)

#store the maintenance logs in the dictionary. key is the id of the machine, value is the log
maintenance_dict = {}
with open(sys.argv[2], 'w') as file:
    for _ in range(number_of_cycles):
        # get the final product from the terminal machine number_of_cycles times and write to the file
        data = common_comm.recv(source = 1)
        file.write(data + '\n')
        #print(data)
        #check if there is a request waiting. If so, receive it. Iprobe is used to facilitate nonblocking
        while common_comm.Iprobe(source=MPI.ANY_SOURCE, tag=13):
            #receive the data
            data = common_comm.recv(source=MPI.ANY_SOURCE, tag=13)
            id_of_log = data.strip().split("-")[0]
            #add to the dictionary
            if(int(id_of_log) in maintenance_dict):
                maintenance_dict[int(id_of_log)].append(data.strip())
            else:
                maintenance_dict[int(id_of_log)] = [data.strip()]
#check if there is a request waiting. If so, receive it. Iprobe is used to facilitate nonblocking
while common_comm.Iprobe(source=MPI.ANY_SOURCE, tag=13):
    #receive the data
    data = common_comm.recv(source=MPI.ANY_SOURCE, tag=13)
    id_of_log = data.strip().split("-")[0]
    #add to the dictionary
    if(int(id_of_log) in maintenance_dict):
        maintenance_dict[int(id_of_log)].append(data.strip())
    else:
        maintenance_dict[int(id_of_log)] = [data.strip()]
#sort the maintenance log dictionary according to keys
sorted_by_keys = dict(sorted(maintenance_dict.items()))
# Traverse and write the values (ie. logs)
with open(sys.argv[2], 'a') as file:
    for key in sorted_by_keys.keys():
        value = sorted_by_keys[key]
        for log in value:
            file.write(log + '\n')
            #print(log)
"""
#testing purposes
with open(sys.argv[2], 'r') as output_file:
    with open("test_output.txt", 'r') as test_file:
        for output_line in output_file:
            test_line = test_file.readline()
            if(output_line.strip() != test_line.strip()):
                print("output", output_line.strip(), "test", test_line)
#testing purposes
with open("test_output.txt", 'r') as output_file:
    with open(sys.argv[2], 'r') as test_file:
        for output_line in output_file:
            test_line = test_file.readline()
            if(output_line.strip() != test_line.strip()):
                print("test", output_line.strip(), "output", test_line)
"""
