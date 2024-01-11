from mpi4py import MPI
import numpy as np

#get the communication from the parent
comm = MPI.Comm.Get_parent()
#merge with the parent's communication world. therefore, parent has the rank zero,
#and children's ranks range from 1 to number of machines
common_comm=comm.Merge(True)
#retrieve the machine id of the current rank
machine_id = common_comm.recv(source = 0)
parent_id = 0
initial_op = ""
initial_product = ""
wearout = 0
wf = np.empty(5, dtype = int)
threshold = 0

#operation functions for each operation.
#they are only called inside the operate() function
#returns the resulting product
def trim(product):
    if(len(product) > 2):
        return product[1:len(product)-1]
    return product
def reverse(product):
    return product[::-1]
def split(product):
    length = int(np.ceil(len(product)/2))
    return product[:length]
def chop(product):
    if(len(product) > 1):
        return product[:len(product)-1]
    return product
def enhance(product):
    first = product[0]
    last = product[len(product)-1]
    return first + product + last

#entry function for machines to operate.
#calls the appropriate operation function
#changes the next operation according to the current operation
#accumulates the wearout
#if the wearout is greater or equal to the threshold, calculate the cost and send the log to the control center
#return the resulting product
def operate(product):
    global wearout, initial_op
    conducted_op = initial_op
    temp = 0
    if(initial_op == "trim"):
        temp = wf[3]
        initial_op = "reverse"
        product = trim(product)
    elif(initial_op == "reverse"):
        temp = wf[1]
        initial_op = "trim"
        product = reverse(product)
    elif(initial_op == "split"):
        temp = wf[4]
        initial_op = "chop"
        product = split(product)
    elif(initial_op == "chop"):
        temp = wf[2]
        initial_op = "enhance"
        product = chop(product)
    elif(initial_op == "enhance"):
        temp = wf[0]
        initial_op = "split"
        product = enhance(product)
    wearout += temp
    if(wearout >= threshold):
        cost = (wearout - threshold + 1) * temp
        msg = f"{machine_id}-{cost}-{cycle}"
        #Iprobe is used in master to facilitate the nonblocking send
        common_comm.send(msg, dest = 0, tag = 13)
        wearout = 0
    return product

# process with rank 1 is the terminal machine however other than the terminal machine
# all machines need a parent id and an initial operation
if(common_comm.Get_rank() != 1):
    parent_id = common_comm.recv(source = 0)
    #print("parent id", parent_id)
    initial_op = common_comm.recv(source = 0)
    #print("op", initial_op)

#retrieve the list of ranks of the children of the current machine. leaf machines receive am empty list
children_ranks = common_comm.recv(source = 0)
child_count = len(children_ranks)

#retrieve wf as numpy array
common_comm.Bcast([wf, MPI.INT], root=0)
#retrieve threshold as an integer
threshold = common_comm.bcast(0, root=0)
#retrieve number of cycles as integer
number_of_cycles = common_comm.bcast(0, root=0)
#retrieve rank_id_parent_op as a dictionary
rank_id_parent_op = common_comm.bcast(0, root=0)
#retrieve ids_ranks as dictionary
ids_ranks = common_comm.bcast(0, root=0)

for cycle in range(1, number_of_cycles+1):
    #if the machine is a leaf, product comes from the control center
    #else it comes from the child machine
    if(child_count == 0):
        #retrieve the initial product from the control center for each cycle
        initial_product = common_comm.recv(source = 0)
        #since leaf nodes have no child, no need to add strings

        #change the initial product according to the current operation
        initial_product = operate(initial_product)
        #use blocking send to send the intermediate product to the parent machine
        common_comm.send(initial_product, dest = ids_ranks[parent_id]) # send the machine id
    else:
        #store the children's ids by using children_ranks and rank_id_parent_op dictionaries
        children_ids = []
        for child in children_ranks:
            children_ids.append(rank_id_parent_op[child][0])
        #sort the children ids so that we can collect the resulting strings from the children in correct order(ie ascending)
        children_ids.sort()
        #calculate resulting product after add operation
        after_add = ""
        for child in children_ids:
            #recieve the intermediate product from the child
            data = common_comm.recv(source = ids_ranks[child])
            after_add = after_add + data
        #change the product according to the current operation
        after_add = operate(after_add)
        if(common_comm.Get_rank() == 1):    # terminal machine
            #in the terminal machine, send the final product to the control center
            common_comm.send(after_add, dest = 0)
        else:
            #if it is not the terminal machine, use blocking send to send the resulting product to the parent machine
            dest = ids_ranks[parent_id]
            common_comm.send(after_add, dest = dest) # send the machine id
