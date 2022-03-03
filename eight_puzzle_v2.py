#!/usr/bin/env python
import numpy as np
from collections import deque

class MyQueue:
    def __init__(self):
       # Write code here
       self.queue = deque([]) 
    
    
    def push(self, x):
        # Write code here
        self.queue.append(x)
    
    
    def pop(self):
        # Write code here
        return self.queue.popleft()
    
    
    def peek(self):
        # Write code here
        return self.queue[0]
    
    
    def empty(self):
        # Write code here
        if len(self.queue) == 0:
            return True
        return False

# -------------------- Node Data Structure / Class ---------------
class Node:
    def __init__(self, node_state, node_index, parent_index):
        self.state = node_state # Node State (3 x 3 Array)
        self.n_i = node_index # Node index (int)
        self.p_i = parent_index # Parent index (int)

# -------------- Function to Print Matrix from .txt file ------------------
def print_matrix(state):
    counter = 0
    for row in range(0, len(state), 3):
        if counter == 0 :
            print("-------------")
        for element in range(counter, len(state), 3):
            if element <= counter:
                print("|", end=" ")
            print(int(state[element]), "|", end=" ")
        counter = counter +1
        print("\n-------------")


# ---------------- Main Class for BFS Algo for 8-puzzle problem ------------- 
class EightPuzzle:
    def __init__(self, start_state, goal_state):
        self.n_i = 1 # Node Index counter
        self.start = Node(start_state, self.n_i, 1) # Start node 
        self.goal_state = goal_state # Goal state matrix
        self.queue = MyQueue() # FIFO Queue data structure to push and pop nodes
        self.queue.push(self.start)
        self.visited = np.array([]) # Keeps track of Node objects
        self.explored_states = np.zeros((1,9)) # Keeps track of explored node states nd written onto Nodes.txt
        self.path = np.reshape(np.transpose(goal_state), (1,9)) # To save path while Backtracking
        self.nodes_info = np.array([[1],[1]]) # Saves Node index and Parent index for writing to NodesInfo.txt
        self.nodes_info.reshape((1,2))
        self.nodes_info = self.nodes_info.T 
        self.exit = False # Boolean to indicate when goal is reached
        self.goal_index = 0 # Goal's Node index is saved into this member
        self.goal_p_i = 0  # Goal's Parent Index is saved into this member
        # The Goal Node index and parent index are used while backtracking  
    
    # ---------- Method to calculate Blank Tile position ------------
    def Calc_Blank(self, current_node):
        arr = current_node.state
        i = np.where(arr == 0)[0][0]
        j = np.where(arr == 0)[1][0]
        i += 1
        j += 1
        return [i,j]

    # ----------------- Move Blank tile to directions -------------
    # Returns:
    # 1) success --> Boolean True if node exists in respective direction and hence motion is possible
    # 2) A Node that represents the outcome of the motion
    def ActionMoveLeft(self, current_node):
        i,j = self.Calc_Blank(current_node) 
        i -= 1
        j -= 1
        a = np.copy(current_node.state)
        if j != 0:
            temp = a[i][j-1]
            a[i][j-1] = a[i][j]
            a[i][j] = temp
            p_i = current_node.n_i
            success = True
            return success, Node(a, 0, p_i)
        else:
            success = False
        return success, Node(np.zeros((3,3)), 0, 0)

    def ActionMoveRight(self, current_node):
        i,j = self.Calc_Blank(current_node) 
        i -= 1
        j -= 1
        a = np.copy(current_node.state)
        if j != 2:
            temp = a[i][j+1]
            a[i][j+1] = a[i][j]
            a[i][j] = temp
            p_i = current_node.n_i
            success = True
            return success, Node(a, 0, p_i)
        else:
            success = False
        return success, Node(np.zeros((3,3)), 0, 0)

    def ActionMoveUp(self, current_node):
        i,j = self.Calc_Blank(current_node)
        i -= 1
        j -= 1
        a = np.copy(current_node.state) 
        if i != 0:
            temp = a[i-1][j]
            a[i-1][j] = a[i][j]
            a[i][j] = temp
            p_i = current_node.n_i
            success = True
            return success, Node(a, 0, p_i)
        else:
            success = False
        return success, Node(np.zeros((3,3)), 0, 0)

    def ActionMoveDown(self, current_node):
        i,j = self.Calc_Blank(current_node) 
        i -= 1
        j -= 1
        a = np.copy(current_node.state)
        if i != 2:
            temp = a[i+1][j]
            a[i+1][j] = a[i][j]
            a[i][j] = temp
            p_i = current_node.n_i
            success = True
            return success, Node(a, 0, p_i)
        else:
            success = False
        return success, Node(np.zeros((3,3)), 0, 0)

    # ------------- Explore the children of a node ---------------
    # Perform the various motions if possible
    # Push the children to the Queue
    # If goal is found in the children, save the corresponding node index and parent index
    # Add to visited but not to explored states
    # Set self.exit to True  
    def ExploreNode(self,curr_node):
        n = curr_node
        L_valid, LeftNode = self.ActionMoveLeft(n)
        U_valid, UpNode = self.ActionMoveUp(n)
        R_valid, RightNode = self.ActionMoveRight(n)
        D_valid, DownNode = self.ActionMoveDown(n)
            
        if L_valid:
            if (LeftNode.state == self.goal_state).all():
                self.n_i += 1
                LeftNode.n_i = self.n_i
                self.visited= np.append(self.visited, LeftNode)
                node_info = np.array([[LeftNode.n_i], [LeftNode.p_i]])
                node_info.reshape((2,1))
                node_info = node_info.T
                self.nodes_info = np.append(self.nodes_info, node_info, axis=0)
                self.goal_index = LeftNode.n_i
                self.goal_p_i = LeftNode.p_i
                self.exit = True
            
            self.queue.push(LeftNode)
        
        if U_valid:
            if (UpNode.state == self.goal_state).all():
                self.n_i += 1
                UpNode.n_i = self.n_i
                self.visited = np.append(self.visited, UpNode)
                node_info = np.array([[UpNode.n_i], [UpNode.p_i]])
                node_info.reshape((2,1))
                node_info = node_info.T
                self.nodes_info = np.append(self.nodes_info, node_info, axis=0)
                self.goal_index = UpNode.n_i
                self.goal_p_i = UpNode.p_i
                self.exit = True
            
            self.queue.push(UpNode)
        
        if R_valid:
            if (RightNode.state == self.goal_state).all():
                self.n_i += 1
                RightNode.n_i = self.n_i
                self.visited = np.append(self.visited, RightNode)
                node_info = np.array([[RightNode.n_i], [RightNode.p_i]])
                node_info.reshape((2,1))
                node_info = node_info.T
                self.nodes_info = np.append(self.nodes_info, node_info, axis=0)
                self.goal_index = RightNode.n_i
                self.goal_p_i = RightNode.p_i
                self.exit = True
            
            self.queue.push(RightNode)
        
        if D_valid:
            if(DownNode.state == self.goal_state).all():
                self.n_i += 1
                RightNode.n_i = self.n_i
                self.visited = np.append(self.visited, DownNode)
                node_info = np.array([[DownNode.n_i], [DownNode.p_i]])
                node_info.reshape((2,1))
                node_info = node_info.T
                self.nodes_info = np.append(self.nodes_info, node_info, axis=0)
                self.goal_index = DownNode.n_i
                self.goal_p_i = DownNode.p_i
                self.exit = True
            
            self.queue.push(DownNode)

    # --------- Function to save the data to the 3 .txt files ----------
    def save_files(self):
        np.savetxt('Nodes.txt', self.explored_states, fmt='%d')
        np.savetxt('nodePath.txt', self.path, fmt='%d')
        np.savetxt('NodesInfo.txt', self.nodes_info, fmt='%d', header='Node_index Parent_Node_index')
    
    # ---------------- Breadth First Search (BFS) Algo -----------------
    def bfs(self):
        j = 0 # Iteration Counter
        self.exit = False # Initialize exit to False
        # Exits while loop when exit becomes True (,i.e., Goal is reached) 
        # or Queue becomes empty
        while ((not self.exit) and (not self.queue.empty())): 
            current_node = self.queue.pop() # Removes and returns top element of FIFO queue
            # Check if current node is already visited
            visited_flag = False 
            if j !=0:
                for i in range(len(self.visited)):
                    if (self.visited[i].state == current_node.state).all():
                        visited_flag = True
                        break
            # If visited, skip iteration
            if(visited_flag):
                continue
            
            # If start node, create explored_states and visited arrays
            # Else, append to the exisiting explored_states and visited arrays
            if (j==0):
                self.explored_states = np.array(np.reshape(np.transpose(current_node.state), (1,9)))
                self.visited = np.array([current_node])    
            else:
                self.n_i += 1
                current_node.n_i = self.n_i
                self.visited = np.append(self.visited, current_node)
                self.explored_states = np.append(self.explored_states, np.reshape(np.transpose(current_node.state), (1,9)), axis = 0)
            if not (current_node.state == self.start.state).all() and not (current_node.state == self.goal_state).all():
                node_info = np.array([[current_node.n_i], [current_node.p_i]])
                node_info.reshape((2,1))
                node_info= node_info.T 
                self.nodes_info = np.append(self.nodes_info, node_info, axis=0)
            
            # Explore, find and push possible Children
            self.ExploreNode(current_node)
            
            j += 1
              
        # --------------- Back-Tracking and finding path ---------------
        p_i = self.goal_p_i
        while(p_i > 1):
           self.path = np.append(self.path, np.reshape(np.transpose(self.visited[p_i-1].state), (1,9)), axis=0)
           p_i = self.visited[p_i -1].p_i
        self.path = np.append(self.path, np.reshape(np.transpose(self.start.state), (1,9)), axis=0)
        self.path = np.flip(self.path, axis=0)

        # ------------- Save to txt files ---------
        self.save_files()

# ------------ Main Function
if __name__ == '__main__':
    # Take Start State as Input
    start_state = np.zeros((3,3))
    print('Enter Start state:')
    for i in range(9):
        start_state[i//3][i%3] = input('Enter ' + 's[' + str(i//3) + '][' + str(i%3) + ']: ')
    print('----------------------------------')
    print('Enter goal state:')
    
    # Take Goal State as Input 
    goal_state = np.zeros((3,3))
    for i in range(9):
        goal_state[i//3][i%3] = input('Enter ' + 'g[' + str(i//3) + '][' + str(i%3) + ']: ')
    
    # Create EightPuzzle object
    bfs = EightPuzzle(start_state, goal_state)
    # Call bfs() method of EightPuzzle class
    bfs.bfs()
    print('----------------------------------')
    print()
    # ----------- Print Path from txt file to terminal -----------
    fname = 'nodePath.txt'
    data = np.loadtxt(fname)
    if len(data[1]) != 9:
        print("Format of the text file is incorrect, retry ")
    else:
        for i in range(0, len(data)):
            if i == 0:
                print("Start Node")
            elif i == len(data)-1:
                print("Achieved Goal Node")
            else:
                print("Step ",i)
            print_matrix(data[i])
            print()
            print()