# Eight-puzzle-bfs
Breadth First Search for Eight Puzzle Problem

The direction peference I have used in the BFS is:
Left-->Up-->Down-->Right (Clockwise)

nodePath.txt format:
If node state is: 
1 4 7
5 0 8
2 3 6
nodePath.txt will have the line: 152403786

Libraries used:
1) numpy
2) collections (deque)

Steps to Run the Code:

1) Make sure you keep the txt files in same folder as the python code
2) Run the code
3) When prompted enter the elements according to the element indices shown in the prompt (which is Row-wise)
4) Enter elements of start and goal states
5) After entering last element and pressing Enter, you must see the path computed by the code using BFS Algorithm.
