from curses import has_key
import sys
from xml.dom.minidom import TypeInfo

import puzz
import pdqpq


GOAL_STATE = puzz.EightPuzzleBoard("012345678")


def solve_puzzle(start_state, flavor):
    """Perform a search to find a solution to a puzzle.
    
    Args:
        start_state (EightPuzzleBoard): the start state for the search
        flavor (str): tag that indicate which type of search to run.  Can be one of the following:
            'bfs' - breadth-first search
            'ucost' - uniform-cost search
            'greedy-h1' - Greedy best-first search using a misplaced tile count heuristic
            'greedy-h2' - Greedy best-first search using a Manhattan distance heuristic
            'greedy-h3' - Greedy best-first search using a weighted Manhattan distance heuristic
            'astar-h1' - A* search using a misplaced tile count heuristic
            'astar-h2' - A* search using a Manhattan distance heuristic
            'astar-h3' - A* search using a weighted Manhattan distance heuristic
    
    Returns: 
        A dictionary containing describing the search performed, containing the following entries:
        'path' - list of 2-tuples representing the path from the start to the goal state (both 
            included).  Each entry is a (str, EightPuzzleBoard) pair indicating the move and 
            resulting successor state for each action.  Omitted if the search fails.
        'path_cost' - the total cost of the path, taking into account the costs associated with 
            each state transition.  Omitted if the search fails.
        'frontier_count' - the number of unique states added to the search frontier at any point 
            during the search.
        'expanded_count' - the number of unique states removed from the frontier and expanded 
            (successors generated)

    """
    if flavor.find('-') > -1:
        strat, heur = flavor.split('-')
    else:
        strat, heur = flavor, None

    if strat == 'bfs':
        return BreadthFirstSolver(GOAL_STATE).solve(start_state)
    elif strat == 'ucost':
        return UniformCostSolver(GOAL_STATE).solve(start_state)
    elif strat == 'greedy':
        return Greedy1Solver(GOAL_STATE).solve(start_state)
    elif strat == 'greedy2':
        return Greedy2Solver(GOAL_STATE).solve(start_state)
    elif strat == 'greedy3':
        return Greedy3Solver(GOAL_STATE).solve(start_state)
    elif strat == 'astar':
        return Astarh1Solver(GOAL_STATE).solve(start_state)
    elif strat == 'astar2':
        return Astarh1Solver(GOAL_STATE).solve(start_state)
    elif strat == 'astar3':
        return Astarh1Solver(GOAL_STATE).solve(start_state)
    else:
        raise ValueError("Unknown search flavor '{}'".format(flavor))


def get_test_puzzles():
    """Return sample start states for testing the search strategies.
    
    Returns:
        A tuple containing three EightPuzzleBoard objects representing start states that have an
        optimal solution path length of 3-5, 10-15, and >=25 respectively.
    
    """ 
    state = [[8,None,2],
             [3,5,6],
             [1,7,4]]
    return state  

def h1(state):
    # compare start state and goal state
    # if one tile moved, h1+=1
    h1 = 0
    if state.get_tile(0, 0) != '6':
        h1 += 1   
    if state.get_tile(0, 1) != '3':
        h1 += 1 
    if state.get_tile(1, 0) != '7':
        h1 += 1
    if state.get_tile(1, 1) != '4':
        h1 += 1
    if state.get_tile(1, 2) != '1':
        h1 += 1
    if state.get_tile(2, 0) != '8':
        h1 += 1
    if state.get_tile(2, 1) != '5':
        h1 += 1
    if state.get_tile(2, 2) != '2':
        h1 += 1
    return  h1


def h2(state):
    # add up total steps moved for every title
    h2 = 0
    h2 += abs(state.find('1')[0] - 1) + abs(state.find('1')[1] - 2)
    h2 += abs(state.find('2')[0] - 2) + abs(state.find('2')[1] - 2)
    h2 += abs(state.find('3')[0] - 0) + abs(state.find('3')[1] - 1)
    h2 += abs(state.find('4')[0] - 1) + abs(state.find('4')[1] - 1)
    h2 += abs(state.find('5')[0] - 2) + abs(state.find('5')[1] - 1)
    h2 += abs(state.find('6')[0] - 0) + abs(state.find('6')[1] - 0)
    h2 += abs(state.find('7')[0] - 1) + abs(state.find('7')[1] - 0)
    h2 += abs(state.find('8')[0] - 2) + abs(state.find('8')[1] - 0)
    return h2

def h3(state):
    # add up total steps moved for every title
    h3 = 0
    h3 += 1**2*(abs(state.find('1')[0] - 1) + abs(state.find('1')[1] - 2))
    h3 += 2**2*(abs(state.find('2')[0] - 2) + abs(state.find('2')[1] - 2))
    h3 += 3**2*(abs(state.find('3')[0] - 0) + abs(state.find('3')[1] - 1))
    h3 += 4**2*(abs(state.find('4')[0] - 1) + abs(state.find('4')[1] - 1))
    h3 += 5**2*(abs(state.find('5')[0] - 2) + abs(state.find('5')[1] - 1))
    h3 += 6**2*(abs(state.find('6')[0] - 0) + abs(state.find('6')[1] - 0))
    h3 += 7**2*(abs(state.find('7')[0] - 1) + abs(state.find('7')[1] - 0))
    h3 += 8**2*(abs(state.find('8')[0] - 2) + abs(state.find('8')[1] - 0))
    return h3

def print_table(flav__results, include_path=False):
    """Print out a comparison of search strategy results.
    
    Args:
        flav__results (dictionary): a dictionary mapping search flavor tags result statistics. See
            solve_puzzle() for detail.
        include_path (bool): indicates whether to include the actual solution paths in the table

    """
    result_tups = sorted(flav__results.items())
    c = len(result_tups)
    na = "{:>12}".format("n/a")
    rows = [  # abandon all hope ye who try to modify the table formatting code...
        "flavor  " + "".join([ "{:>12}".format(tag) for tag, _ in result_tups]),
        "--------" + ("  " + "-"*10)*c,
        "length  " + "".join([ "{:>12}".format(len(res['path'])) if 'path' in res else na 
                                for _, res in result_tups ]),
        "cost    " + "".join([ "{:>12,}".format(res['path_cost']) if 'path_cost' in res else na 
                                for _, res in result_tups ]),
        "frontier" + ("{:>12,}" * c).format(*[res['frontier_count'] for _, res in result_tups]),
        "expanded" + ("{:>12,}" * c).format(*[res['expanded_count'] for _, res in result_tups])
    ]
    if include_path:
        rows.append("path")
        longest_path = max([ len(res['path']) for _, res in result_tups if 'path' in res ])
        print("longest", longest_path)
        for i in range(longest_path):
            row = "        "
            for _, res in result_tups:
                if len(res.get('path', [])) > i:
                    move, state = res['path'][i]
                    row += " " + move[0] + " " + str(state)
                else:
                    row += " "*12
            rows.append(row)
    print("\n" + "\n".join(rows), "\n")


class PuzzleSolver:
    """Base class for 8-puzzle solver."""

    def __init__(self, goal_state):
        self.parents = {}  # state -> parent_state
        self.expanded_count = 0
        self.frontier_count = 0
        self.goal = goal_state

    def get_path(self, state):
        """Return the solution path from the start state of the search to a target.
        
        Results are obtained by retracing the path backwards through the parent tree to the start
        state for the serach at the root.
        
        Args:
            state (EightPuzzleBoard): target state in the search tree
        
        Returns:
            A list of EightPuzzleBoard objects representing the path from the start state to the
            target state

        """
        path = []
        while state is not None:
            path.append(state)
            state = self.parents[state]
        path.reverse()
        return path

    def get_cost(self, state): 
        """Calculate the path cost from start state to a target state.
        
        Transition costs between states are equal to the square of the number on the tile that 
        was moved. 

        Args:
            state (EightPuzzleBoard): target state in the search tree
        
        Returns:
            Integer indicating the cost of the solution path

        """
        cost = 0
        moveCost = 0
        path = self.get_path(state)
        for i in range(1, len(path)):
            x, y = path[i-1].find(None)  # the most recently moved tile leaves the blank behind
            tile = path[i].get_tile(x, y) 
            #moveCost = self.costHelper(tile, state)  
            cost += int(tile)**2
        return cost #+ moveCost

    def costHelper(self, tile, state):
        s_cost = 0
        upcost=100
        dcost=100
        rcost=100
        lcost=100
        dic = state.successors()
        for key in dic:
            if state.get_move(dic[key]) == "up":
                upcost = int(tile)**2
            if state.get_move(dic[key]) == "down":
                dcost = int(tile)**2
            if state.get_move(dic[key]) == "right":
                rcost = int(tile)**2
            if state.get_move(dic[key]) == "left":
                lcost = int(tile)**2
            s_cost=min(upcost,dcost,rcost,lcost)
        return s_cost


    def get_results_dict(self, state):
        """Construct the output dictionary for solve_puzzle()
        
        Args:
            state (EightPuzzleBoard): final state in the search tree
        
        Returns:
            A dictionary describing the search performed (see solve_puzzle())

        """
        results = {}
        results['frontier_count'] = self.frontier_count
        results['expanded_count'] = self.expanded_count
        if state:
            results['path_cost'] = self.get_cost(state)
            path = self.get_path(state)
            moves = ['start'] + [ path[i-1].get_move(path[i]) for i in range(1, len(path)) ]
            results['path'] = list(zip(moves, path))
        return results

    def solve(self, start_state):
        """Carry out the search for a solution path to the goal state.
        
        Args:
            start_state (EightPuzzleBoard): start state for the search 
        
        Returns:
            A dictionary describing the search from the start state to the goal state.

        """
        raise NotImplementedError('Classed inheriting from PuzzleSolver must implement solve()')
        

class UniformCostSolver(PuzzleSolver):
    """Implementation of UniformCostSolver Search based on PuzzleSolver"""

    def __init__(self, goal_state):
        self.frontier = pdqpq.PriorityQueue()
        self.explored = set()
        super().__init__(goal_state)

    def add_to_frontier(self, node, cost):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node, cost)
        self.frontier_count += 1
        index = node._board.index("0")
        cost = pow(int(node._board[index]), 2) + cost

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def solve(self, start_state):
        self.parents[start_state] = None
        self.add_to_frontier(start_state, 0)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            if node == self.goal:
                return self.get_results_dict(succ)
            succs = self.expand_node(node)
            for move, succ in succs.items():
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.add_to_frontier(succ, self.get_cost(node))
                    self.parents[succ] = node
                elif (succ in self.frontier) and (self.frontier.get(succ) > self.get_cost(start_state)):
                    self.frontier.add(succ, self.get_cost(start_state))
                    self.parents[succ] = node
    
class Astarh1Solver(PuzzleSolver):
    """Implementation of Breadth-First Search based on PuzzleSolver"""

    def __init__(self, goal_state):
        self.frontier = pdqpq.FifoQueue()
        self.explored = set()
        super().__init__(goal_state)

    def add_to_frontier(self, node, cost):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node, cost)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def solve(self, start_state):
        self.parents[start_state] = None
        self.add_to_frontier(start_state, 0)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            if succ == self.goal:
                return self.get_results_dict(succ)
            succs = self.expand_node(node)

            for move, succ in succs.items():
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    self.add_to_frontier(succ, h1(start_state)+self.costHelper(succ))
                elif (succ in self.frontier) and (self.frontier.get(succ) > h1(start_state) + self.costHelper(succ)):
                    self.frontier.add(succ, h1(start_state)+self.costHelper(succ))

        # if we get here, the search failed
        return self.get_results_dict(None) 

class Astarh2Solver(PuzzleSolver):
    """Implementation of Breadth-First Search based on PuzzleSolver"""

    def __init__(self, goal_state):
        self.frontier = pdqpq.FifoQueue()
        self.explored = set()
        super().__init__(goal_state)

    def add_to_frontier(self, node, cost):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node, cost)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def solve(self, start_state):
        self.parents[start_state] = None
        self.add_to_frontier(start_state, 0)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            if succ == self.goal:
                return self.get_results_dict(succ)
            succs = self.expand_node(node)

            for move, succ in succs.items():
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    self.add_to_frontier(succ, h2(start_state)+self.costHelper(succ))
                elif (succ in self.frontier) and (self.frontier.get(succ) > h2(start_state) + self.costHelper(succ)):
                    self.frontier.add(succ, h2(start_state)+self.costHelper(succ))

        # if we get here, the search failed
        return self.get_results_dict(None)

class Astarh3Solver(PuzzleSolver):
    """Implementation of Breadth-First Search based on PuzzleSolver"""

    def __init__(self, goal_state):
        self.frontier = pdqpq.FifoQueue()
        self.explored = set()
        super().__init__(goal_state)

    def add_to_frontier(self, node, cost):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node, cost)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def solve(self, start_state):
        self.parents[start_state] = None
        self.add_to_frontier(start_state, 0)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            if succ == self.goal:
                return self.get_results_dict(succ)
            succs = self.expand_node(node)

            for move, succ in succs.items():
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    self.add_to_frontier(succ, h3(start_state)+self.costHelper(succ))
                elif (succ in self.frontier) and (self.frontier.get(succ) > h3(start_state) + self.costHelper(succ)):
                    self.frontier.add(succ, h3(start_state)+self.costHelper(succ))

        # if we get here, the search failed
        return self.get_results_dict(None)

class Greedy1Solver(PuzzleSolver):
    """Implementation of Breadth-First Search based on PuzzleSolver"""

    def __init__(self, goal_state):
        self.frontier = pdqpq.PriorityQueue()
        self.explored = set()
        super().__init__(goal_state)

    def add_to_frontier(self, node, cost):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node, cost)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def solve(self, start_state):
        self.parents[start_state] = None
        self.add_to_frontier(start_state, 0)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            if node == self.goal:
                return self.get_results_dict(succ)
            succs = self.expand_node(node)

            for move, succ in succs.items():
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    self.add_to_frontier(succ, h1(start_state))
                elif (succ in self.frontier) and (self.frontier.get(succ) > h1(start_state)):
                    self.frontier.add(succ, h1(start_state))

        # if we get here, the search failed
        return self.get_results_dict(None) 

class Greedy2Solver(PuzzleSolver):
    """Implementation of Breadth-First Search based on PuzzleSolver"""

    def __init__(self, goal_state):
        self.frontier = pdqpq.PriorityQueue()
        self.explored = set()
        super().__init__(goal_state)

    def add_to_frontier(self, node, cost):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node, cost)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def solve(self, start_state):
        self.parents[start_state] = None
        self.add_to_frontier(start_state, 0)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            if node == self.goal:
                return self.get_results_dict(succ)
            succs = self.expand_node(node)

            for move, succ in succs.items():
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    self.add_to_frontier(succ, h2(start_state))
                elif (succ in self.frontier) and (self.frontier.get(succ) > h2(start_state)):
                    self.frontier.add(succ, h2(start_state))

        # if we get here, the search failed
        return self.get_results_dict(None) 

class Greedy3Solver(PuzzleSolver):
    """Implementation of Breadth-First Search based on PuzzleSolver"""

    def __init__(self, goal_state):
        self.frontier = pdqpq.PriorityQueue()
        self.explored = set()
        super().__init__(goal_state)

    def add_to_frontier(self, node, cost):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node, cost)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def solve(self, start_state):
        self.parents[start_state] = None
        self.add_to_frontier(start_state, 0)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            if node == self.goal:
                return self.get_results_dict(succ)
            succs = self.expand_node(node)

            for move, succ in succs.items():
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node
                    self.add_to_frontier(succ, h3(start_state))
                elif (succ in self.frontier) and (self.frontier.get(succ) > h3(start_state)):
                    self.frontier.add(succ, h3(start_state))

        # if we get here, the search failed
        return self.get_results_dict(None) 


class BreadthFirstSolver(PuzzleSolver):
    """Implementation of Breadth-First Search based on PuzzleSolver"""

    def __init__(self, goal_state):
        self.frontier = pdqpq.FifoQueue()
        self.explored = set()
        super().__init__(goal_state)

    def add_to_frontier(self, node):
        """Add state to frontier and increase the frontier count."""
        self.frontier.add(node)
        self.frontier_count += 1

    def expand_node(self, node):
        """Get the next state from the frontier and increase the expanded count."""
        self.explored.add(node)
        self.expanded_count += 1
        return node.successors()

    def solve(self, start_state):
        self.parents[start_state] = None
        self.add_to_frontier(start_state)

        if start_state == self.goal:  # edge case        
            return self.get_results_dict(start_state)

        while not self.frontier.is_empty():
            node = self.frontier.pop()  # get the next node in the frontier queue
            succs = self.expand_node(node)

            for move, succ in succs.items():
                if (succ not in self.frontier) and (succ not in self.explored):
                    self.parents[succ] = node

                    # BFS checks for goal state _before_ adding to frontier
                    if succ == self.goal:
                        return self.get_results_dict(succ)
                    else:
                        self.add_to_frontier(succ)

        # if we get here, the search failed
        return self.get_results_dict(None) 


############################################

if __name__ == '__main__':

    # parse the command line args
    #start = puzz.EightPuzzleBoard(sys.argv[1])

    start = puzz.EightPuzzleBoard("802356174")
    #print(h3(start))
    # if sys.argv[2] == 'all':
    #     flavors = ['bfs', 'ucost', 'greedy-h1', 'greedy-h2',
    #                'greedy-h3', 'astar-h1', 'astar-h2', 'astar-h3']
    # else:
    #     flavors = sys.argv[2:]
    #flavors = ['greedy']
    #flavors = ['greedy2']
    #flavors = ['bfs']
    flavors = ['ucost']
    # run the search(es)
    results = {}
    for flav in flavors:
        print("solving puzzle {} with {}".format(start, flav))
        results[flav] = solve_puzzle(start, flav)

    # change to True to see the paths!
    print_table(results, include_path=False)
    


