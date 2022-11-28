# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    stack = util.Stack()
    stack.push((problem.getStartState(), []))  # Element and path till node
    nodesVisited = set()
    currentNode = (problem.getStartState(), [])
    while stack:
        currentNode = stack.pop()
        if problem.isGoalState(currentNode[0]):
            break
        elif currentNode[0] not in nodesVisited:
            nodesVisited.add(currentNode[0])
            successors = problem.getSuccessors(currentNode[0])
            for sucessor in successors:
                childNode, childPath = sucessor[0], sucessor[1]
                stack.push((childNode, currentNode[1] + [childPath]))
    return currentNode[1]


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    queue.push((problem.getStartState(), []))  # Element and path till node
    # print(problem.getStartState())
    nodesVisited = set()
    currentNode = (problem.getStartState(), [])
    while queue:
        # print(currentNode[0])
        currentNode = queue.pop()
        if problem.isGoalState(currentNode[0]):
            break
        elif currentNode[0] not in nodesVisited:
            nodesVisited.add(currentNode[0])
            successors = problem.getSuccessors(currentNode[0])
            for sucessor in successors:
                childNode, childPath = sucessor[0], sucessor[1]
                queue.push((childNode, currentNode[1] + [childPath]))
    return currentNode[1]


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    queue = util.PriorityQueue()
    queue.push((problem.getStartState(), [], 0), 0)  # Element and path till node combined with cost till now
    nodesVisited = set()
    currentNode = (problem.getStartState(), [], 0)
    while queue:
        currentNode = queue.pop()
        if problem.isGoalState(currentNode[0]):
            break
        elif currentNode[0] not in nodesVisited:
            nodesVisited.add(currentNode[0])
            successors = problem.getSuccessors(currentNode[0])
            for sucessor in successors:
                childNode, childPath, childCost = sucessor[0], sucessor[1], sucessor[2]
                queue.push((childNode, currentNode[1] + [childPath], currentNode[2] + childCost),
                           currentNode[2] + childCost)
    return currentNode[1]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    queue = util.PriorityQueue()
    queue.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(),
                                                           problem) + 0)  # Element and path till node combined with cost till now
    nodesVisited = set()
    currentNode = (problem.getStartState(), [], 0)
    while queue:
        currentNode = queue.pop()
        if problem.isGoalState(currentNode[0]):
            break
        elif currentNode[0] not in nodesVisited:
            nodesVisited.add(currentNode[0])
            successors = problem.getSuccessors(currentNode[0])
            for sucessor in successors:
                childNode, childPath, childCost = sucessor[0], sucessor[1], sucessor[2]
                queue.push((childNode, currentNode[1] + [childPath], currentNode[2] + childCost),
                           currentNode[2] + childCost + heuristic(childNode, problem))
    return currentNode[1]


def dStarSearch(problem):
    def manhattanDistance(s1, s2):
        return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])

    def CalculateKey(u):
        return [min(g[u], rhs[u]) + manhattanDistance(start_state, u), min(g[u], rhs[u])]

    def UpdateVertex(u):
        if not problem.isGoalState(u):
            rhs[u] = float('inf')
            for s, a, d in problem.getSuccessors(u):
                rhs[u] = min(rhs[u], g[s] + 1)

        if U.CheckPresence(u):
            U.Remove(u)

        if g[u] != rhs[u]:
            U.update(u, CalculateKey(u))

    def ComputeShortestPath():
        while U.TopKey() < CalculateKey(start_state) or rhs[start_state] != g[start_state]:

            u = U.Top()
            k_old = U.TopKey()
            k_new = CalculateKey(u)

            U.pop()
            # print(rhs[start_state])

            if (k_old < k_new):  # this means some edges have become unusable
                U.update(u, k_new)

            elif (g[u] > rhs[u]):  # This means a shorter path has been found
                g[u] = rhs[u]
                for s, a, d in problem.getSuccessors(u):
                    UpdateVertex(s)
            else:  # This means g[u]=rhs[u] i.e vertex is locally consistent
                g_old = g[u]
                g[u] = float('inf')
                UpdateVertex(u)

                for s, a, d in problem.getSuccessors(u):
                    UpdateVertex(s)

    def senseWallAt(problem, s):
        return problem.actual_walls[s[0]][s[1]]

    def knowWallAt(problem, s):
        return problem.walls[s[0]][s[1]]

    from util import PriorityQueue

    # Initialize
    U = PriorityQueue()
    km = 0
    start_state = problem.getStartState()
    last_state = start_state  # A copy is created, not a reference
    problem.getNeighboringWalls(start_state)  # Update the agent model
    rhs = {}
    g = {}

    for state in problem.AllStates():
        rhs[state] = float('inf')
        g[state] = float('inf')

    goal = (1, 1)  # Predefined
    rhs[goal] = 0
    U.push(goal, [manhattanDistance(goal, start_state), 0])

    ComputeShortestPath()

    actions = []

    while (not problem.isGoalState(start_state)):
        min_successor_value = float('inf')
        current_action = []

        for s, a, discombombulation in problem.getSuccessors(start_state):

            if (1 + g[s] < min_successor_value):
                start_state = s
                current_action = a
                min_successor_value = 1 + g[s]

        actions.append(current_action)

        # Scan for edge-weight changes after moving to the new start_state
        changes = problem.getNeighboringWalls(start_state)

        if (changes):
            km += manhattanDistance(last_state, start_state)
            last_state = start_state
            for wall in changes:
                g[wall] = float('inf')  # Since there's a wall, it's impossible to reach from the goal node

                # When a wall is detected, upto 4 edges become infinitely weighted. Update all their successors (predecessors)
                successors = problem.getSuccessors(wall)  # No edge weight has gone down
                for s1, a1, discombombulation in successors:
                    UpdateVertex(s1)

        ComputeShortestPath()
    return actions

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
dstar = dStarSearch
