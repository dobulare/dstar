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
from game import Actions, Directions
import time

INFINITE_VALUE = float("inf")

class dStarParams:
    def __init__(self, problem, heuristic):
        self.start = problem.getStartState()
        self.prev = problem.getStartState()
        self.goal = problem.getGoalState()
        self.finalPath = []
        self.updatedVertices = []
        self.removedVertices = 0
        self.width, self.height = problem.getDims()
        self.discoveredWalls = problem.get_walls_discovered()
        self.heuristic_func = heuristic

        # Procedure Initialize steps are executed below
        self.U = util.dpQueue()
        self.k_m = 0
        self.gNrhs = [[[INFINITE_VALUE, INFINITE_VALUE] for i in range(self.height)] for j in range(self.width)]
        self.assignGorRHS(self.goal, (INFINITE_VALUE, 0))

    def assignGorRHS(self, state, weight):
        if weight[0] is not None:
            self.gNrhs[state[0]][state[1]][0] = weight[0]
        if weight[1] is not None:
            self.gNrhs[state[0]][state[1]][1] = weight[1]

    # Procedure Calculate keys from the paper
    def calculateKeys(self, state):
        sKey = min(self.gNrhs[state[0]][state[1]])
        pKey = self.k_m + self.heuristic_func(self.start, state) + sKey
        return pKey, sKey

    # Procedure Compute Shortest Path from the paper
    def computeShortestPath(self):
        g, rhs = self.gNrhs[self.start[0]][self.start[1]]
        while g != rhs or (self.U.size() > 0 and compKeys(self.U.peek(), self.calculateKeys(self.start))):
            previousvertexvalue = self.U.peek()
            vertex = self.U.pop()[0]
            self.removedVertices += 1
            g_vertex, rhs_vertex = self.gNrhs[vertex[0]][vertex[1]]
            if compKeys(previousvertexvalue, self.calculateKeys(vertex)):
                pKey, sKey = self.calculateKeys(vertex)
                self.U.Insert(vertex, pKey, sKey)
            elif g_vertex > rhs_vertex:
                self.assignGorRHS(vertex, (rhs_vertex, INFINITE_VALUE))
                for neighbour in self.retrieveNeighbours(vertex):
                    self.updateVertex(neighbour, self.goal)
            else:
                self.assignGorRHS(vertex, (INFINITE_VALUE, INFINITE_VALUE))
                for neighbour in self.retrieveNeighbours(vertex):
                    self.updateVertex(neighbour, self.goal)
                self.updateVertex(vertex, self.goal)
            g, rhs = self.gNrhs[self.start[0]][self.start[1]]

    # Procedure UpdateVertex from the paper
    def updateVertex(self, u, ExcludedVertex=None):
        if self.goal != u:
            newRHS = INFINITE_VALUE
            if not self.discoveredWalls[u[0]][u[1]]:
                for neighbour in self.retrieveNeighbours(u):
                    g, Rhs = self.gNrhs[neighbour[0]][neighbour[1]]
                    newRHS = min(g + 1, newRHS)
            self.assignGorRHS(u, (INFINITE_VALUE, newRHS))
        self.U.Remove(u)
        g, RHS = self.gNrhs[u[0]][u[1]]
        if RHS != g:
            pKey, sKey = self.calculateKeys(u)
            self.U.Insert(u, pKey, sKey)

    # Retrives All the Neighbours at a cell
    def retrieveNeighbours(self, node):
        neighbours = []
        for n in [(node[0], node[1] + 1), (node[0], node[1] - 1), (node[0] + 1, node[1]), (node[0] - 1, node[1])]:
            if 0 <= n[1] < self.height and 0 <= n[0] < self.width:
                neighbours.append(n)
        return neighbours

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

def getLocCoordinate(x, y, action):
    dX, dY = Actions.directionToVector(action)
    newX = int(x + dX)
    newY = int(y + dY)
    return newX, newY


def getDirection(cell, nextCell):
    x = cell[0]
    y = cell[1]
    nextX = nextCell[0]
    nextY = nextCell[1]
    if nextX - x > 0:
        return Directions.EAST
    elif nextY - y > 0:
        return Directions.NORTH
    elif x - nextX > 0:
        return Directions.WEST
    else:
        return Directions.SOUTH


def cListAList(locList):
    directions = []
    for cell in range(len(locList) - 1):
        directions.append(getDirection(locList[cell], locList[cell + 1]))
    return directions


def naiveAStarSearch(problem, heuristic):
    start_Time = int(round(time.time() * 1000))
    x, y = problem.getStartState()

    finalpath = []
    listOfActions = aStarSearch(problem, heuristic)
    while not problem.isGoalState((x, y)):
        finalpath.append((x, y))
        nextX, nextY = getLocCoordinate(x, y, listOfActions.pop(0))
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            adjX, adjY = getLocCoordinate(x, y, direction)
            if not problem.is_wall_discovered(adjX, adjY) and problem.isWall(adjX, adjY):
                problem.set_walls_discovered(adjX, adjY)
        if problem.is_wall_discovered(nextX, nextY):
            listOfActions = aStarSearch(problem, heuristic)
            nextX, nextY = getLocCoordinate(x, y, listOfActions.pop(0))
        x, y = nextX, nextY
        problem.setStartState(x, y)
    finalpath.append((x, y))
    actions = cListAList(finalpath)
    problem.setStartState(x, y)
    time_of_task_completion = int(round(time.time() * 1000)) - start_Time
    print("Total Time for the task completion: ", time_of_task_completion, " msecs")
    return actions





def traverseAgent(dtarParams):
    if dtarParams.start == dtarParams.goal:
        return dtarParams.start
    g_start, rhs_start = dtarParams.gNrhs[dtarParams.start[0]][dtarParams.start[1]]
    if g_start == INFINITE_VALUE:
        return dtarParams.start
    cell_weight = (None, INFINITE_VALUE)
    for neighbour in dtarParams.reachableNeighbours(dtarParams.start):
        weight = 1 + dtarParams.gNrhs[neighbour[0]][neighbour[1]][0]
        if weight < cell_weight[1]:
            cell_weight = (neighbour, weight)
    dtarParams.finalPath.append(dtarParams.start)
    dtarParams.start = cell_weight[0]
    return dtarParams.start

def compKeys(leftState, rightState):
    if len(leftState) == 2:
        p_1, s_1 = leftState
    elif len(leftState) == 3:
        l_1, p_1, s_1 = leftState
    if len(rightState) == 2:
        p_2, s_2 = rightState
    elif len(rightState) == 3:
        l_2, p_2, s_2 = rightState

    if p_1 < p_2:
        return True
    elif p_1 > p_2:
        return False
    else:
        return s_1 < s_2





# Procedure for the DStarLite Search
def dStarLiteSearch(problem, heuristic):
    # Start and end time for the calculation of execution time in msec
    start_Time = (int(round(time.time() * 1000)))
    x, y = problem.getStartState()

    # Procedure Initialize
    dStarObj = dStarParams(problem, heuristic)

    pKey, sKey = dStarObj.calculateKeys(dStarObj.goal)
    dStarObj.U.Insert(dStarObj.goal, pKey, sKey)
    dStarObj.computeShortestPath()
    while (x, y) != problem.getGoalState():
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if problem.isWall(nextx, nexty):
                dStarObj.updatedVertices.append((nextx, nexty))
                for neighbour in dStarObj.retrieveNeighbours((nextx, nexty)):
                    dStarObj.updatedVertices.append(neighbour)
                dStarObj.k_m += dStarObj.heuristic_func(dStarObj.prev, dStarObj.start)
                dStarObj.prev = dStarObj.start
                dStarObj.discoveredWalls[nextx][nexty] = True
                for updatedVertices in dStarObj.updatedVertices:
                    dStarObj.updateVertex(updatedVertices, dStarObj.goal)
                dStarObj.updatedVertices = []
                dStarObj.computeShortestPath()
        x, y = traverseAgent(dStarObj)
    finalPath = list(dStarObj.finalPath)
    finalPath.append(dStarObj.start)
    finalActions = cListAList(finalPath)
    problem._expanded = dStarObj.removedVertices
    time_of_task_completion = int(round(time.time() * 1000)) - start_Time
    print("Total Time for the task completion: ", time_of_task_completion, " msecs")
    return finalActions


# Abbreviations
nastar = naiveAStarSearch
dstar = dStarLiteSearch
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
