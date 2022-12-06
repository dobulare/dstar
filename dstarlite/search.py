# coding=utf-8
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
from collections import defaultdict
from util import compareTwoStates

INFINITE = float("inf")


def returninf():
    return INFINITE


class DStarLite:
    # Procedure Initialize from the Paper.
    def __init__(self, problem, heuristic):
        self.U = util.priorityQModified()
        self.k_m = 0
        self.g = defaultdict(returninf)
        self.rhs = defaultdict(returninf)

        # Helper variables while computing the actions
        self.start = problem.getStartState()
        self.last = problem.getStartState()
        self.goal = problem.getGoalState()
        self.h = heuristic

        self.rhs[self.goal] = 0
        self.U.Insert(self.goal, self.calculateKey(self.goal))

        self.finalVisitedNode = ""
        self.actions = []
        self.countOfExpanded = 0
        self.width, self.height = problem.getDims()
        self.discovered_walls = problem.get_walls_explored()
        self.unitcost = 1

    # Procedure Calculate Key from the paper
    def calculateKey(self, state):
        return min(self.g[state], self.rhs[state]) + self.h(self.start, state) + self.k_m, min(self.g[state],
                                                                                               self.rhs[state])

    # Retrives All the Neighbours at a cell
    def reachableNeighbours(self, node):
        neighbours = []
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dir in dirs:
            newNode = (node[0] + dir[0], node[1] + dir[1])
            if 0 <= newNode[1] < self.height and 0 <= newNode[0] < self.width:
                neighbours.append(newNode)
        return neighbours

    def gEqualRhs(self, state):
        return self.g[state] == self.rhs[state]

    # Procedure Compute Shortest Path from the paper
    def computeShortestPath(self):
        while (self.U.size() > 0 and compareTwoStates(self.U.topKey(),
                                                      self.calculateKey(self.start))) or not self.gEqualRhs(self.start):
            k_old = self.U.topKey()
            u = self.U.pop()[0]
            self.countOfExpanded += 1

            if compareTwoStates(k_old, self.calculateKey(u)):
                self.U.Insert(u, self.calculateKey(u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for neighbour in self.reachableNeighbours(u):
                    self.updateVertex(neighbour)
            else:
                self.g[u] = INFINITE
                for neighbour in self.reachableNeighbours(u):
                    self.updateVertex(neighbour)
                self.updateVertex(u)

    # Procedure Update Vertex from the paper
    def updateVertex(self, u):
        if u != self.goal:
            minRhs = INFINITE
            if not self.discovered_walls[u[0]][u[1]]:
                for sdash in self.reachableNeighbours(u):
                    minRhs = min(self.unitcost + self.g[sdash], minRhs)
            self.rhs[u] = minRhs

        self.U.Remove(u)

        if self.g[u] != self.rhs[u]:
            self.U.Insert(u, self.calculateKey(u))

    def moveToStart(self):
        if self.start == self.goal:
            return self.start

        if self.g[self.start] == INFINITE:
            return self.start

        minCost = (None, INFINITE)
        for neighbour in self.reachableNeighbours(self.start):
            if self.unitcost + self.g[neighbour] < minCost[1]:
                minCost = (neighbour, self.unitcost + self.g[neighbour])

        if self.finalVisitedNode != "":
            self.actions.append(getDirection(self.finalVisitedNode, self.start))
        self.finalVisitedNode = self.start

        self.start = minCost[0]
        return self.start


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


def depthFirstSearch(problem, heuristic):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    already_visisted = set()
    curr_state = problem.getStartState()
    dfs_stack = util.Stack()
    dfs_stack.push((curr_state, []))
    next_states = []
    while dfs_stack.isEmpty() != True:
        state = dfs_stack.pop()
        curr_state = state[0]
        next_states = state[1]
        if problem.isGoalState(curr_state) == True:
            return next_states
        if already_visisted.__contains__(curr_state) == False:
            already_visisted.add(curr_state)
            # already_visisted.add(curr_state)
            for states in problem.getSuccessors(curr_state):
                if already_visisted.__contains__(states[0]) == False:
                    # print states
                    dfs_stack.push((states[0], next_states + [states[1]]))
    return next_states


def breadthFirstSearch(problem, heuristic):
    already_visisted = set()
    curr_state = problem.getStartState()
    bfs_queue = util.Queue()
    bfs_queue.push((curr_state, []))
    while bfs_queue.isEmpty() != True:
        state = bfs_queue.pop()
        # print(state)
        curr_state = state[0]
        next_states = state[1]
        # print next_states
        if problem.isGoalState(curr_state) == True:
            # print "In Final"
            # print next_states
            return next_states
        # print(curr_state)
        if already_visisted.__contains__(str(curr_state)) == False:
            already_visisted.add(str(curr_state))
            for states in problem.getSuccessors(curr_state):
                if already_visisted.__contains__(str(states[0])) == False:
                    bfs_queue.push((states[0], next_states + [states[1]]))


def uniformCostSearch(problem, heuristic):
    already_visisted = set()
    curr_state = problem.getStartState()
    bfs_pr_queue = util.PriorityQueue()
    bfs_pr_queue.push((curr_state, []), 0)
    while bfs_pr_queue.isEmpty() != True:
        state = bfs_pr_queue.pop()
        # print state
        curr_state = state[0]
        next_states = state[1]
        # value_curr = state[1]
        if problem.isGoalState(curr_state) == True:
            return next_states
        if already_visisted.__contains__(curr_state) == False:
            already_visisted.add(curr_state)
            for states in problem.getSuccessors(curr_state):
                if already_visisted.__contains__(states[0]) == False:
                    curr_cost = problem.getCostOfActions(next_states + [states[1]])
                    bfs_pr_queue.push((states[0], next_states + [states[1]]), curr_cost)


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


def getNextState(state, action):
    dX, dY = Actions.directionToVector(action)
    return int(state[0] + dX), int(state[1] + dY)


def getDirection(cell, nextCell):
    x, y = cell
    nextX, nextY = nextCell
    if nextX - x > 0:
        return Directions.EAST
    elif nextY - y > 0:
        return Directions.NORTH
    elif x - nextX > 0:
        return Directions.WEST
    else:
        return Directions.SOUTH


def ReplanningSearch(problem, heuristic, function=aStarSearch):
    initialState = problem.getStartState()
    currState = problem.getStartState()
    actions = []
    lastState = ""
    actionsGenerated = function(problem, heuristic)
    while not problem.isGoalState(currState):
        if lastState != "":
            actions.append(getDirection(lastState, currState))
        nextState = getNextState(currState, actionsGenerated.pop(0))
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            adjacent = getNextState(currState, direction)
            if not problem.is_wall_found(adjacent) and problem.checkWall(adjacent):
                problem.set_walls_explored(adjacent)
        if problem.is_wall_found(nextState):
            actionsGenerated = function(problem, heuristic)
            nextState = getNextState(currState, actionsGenerated.pop(0))
        lastState = currState
        currState = nextState
        problem.setStartState(currState)
    actions.append(getDirection(lastState, currState))
    problem.setStartState(initialState)
    return actions


def replanningAStar(problem, heuristic):
    return ReplanningSearch(problem, heuristic, aStarSearch)


def replanningBFS(problem, heuristic):
    return ReplanningSearch(problem, heuristic, breadthFirstSearch)


def replanningDFS(problem, heuristic):
    return ReplanningSearch(problem, heuristic, depthFirstSearch)


def replanningUCS(problem, heuristic):
    return ReplanningSearch(problem, heuristic, uniformCostSearch)


# Procedure for Main in the DStarLite Search
def dStarLiteSearch(problem, heuristic):
    # Procedure Initialize
    # s_start and s_last are initialized in the init function of the DStarLite
    obj = DStarLite(problem, heuristic)

    obj.computeShortestPath()

    while not problem.isGoalState(obj.start):
        obj.moveToStart()
        for next in obj.reachableNeighbours(obj.start):

            updateVertices = []
            # Updating the edge costs whenever we discovered a wall
            if problem.checkWall(next):
                updateVertices.append(next)
                for neighbour in obj.reachableNeighbours(next):
                    updateVertices.append(neighbour)
                obj.k_m += heuristic(obj.last, obj.start)
                obj.last = obj.start
                obj.discovered_walls[next[0]][next[1]] = True
                for updateVertex in updateVertices:
                    obj.updateVertex(updateVertex)
                obj.computeShortestPath()

    obj.actions.append(getDirection(obj.finalVisitedNode, obj.start))
    problem._expanded = obj.countOfExpanded
    return obj.actions


# Abbreviations
rastar = replanningAStar
rbfs = replanningBFS
rdfs = replanningDFS
rucs = replanningUCS
dstar = dStarLiteSearch
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
