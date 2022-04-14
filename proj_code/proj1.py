# By RJ Gleaton, Jan 2021
from typing import List, Tuple, Set, Dict, Optional, cast
from environments.environment_abstract import Environment, State
from environments.farm_grid_world import FarmState
from heapq import heappush, heappop
import pdb


class Node:
    def __init__(self, state: State, path_cost: float, parent_action: Optional[int], parent, depth):
        self.state: State = state
        self.parent: Optional[Node] = parent
        self.path_cost: float = path_cost
        self.parent_action: Optional[int] = parent_action
        self.depth: int = depth

    def __hash__(self):
        return self.state.__hash__()

    def __gt__(self, other):
        return self.path_cost < other.path_cost

    def __eq__(self, other):
        return self.state == other.state


def get_next_state_and_transition_cost(env: Environment, state: State, action: int) -> Tuple[State, float]:
    rw, states_a, _ = env.state_action_dynamics(state, action)
    state: State = states_a[0]
    transition_cost: float = -rw

    return state, transition_cost


def expand_node(env: Environment, parent_node: Node) -> List[Node]:
    actions = env.get_actions()
    nodeList = []
    for i in actions:
        nextNodeState = get_next_state_and_transition_cost(env, parent_node.state, i)
        newNode = Node(nextNodeState[0], nextNodeState[1]+parent_node.path_cost, i, parent_node, parent_node.depth+1)
        nodeList.append(newNode)
    return nodeList
    pass


def get_soln(node: Node) -> List[int]:
    actionsList = []
    while True:
        if node.parent is None and node.parent_action is None:
            actionsList.reverse()
            return actionsList
        actionsList.append(node.parent_action)
        node = node.parent
    pass


def is_cycle(node: Node) -> bool:
    stateToCheck = node.state
    while True:
        if node.parent is None:
            #pdb.set_trace()
            return False
        if node.parent.state == stateToCheck:
            #pdb.set_trace()
            return True
        node = node.parent
    pass


def get_heuristic(node: Node) -> float:
    state: FarmState = cast(FarmState, node.state)
    return (abs(state.agent_idx[0]-state.goal_idx[0])+abs(state.agent_idx[1]-state.goal_idx[1]))
    pass

def get_cost(node: Node, heuristic: float, weight_g: float, weight_h: float) -> float:
    return ((weight_g*node.path_cost)+(weight_h*heuristic))
    pass


class BreadthFirstSearch:

    def __init__(self, state: State, env: Environment):
        self.env: Environment = env

        self.open: Set[Node] = set()
        self.fifo: List[Node] = []
        self.closed_set: Set[State] = set()

        # compute cost
        root_node: Node = Node(state, 0.0, None, None, 0)

        # push to open
        self.fifo.append(root_node)
        self.closed_set.add(root_node.state)

    def step(self):
        if len(self.fifo) > 0:
            curNode = self.fifo.pop(0)
        for i in expand_node(self.env, curNode):
            s = i.state
            if self.env.is_terminal(s):
                #pdb.set_trace()
                return i
            if not s in self.closed_set:
                self.closed_set.add(s)
                self.fifo.append(i)
        pass


class DepthLimitedSearch:

    def __init__(self, state: State, env: Environment, limit: float):
        self.env: Environment = env
        self.limit: float = limit

        self.lifo: List[Node] = []
        self.goal_node: Optional[Node] = None

        root_node: Node = Node(state, 0.0, None, None, 0)

        self.lifo.append(root_node)

    def step(self):
        #pdb.set_trace()
        currNode = self.lifo.pop()
        if self.env.is_terminal(currNode.state):
            #pdb.set_trace()
            return currNode
        if currNode.depth >= self.limit:
            #pdb.set_trace()
            return None
        elif not is_cycle(currNode):
            #pdb.set_trace()
            for i in expand_node(self.env, currNode):
                #if not i in visitedNodeList:
                    self.lifo.insert(len(self.lifo),i)
        pass


OpenSetElem = Tuple[float, Node]


class BestFirstSearch:

    def __init__(self, state: State, env: Environment, weight_g: float, weight_h: float):
        self.env: Environment = env
        self.weight_g: float = weight_g
        self.weight_h: float = weight_h

        self.priority_queue: List[OpenSetElem] = []
        self.closed_dict: Dict[State, Node] = dict()

        root_node: Node = Node(state, 0.0, None, None, 0)

        self.closed_dict[state] = root_node

        heuristic = get_heuristic(root_node)
        cost = get_cost(root_node, heuristic, self.weight_g, self.weight_h)
        heappush(self.priority_queue, (cost, root_node))

    def step(self):
        currNode = heappop(self.priority_queue)[1]
        if self.env.is_terminal(currNode.state):
            #pdb.set_trace()
            return currNode
        for i in expand_node(self.env, currNode):
            state = i.state
            if state not in self.closed_dict or i.path_cost < self.closed_dict[state].path_cost:
                self.closed_dict[state] = i
                newheuristic = get_heuristic(currNode)
                newCost = get_cost(i, newheuristic, self.weight_g, self.weight_h)
                heappush(self.priority_queue, (newCost, i))
        pass
