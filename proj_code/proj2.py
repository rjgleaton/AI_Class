from environments.connect_four import ConnectFourState, ConnectFour
#import scipy.signal
#import numpy as np
import pdb

def heuistic(state: ConnectFourState) -> float:
    """
    Returns the heuristic value for the given state
    :param state: connect four game state
    :return: heuristic value
    """
    '''
    #Verticle, horizontal, top-left-to-top-right, top-right-to-top-left
    kernels = [ [[1],[1],[1]], [[1,1,1]], np.eye(3, dtype=int), np.fliplr(np.eye(3, dtype=int))]

    #Track num of pieces in a row
    maxDub = 0
    minDub = 0
    for kernel in  kernels:
        #print("Grid: ")
        #print(state.grid)
        #print(" ")
        conv2D = scipy.signal.convolve2d(state.grid, kernel, mode="valid")
        #print(conv2D)
        if (conv2D == 3).any():
            maxDub += 1
        if (conv2D == -3).any():
            minDub += 1

    #pdb.set_trace()
    #print("Max: ",maxDub,", Min: ",minDub)
    return maxDub - minDub
    '''

    maxDub = 0
    minDub = 0
    try:
        for i in range(len(state.grid)):
            for j in range(len(state.grid[0]), 0, -1):
                if state.grid[i][j-1] == 1:
                    if (state.grid[i][j-2] == 1) and (state.grid[i][j-3] == 1):
                        maxDub += 1
                    if (state.grid[i+1][j-2] == 1) and (state.grid[i+2][j-3] == 1):
                        maxDub += 1
                    if (state.grid[i-1][j-2] == 1) and (state.grid[i-2][j-3] == 1):
                        maxDub += 1

                if state.grid[i][j-1] == -1:
                    if (state.grid[i][j-2] == -1) and (state.grid[i][j-3] == -1):
                        minDub += 1
                    if (state.grid[i+1][j-2] == -1) and (state.grid[i+2][j-3] == -1):
                        minDub += 1
                    if (state.grid[i-1][j-2] == -1) and (state.grid[i-2][j-3] == -1):
                        minDub += 1
    except IndexError:
        pass
    return maxDub - minDub
    pass


def heuristic_minimax_search(state: ConnectFourState, env: ConnectFour, depth: int) -> int:
    """
    Returns the action to take
    :param state: connect four game state
    :param env: connect four environment
    :param depth: maximum search depth
    :return: action to take
    """
    value, move = max_value(state, env, depth)
    return move
    pass


def max_value(state: ConnectFourState, env: ConnectFour, depth: int):
        #pdb.set_trace()
    if env.is_terminal(state) and depth == 0:
            return env.utility(state), None

    if depth == 0 and not env.is_terminal(state):
        return heuistic(state), 0

    v = float('-inf')
    #pdb.set_trace()
    for actions in env.get_actions(state):
        v2, actions2 = min_value(env.next_state(state, actions), env, depth-1)
        if v2 > v:
            v, move = v2, actions

    #pdb.set_trace()
    return v, move


def min_value(state: ConnectFourState, env: ConnectFour, depth: int):
    if env.is_terminal(state) and depth == 0:
            return env.utility(state), None

    if depth == 0 and not env.is_terminal(state):
        return heuistic(state), 0

    #pdb.set_trace()
    v = float('inf')
    for actions in env.get_actions(state):
        v2, actions2 = max_value(env.next_state(state, actions), env, depth-1)
        if v2 < v:
            v, move = v2, actions

    #pdb.set_trace()
    return v, move


