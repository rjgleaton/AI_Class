from typing import List, Dict
from environments.environment_abstract import Environment, State
import numpy as np
import pdb
import random


def policy_evaluation(env: Environment, states: List[State], state_values: Dict[State, float],
                      policy: Dict[State, List[float]], discount: float) -> Dict[State, float]:

    delta = float('inf')
    actions = env.get_actions()

    while delta > 0.0:
        delta = 0.0

        for state in states:
            values = []
            v = state_values[state]

            for action in actions:
                eval = env.state_action_dynamics(state, action)
                sum = 0.0

                for i, new_state in enumerate(eval[1]):
                    sum += (eval[2][i]*state_values[new_state])
                values.append(policy[state][action]*(eval[0]+discount*sum))

            state_values[state] = np.sum(values)

            values.clear()

            delta = max(delta, abs(v-state_values[state]))

    return state_values


def policy_improvement(env: Environment, states: List[State], state_values: Dict[State, float],
                       discount: float) -> Dict[State, List[float]]:
    policy_new: Dict[State, List[float]] = dict()
    print(policy_new)
    values_list = []

    for state in states:

        for action in env.get_actions():
            eval = env.state_action_dynamics(state, action)
            sum = 0.0

            for i, new_state in enumerate(eval[1]):
                sum += eval[2][i]*state_values[new_state]
            values_list.append(eval[0]+discount*sum)

        #There's better ways to do this but I have to leave for work 5 minutrs ago
        policy_new[state] = [0.0]*len(values_list)
        policy_new[state][np.argmax(values_list)] = 1.0

        values_list.clear()

    print("returning new policy")
    return policy_new

def q_learning_step(env: Environment, state: State, action_values: Dict[State, List[float]], epsilon: float,
                    learning_rate: float, discount: float):

    # TODO implement
    action = 0
    random_num = np.random.uniform(0.0,1.0)
    if random_num <= epsilon:
        action = random.choice(env.get_actions())
    else:
        action = np.argmax(action_values[state])

    Snew, R = env.sample_transition(state, action)

    Qnew = []
    for action_list in env.get_actions():
        Qnew.append(action_values[Snew][action_list])

    action_values[state][action] = (action_values[state][action] + learning_rate*(R+discount*max(Qnew)-action_values[state][action]))

    return Snew, action_values
