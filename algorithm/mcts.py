import copy
import math
import random
import time
from collections import defaultdict
from copy import copy

TOTAL_STORAGE = 0
STORAGE_THRESHOLD = 0
AVAILABLE_CHOICES = None
ATOMIC_CHOICES = None
# available_choices: {I1, I2, I3} --> atomic_choices: {(), (I1), (I2), (I3), (I1, I2), (I1, I3), ...}
WORKLOAD = None
MAX_INDEX_NUM = 0
CANDIDATE_SUBSET = defaultdict(list)
CANDIDATE_SUBSET_BENEFIT = defaultdict(list)


def find_best_benefit(choices):
    if choices[-1] in CANDIDATE_SUBSET.keys() and set(choices[:-1]) in CANDIDATE_SUBSET[choices[-1]]:
        return CANDIDATE_SUBSET_BENEFIT[choices[-1]][CANDIDATE_SUBSET[choices[-1]].index(set(choices[:-1]))]

    total_benefit = infer_workload_benefit(WORKLOAD, choices, ATOMIC_CHOICES)

    CANDIDATE_SUBSET[choices[-1]].append(set(choices[:-1]))
    CANDIDATE_SUBSET_BENEFIT[choices[-1]].append(total_benefit)
    return total_benefit


def get_diff(available_choices, choices):
    return set(available_choices).difference(set(choices))


class State:
    def __init__(self):
        self.current_storage = 0.0
        self.current_benefit = 0.0
        # record the sum of choices up to the current state
        self.accumulation_choices = []
        # record available choices of current state
        self.available_choices = []
        self.displayable_choices = []

    def reset_state(self):
        # TODO. 需要在此生成候选索引子集
        self.set_available_choices(set(AVAILABLE_CHOICES).difference(self.accumulation_choices))

    def get_available_choices(self):
        return self.available_choices

    def set_available_choices(self, choices):
        self.available_choices = choices

    def get_current_storage(self):
        return self.current_storage

    def set_current_storage(self, value):
        self.current_storage = value

    def get_current_benefit(self):
        return self.current_benefit

    def set_current_benefit(self, value):
        self.current_benefit = value

    def get_accumulation_choices(self):
        return self.accumulation_choices

    def set_accumulation_choices(self, choices):
        self.accumulation_choices = choices

    def is_terminal(self):
        # The current node is a leaf node.
        return len(self.accumulation_choices) == MAX_INDEX_NUM

    def take_action(self, action):
        if not self.available_choices:
            return None
        self.available_choices.remove(action)
        choices = copy.copy(self.accumulation_choices)
        choices.append(action)
        benefit = find_best_benefit(choices) + self.current_benefit
        # If the current choice does not satisfy restrictions, then continue to get the next choice.
        if benefit <= self.current_benefit or \
                self.current_storage + action.get_storage() > STORAGE_THRESHOLD:
            return self.take_action(random.choice(self.available_choices))

        next_state = State()
        # Initialize the properties of the new state.
        next_state.set_accumulation_choices(choices)
        next_state.set_current_benefit(benefit)
        next_state.set_current_storage(self.current_storage + action.get_storage())
        next_state.set_available_choices(get_diff(AVAILABLE_CHOICES, choices))
        return next_state

    def __repr__(self):
        self.displayable_choices = ['{}: {}'.format(choice.get_table(), choice.get_columns())
                                    for choice in self.accumulation_choices]
        return "benefit: {}, storage :{}, choices: {}".format(
            self.current_benefit, self.current_storage, self.displayable_choices)


def randomPolicy(node):
    t1 = 0
    startTime = time.time()
    while not node.isTerminal:
        try:
            temp = node.state.get_available_choices()
            action = random.choice(list(temp))
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(node.state))
        newNode = treeNode(node.state.take_action(action), node)
        node.children[action] = newNode
        if len(node.state.get_available_choices()) == len(node.children):
            node.isFullyExpanded = True
        node = newNode

    benefit = find_best_benefit(node.state.get_accumulation_choices())
    t1 += time.time() - startTime
    return node, benefit, t1


class mcts:
    def __init__(self, iteration_limit=None, exploration_constant=1 / math.sqrt(16),
                 rollout_policy=randomPolicy):
        self.root = None
        if iteration_limit is None:
            raise ValueError("Must have either a time limit or an iteration limit")
        # number of iterations of the search
        if iteration_limit < 1:
            raise ValueError("Iteration limit must be greater than one")
        self.searchLimit = iteration_limit
        self.explorationConstant = exploration_constant
        self.rollout = rollout_policy
        self.nntime = 0
        self.nntime_no_feature = 0
        global getPossibleActionsTime
        getPossibleActionsTime = 0
        global takeActionTime
        takeActionTime = 0

    def search(self, initial_state):
        self.root = treeNode(initial_state, None)
        for i in range(self.searchLimit):
            self.executeRound()

    def continue_search(self):
        for i in range(self.searchLimit):
            self.executeRound()

    def executeRound(self):
        node = self.select_node(self.root)
        # newState = deepcopy(node.state)
        startTime = time.time()
        node, reward, nntime_no_feature = self.rollout(node)
        self.nntime += time.time() - startTime
        self.nntime_no_feature += nntime_no_feature
        self.back_propogate(node, reward)

    def select_node(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.get_best_child(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        # expand 方法目前简单地选择了第一个可用的动作。这是一个简单的策略，但并不一定是最优的。
        # 可以考虑一些更复杂的策略来选择扩展哪个动作，如贪婪策略、UCB1算法等。
        actions = node.state.get_available_choices()
        for action in actions:
            if action not in node.children:
                newNode = treeNode(node.state.take_action(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                # if newNode.isTerminal:
                #     print(newNode)
                return newNode
        print(len(actions), len(node.children))
        raise Exception("Should never reach here")

    def back_propogate(self, node, reward):
        # print(reward)
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def get_best_child(self, node, exploration_value):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = child.totalReward / child.numVisits + exploration_value * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def get_action(self, root, best_child):
        for action, node in root.children.items():
            if node is best_child:
                return action


class treeNode:
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.is_terminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}
