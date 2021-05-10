# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# kaggle_environments licensed under Copyright 2020 Kaggle Inc. and the Apache License, Version 2.0
# (see https://github.com/Kaggle/kaggle-environments/blob/master/LICENSE for details)

# wrapper of Hungry Geese environment from kaggle

import random
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# You need to install kaggle_environments, requests
from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import *
from ...environment import BaseEnvironment


def override_interpreter(state, env):
    print("!")
    configuration = Configuration(env.configuration)
    columns = configuration.columns
    rows = configuration.rows
    min_food = configuration.min_food
    state[0].observation = shared_observation = Observation(state[0].observation)
    hits = 0

    # Reset the environment.
    if env.done:
        agent_count = len(state)
        heads = sample(range(columns * rows), agent_count)
        shared_observation["geese"] = [[head] for head in heads]
        food_candidates = set(range(columns * rows)).difference(heads)
        # Ensure we only place as many food as there are open squares
        min_food = min(min_food, len(food_candidates))
        shared_observation["food"] = sample(food_candidates, min_food)
        return state

    geese = shared_observation.geese
    food = shared_observation.food

    # If there is no last state, reuse current state so that current action is never the opposite of the last action.
    last_state = env.steps[-1] if len(env.steps) > 1 else state
    # Apply the actions from active agents.
    for index, agent in enumerate(state):
        if agent.status != "ACTIVE":
            if agent.status != "INACTIVE" and agent.status != "DONE":
                # ERROR, INVALID, or TIMEOUT, remove the goose.
                geese[index] = []
            continue

        action = Action[agent.action]

        # Check action direction
        last_agent = last_state[index]
        last_action = Action[last_agent["action"]] if "action" in last_agent else action
        if last_action == action.opposite():
            env.debug_print(f"Opposite action: {agent.observation.index, action, last_action}")
            agent.status = "DONE"
            geese[index] = []
            hits +=1            
            continue

        goose = geese[index]
        head = translate(goose[0], action, columns, rows)

        # Consume food or drop a tail piece.
        if head in food:
            food.remove(head)
        else:
            goose.pop()

        # Self collision.
        if head in goose:
            env.debug_print(f"Body Hit: {agent.observation.index, action, head, goose}")
            agent.status = "DONE"
            geese[index] = []
            hits +=1
            continue

        while len(goose) >= configuration.max_length:
            # Free a spot for the new head if needed
            goose.pop()
        # Add New Head to the Goose.
        goose.insert(0, head)

        # If hunger strikes remove from the tail.
        if len(env.steps) % configuration.hunger_rate == 0:
            if len(goose) > 0:
                goose.pop()
            if len(goose) == 0:
                env.debug_print(f"Goose Starved: {action}")
                agent.status = "DONE"
                hits +=1
                continue

    goose_positions = histogram(
        position
        for goose in geese
        for position in goose
    )

    # Check for collisions.
    for index, agent in enumerate(state):
        goose = geese[index]
        if len(goose) > 0:
            head = geese[index][0]
            if goose_positions[head] > 1:
                env.debug_print(f"Goose Collision: {agent.action}")
                agent.status = "DONE"
                geese[index] = []
                hits +=1

    # Add food if min_food threshold reached.
    needed_food = min_food - len(food)
    if needed_food > 0:
        collisions = {
            position
            for goose in geese
            for position in goose
        }
        available_positions = set(range(rows * columns)).difference(collisions).difference(food)
        # Ensure we don't sample more food than available positions.
        needed_food = min(needed_food, len(available_positions))
        food.extend(sample(available_positions, needed_food))

    # Set rewards after deleting all geese to ensure that geese don't receive a reward on the turn they perish.
    for index, agent in enumerate(state):
        if agent.status == "ACTIVE":
            agent.reward = (configuration.max_length + 1)*(hits+1) + len(geese[index]) 

            # Adding 1 to len(env.steps) ensures that if an agent gets reward 4507, it died on turn 45 with length 7.

    # If only one ACTIVE agent left, set it to DONE.
    active_agents = [a for a in state if a.status == "ACTIVE"]
    if len(active_agents) == 1:
        agent = active_agents[0]
        agent.status = "DONE"

    return state

class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = torch.cat([x[:,:,:,-self.edge_size[1]:], x, x[:,:,:,:self.edge_size[1]]], dim=3)
        h = torch.cat([h[:,:,-self.edge_size[0]:], h, h[:,:,:self.edge_size[0]]], dim=2)
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h


class GeeseNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32

        self.conv0 = TorusConv2d(17, filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 4, bias=False)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x, _=None):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = self.head_p(h_head)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))

        return {'policy': p, 'value': v}


class Environment(BaseEnvironment):
    ACTION = ['NORTH', 'SOUTH', 'WEST', 'EAST']
    NUM_AGENTS = 4

    def __init__(self, args={}):
        super().__init__()
        self.env = make("hungry_geese")
        self.env.interpreter = override_interpreter
        self.reset()

    def reset(self, args={}):
        obs = self.env.reset(num_agents=self.NUM_AGENTS)
        self.update((obs, {}), True)

    def update(self, info, reset):
        obs, last_actions = info
        if reset:
            self.obs_list = []
        self.obs_list.append(obs)
        self.last_actions = last_actions

    def action2str(self, a, player=None):
        return self.ACTION[a]

    def str2action(self, s, player=None):
        return self.ACTION.index(s)

    def direction(self, pos_from, pos_to):
        if pos_to is None:
            return None
        x_from, y_from = pos_from // 11, pos_from % 11
        x_to, y_to = pos_to // 11, pos_to % 11
        if x_from == x_to:
            if (y_from + 1) % 11 == y_to:
                return 3
            if (y_from - 1) % 11 == y_to:
                return 2
        if y_from == y_to:
            if (x_from + 1) % 7 == x_to:
                return 1
            if (x_from - 1) % 7 == x_to:
                return 0

    def __str__(self):
        # output state
        obs = self.obs_list[-1][0]['observation']
        colors = ['\033[33m', '\033[34m', '\033[32m', '\033[31m']
        color_end = '\033[0m'

        def check_cell(pos):
            for i, geese in enumerate(obs['geese']):
                if pos in geese:
                    if pos == geese[0]:
                        return i, 'h'
                    if pos == geese[-1]:
                        return i, 't'
                    index = geese.index(pos)
                    pos_prev = geese[index - 1] if index > 0 else None
                    pos_next = geese[index + 1] if index < len(geese) - 1 else None
                    directions = [self.direction(pos, pos_prev), self.direction(pos, pos_next)]
                    return i, directions
            if pos in obs['food']:
                return 'f'
            return None

        def cell_string(cell):
            if cell is None:
                return '.'
            elif cell == 'f':
                return 'f'
            else:
                index, directions = cell
                if directions == 'h':
                    return colors[index] + '@' + color_end
                elif directions == 't':
                    return colors[index] + '*' + color_end
                elif max(directions) < 2:
                    return colors[index] + '|' + color_end
                elif min(directions) >= 2:
                    return colors[index] + '-' + color_end
                else:
                    return colors[index] + '+' + color_end

        cell_status = [check_cell(pos) for pos in range(7 * 11)]

        s = 'turn %d\n' % len(self.obs_list)
        for x in range(7):
            for y in range(11):
                pos = x * 11 + y
                s += cell_string(cell_status[pos])
            s += '\n'
        for i, geese in enumerate(obs['geese']):
            s += colors[i] + str(len(geese) or '-') + color_end + ' '
        return s

    def step(self, actions):
        # state transition
        state = self.env.step([self.action2str(actions.get(p, None) or 0) for p in self.players()])

        self.update((state, actions), False)

    def diff_info(self, _):
        return self.obs_list[-1], self.last_actions

    def turns(self):
        # players to move
        return [p for p in self.players() if self.obs_list[-1][p]['status'] == 'ACTIVE']

    def terminal(self):
        # check whether terminal state or not
        for obs in self.obs_list[-1]:
            if obs['status'] == 'ACTIVE':
                return False
        return True

    def outcome(self):
        # return terminal outcomes
        # 1st: 1.0 2nd: 0.33 3rd: -0.33 4th: -1.00
        rewards = {o['observation']['index']: o['reward'] for o in self.obs_list[-1]}
        outcomes = {p: 0 for p in self.players()}
        for p, r in rewards.items():
            for pp, rr in rewards.items():
                if p != pp:
                    if r > rr:
                        outcomes[p] += 1 / (self.NUM_AGENTS - 1)
                    elif r < rr:
                        outcomes[p] -= 1 / (self.NUM_AGENTS - 1)
        return outcomes

    def legal_actions(self, player):
        # return legal action list
        return list(range(len(self.ACTION)))

    def action_length(self):
        # maximum action label (it determines output size of policy function)
        return len(self.ACTION)

    def players(self):
        return list(range(self.NUM_AGENTS))

    def rule_based_action(self, player):
        from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, GreedyAgent
        action_map = {'N': Action.NORTH, 'S': Action.SOUTH, 'W': Action.WEST, 'E': Action.EAST}

        agent = GreedyAgent(Configuration({'rows': 7, 'columns': 11}))
        agent.last_action = action_map[self.ACTION[self.last_actions[player]][0]] if player in self.last_actions else None
        obs = {**self.obs_list[-1][0]['observation'], **self.obs_list[-1][player]['observation']}
        action = agent(Observation(obs))
        return self.ACTION.index(action)

    def net(self):
        return GeeseNet

    def observation(self, player=None):
        if player is None:
            player = 0

        b = np.zeros((self.NUM_AGENTS * 4 + 1, 7 * 11), dtype=np.float32)
        obs = self.obs_list[-1][0]['observation']

        for p, geese in enumerate(obs['geese']):
            # head position
            for pos in geese[:1]:
                b[0 + (p - player) % self.NUM_AGENTS, pos] = 1
            # tip position
            for pos in geese[-1:]:
                b[4 + (p - player) % self.NUM_AGENTS, pos] = 1
            # whole position
            for pos in geese:
                b[8 + (p - player) % self.NUM_AGENTS, pos] = 1

        # previous head position
        if len(self.obs_list) > 1:
            obs_prev = self.obs_list[-2][0]['observation']
            for p, geese in enumerate(obs_prev['geese']):
                for pos in geese[:1]:
                    b[12 + (p - player) % self.NUM_AGENTS, pos] = 1

        # food
        for pos in obs['food']:
            b[16, pos] = 1

        return b.reshape(-1, 7, 11)



if __name__ == '__main__':
    e = Environment()
    for _ in range(100):
        e.reset()
        while not e.terminal():
            print(e)
            actions = {p: e.legal_actions(p) for p in e.turns()}
            print([[e.action2str(a, p) for a in alist] for p, alist in actions.items()])
            e.step({p: random.choice(alist) for p, alist in actions.items()})
        print(e)
        print(e.outcome())
