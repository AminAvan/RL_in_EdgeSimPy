from collections import deque
import random
import numpy as np
from Agents.MAD4PG.agent_distributed_mad4pg import DistributedD4PG
from Agents.MAD4PG.networks_mad4pg import make_default_networks


class MAD4PGServiceProvisioning:
    def __init__(self, agent_number, agent_action_size, environment_wrapper):
        self.agent_number = agent_number
        self.agent_action_size = agent_action_size
        self.environment_wrapper = environment_wrapper

        self.networks = make_default_networks(
            agent_number=self.agent_number,
            action_spec=self._get_action_spec(),
        )

        self.agent = DistributedD4PG(
            agent_number=self.agent_number,
            agent_action_size=self.agent_action_size,
            environment_wrapper=self.environment_wrapper,
            networks=self.networks,
            # ... other parameters as in run_mad4pg.py
        )

        self.program = self.agent.build()

        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64
        self.learning_frequency = 5
        self.step_counter = 0

    def __call__(self, parameters):
        if self.environment_wrapper.done:
            state = self.environment_wrapper.reset()
        else:
            state = self.environment_wrapper.state

        action = self.agent.select_action(state)
        next_state, reward, done, _ = self.environment_wrapper.step(action)

        self.replay_buffer.append((state, action, reward, next_state, done))

        self.step_counter += 1
        if self.step_counter % self.learning_frequency == 0:
            self.learn()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        self.agent.learner.step(states, actions, rewards, next_states, dones)

    def _get_action_spec(self):
        # Implement this based on your action space
        return np.zeros(self.agent_action_size)