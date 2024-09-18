from collections import deque
import random
import numpy as np
from .Agents.MAD4PG_agent_distributed import DistributedD4PG
from .Agents.MAD4PG_networks import make_default_networks
from acme import specs

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
            # environment_wrapper=None,
            networks=self.networks,
            num_actors=1,
            batch_size=256,
            prefetch_size=4,
            min_replay_size=1000,
            max_replay_size=1000000,
            samples_per_insert=8.0,
            n_step=1,
            sigma=0.3,
            discount=0.996,
            target_update_period=100,
            variable_update_period=1000,
            max_actor_steps=None,
            log_every=5.0,
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

    # def _get_action_spec(self):
    #     # Implement this based on your action space
    #     return np.zeros(self.agent_action_size)
    def _get_action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        action_shape = (self.agent_action_size, )
        return specs.BoundedArray(
            shape=action_shape,
            dtype=float,
            minimum=np.zeros(action_shape),
            maximum=np.ones(action_shape),
            name='actions'
        )