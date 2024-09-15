import numpy as np
from edge_sim_py import *


class EdgeSimPyWrapper:
    def __init__(self, simulator, input_file):
        self.simulator = simulator
        self.input_file = input_file
        self.state = None
        self.reward = 0
        self.done = False
        self.current_step = 0

    def reset(self):
        self.simulator.initialize(self.input_file)
        self.state = self._get_state()
        return self.state

    def step(self, action):
        # Apply the action
        self._apply_action(action)

        self.simulator.resource_management_algorithm(None)

        # Advance the simulation
        self.simulator.step()

        # current step
        self.current_step = self.simulator.schedule.steps

        next_state = self._get_state()
        reward = self._calculate_reward()
        done = self.simulator.stopping_criterion(self.simulator)

        self.state = next_state
        self.reward = reward
        self.done = done

        return next_state, reward, done, {}

    def _get_state(self):
        # Implement this based on EdgeSimPy's state representation
        # You'll need to extract relevant information from the simulator
        # This is a placeholder implementation
        edge_server_state = self._get_edge_server_state()
        service_state = self._get_service_state()
        network_state = self._get_network_state()
        return np.concatenate([edge_server_state, service_state, network_state])

    def _get_edge_server_state(self):
        # Extract state information about edge servers
        # return np.array([server.cpu.usage for server in EdgeServer.all()])  ### was
        return np.array([server.total_cpu_utilization for server in EdgeServer.all()])  ### is

    def _get_service_state(self):
        # Extract state information about services
        return np.array([service.processing_power_demand for service in Service.all()])

    def _get_network_state(self):
        # Extract state information about the network
        return np.array([link.bandwidth for link in NetworkFlow.all()])

    def _apply_action(self, action):
        # Implement this to apply the action in EdgeSimPy
        # This is a placeholder implementation
        for i, service in enumerate(Service.all()):
            if i < len(action):
                target_server = EdgeServer.all()[action[i]]
                service.provision(target_server=target_server)

    def _calculate_reward(self):
        total_usage = sum(server.total_cpu_utilization for server in EdgeServer.all())
        # total_capacity = sum(server.cpu.capacity for server in EdgeServer.all())
        total_capacity = len(EdgeServer.all())

        print(f"total_usage: {total_usage}")
        print(f"total_capacity: {total_capacity}")
        return -total_usage / total_capacity  # Negative reward for high usage