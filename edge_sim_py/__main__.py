from edge_sim_py import *
import os
import networkx as nx
import msgpack
import itertools
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import requests
import ast
import time
import psutil
from functools import wraps
# from pyrapl import measurement
import subprocess

### rl down ###
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys

import my_rl
####################################################

class ResourceTracker:
    def __init__(self):
        self.total_memory = 0
        self.total_power = 0
        self.call_count = 0
        self.start_time = time.time()

    def update(self, memory):
        self.total_memory += memory
        self.call_count += 1

        # Define your CPU's idle and max power consumption (in watts)
        P_idle = 5  # Example idle power in watts (adjust based on your CPU)
        P_max = 98  # Example max power in watts at full load (adjust based on your CPU)

        # Estimate power consumption based on CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        # Apply the CPU power consumption formula
        power_estimate = P_idle + (P_max - P_idle) * (cpu_percent / 100)
        # Add the power estimate to the total power consumption
        self.total_power += power_estimate

    def report(self):
        total_time = time.time() - self.start_time
        print(f"runtime: {total_time:.2f} seconds")
        print(f"memory consumption: {self.total_memory / (1024 * 1024):.2f} MB until {total_time:.2f} seconds")
        print(f"power consumption: {self.total_power:.2f} Watt-seconds until {total_time:.2f} seconds")
        # print(f"Number of calls: {self.call_count}")


    def final_report(self):
        print(f"Total memory consumption: {self.total_memory / (1024 * 1024):.2f} MB")
        print(f"Total power consumption: {self.total_power:.2f} Watt-seconds")

resource_tracker = ResourceTracker()

#################################################################
# Adjust display settings to show all columns and increase width
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# Set the maximum number of rows to display
pd.set_option('display.max_rows', 100000)  # Set this to the desired number, or None for unlimited rows
# Adjust the print options to display the full array
np.set_printoptions(threshold=np.inf)
#################################################################


# The function "has_capacity_to_host_only_resources(self, service)" is a modified version of the original
# "has_capacity_to_host" function in edge_sim_py/components/edge_server.py. Unlike "has_capacity_to_host",
# which only considers the number of cores and assumes a service requires a specific number of cores,
# "has_capacity_to_host_only_resources(self, service)" function also accounts for the clock frequency of the cores.
# This modification provides a more fair way of comparison of the original method in EdgeSimPy with our new method
# of investigating the proper edge server for the edge user application.
def has_capacity_to_host_only_resources(self, service):
    # Calculating the additional disk demand that would be incurred to the edge server
    additional_disk_demand = self._get_disk_demand_delta(service=service)

    # Calculating the edge server's free resources
    free_processing_power = self.processing_power - self.processing_power_demand
    free_memory = self.memory - self.memory_demand
    free_disk = self.disk - self.disk_demand

    # Checking if the host would have resources to host
    if (free_processing_power >= service.processing_power_demand and free_memory >= service.memory_demand and free_disk >= additional_disk_demand):
        EdgeServer.is_potential_host = EdgeServer.is_potential_host + 1
        # print(f"EdgeServer.is_potential_host {EdgeServer.is_potential_host}")
        self.execution_time_of_service[str(service.id)] = (service.processing_power_demand / self.processing_power)
        can_host = True
    else:
        can_host = False

    return can_host

# Best-fit is used as a baseline for scheduling services of edge user applications across edge servers.
def Best_Fit_Service_Provisioning(parameters):
    # Override 'has_capacity_to_host'
    EdgeServer.has_capacity_to_host = has_capacity_to_host_only_resources

    # We can always call the 'all()' method to get a list with all created instances of a given class
    for service in Service.all():
        # We don't want to migrate services are already being migrated
        if service.server == None and not service.being_provisioned:

            edge_servers = sorted(
            EdgeServer.all(),
            key=lambda s: ((s.processing_power - s.processing_power_demand) * (s.memory - s.memory_demand) * (s.disk - s.disk_demand)) ** (1 / 3),
            reverse=False,
            )

            # Let's iterate over the list of edge servers to find a suitable host for our service
            for edge_server in edge_servers:

                # We must check if the edge server has enough resources to host the service
                if edge_server.has_capacity_to_host(service=service):
                    # Start provisioning the service in the edge server
                    service.provision(target_server=edge_server)

                    # After start migrating the service we can move on to the next service
                    break


# Our proposed mechanism to check properly with considering multi-attributes in a proper way to realize if the candidate edge server poses enough resources (including: processor, memory, locak disk, communication latency) for the services of applications of edge users.
# The total amount of 'processing power demand' of a service is calculated by 'processor cycles per megabyte * required memory size'.
# Then, we calculate the 'exection time' of the service by 'processing_power_demand / free_processing_power', where the 'free_processing_power' is the total processing power of edge server that is obtained by 'self.cpu_cycle * self.cpu' (where self.cpu is the number of core of each edge server)
# Then, by having the 'exection time' of the service on this specific edge server, we calculate the service_utilization based on 'exection time / deadline'
# now we have the 'service utilization' now we check if the processor of the edge server has enough space for the service, simultanesouly the memory demand of the service and the disk demand of the service would be check to see if the edge server has enough space
# after all, if the edge server has enough capacity regarding processor, memory, and disk; from this function
def has_capacity_to_host_proposed(self, service: object) -> bool:
    # Calculating the additional disk demand that would be incurred to the edge server
    additional_disk_demand = self._get_disk_demand_delta(service=service)

    # Calculating the edge server's free resources
    # free_cpu_cycle = (self.cpu_cycle * self.cpu) - service.processing_power_demand
    free_memory = self.memory - self.memory_demand
    free_disk = self.disk - self.disk_demand
    free_processing_power = self.processing_power
    

    user_service_deadline = next(iter(service.application.users[0].delay_slas.values()))
    user_service_exe_time = (service.processing_power_demand / free_processing_power)
    user_service_utilization = (user_service_exe_time / user_service_deadline)
    free_cpu_utilization = self.total_cpu_utilization + user_service_utilization

    # Checking if the host would have resources to host the registry and its (additional) layers
    if (free_cpu_utilization <= 1 and free_memory >= service.memory_demand and free_disk >= additional_disk_demand):  ### if (free_cpu >= service.cpu_demand and free_memory >= service.memory_demand and free_disk >= additional_disk_demand):
        # print(f"free_cpu_utilization: {free_cpu_utilization}, free_memory: {free_memory}, free_disk: {free_disk}")
        # calculating true runtime of service on the host server
        self.execution_time_of_service[str(service.id)] = user_service_exe_time
        self.total_cpu_utilization = (self.total_cpu_utilization + user_service_utilization)

        # # print(f"free_cpu_cycle of {self}:{free_cpu_cycle}")
        # if (free_cpu_cycle < 0):
        #     EdgeServer.is_negative_freq_capacity = EdgeServer.is_negative_freq_capacity + 1

        EdgeServer.is_potential_host = EdgeServer.is_potential_host + 1
        # print(f"EdgeServer.is_potential_host: {EdgeServer.is_potential_host}.")
        if(len(service.all()) == EdgeServer.is_potential_host):
            # print(f"All services are hosted!")
            # print(f"missed applications: {EdgeServer.is_negative_freq_capacity}, users: {len(User.all())}, miss-ratio of users: {EdgeServer.is_negative_freq_capacity/(len(User.all()))}")
            s_total_cpu_util = 0
            s_total_mem_util = 0
            for s in EdgeServer.all():
                s_total_cpu_util += round(s.total_cpu_utilization, 2)
                s_total_mem_util += round((s.memory_demand/s.memory), 2)
            # print(f"Average CPU load of all edge servers: {((s_total_cpu_util/len(EdgeServer.all()))*100)}%")
            # print(f"Average memory load of all edge servers: {((s_total_mem_util / len(EdgeServer.all())) * 100)}%\n")

        can_host = True
    else:
        can_host = False
    return can_host


def MASS(parameters):
    # Override 'has_capacity_to_host' for all instances of the EdgeServer class
    EdgeServer.has_capacity_to_host = has_capacity_to_host_proposed

    priorities_list = []
    for usr in User.all():
        # Calculating the urgency of each user's deadline
        priority = 1 / list(usr.delay_slas.values())[0]

        # Assign users along sith their deadline-priority
        priorities_list.append((usr, priority))

    # Sort the priorities_list based on deadline
    sorted_priorities_list = sorted(priorities_list, key=lambda x: (x[1]), reverse=True)

    for user in sorted_priorities_list:
        for service in user[0].applications[0].services:
            # We don't want to migrate services are already being migrated
            if service.server == None and not service.being_provisioned:

                lowest_rtt = []
                rtt = user[0].base_station.wireless_delay
                for edge_server in EdgeServer.all():
                    rtt = rtt + edge_server.base_station.wireless_delay
                    server_priority = 1 / (rtt + edge_server.base_station.wireless_delay)

                    # Assign servers along with their rtt
                    lowest_rtt.append((edge_server, priority))

                # Sort the lowest_rtt based on rtt
                sorted_lowest_rtt = sorted(lowest_rtt, key=lambda x: (x[1]), reverse=True)

                # Let's iterate over the list of edge servers to find a suitable host for our service
                for edge_server in sorted_lowest_rtt:

                    # We must check if the edge server has enough resources to host the service
                    if edge_server[0].has_capacity_to_host(service=service):
                        # Start provisioning the service in the edge server
                        service.provision(target_server=edge_server[0])

                        # After start migrating the service we can move on to the next service
                        break


def lapse(parameters):
    """A cost-based heuristic algorithm to optimize the placement of applications on
        heterogeneous edge computing infrastructures [1].

        [1] Kayser, Carlos Henrique, Marcos Dias de Assunção, and Tiago Ferreto.
        "Lapse: Latency & Power-Aware Placement of Data Stream Applications on Edge Computing."
        CLOSER. 2024.

        Args:
            parameters (dict, optional): Algorithm parameters. Defaults to {}.
    """

    EdgeServer.has_capacity_to_host = has_capacity_to_host_proposed

    def find_shortest_path(origin_network_switch: object, target_network_switch: object) -> int:
        topology = origin_network_switch.model.topology
        path = []

        if not hasattr(topology, "delay_shortest_paths"):
            topology.delay_shortest_paths = {}

        key = (origin_network_switch, target_network_switch)

        if key in topology.delay_shortest_paths.keys():
            path = topology.delay_shortest_paths[key]
        else:
            path = nx.shortest_path(G=topology, source=origin_network_switch, target=target_network_switch,
                                    weight="delay")
            topology.delay_shortest_paths[key] = path

        return path

    def calculate_path_delay(origin_network_switch: object, target_network_switch: object) -> int:
        topology = origin_network_switch.model.topology

        path = find_shortest_path(origin_network_switch=origin_network_switch,
                                  target_network_switch=target_network_switch)
        delay = topology.calculate_path_delay(path=path)

        return delay

    def get_all_shortest_path_between(origin_network_switch: object, target_network_switch: object) -> int:
        topology = origin_network_switch.model.topology

        paths = list(
            nx.all_shortest_paths(topology, source=origin_network_switch, target=target_network_switch, weight="delay"))

        return paths

    def get_edge_servers_metadata(source: object, app: object, edge_servers: object = None) -> list:
        metadata = []

        # edge_servers_list = edge_servers if edge_servers else EdgeServer.all()

        # for edge_server in edge_servers_list:
        for edge_server in EdgeServer.all():
            # Compute the percentage of services that can be hosted on the edge server
            app_demand = 0
            for service in app.services:
                if not service.server:
                    # app_demand += service.input_event_rate * service.mips_demand ### was
                    app_demand += service.cpu_demand * service.cpu_cycles_demand * service.memory_demand

            edge_server_attrs = {
                "object": edge_server,
                "path_delay_source": calculate_path_delay(source, edge_server.network_switch),
                # "path_delay_sink": calculate_path_delay(app.services[-1].server.network_switch,
                #                                         edge_server.network_switch),
                # "path_delay_sink": calculate_path_delay(edge_server.network_switch, edge_server.network_switch),
                "max_power_consumption": edge_server.power_model_parameters["max_power_consumption"],
            }

            metadata.append(edge_server_attrs)

        return metadata

    def get_app_total_demand(application: Application) -> float:
        app_demand = 0
        for service in application.services:
            # app_demand += service.input_event_rate * service.mips_demand  ## was
            app_demand += service.cpu_demand * service.cpu_cycles_demand * service.memory_demand
            # app_demand += service.cpu_demand * service.cpu_cycles_demand

        return app_demand

    def get_edge_servers_between(source: object, target: object) -> list:
        topology = source.model.topology

        best_path_servers = 0
        possible_edge_servers = None

        paths_between_sensor_and_target = get_all_shortest_path_between(source, target)

        for path in paths_between_sensor_and_target:
            edge_servers_in_path = []

            # search for edge servers in the path
            for switch in path:
                for es in switch.edge_servers:
                    if es not in edge_servers_in_path:
                        edge_servers_in_path.append(es)

                # search for edge servers on neighboors network switches
                for neighbor in list(topology.neighbors(switch)):
                    for es in neighbor.edge_servers:
                        if es not in edge_servers_in_path:
                            edge_servers_in_path.append(es)

            # Choose the path with the most edge servers
            if len(edge_servers_in_path) > best_path_servers:
                best_path_servers = len(edge_servers_in_path)
                possible_edge_servers = edge_servers_in_path

        return possible_edge_servers

    def find_minimum_and_maximum(metadata: list):
        min_and_max = {
            "minimum": {},
            "maximum": {},
        }

        for metadata_item in metadata:
            for attr_name, attr_value in metadata_item.items():
                if attr_name != "object":
                    # Updating the attribute's minimum value
                    if (
                            attr_name not in min_and_max["minimum"]
                            or attr_name in min_and_max["minimum"]
                            and attr_value < min_and_max["minimum"][attr_name]
                    ):
                        min_and_max["minimum"][attr_name] = attr_value

                    # Updating the attribute's maximum value
                    if (
                            attr_name not in min_and_max["maximum"]
                            or attr_name in min_and_max["maximum"]
                            and attr_value > min_and_max["maximum"][attr_name]
                    ):
                        min_and_max["maximum"][attr_name] = attr_value

        return min_and_max

    def min_max_norm(x, min, max):
        if min == max:
            return 1
        return (x - min) / (max - min)

    def get_norm(metadata: dict, attr_name: str, min: dict, max: dict) -> float:
        normalized_value = min_max_norm(x=metadata[attr_name], min=min[attr_name], max=max[attr_name])
        return normalized_value

    ### main LAPSE ###
    apps = Application.all()

    # Sorts applications based on their processing time SLA (from lowest to highest),
    # number of services (from highest to lowest), and input demand (from highest to lowest)
    apps = sorted(
        apps,
        key=lambda app: (
            # -get_app_total_demand(app),   ## was
            # app.processing_latency_sla,   ## was
            get_app_total_demand(app),
            -list(app.users[0].delay_slas.values())[0]
        ),
    )

    for app in apps:
        # for base_station in BaseStation.all() if (len(base_station.edgeservers) > 0):
        source = app.users[0].base_station.network_switch
        # print(f"app.users[0]: {app.users[0]}")
        # sink = app.services[-1].server.network_switch
        # sink = edgeserver.base_station.network_switch

        # possible_edge_servers = get_edge_servers_between(source, sink)   ### was
        possible_edge_servers = EdgeServer.all()

        for service in app.services:
            # if service.server:
            if service.server != None and service.being_provisioned:
                continue

            # while not service.server:
            while service.server == None and not service.being_provisioned:
                edge_servers_metadata = get_edge_servers_metadata(source, app, edge_servers=possible_edge_servers)
                min_and_max = find_minimum_and_maximum(metadata=edge_servers_metadata)
                edge_servers_metadata = sorted(
                    edge_servers_metadata,
                    key=lambda m: (
                        get_norm(m, "path_delay_source", min=min_and_max["minimum"], max=min_and_max["maximum"])
                        # + get_norm(m, "path_delay_sink", min=min_and_max["minimum"], max=min_and_max["maximum"])
                        + get_norm(m, "max_power_consumption", min=min_and_max["minimum"], max=min_and_max["maximum"]),
                    ),
                )

                for es_metadata in edge_servers_metadata:
                    edge_server = es_metadata["object"]

                    if edge_server.has_capacity_to_host(service=service):
                        service.provision(target_server=edge_server)
                        source = edge_server.network_switch
                        break

                if service.server == None and not service.being_provisioned:
                    possible_edge_servers = EdgeServer.all()


def EDF_algorithm(parameters):
    # Override 'has_capacity_to_host' for all instances of the EdgeServer class
    EdgeServer.has_capacity_to_host = has_capacity_to_host_only_resources  ## this for baseline

    priorities_list = []
    for usr in User.all():
        # Calculating the urgency of each user's deadline
        priority = 1 / list(usr.delay_slas.values())[0]

        # Assign users along sith their deadline-priority
        priorities_list.append((usr, priority))

    # Sort the priorities_list based on deadline
    sorted_priorities_list = sorted(priorities_list, key=lambda x: (x[1]), reverse=True)
    for user in sorted_priorities_list:

        for service in user[0].applications[0].services:
            # We don't want to migrate services are already being migrated
            if service.server == None and not service.being_provisioned:

                # Let's iterate over the list of edge servers to find a suitable host for our service
                for edge_server in EdgeServer.all():

                    # We must check if the edge server has enough resources to host the service
                    if edge_server.has_capacity_to_host(service=service):
                        # Start provisioning the service in the edge server
                        service.provision(target_server=edge_server)

                        # After start migrating the service we can move on to the next service
                        break

rl_start = True
# rl_initialize = False
def rl(parameters):

    global rl_start, rl_initialize
    if (rl_start == True):
        my_rl.initialize()
        # rl_initialize = True
        # my_rl.initialize(rl_initialize)
        rl_start = False

    # # Override 'has_capacity_to_host' for all instances of the EdgeServer class
    EdgeServer.has_capacity_to_host = has_capacity_to_host_proposed
    my_rl.rl_training()

    if edge_server.has_capacity_to_host(service=service):
        # Start provisioning the service in the edge server
        service.provision(target_server=edge_server)

        # After start migrating the service we can move on to the next service
    # # if edge_server.has_capacity_to_host(service=service):



def my_rl_in_edgesimpy(parameters):
    # Override 'has_capacity_to_host' for all instances of the EdgeServer class
    EdgeServer.has_capacity_to_host = has_capacity_to_host_proposed

    env = gym.make("CartPole-v1")

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    class ReplayMemory(object):

        def __init__(self, capacity):
            self.rl_memory = deque([], maxlen=capacity)

        def push(self, *args):
            """Save a transition"""
            self.rl_memory.append(Transition(*args))

        def sample(self, batch_size):
            return random.sample(self.rl_memory, batch_size)

        def __len__(self):
            return len(self.rl_memory)

    class DQN(nn.Module):

        def __init__(self, n_observations, n_actions):
            super(DQN, self).__init__()
            self.layer1 = nn.Linear(n_observations, 128)
            self.layer2 = nn.Linear(128, 128)
            self.layer3 = nn.Linear(128, n_actions)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return self.layer3(x)

    priorities_list = [] ## amin
    for usr in User.all():  ## amin
        # Calculating the urgency of each user's deadline       ## amin
        priority = 1 / list(usr.delay_slas.values())[0]     ## amin
        # Assign users along sith their deadline-priority   ## amin
        priorities_list.append((usr, priority)) ## amin
    # Sort the priorities_list based on deadline    ## amin
    sorted_priorities_list = sorted(priorities_list, key=lambda x: (x[1]), reverse=True)    ## amin

    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    def map_action_to_task_server(action):
        """
        Maps an action index to a task and server.

        Args:
            action (int): The action index.
            total_num_tasks (int): Total number of tasks.
            total_num_servers (int): Total number of servers.

        Returns:
            (int, int): A tuple of (task_index, server_index).
        """
        total_num_tasks = len(Service.all())
        total_num_servers = len(EdgeServer.all())

        # Determine the task and server indices
        task_index = (action // total_num_servers) + 1 ## the task(service) 0 represents the first service which its ID is '1'
        server_index = (action % total_num_servers) + 1 ## the server 0 represents the first server which its ID is '1'

        # Validate indices
        if task_index >= total_num_tasks:
            raise ValueError("Action index out of bounds for the given number of tasks and servers.")

        return task_index, server_index
    
    # Get number of actions from gym action space
    # n_actions = env.action_space.n ## was
    n_actions = (len(Service.all())*len(EdgeServer.all()))  ## amin

    # Get the number of state observations
    services_status_values = [  ## amin
        service.server if service.server is not None or service.being_provisioned else 0
        for service in Service.all()
    ]
    state = services_status_values  ## amin
    # state, info = env.reset() ## was
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    rl_memory = ReplayMemory(10000)

    steps_done = 0

    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    episode_durations = []

    def plot_durations(show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def optimize_model():
        if len(rl_memory) < BATCH_SIZE:
            return
        transitions = rl_memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=device, dtype=torch.bool) ## was
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool) ## amin
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 500

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        services_status_values = [  ## amin
            service.server if service.server is not None or service.being_provisioned else 0  ## amin
            for service in Service.all()  ## amin
        ]  ## amin
        state = services_status_values  ## amin
        # state, info = env.reset() ## was
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state) ## amin
            # print(f"action x: {action}") ## amin
            # print(f"action.item(): {action.item()}") ## amin

            rl_task, rl_server = map_action_to_task_server(action.item()) ## amin
            # print(f"Action {action.item()} corresponds to Task {rl_task} and Server {rl_server}.") ## amin

            rl_selected_service = next((s for s in Service._instances if s.id == (rl_task)), None) ## amin
            # rl_selected_application = next((s for s in Application._instances if s.id == (rl_task)), None)  ## amin
            rl_selected_application = next(    ## amin
                (app for app in Application._instances if rl_task in [service.id for service in app.services]),  ## amin
                None      ## amin
            )     ## amin
            rl_selected_user = next((user for user in User._instances if rl_selected_application in user.applications), None)
            rl_selected_server = next((s for s in EdgeServer._instances if s.id == (rl_server)), None)  ## amin

            # print(f"sorted_priorities_list[0][0]: {sorted_priorities_list[0][0]}")  ## amin
            # print(f"cpu U of {rl_selected_server}: {rl_selected_server.total_cpu_utilization}")

            if rl_selected_server.has_capacity_to_host(service=rl_selected_service):  ## amin
                # bool_not_overload===True ## put some positive reward in reward-function
                if sorted_priorities_list[0][0] == rl_selected_user: ## amin ##
                    # bool_EDF===True ## the earliest deadline task is allocated
                    rl_selected_service.provision(target_server=rl_selected_server)     ## amin
                    print(f"can host and service {rl_selected_service} is the earliest service")       ## amin
                    ################################################################
                    ##############calculating the response time##################
                    #### Calculate the one-way delay from the user to the candidate edge server for the service
                    communication_paths = []
                    topology = Topology.first()
                    communication_chain = [rl_selected_user.base_station, rl_selected_server.base_station]
                    for i in range(len(communication_chain) - 1):

                        # Defining origin and target nodes
                        origin = communication_chain[i]
                        target = communication_chain[i + 1]

                        # Finding and storing the best communication path between the origin and target nodes
                        if origin == target:
                            path = []
                        else:
                            path = nx.shortest_path(
                                G=topology,
                                source=origin.network_switch,
                                target=target.network_switch,
                                weight="delay",
                                method="dijkstra",
                            )

                        # Adding the best path found to the communication path
                        communication_paths.append([network_switch.id for network_switch in path])
                        ########
                    delay = 0.0
                    roundtrip_time = 0.0
                    # Initializes the application's delay with the time it takes to communicate its client and his base station
                    delay = rl_selected_user.base_station.wireless_delay
                    for path in communication_paths:
                        delay += topology.calculate_path_delay(path=[NetworkSwitch.find_by_id(i) for i in path])

                    roundtrip_time = (2 * delay)
                    response_time_for_service = round(
                        (roundtrip_time + rl_selected_server.execution_time_of_service[str(rl_selected_service.id)]), 2)
                    print(f"response_time_for_service: {response_time_for_service}")
                    #################################################
                    if (response_time_for_service < list(rl_selected_user.delay_slas.values())[0]):
                        bool_deadline_will_be_met = True
            else:
                print(f"can host but service {rl_selected_service} is NOT the earliest service")  ## amin
            #######################################################################################
            # def compute_reward(task, server, allocation_success, metrics):
            #     """
            #     Compute the reward for the RL agent.
            #
            #     Args:
            #         task (Task): The task being allocated.
            #         server (Server): The server to which the task is allocated.
            #         allocation_success (bool): Whether the allocation succeeded.
            #         metrics (dict): Additional metrics (e.g., latency, server utilization, deadline met).
            #
            #     Returns:
            #         float: The reward for the action.
            #     """
            #     reward = 0
            #
            #     # Positive reward for successful allocation
            #     if allocation_success:
            #         reward += 1  # Base reward for success
            #
            #         # Reward for efficient utilization
            #         utilization_factor = metrics.get("utilization", 0)
            #         reward += 0.5 * utilization_factor  # Adjust weight as needed
            #
            #         # Reward for low latency
            #         latency = metrics.get("latency", float("inf"))
            #         reward += 1.0 / max(latency, 1)  # Inverse of latency
            #
            #     # Negative reward for missing deadlines
            #     if not metrics.get("deadline_met", True):
            #         reward -= 2  # Penalty for missing a deadline
            #
            #     # Negative reward for server overload
            #     if metrics.get("server_overloaded", False):
            #         reward -= 1  # Penalty for causing overload
            #
            #     # Negative reward for excessive migrations
            #     if metrics.get("migration_cost", 0) > 0:
            #         reward -= 0.1 * metrics["migration_cost"]  # Weight migration penalty
            #
            #     return reward
            #######################################################################################

            sys.exit(0) ##################????????????!!!!!!!!!!!!!!!!!!????################

            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in rl_memory
            rl_memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                if (i_episode > 0) and (i_episode % 50 == 0):
                    plot_durations()
                break

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()

##########################################################################################################
## my proposed RL-approach for scheduling - Deadline and Resource Aware State Pruning (DRASP)
# In 'def My_Proposed_Pruned_Method_For_RL_train(parameters)' I train the RL-agent to obtain the optimal policy via my method
def My_Proposed_Pruned_Method_For_RL_train(parameters):
    # Override 'has_capacity_to_host' for all instances of the EdgeServer class
    EdgeServer.has_capacity_to_host = has_capacity_to_host_proposed

    # calculating number of cell in the matrix of each node of MDP based on the users and edgeservers
    number_of_cells = (len(EdgeServer.all())) * (len(User.all()))

    # original MDP graph of users
    nodes_in_org_graph = 2 ** (number_of_cells)
    edges_in_org_graph = nodes_in_org_graph * (len(Service.all()))

    # to determine most prior users
    priorities_list = []

    for usr in User.all():
        freq, memory = 0, 0
        priority = 1 / list(usr.delay_slas.values())[0]

        for i in range(len(usr.applications[0].services)):
            memory += usr.applications[0].services[i].memory_demand
            freq += usr.applications[0].services[i].cpu_cycles_demand

        comp_user = freq * memory
        # last item in 'priorities_list' represent 'being assign or not', where '0' means not assigned yet
        priorities_list.append((usr, priority, comp_user, memory, 0))

    # print(f"Priority list: {priorities_list}")

    # Sort the priorities_list based on the fourth item (priority) in descending order
    sorted_priorities_list = sorted(priorities_list, key=lambda x: (x[1], -x[2]), reverse=True)
    # print(f"Sorted priority list: {sorted_priorities_list}")

    # Function to dynamically update the tuple
    def update_tuple(original_tuple, user, server):
        # Convert the tuple to a list to allow modification
        mutable_list = list(original_tuple)

        # Each user has 4 servers, so calculate the index based on user and server
        index = (user - 1) * len(EdgeServer.all()) + (server - 1)

        # Update the value at the calculated index
        mutable_list[index] = 1

        # Convert the list back to a tuple
        updated_tuple = tuple(mutable_list)

        return updated_tuple

    # global best_nodes, worst_nodes
    all_sudo_permu = []
    G_small = nx.DiGraph()
    # Define global lists for best nodes & worst nodes of MDP graph
    global best_nodes
    weight_value_for_best_nodes = ((len(EdgeServer.all())) * (len(User.all())) * (+1))
    global worst_nodes
    weight_value_for_worst_nodes = ((len(EdgeServer.all())) * (len(User.all())) * (-1))
    # Start the traversal from node "0"
    start_node = tuple(0 for _ in range(number_of_cells))

    # Generate all permutations where one of the zeros is changed to "1" at a time
    for i in range(len(start_node)):
        if (start_node[i] == 0):
            sudo_permu = list(start_node)
            sudo_permu[i] = 1
            tuple_sudo_permu = tuple(sudo_permu)
            all_sudo_permu.append(tuple_sudo_permu)

    # Add edges from start_node to all_sudo_permu
    for node in all_sudo_permu:
        G_small.add_edge(start_node, node)

    visited_nodes = set()
    queue = [start_node]


    # We can always call the 'all()' method to get a list with all created instances of a given class
    # for user in sorted_priorities_list:
    for idx, user in enumerate(sorted_priorities_list):
        for service in user[0].applications[0].services:
            # We don't want to migrate services are already being migrated
            if service.server == None and not service.being_provisioned:

                # Let's iterate over the list of edge servers to find a suitable host for our service
                for edge_server in EdgeServer.all():

                    # We must check if the edge server has enough resources to host the service
                    if edge_server.has_capacity_to_host(service=service):
                        # Start provisioning the service in the edge server
                        service.provision(target_server=edge_server)

                        # create best_node tuple
                        # print(f"user[0].id: {user[0].id}")
                        # print(f"edge_server.id: {edge_server.id}")
                        best_nodes.append(update_tuple(start_node, user[0].id, edge_server.id))
                        # print(f"best node:{update_tuple(start_node, user[0].id, edge_server.id)}")
                        if idx == len(sorted_priorities_list) - 1:
                            # print("Processing the last user:", user)
                            for s in EdgeServer.all():
                                # print(f"{s}.cpu_U: {s.total_cpu_utilization}")
                                # print(f"{s}.memory_U: {(s.memory_demand / s.memory)}")
                                if ((s.memory < (s.memory_demand + service.memory_demand)) and (s.total_cpu_utilization + service.cpu_demand > 1)):
                                    worst_node = update_tuple(start_node, user[0].id, s.id)
                                elif ((s.memory < (s.memory_demand + service.memory_demand)) or (s.total_cpu_utilization + service.cpu_demand > 1)):
                                    worst_node = update_tuple(start_node, user[0].id, s.id)
                            worst_nodes.append(worst_node)
                        elif idx != len(sorted_priorities_list) - 1:
                            last_user = sorted_priorities_list[-1]  # Access the last user
                            worst_nodes.append(update_tuple(start_node, last_user[0].id, edge_server.id))
                            # print(f"worst node:{update_tuple(start_node, last_user[0].id, edge_server.id)}")

                        # After start migrating the service we can move on to the next service
                        break

# In 'def Testing_My_Proposed_Pruned_Method_For_RL(parameters)' I test the RL-agent that was trained by 'My_Proposed_Pruned_Method_For_RL_train'.
def Testing_My_Proposed_Pruned_Method_For_RL(parameters):
    if (EdgeServer.is_provisioned <= len(Service.all())):
        # Load the Q_values from the .npy file
        Q_values_loaded = np.load(r'C:\Users\100807003\PycharmProjects\EdgeSimPy\edge_sim_py\Q_values.npy')
        max_steps = len(Q_values_loaded)

        # Load the graph from the GraphML file
        pruned_G = nx.read_graphml(r'C:\Users\100807003\PycharmProjects\EdgeSimPy\edge_sim_py\pruned_G.graphml')

        # Now you can use Q_values_loaded in your test code
        start_state = next(iter(pruned_G.nodes()))  # Get the first node (tuple or other representation)
        # print(f"Start state: {start_state}")

        # Example function to derive the optimal policy
        def get_optimal_policy(Q_values_loaded):
            optimal_policy = []
            for state in range(len(Q_values_loaded)):
                best_action = np.argmax(Q_values_loaded[state])  # Get the action with the highest Q-value
                optimal_policy.append(best_action)
            return optimal_policy

        # Example function to test the RL algorithm
        def test_rl_algorithm(Q_values_loaded, pruned_G, start_state, max_steps):
            current_state = start_state  # Start from the initial state
            optimal_policy = get_optimal_policy(Q_values_loaded)  # Extract the optimal policy from Q-values

            # print(f"Starting test from state {current_state}")

            # Convert nodes to a list for index lookups
            node_list = list(pruned_G.nodes())

            for step in range(max_steps):
                try:
                    current_state_index = node_list.index(current_state)  # Find index of current_state
                except ValueError:
                    print(f"Error: State {current_state} not found in graph. Ending test.")
                    break

                # Use the optimal policy to decide the action for the current state
                action = optimal_policy[current_state_index]
                # print(f"Step {step + 1}: In state {current_state}")

                # Simulate moving to the next state by using the loaded graph (pruned_G)
                if current_state in pruned_G:
                    neighbors = list(pruned_G.neighbors(current_state))
                    if neighbors:
                        # Use the action to select the next state from the neighbors
                        next_state = neighbors[action % len(neighbors)]
                        # print(f"Moving to {next_state}")

                        # Convert the string to an actual tuple using ast.literal_eval
                        next_state_tuple = ast.literal_eval(next_state)
                        # Access the first element, which is the large tuple
                        first_tuple = next_state_tuple[0]
                        # Find the index of '1' in the first tuple
                        index_of_one = first_tuple.index(1)
                        # print(f"Index of '1': {index_of_one}")

                        # Reverse the formula to get user and server
                        selected_user = (index_of_one // len(EdgeServer.all())) + 1  # Integer division to get user
                        selected_server = (index_of_one % len(EdgeServer.all())) + 1  # Modulus to get server

                        selected_user = next((user for user in User.all() if user.id == selected_user), None)
                        selected_server = next((edge_server for edge_server in EdgeServer.all() if edge_server.id == selected_server), None)
                        # print(f"SELECTED user: {selected_user}")
                        # print(f"SELECTED server: {selected_server}")

                        # for service in selected_user.applications[0].services:
                        #     print(f"SERVICE: {service} of {selected_user}")
                            # if service.server == None and not service.being_provisioned:

                        selected_service = next((service for service in selected_user.applications[0].services
                                                 if service.server is None and not service.being_provisioned), None)

                        if selected_service is not None:
                            user_service_exe_time = (selected_service.processing_power_demand / selected_server.processing_power)
                            selected_server.execution_time_of_service[str(selected_service.id)] = user_service_exe_time
                            selected_service.provision(target_server=selected_server)
                            EdgeServer.is_provisioned = EdgeServer.is_provisioned + 1
                            # print(f"SERVICE: {selected_service} of {selected_user} on {selected_server}")
                        # else:
                        #     print("No matching service found.")
                    else:
                        # print(f"No neighbors for state {current_state}, ending test.")
                        break
                else:
                    print(f"State {current_state} not found in pruned_G. Ending test.")
                    break

                # Check if the action leads to an invalid state in the Q-values
                if Q_values_loaded[current_state_index][action] == -np.inf:
                    print(f"Action {action} is invalid in state {current_state}. Ending test.")
                    break

                # Move to the next state
                current_state = next_state  # Continue using the tuple-based states

                # Add termination condition (e.g., if you reach a terminal state)
                next_state_index = node_list.index(current_state)
                if Q_values_loaded[next_state_index][action] == 0:
                    print(f"Reached terminal state {current_state}. Ending test.")
                    break
            else:
                print("Reached max steps.")

        # Run the test from a start state
        test_rl_algorithm(Q_values_loaded, pruned_G, start_state=start_state, max_steps=max_steps)



scheduling_time_exceeded = False
service_scheduling_duration = time.time()
old_provisioned_services = 0
def stopping_criterion(model: object):
    # Defining a variable that will help us to count the number of services successfully provisioned within the infrastructure
    provisioned_services = 0
    scheduling_time_limitation = (1 / 30) * (len(Service.all()) * len(EdgeServer.all()))
    global old_provisioned_services, service_scheduling_duration, scheduling_time_exceeded

    # Iterating over the list of services to count the number of services provisioned within the infrastructure
    for service in Service.all():
        # Initially, services are not hosted by any server (i.e., their "server" attribute is None).
        # Once that value changes, we know that it has been successfully provisioned inside an edge server.
        if service.server != None:
            provisioned_services += 1

    # Since services constitute the components of applications, the maximum allowable time for scheduling
    # a single service is inspired by the '30 FPS' benchmark for real-time responsiveness. Accordingly,
    # the maximum scheduling time is defined as (1/30) * (total number of services * total number of edge servers) [2].
    # Therefore, the value of 'service_scheduling_duration' get updated as soon as a service is succesfully scheduled by
    # the 'if (old_provisioned_services < provisioned_services):' condition.
    # [2]: Wang, J., Shi, R., Zheng, W., Xie, W., Kao, D., & Liang, H. N. (2023).
    #       Effect of frame rate on user experience, performance, and simulator sickness in virtual reality.
    #       IEEE Transactions on Visualization and Computer Graphics, 29(5), 2478-2488.

    elapsed_time = time.time() - service_scheduling_duration
    if elapsed_time >= scheduling_time_limitation:
        scheduling_time_exceeded = True

    # This condition checks whether the service can be properly provisioned to the edge server based on
    # the decision made by the scheduling algorithm
    if (old_provisioned_services < provisioned_services):
        service_scheduling_duration = time.time()
        old_provisioned_services = provisioned_services
        print(f"{old_provisioned_services} out of {len(Service.all())} services are successfully scheduled.")
        # print(f"Time Step of EdgeSimPy: {simulator.schedule.steps}")
        # semi_end_time_edgesimpy = time.time()
        # semi_duration_edgesimpy = semi_end_time_edgesimpy - start_time_edgesimpy
        # print(f"runtime: {semi_duration_edgesimpy:.2f} seconds")
        resource_tracker.report()
        print()

    return (provisioned_services == Service.count()) or (provisioned_services == EdgeServer.is_potential_host) or (scheduling_time_exceeded == True)

#######################################################################################################################
##################################################
## Determining the name of Scheduling Algorithm ##
##################################################

# Map algorithm names to functions
algorithm_functions = {
    "lapse": lapse,
    "MASS": MASS,
    "BestFit": Best_Fit_Service_Provisioning,
    "EDF": EDF_algorithm,
    "rl": rl,
    "my_rl_in_edgesimpy": my_rl_in_edgesimpy
}
# Define the name of the scheduling algorithm, that could be "lapse", "MASS", "BestFit", "EDF"
# scheduling_algorithm = "rl"
# scheduling_algorithm = "EDF"
scheduling_algorithm = "my_rl_in_edgesimpy"

# @measure_memory
def wrapped_Service_Provisioning(parameters, algorithm_name=scheduling_algorithm):
    # Get the function based on the algorithm name
    result = algorithm_functions[algorithm_name](parameters)
    process = psutil.Process(os.getpid())
    resource_tracker.update(process.memory_info().rss)
    return result


# Define variables as a global variables outside of the functions
best_nodes = []
worst_nodes = []

# logs_directory = f"logs/algorithm=FFSP;dataset=sample_dataset2;"  ## baseline alg with example dataset
logs_directory = f"logs/algorithm={scheduling_algorithm};dataset=dataset1;"  ## baseline alg with mine datasets

# Creating a Simulator object
simulator = Simulator(
    dump_interval=5,
    tick_duration=1,
    tick_unit="seconds",
    stopping_criterion=stopping_criterion,
    resource_management_algorithm=wrapped_Service_Provisioning,
    logs_directory=logs_directory,
)

#########################
## Loading the dataset ##
#########################
# Get the directory of the current script
current_dir = os.path.dirname(__file__)
# Define the relative path to the dataset, using the script's directory as the base
dataset_path = os.path.join(current_dir, "dataset_generator", "datasets", "dataset1.json")
# Initialize the simulator with the absolute path
simulator.initialize(input_file=dataset_path)


# Start the timer
start_time_edgesimpy = time.time()

# Executing the simulation
simulator.run_model()

# End the timer and calculate the duration
end_time_edgesimpy = time.time()
duration_edgesimpy = end_time_edgesimpy - start_time_edgesimpy
print(f"Total runtime: {duration_edgesimpy:.2f} seconds")

resource_tracker.final_report()

