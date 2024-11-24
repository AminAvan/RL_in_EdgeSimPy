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

import my_rl
import rl_in_edgesimpy
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


my_rl_in_edgesimpy_start = True
def my_rl_in_edgesimpy(parameters):
    global my_rl_in_edgesimpy_start
    if (my_rl_in_edgesimpy_start == True):
        rl_in_edgesimpy.initialize()
        my_rl_in_edgesimpy_start = False

    rl_in_edgesimpy.rl_training()

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
scheduling_algorithm = "rl"
# scheduling_algorithm = "my_rl_in_edgesimpy"

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

