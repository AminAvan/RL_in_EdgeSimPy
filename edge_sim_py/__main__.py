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
        print(f"Total memory consumed: {self.total_memory / (1024 * 1024):.2f} MiB")
        print(f"Total power consumption: {self.total_power:.2f} Watt-seconds")
        print(f"Number of calls: {self.call_count}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Average power over time: {(self.total_power / total_time):.2f} Watts")

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
        # calculating true execution time of service on the host server
        self.execution_time_of_service[str(service.id)] = user_service_exe_time
        self.total_cpu_utilization = (self.total_cpu_utilization + user_service_utilization)

        # # print(f"free_cpu_cycle of {self}:{free_cpu_cycle}")
        # if (free_cpu_cycle < 0):
        #     EdgeServer.is_negative_freq_capacity = EdgeServer.is_negative_freq_capacity + 1

        EdgeServer.is_potential_host = EdgeServer.is_potential_host + 1
        print(f"EdgeServer.is_potential_host: {EdgeServer.is_potential_host}.")
        if(len(service.all()) == EdgeServer.is_potential_host):
            print(f"All services are hosted!")
            # print(f"missed applications: {EdgeServer.is_negative_freq_capacity}, users: {len(User.all())}, miss-ratio of users: {EdgeServer.is_negative_freq_capacity/(len(User.all()))}")
            s_total_cpu_util = 0
            s_total_mem_util = 0
            for s in EdgeServer.all():
                s_total_cpu_util += round(s.total_cpu_utilization, 2)
                s_total_mem_util += round((s.memory_demand/s.memory), 2)
            print(f"Average CPU load of all edge servers: {((s_total_cpu_util/len(EdgeServer.all()))*100)}%")
            print(f"Average memory load of all edge servers: {((s_total_mem_util / len(EdgeServer.all())) * 100)}%\n")

        can_host = True
    else:
        can_host = False
    return can_host


def MARS(parameters):
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


    def get_app_total_demand(application: Application) -> float:
        app_demand = 0
        for service in application.services:
            app_demand += service.cpu_demand * service.cpu_cycles_demand * service.memory_demand

        return app_demand


    def get_all_shortest_path_between(origin_network_switch: object, target_network_switch: object) -> int:
        topology = origin_network_switch.model.topology

        paths = list(
            nx.all_shortest_paths(topology, source=origin_network_switch, target=target_network_switch, weight="delay"))

        return paths


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

    def lapse_calculate_path_delay(origin_network_switch: object, target_network_switch: object) -> int:
        topology = origin_network_switch.model.topology

        path = find_shortest_path(origin_network_switch=origin_network_switch,
                                  target_network_switch=target_network_switch)
        delay = topology.calculate_path_delay(path=path)

        return delay

    def get_edge_servers_metadata(source: object, sink: object, app: object, edge_servers: object = None) -> list:
        metadata = []

        edge_servers_list = edge_servers if edge_servers else EdgeServer.all()

        for edge_server in edge_servers_list:
            # Compute the percentage of services that can be hosted on the edge server
            app_demand = 0
            for service in app.services:
                if not service.server:
                    # app_demand += service.input_event_rate * service.mips_demand
                    app_demand += service.cpu_demand * service.cpu_cycles_demand * service.memory_demand

            edge_server_attrs = {
                "object": edge_server,
                "path_delay_source": lapse_calculate_path_delay(source, edge_server.network_switch),
                "path_delay_sink": lapse_calculate_path_delay(sink, edge_server.network_switch),
                "max_power_consumption": edge_server.power_model_parameters["max_power_consumption"],
            }

            metadata.append(edge_server_attrs)

        return metadata

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
    #############################################
    ###########################################################
    #########################################################
    # if (EdgeServer.is_potential_host < 261): ## commented by Amin
    apps = Application.all()

    # Sorts applications based on their processing time SLA (from lowest to highest),
    # number of services (from highest to lowest), and input demand (from highest to lowest)
    apps = sorted(
        apps,
        key=lambda app: (
            get_app_total_demand(app),
            -list(app.users[0].delay_slas.values())[0]
        ),
    )


    for app in apps:
        for base_station in BaseStation.all():
        #
        #     if len(base_station.edge_servers) > 0: ## commented by Amin
            source = app.users[0].base_station.network_switch
            # sink = app.services[-1].server.network_switch ## point
            sink = base_station.network_switch

            possible_edge_servers = get_edge_servers_between(source, sink)

            for service in app.services:
                # print(f"service: {service}")
                # print(f"service.server: {service.server}")
                if service.server != None and service.being_provisioned:  ## uncommented by amin
                    continue        ## uncommented by amin

                # if service.server == None and not service.being_provisioned: ## commented by amin
                while service.server == None and not service.being_provisioned:

                    edge_servers_metadata = get_edge_servers_metadata(source, sink, app, edge_servers=possible_edge_servers)

                    min_and_max = find_minimum_and_maximum(metadata=edge_servers_metadata)

                    edge_servers_metadata = sorted(
                        edge_servers_metadata,
                        key=lambda m: (
                            get_norm(m, "path_delay_source", min=min_and_max["minimum"], max=min_and_max["maximum"])
                            + get_norm(m, "path_delay_sink", min=min_and_max["minimum"], max=min_and_max["maximum"])
                            + get_norm(m, "max_power_consumption", min=min_and_max["minimum"], max=min_and_max["maximum"]),
                        ),
                    )

                    for es_metadata in edge_servers_metadata:
                        edge_server = es_metadata["object"]

                        # if has_capacity_to_host(edge_server, service):
                        if edge_server.has_capacity_to_host(service=service):
                            # print(f"service:{service} {service.application}, edge_server:{edge_server}")
                            # print()
                            # print(f"service.server: {service.server}")
                            # place(service=service, edge_server=edge_server)
                            service.provision(target_server=edge_server)
                            source = edge_server.network_switch
                            # Start provisioning the service in the edge server
                            break

                    # if not service.server: ## comment by Amin
                    if service.server == None and not service.being_provisioned:
                        possible_edge_servers = EdgeServer.all()
#############################################################################

def lapse_ICCPS(parameters):
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

                for es_metadata in edge_servers_metadata:  ## was and stuck
                    edge_server = es_metadata["object"]   ## was and stuck
                    # print(f"edge_server type f1: {type(edge_server)}")
                # for edge_server in EdgeServer.all():        ## is & not stuck
                    # print(f"edge_server type f2: {type(edge_server)}")
                    # if has_capacity_to_host(edge_server, service):
                    if edge_server.has_capacity_to_host(service=service):
                        service.provision(target_server=edge_server)
                        source = edge_server.network_switch
                        # place(service=service, edge_server=edge_server)  ## was
                        # source = edge_server.network_switch   ## was
                        break

                # if not service.server:
                if service.server == None and not service.being_provisioned:
                    possible_edge_servers = EdgeServer.all()
#############################################################################
#############################################################################
#############################################################################
#############################################################################

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
    # for i in sorted_priorities_list:
    #     print(f"Sorted priority list: {i}")
    for user in sorted_priorities_list:
        # print(user[0])
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



#############################################################################
#############################################################################
#############################################################################
"""
from edge_sim_py import Simulator
from edgesimpy_wrapper import EdgeSimPyWrapper
from mad4pg_service_provisioning import MAD4PGServiceProvisioning

def stopping_criterion(model):
    # Implement your stopping criterion here
    return False  # placeholder

if __name__ == "__main__":
    environment_file = "path_to_your_environment_file.json"
    
    simulator = Simulator(
        dump_interval=5,
        tick_duration=1,
        tick_unit="seconds",
        stopping_criterion=stopping_criterion,
        resource_management_algorithm=None,
        logs_directory="logs/MAD4PG_simulation/"
    )
    
    simulator.initialize(input_file=environment_file)
    
    env_wrapper = EdgeSimPyWrapper(simulator)
    
    mad4pg_provisioning = MAD4PGServiceProvisioning(
        agent_number=9,
        agent_action_size=27,
        environment_wrapper=env_wrapper
    )
    
    simulator.resource_management_algorithm = mad4pg_provisioning
    
    simulator.run_model()
"""
#############################################################################
# from edge_sim_py.mad4pg_files.edgesimpy_wrapper_mad4pg import EdgeSimPyWrapper
# from edge_sim_py.mad4pg_files.mad4pg_service_provisioning import MAD4PGServiceProvisioning
# def mad4pg_stopping_criterion(model):
#     return model.current_tick >= 1000

#############################################################################


old_provisioned_services = 0
def stopping_criterion(model: object):
    # Defining a variable that will help us to count the number of services successfully provisioned within the infrastructure
    provisioned_services = 0
    global old_provisioned_services
    global time_to_stop

    # Iterating over the list of services to count the number of services provisioned within the infrastructure
    for service in Service.all():
        # Initially, services are not hosted by any server (i.e., their "server" attribute is None).
        # Once that value changes, we know that it has been successfully provisioned inside an edge server.
        if service.server != None:
            provisioned_services += 1

        # print(f"service.server: {service.server.}")


    if (old_provisioned_services < provisioned_services):
        old_provisioned_services = provisioned_services
        print(f"provisioned_services:{old_provisioned_services}")
        print(f"EdgeServer.is_potential_host: {EdgeServer.is_potential_host}")
        print(f"Time Step: {simulator.schedule.steps}")
        # for service in Service.all():
        #     if service.server == None:
        #         print(f"{service}: {service._Service__migrations}")
        semi_end_time_edgesimpy = time.time()
        semi_duration_edgesimpy = semi_end_time_edgesimpy - start_time_edgesimpy
        print(f"Execution time: {semi_duration_edgesimpy:.2f} seconds")
        resource_tracker.report()
        print()

    #################################################
    #################################################
    #################################################
    # for my RL algorithm to construct prunned MDP
    # if ((provisioned_services == Service.count()) or (provisioned_services == EdgeServer.is_potential_host)):
    #     global best_nodes
    #     global worst_nodes
    #     number_of_cells = (len(EdgeServer.all())) * (len(User.all()))
    #     weight_value_for_best_nodes = (2 * (number_of_cells) * (+1))
    #     weight_value_for_worst_nodes = (2 * (number_of_cells) * (-1))
    #
    #     # Create a pruned directed graph from the best nodes and worst nodes
    #     pruned_G = nx.DiGraph()
    #
    #     # Add the starter node for the pruned directed graph
    #     starter_node_in_pruned_graph = tuple(0 for _ in range(number_of_cells))
    #     pruned_G.add_node(starter_node_in_pruned_graph)
    #
    #     print(f"len(best_nodes): {len(best_nodes)}")
    #     print(f"len(worst_nodes): {len(worst_nodes)}")
    #
    #     ## old
    #     # # constructing the edges and nodes of the pruned directed graph
    #     # for i in range(len(best_nodes)):
    #     #     pruned_G.add_node(best_nodes[i])
    #     #     pruned_G.add_edge(starter_node_in_pruned_graph, best_nodes[i], weight=weight_value_for_best_nodes)
    #     #     # print(f"best_nodes[i]: {best_nodes[i]}")
    #     #     # print()
    #     #     pruned_G.add_node(worst_nodes[i])
    #     #     pruned_G.add_edge(starter_node_in_pruned_graph, worst_nodes[i], weight=weight_value_for_worst_nodes)
    #     #     # print(f"worst_nodes[i]: {worst_nodes[i]}")
    #     #     starter_node_in_pruned_graph = best_nodes[i]
    #
    #     # Create unique node identifiers by combining node content with an index
    #     for i in range(len(best_nodes)):
    #         unique_best_node = (best_nodes[i], i)  # Append index to make each node unique
    #         unique_worst_node = (worst_nodes[i], i)
    #
    #         pruned_G.add_node(unique_best_node)  # Add the uniquely identified node
    #         pruned_G.add_edge(starter_node_in_pruned_graph, unique_best_node, weight=weight_value_for_best_nodes)
    #
    #         pruned_G.add_node(unique_worst_node)
    #         pruned_G.add_edge(starter_node_in_pruned_graph, unique_worst_node, weight=weight_value_for_worst_nodes)
    #
    #         starter_node_in_pruned_graph = unique_best_node  # Update for next iteration
    #
    #     print("pruned_G", pruned_G)
    #     # Save pruned_G to a GraphML file
    #     nx.write_graphml(pruned_G, r'C:\Users\100807003\PycharmProjects\EdgeSimPy\edge_sim_py\pruned_G.graphml')
    #     print("=========================================")
    #
    #     ## starting procedure of finding optimal policy based on the MDGP ##
    #     # tranforming the nx-graph into a MDP template
    #     # tranforming the nx-graph into a MDP template
    #     def graph_to_mdp(pruned_G):
    #         num_nodes = len(pruned_G.nodes)
    #         transition_probabilities = np.full((num_nodes, 2, num_nodes), 0.0, dtype=object)
    #         rewards = np.zeros((num_nodes, 2, num_nodes), dtype=int)
    #         possible_actions = []
    #         nodes_list_of_pruned_G = list(pruned_G.nodes)
    #
    #         for node in pruned_G.nodes:
    #             successors = list(pruned_G.successors(node))
    #             # print("node", node)
    #             # print("successors", successors)
    #
    #             if successors:
    #                 # Define possible actions based on the number of successors
    #                 possible_actions.append(list(range(len(successors))))
    #
    #                 # Iterate over successors only, not always assuming there are exactly 2 actions
    #                 for action, next_node in enumerate(successors):
    #                     # Update transition probabilities and rewards
    #                     transition_probabilities[
    #                         nodes_list_of_pruned_G.index(node), action, nodes_list_of_pruned_G.index(next_node)
    #                     ] = 1.0
    #                     rewards[
    #                         nodes_list_of_pruned_G.index(node), action, nodes_list_of_pruned_G.index(next_node)
    #                     ] = pruned_G[node][next_node]['weight']
    #
    #             else:
    #                 # If no successors, only one action possible (e.g., staying in the same state or a terminal state)
    #                 possible_actions.append([0])
    #
    #         return transition_probabilities, rewards, possible_actions
    #
    #     transition_probabilities, rewards, possible_actions = graph_to_mdp(pruned_G)
    #     print(f"graph_to_mdp(pruned_G) is DONE")
    #     # print("Transition Probabilities:")
    #     # print(transition_probabilities)
    #     # print("\nRewards:")
    #     # print(rewards)
    #     # print("\nPossible Actions:")
    #     # print(possible_actions)
    #     # print("")
    #
    #     # Q-value approach
    #     Q_values = np.full(((len(pruned_G.nodes)), (len(pruned_G.nodes))), -np.inf)  # -np.inf for impossible actions
    #     # print(f"before Q_values: {Q_values}")
    #     for state, actions in enumerate(possible_actions):
    #         Q_values[state, actions] = 0.0  # for all possible actions
    #     # print(f"after Q_values: {Q_values}")
    #     # print("")
    #
    #     # # now, Q-value iteration algorithm (it applies Equation 18-3 repeatedly) to all Q-values, for every state and every possible action
    #     gamma = 0.95  # the discount factor
    #     print(f"gamma: {gamma}")
    #     for iteration in range(90):
    #         Q_prev = Q_values.copy()
    #         for s in range(len(pruned_G.nodes)):
    #             for a in possible_actions[s]:
    #                 Q_values[s, a] = np.sum([
    #                     transition_probabilities[s][a][sp]
    #                     * (rewards[s][a][sp] + gamma * Q_prev[sp].max())
    #                     for sp in range(len(pruned_G.nodes))])
    #     print(f"end of for iteration in range(90):")
    #     # the resulting Q-values, which give us the optimal policy for this MDP when using a discount factor
    #     # print(f'Q_values giving us optimal policy for this MDP when using discount factor (gamma={gamma}):\n\n',
    #     #       Q_values)
    #     # print(f'Q_values giving us optimal policy for this MDP when using discount factor (gamma={gamma}).')
    #     print("==========================================================")
    #
    #     # For example, when the agent is in state s0 and it chooses action a0, the expected sum of discounted future rewards is as below:
    #     # print("Sum of discounted future rewards when agent is in state s0 and it chooses action a0:", Q_values[0][0])
    #
    #     # For each state, it is possible to find the action that has the highest Q-value:
    #     # print("optimal action for each state:", Q_values.argmax(axis=1))
    #     np.save(r'C:\Users\100807003\PycharmProjects\EdgeSimPy\edge_sim_py\Q_values.npy', Q_values)
    ###################################
    ##########################################
    ###############################################


    # As EdgeSimPy will halt the simulation whenever this function returns True, its output will be a boolean expression
    # that checks if the number of provisioned services equals to the number of services spawned in our simulation


    return (provisioned_services == Service.count()) or (provisioned_services == EdgeServer.is_potential_host)

###################################################################################

# @measure_memory
def wrapped_Service_Provisioning(parameters):
    # result = My_Second_Service_Provisioning(parameters)
    # result = Best_Fit_Service_Provisioning(parameters)
    # result = lapse(parameters)
    # result = lapse_ICCPS(parameters)
    # result = DAMR(parameters)
    result = MARS(parameters)
    # result = EDF_algorithm(parameters)
    process = psutil.Process(os.getpid())
    resource_tracker.update(process.memory_info().rss)
    return result



# Define variables as a global variables outside of the functions
best_nodes = []
worst_nodes = []

# logs_directory = f"logs/algorithm=FFSP;dataset=sample_dataset2;"  ## baseline alg with example dataset
logs_directory = f"logs/algorithm=FFSP;dataset=dataset1;"  ## baseline alg with mine datasets
# input_file="/mnt/c/Users/100807003/PycharmProjects/EdgeSimPy/edge_sim_py/dataset_generator/datasets/dataset1.json" ## mad4pg

# Creating a Simulator object
simulator = Simulator(
    dump_interval=5,
    tick_duration=1,
    tick_unit="seconds",
    stopping_criterion=stopping_criterion,     ## normal
    # stopping_criterion=mad4pg_stopping_criterion,   ## for mad4pg
    # resource_management_algorithm=First_Fit_Service_Provisioning,    ## all services are provisioned 262/262
    # resource_management_algorithm=Best_Fit_Service_Provisioning,       ## 197 services are provisioned out of 262
    # resource_management_algorithm=My_Second_Service_Provisioning,
    resource_management_algorithm=wrapped_Service_Provisioning,
    # resource_management_algorithm=Test_My_Second_Service_Provisioning,
    # resource_management_algorithm=mad4pg_provisioning,
    # resource_management_algorithm=lapse,
    logs_directory=logs_directory,
)

### Loading a sample dataset from GitHub
# simulator.initialize(input_file=r"C:\Users\100807003\PycharmProjects\EdgeSimPy\datasets\sample_dataset2.json")
simulator.initialize(input_file=r"C:\Users\100807003\PycharmProjects\EdgeSimPy\edge_sim_py\dataset_generator\datasets\dataset1.json")    ## run on windows
# simulator.initialize(input_file="/mnt/c/Users/100807003/PycharmProjects/EdgeSimPy/edge_sim_py/dataset_generator/datasets/dataset1.json") ## run in wsl
# simulator.initialize(input_file=input_file) ## run in wsl


# Start the timer
start_time_edgesimpy = time.time()

# Executing the simulation
simulator.run_model()

# End the timer and calculate the duration
end_time_edgesimpy = time.time()
duration_edgesimpy = end_time_edgesimpy - start_time_edgesimpy
print(f"Execution time: {duration_edgesimpy:.2f} seconds")

resource_tracker.report()