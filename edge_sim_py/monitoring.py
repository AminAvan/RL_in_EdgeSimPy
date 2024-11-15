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
####################################################

"""
The line
datasets[file.replace(".msgpack", "")] = pd.DataFrame(msgpack.unpackb(data_file.read(), strict_map_key=False))
does not create new files on disk.
Instead, it reads the content of existing MessagePack files, converts them into DataFrames, and
stores these DataFrames in a dictionary in memory.
"""
# Adjust display settings to show all columns and increase width
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# Set the maximum number of rows to display
pd.set_option('display.max_rows', 100000)  # Set this to the desired number, or None for unlimited rows

##############################################################################################################################
def load_msgpack_logs(logs_directory):
    """
    Load msgpack log files from the specified directory into pandas DataFrames.

    Args:
        logs_directory (str): Path to the directory containing msgpack log files.

    Returns:
        dict: Dictionary of pandas DataFrames, where the keys are the log file names.
    """
    datasets = {}
    dataset_files = [file for file in os.listdir(logs_directory) if ".msgpack" in file]

    for file in dataset_files:
        try:
            with open(os.path.join(logs_directory, file), 'rb') as data_file:
                datasets[file.replace(".msgpack", "")] = pd.DataFrame(
                    msgpack.unpackb(data_file.read(), strict_map_key=False))
        except (msgpack.UnpackException, ValueError) as e:
            print(f"Error loading file {file}: {e}")

    return datasets

##################################
### Determining name algorithm ###
##################################

scheduling_algorithm = "lapse"
logs_directory = f"{os.getcwd()}/logs/algorithm={scheduling_algorithm};dataset=dataset1;"
datasets = load_msgpack_logs(logs_directory)
##############################################################################################################################

# Count the number of DataFrames created
# num_dataframes = len(datasets)
# print(f"Number of DataFrames created: {num_dataframes}")
# show all the datasets of entities
# for item in datasets:
#     print(f"{item}")

"""
EdgeServer
"""
# Access the last row of the DataFrame
last_row = datasets["EdgeServer"].iloc[-1]

# Retrieve the value of the 'Time Step' attribute
time_step_value = last_row['Time Step']

# Find the largest number in the 'Instance ID' column
max_instance_id = datasets["EdgeServer"]["Instance ID"].max()

# ## showing all the columns of the "EdgeServer"
# print(datasets["EdgeServer"].columns)
selected_columns = ['Object', 'Time Step', 'Instance ID', 'Coordinates', 'Available', 'CPU', 'RAM', 'Disk', 'CPU Demand', 'RAM Demand', 'Disk Demand', 'Services', 'Registries', 'Layers', 'Images', 'Download Queue', 'Waiting Queue', 'Power Consumption']
# print(datasets["EdgeServer"][selected_columns])
# print()
### Filter rows with specific condition - last "Time Step"
filtered_df = datasets["EdgeServer"][(datasets["EdgeServer"]["Time Step"] == time_step_value)]
filtered_selected_df = filtered_df[selected_columns]
print(filtered_selected_df)
print()


# Sum of the 'Power Consumption' of all edge servers and number of services in edge computing
total_power_consumption = round((filtered_df[selected_columns]['Power Consumption'].sum()), 2)
total_unique_services = datasets["Service"]["Instance ID"].nunique()
print(f"Total Power Consumption of EdgeServers is {total_power_consumption} with {total_unique_services} services.")
print()

# # print(datasets["EdgeServer"][selected_columns])
#
# print()
# print()
#



"""
service
"""
# Access the last row of the DataFrame
last_row = datasets["Service"].iloc[-1]

# Retrieve the value of the 'Time Step' attribute
time_step_value = last_row['Time Step']

# Find the largest number in the 'Instance ID' column
max_instance_id = datasets["Service"]["Instance ID"].max()
# showing all the columns of the "Service"
# print(datasets["Service"].columns)
# selecting specific columns
selected_columns_service = ['Object', 'Time Step', 'Instance ID', 'Available', 'Application', 'Server', 'Being Provisioned', 'Last Migration']
### filtering the rows
filtered_col_service = datasets["Service"][(datasets["Service"]["Time Step"] == (time_step_value))]
print(filtered_col_service[selected_columns_service])
# Initialize a counter for 'miss'
service_miss_count = 0

# Iterate over each row in the 'Deadline Status' column
for status in filtered_col_service[selected_columns_service]['Available']:
    # Check if 'miss' is present in the dictionary values
    if status == False:
        service_miss_count += 1

# Print the count of 'miss'
# print(f"\nNumber of 'miss-service': {service_miss_count} -- miss-rate: {(service_miss_count) / (len(filtered_col_service[selected_columns_service]['Available']))}")
print()

# print()



"""
user
"""
# Access the last row of the DataFrame
last_row = datasets["User"].iloc[-1]

# Retrieve the value of the 'Time Step' attribute
time_step_value = last_row['Time Step']

# Find the largest number in the 'Instance ID' column
max_instance_id = datasets["User"]["Instance ID"].max()

# Initialize a counter for users who met the deadline and total users
meet_deadline_count = 0
total_users = 0
# #
# # # showing all the columns of the "User"
# # # print(datasets["User"].columns)
# # # print()
# # # selecting specific columns
selected_columns_user = ['Object', 'Time Step', 'Instance ID', 'Required Deadline', 'Round Trip Time', 'In Server Processing Time', 'Response Time', 'Deadline Status']
# # printing the selected columns from User
## print(datasets["User"][selected_columns_user])
# # # filtering the rows
filtered_col_user = datasets["User"][(datasets["User"]["Time Step"] == time_step_value)]
print(filtered_col_user[selected_columns_user])

# # Initialize a counter for 'miss'
# miss_count = 0
#
# # Iterate over each row in the 'Deadline Status' column
# for status in filtered_col_user[selected_columns_user]['Deadline Status']:
#     # Check if 'miss' is present in the dictionary values
#     if 'miss' in status.values():
#         miss_count += 1
#
# # Print the count of 'miss'
# # print(f"\nNumber of 'miss': {miss_count} -- miss-rate: {(miss_count) / (len(filtered_col_user[selected_columns_user]['Deadline Status']))}")

print()

print(f"\nnumber of missed services {service_miss_count}, total services {(len(filtered_col_service[selected_columns_service]['Available']))} ==> service-miss-rate: {(service_miss_count) / (len(filtered_col_service[selected_columns_service]['Available']))}")

# # # Iterate over each row in the "User" dataset
# # for index, row in datasets["User"].iterrows():
# #     # Access the last row of the DataFrame
# #     last_row = datasets["User"].iloc[-1]
# #     # Retrieve the value of the 'Time Step' attribute
# #     time_step_value_last_row = last_row['Time Step']
# #
# #     # Find the largest number in the 'Instance ID' column
# #     max_instance_id = datasets["User"]["Instance ID"].max()
#
#
# # Iterate over each user in the range of Instance IDs
# for instance_id in range(1, max_instance_id + 1):
#     # Filter the dataset for the current user and time step
#     user_data = datasets["User"][(datasets["User"]["Instance ID"] == instance_id) &
#                                  (datasets["User"]["Time Step"] == time_step_value)]
#
#     # If there's data for the user at the specified time step
#     if not user_data.empty:
#         # Access the 'Deadline Status' dictionary
#         deadline_status_dict = user_data.iloc[0]['Deadline Status']
#
#         # Ensure it's a dictionary and access the status value
#         if isinstance(deadline_status_dict, dict):
#             deadline_status = list(deadline_status_dict.values())[0]
#
#             # Check if the deadline was met
#             if deadline_status == 'meet':
#                 meet_deadline_count += 1
#
#         total_users += 1
#
# # Calculate the hit-ratio
# hit_ratio = meet_deadline_count / total_users if total_users > 0 else 0
#
# print(f"Hit Ratio: {hit_ratio:.3f}")


# """ ContainerRegistry """
# # print(datasets["ContainerRegistry"].columns)
# selected_columns = ['Object', 'Time Step', 'Available', 'CPU Demand', 'RAM Demand', 'Server', 'Images', 'Layers']
# print(datasets["ContainerRegistry"][selected_columns])
# # # Filter rows where "Time Step" is 8 or 7
# # filtered_df = datasets["EdgeServer"][(datasets["EdgeServer"]["Time Step"] == 6) | (datasets["EdgeServer"]["Time Step"] == 6)]
# # # Select specific columns from the filtered DataFrame
# # filtered_selected_df = filtered_df[selected_columns]
# # # Print the filtered DataFrame with selected columns