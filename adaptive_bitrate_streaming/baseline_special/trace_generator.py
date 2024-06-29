#!/usr/bin/python
import sys
import os
import csv
import numpy as np
import ast
import random

# RANDOM_SEED = 101
# args = sys.argv

# '''
# args[1]: config dictionary

# TODO: This might need to take in a configs.json with multiple configurations?
# '''

####################
####################
# Argument Loading #
####################
####################
# May be helpful for passing dictionaries:
# https://stackoverflow.com/questions/3780468/passing-dictionaries-to-a-python-script-through-the-command-line
# 
# if len(args) < 2:
    #print("Usage: python {} <configuration dictionary>".format(args[0]))
    # sys.exit()

#print("Generating random traces with arguments...")
# config = ast.literal_eval(args[1])
#print("Running on config: ", config)

#################
# TraceGenerator
#################

class TraceGenerator(object):
    """
    Trace generator used to simulate a network trace.
    non-MDP logic
    """
    def __init__(self, T_l, T_s, cov, duration, steps, min_throughput, max_throughput, seed):
        """Construct a trace generator.
        T_s: frequency of changing the bandwidth
        duration: the duration of the trace in time
        """
        self.T_s = T_s
        self.duration = duration
        self.min_throughput = min_throughput
        self.max_throughput = max_throughput

        # The following vars are unused, but are kept around for backwards compatibility with
        # previous code.
        self.T_l = T_l # unused
        self.cov = cov # unused
        self.steps = steps # unused
        self.seed = seed # unused

        # set numpy random seed
        np.random.seed(self.seed)

    def generate_trace(self):
        """Generate a network trace."""
        round_digit = 2
        ts = 0 # timestamp
        cnt = 0
        trace_time = []
        trace_bw = []
        assert self.min_throughput is not None
        assert self.max_throughput is not None        
        last_val = round(np.random.uniform(self.min_throughput, self.max_throughput), round_digit)

        while ts < self.duration:
            if cnt <= 0:
                bw_val = round(np.random.uniform(self.min_throughput, self.max_throughput), round_digit)
                cnt = np.random.randint(1, self.T_s + 1)

            elif cnt >= 1:
                bw_val = last_val
            else:
                bw_val = round(np.random.uniform(self.min_throughput, self.max_throughput ), round_digit)

            cnt -= 1
            last_val = bw_val
            time_noise = np.random.uniform(0.1, 3.5)
            ts += time_noise
            ts = round(ts, 2)
            trace_time.append(ts)
            trace_bw.append(bw_val)

        return trace_time, trace_bw

#################
# Create Generator
#################

# tg = TraceGenerator(config["T_l"],
#                     config["T_s"],
#                     config["cov"],
#                     config["duration"],
#                     config["step"],
#                     config["min_throughput"],
#                     config["max_throughput"],
#                     RANDOM_SEED)

# #################
# # Write out traces to directory
# #################
# output_dir = config["trace_dir"]

# I comment this because when BO test generate traces, I do not want to save them
# if not os.path.exists(output_dir):
# os.makedirs(output_dir, exist_ok=True)
# for i in range(config["num_traces"]):
#     output_writer = csv.writer(open(output_dir + 'trace_' + str(i), 'w', 1), delimiter='\t')
#     random_traces_time, random_traces_bw = tg.generate_trace()
#     for ts, bw in zip(random_traces_time, random_traces_bw):
#         output_writer.writerow([ts, bw])
# else:
    #print("INFO: Traces directory exists already. Skipping generation.")
    
#print("INFO: End trace generation script.")

