# %% [markdown]
# # NYC Bike Share Simulation

# %% [markdown]
# ## 1. Scenario

# %% [markdown]
# This project will involve the implementation of and experimentation with a discrete event simualtion of a bike sharing service in NYC. The simulation will involve riders and bike stations where riders can pick up and drop off bikes.

# %% [markdown]
# ## 2.1 Simulator Implementation

# %% [markdown]
# This section will detail the implementation of the simulator. This simulator is implemented using three classes, Events, Stations, and Riders. Station and Rider objects will each hold data corresponding to bike stations and bike riders, respectively. Event objects will hold data corresponding to either an arrival of a rider at a station or the return of a rider to a station. Once at a station, a rider will either take or return a bike, or enter the correpsonding queue if necessary. A rider will need to enter a queue at a station if the station either does not have enough bikes for the rider that is arriving, or if it is at capacity when a rider returns. For this simulator, there will be m stations and n riders. the arrival times of riders will be given by an exponential distribution with mean rate $\lambda$. A rider will select station i to initially go to with probability $p_i$, and will return to station j with probability $q_{i,j}$ The duration of a rider's use of the bike will be generated using a log-normal distribution with mean $\mu$ and standard deviation $\sigma$.

# %%
import numpy as np
import heapq

# %%
class Event:
  def __init__(self, time, event_type, rider, source=None, destination=None):
    self.time = time
    self.event_type = event_type
    self.rider = rider
    self.source = source
    self.destination = destination

  def __lt__(self, other):
    return self.time < other.time

class Station:
  def __init__(self, id, initial_num, capacity):
    self.id = id
    self.num_bikes = initial_num
    self.capacity = capacity
    self.arrival_queue = []
    self.return_queue = []

class Rider:
  def __init__(self, id):
    self.id = id
    self.completed_ride = False
    self.arrival_time = None
    self.departure_time = None
    self.attempted_return_time = None
    self.return_time = None
    self.source = None
    self.destination = None


# %%
def arrival(time, rider, arrival_rate, num_stations, arrival_probabilities, stations):
  time += np.random.exponential(1/arrival_rate)
  source = stations[np.random.choice(num_stations, p=arrival_probabilities)]
  rider.source = source
  rider.arrival_time = time
  return Event(time, "arrival", rider, source)

def attempt_return(time, rider, num_stations, duration_mean, duration_std, destination_probabilities, stations):
  duration = np.random.lognormal(mean=duration_mean, sigma=duration_std)
  time += duration
  destination = stations[np.random.choice(num_stations, p=destination_probabilities[rider.source.id])]
  rider.destination = destination
  rider.attempted_return_time = time
  return Event(time, "return", rider, rider.source, destination)


# %%
def simulate(num_riders, duration_mean, duration_std, arrival_rate, num_stations,
             stations_initial, station_capacities, arrival_probabilities, destination_probabilities, sim_time, sim=1):
  stations = [Station(i, stations_initial[i], station_capacities[i]) for i in range(num_stations)]
  riders = [Rider(i) for i in range(num_riders)]
  event_list = []
  num_bikes_needed = stations_initial.copy()
  time = 0
  heapq.heappush(event_list, arrival(time, riders[0], arrival_rate, num_stations, arrival_probabilities, stations))

  while time <= sim_time * 60 and len(event_list) > 0:
    current_event = heapq.heappop(event_list)
    time = current_event.time
    if current_event.event_type == "arrival":
      station = current_event.source
      rider = current_event.rider
      #print(f"Rider {rider.id} arrived at station {station.id}  at time {time}")
      if (rider.id < num_riders - 1):
        heapq.heappush(event_list, arrival(time, riders[rider.id + 1], arrival_rate, num_stations, arrival_probabilities, stations))
      if station.num_bikes > 0:
        #print(f"Rider {rider.id} departed station {station.id} at time {time}")
        rider.departure_time = time
        station.num_bikes -= 1
        heapq.heappush(event_list, attempt_return(time, rider, num_stations, duration_mean, duration_std, destination_probabilities, stations))
      else:
        #print(f"Rider {rider.id} entered arrival queue at station {station.id} at time {time}")
        if (sim == 1):
          station.arrival_queue.append(rider)
        else:
          num_bikes_needed[station.id] += 1
          rider.departure_time = time
          heapq.heappush(event_list, attempt_return(time, rider, num_stations, duration_mean, duration_std, destination_probabilities, stations))
      if len(station.return_queue) > 0:
        station.num_bikes += 1
        rider = station.return_queue.pop(0)
        rider.completed_ride = True
        rider.return_time = time
    else:
      station = current_event.destination
      rider = current_event.rider
      #print(f"Rider {rider.id} returned to station {station.id} at time {time}")
      if station.num_bikes < station.capacity:
        #print(f"Rider {rider.id} completed ride at {station.id} at time {time}")
        station.num_bikes += 1
        rider.completed_ride = True
        rider.return_time = time
      else:
        #print(f"Rider {rider.id} entered return queue at station {station.id} at time {time}")
        station.return_queue.append(rider)
      if len(station.arrival_queue) > 0:
        rider = station.arrival_queue.pop(0)
        #print(f"Rider {rider.id} departed station {station.id} at time {time}")
        rider.departure_time = time
        station.num_bikes -= 1
        heapq.heappush(event_list, attempt_return(time, rider, num_stations, duration_mean, duration_std, destination_probabilities, stations))


  #Simulation data

  #End number of Bikes at each station
  #for station in stations:
    #print(f"Number of bikes at station {station.id}: {station.num_bikes}")

  #Average ride time for riders who obtained a bike and attempted to return it and number riders who got a bike
  return_time = 0
  num_successful_riders = 0
  for rider in riders:
    if (rider.departure_time is not None and rider.attempted_return_time is not None):
      return_time += (rider.attempted_return_time - rider.departure_time)
      num_successful_riders += 1
  return_time /= num_successful_riders
  #print(f"Average ride time: {return_time}")

  #Percent of successful riders
  average_riders = num_successful_riders / num_riders
  #print(f"Probability of successful ride: {average_riders}")

  #Average waiting time
  wait_time = 0
  for rider in riders:
    if (rider.departure_time is not None and rider.arrival_time is not None):
      wait_time += (rider.departure_time - rider.arrival_time)
  wait_time /= num_successful_riders
  #print(wait_time)

  return(average_riders, wait_time, num_bikes_needed)


# %% [markdown]
# ### Testing

# %% [markdown]
# This simulator was verified using several simulations with varying parameters and checking that the behavior matches the expected behavior

# %% [markdown]
# For example, in the following simulation the expected behavior is that each of the seven riders will arrive to one of the three stations, attempt to take a bike, and then depart the station. At some point in the future, the rider should return to a station and exit the system. Here, the intitial number of bikes at each station is enough so that no rider should have to enter the arrival queue to wait for a bike. Additionally, no rider should have to wait to return a bike since the capacities are high enough for these stations.

# %% [markdown]
# As can be seen in the output, this simulation results in expected behavior.

# %%
num_riders = 7
num_stations = 3
duration_mean = 2.78
duration_std = 0.619
arrival_rate = 2.38
stations_initial = [7, 7, 7]
station_capacities = [10, 10, 10]
arrival_probabilities = [0.3, 0.2, 0.5]
destination_probabilities = [[0.1, 0.4, 0.5], [0.2, 0.5, 0.3], [0.3, 0.3, 0.4]]
sim_time = 5
results = simulate(num_riders, duration_mean, duration_std, arrival_rate, num_stations, stations_initial, station_capacities, arrival_probabilities, destination_probabilities, sim_time)

# %% [markdown]
# Here, the arrival and destination probabilites are varied so that every rider should arrive at station 0 (here, 0 corresponds to whichever the first station is), and then return to station 2. Since there are only five bikes at station 0, the last 2 riders will need to enter the arrival queue. Additionally, each rider will need to enter the return queue at station 2 since it is at capactity. Since no bikes arrive at station 0 and no bikes return to station 2, the riders in the queue will remain there.

# %%
num_riders = 7
num_stations = 3
duration_mean = 2.78
duration_std = 0.619
arrival_rate = 2.38
stations_initial = [5, 5, 5]
station_capacities = [10, 10, 5]
arrival_probabilities = [1.0, 0.0, 0.0]
destination_probabilities = [[0.0, 0.0, 1.0], [0.2, 0.5, 0.3], [0.3, 0.3, 0.4]]
sim_time = 5
results = simulate(num_riders, duration_mean, duration_std, arrival_rate, num_stations, stations_initial, station_capacities, arrival_probabilities, destination_probabilities, sim_time)

# %% [markdown]
# In the process of creating this simulator, several basic simulations like the ones above were run to verify the simulator.

# %% [markdown]
# ## 2.2 A Baseline Experiment

# %% [markdown]
# In this section, an experiment will be run using given data for starting station probabilities and the number of trips made between stations from June 2022. In this experiment, there will be 3500 riders. The arrival rate in the exponential distribution of rider interarrival times will be $\lambda = 2.38$ riders per minute, and the parameters for ride duration will be $\mu = 2.78$ and $\sigma = 0.619$. Note that these are the same parameters used in the testing simulations. Also, note that the print statements in the simulation code, which were used for testing, are commented out for this section.

# %% [markdown]
# This simulation will be used to estimate both percentage of successful riders, that is riders who obtained a bike, and the average waiting time amongst successful riders when each station starts with 10 bikes.

# %% [markdown]
# Locations will be imported from the given CSV files. Each location will be given an index corresponding to its position in start_station_probs.csv

# %%
import csv

# %%
locations = {}

with open('start_station_probs.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for index, row in enumerate(reader):
        location = row[0]
        locations[location] = index

print(locations)

# %%
arrival_probabilities = []

with open('start_station_probs.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        arrival_probabilities.append(float(row[1]))

print(arrival_probabilities)

# %% [markdown]
# Using the indices for each location, a list is generated for return station probabilities, and each of these lists is stored in a list. Note that if an end location (such as any beginning with 6 ave in trip_stats.csv) is not in start_station_probs.csv, then it will not be considered.

# %%
num_locations = len(locations)
trip_counts = np.zeros((num_locations, num_locations), dtype=int)

with open('trip_stats.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        start_location = row['start']
        end_location = row['end']
        trip_count = int(row['count'])
        start_index = locations.get(start_location)
        end_index = locations.get(end_location)
        if start_index is not None and end_index is not None:
            trip_counts[start_index][end_index] = trip_count


total_trips_from_location = np.sum(trip_counts, axis=1, keepdims=True)
trip_probabilities = trip_counts / total_trips_from_location
destination_probabilities = list(trip_probabilities)

for i in range(num_locations):
    start_location = list(locations.keys())[i]

# %% [markdown]
# 10 simulations will be run using the inputs described above, and the proportion of successful riders and their average waiting time will be output.

# %%
num_riders = 3500
num_stations = len(arrival_probabilities)
duration_mean = 2.78
duration_std = 0.619
arrival_rate = 2.38
stations_initial = [10 for _ in range(num_stations)]
station_capacities = [10 for _ in range(num_stations)]
sim_time = 24
successful_riders = []
wait_times = []
num_sims = 15
simulate(num_riders, duration_mean, duration_std, arrival_rate, num_stations, stations_initial, station_capacities, arrival_probabilities, destination_probabilities, sim_time, sim=1)
for i in range(num_sims):
    results = simulate(num_riders, duration_mean, duration_std, arrival_rate, num_stations, stations_initial, station_capacities, arrival_probabilities, destination_probabilities, sim_time, sim=1)
    successful_riders.append(results[0])
    wait_times.append(results[1])

print(successful_riders)
print(wait_times)

# %% [markdown]
# ### Confidence Interval Estimation

# %% [markdown]
# Using the CLT and a student's t distribution, 90% confidence intervals will be estimated for each of the outputs in the above simulation.

# %%
from scipy.stats import t

# %%
def students_t_interval(data, confidence_level):
    sample_mean = np.mean(data)
    sample_std = np.std(data)
    deg = len(data) - 1
    alpha = 1 - confidence_level
    t_value = t.ppf(1 - alpha/2, deg)
    margin_of_error = t_value * (sample_std / np.sqrt(len(data) - 1))
    confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
    return confidence_interval

# %%
print(f'90% Confidence Interval for the Proportion of Successful Riders: {students_t_interval(successful_riders, 0.9)}')
print(f'90% Confidence Interval for the Average Wait Time of a Successful Rider: {students_t_interval(wait_times, 0.9)}')

# %% [markdown]
# ## 2.3 An Idealized Experiment

# %% [markdown]
# In this section, the simulator will be used to determine the minimum number of bikes needed to meet demand fully. Here, meet demand fully will mean that the average wait time will be zero. Note that since the interarrival time of riders is 2.38 riders per minute, it is impossible to guarantee that all riders will receive a bike, nor is it typical that every rider will recieve a bike (since 2.38/minute is 3427 riders over the 24 hour period the simulation is run).

# %% [markdown]
# This experiment will be done by running the simulation 50 times, and keeping track of the number of bikes needed at each station to prevent any riders from needing to wait. Then, the max number of bikes for each station will be used, since this gaurantees that over all of the previous simulations no riders would have entered the arrival queue.

# %%
num_riders = 3500
num_stations = len(arrival_probabilities)
duration_mean = 2.78
duration_std = 0.619
arrival_rate = 2.38
stations_initial = [0 for _ in range(num_stations)]
station_capacities = [np.inf for _ in range(num_stations)]
sim_time = 24
num_sims = 50
bikes_needed = np.zeros((num_sims, num_locations), dtype=int)
for i in range(num_sims):
    results = simulate(num_riders, duration_mean, duration_std, arrival_rate, num_stations, stations_initial, station_capacities, arrival_probabilities, destination_probabilities, sim_time, sim=3)
    bikes_needed[i, :] = results[2]

max_values = np.max(bikes_needed, axis=0)
num_bikes_result = np.array(max_values)
print(num_bikes_result)

# %% [markdown]
# As can be seen in the below code, the above values for the number of intial bikes at each station gives a wait time of 0 minutes for the 3500 riders. Note that the randomness of the simulation makes it so these initial values do not guarantee that the wait time will always be 0.

# %%
stations_initial = num_bikes_result
result = simulate(num_riders, duration_mean, duration_std, arrival_rate, num_stations, stations_initial, station_capacities, arrival_probabilities, destination_probabilities, sim_time, sim=1)
print(f"Proportion of Successful Riders: {result[0]}")
print(f"Average Wait Time for Successful Riders: {result[1]}")


