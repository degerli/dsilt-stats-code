#Python code for chapter 7 of DSILT: Statistics

#-------------------------------------------------------------------------------------------------#
#---------------------Traveling Salesman Optimization with Simulated Annealing--------------------#
#-------------------------------------------------------------------------------------------------#

from math import sin, cos, sqrt, atan2, radians
from random import random
import numpy as np

#Create a dictionary of cities with lat/lon coordinates
cities = {
    'New York': (40.71, 74.00),
    'Washington D.C.': (38.91, 77.04),
    'Chicago': (41.88, 87.63),
    'Los Angeles': (34.05, 118.24),
    'San Francisco': (37.77, 122.42),
    'Dallas': (32.78, 96.80),
    'Miami': (25.76, 80.19),
    'Houston': (29.76, 95.37),
    'Atlanta': (33.75, 84.39),
    'Philadelphia': (39.95, 75.17),
    'Detroit': (42.33, 83.05),
    'Salt Lake City': (40.76, 111.89),
    'Seattle': (47.61, 122.33),
    'Austin': (30.27, 97.74),
    'Columbus': (39.96, 83.00),
    'Indianapolis': (39.77, 86.16)
    }

#Create a function to calculate distance between coordinates
def distance(c1, c2):
    r = 3959.0  #Radius of the Earth in miles
    lat1 = radians(c1[0])
    lat2 = radians(c2[0])
    lon1 = radians(c1[1])
    lon2 = radians(c2[1])
    dist_lat = lat2 - lat1
    dist_lon = lon2 - lon1
    #Use Haversine formula to get shortest distance over earth's surface
    a = sin(dist_lat/2)**2 + cos(lat1) * cos(lat2) * sin(dist_lon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    return round(r*c, 2)

#print(distance(cities['New York'], cities['Chicago']))

#Calculate distances between every pair of cities
distances = {}
for key_a, val_a in cities.items():
    distances[key_a] = {}
    for key_b, val_b in cities.items():
        if key_b == key_a:
            distances[key_a][key_b] = 0.0
        else:
            distances[key_a][key_b] = distance(val_a, val_b)

#print(distances)

#Define an objective function (the total distance for a sequence of cities)
def objective_fxn(city_list):
    total_dist = 0
    for i, c in enumerate(city_list):
        if i == len(city_list)-1:
            next
        else:
            total_dist = total_dist + distances[city_list[i]][city_list[i+1]]
    return round(total_dist, 2)

#print(objective_fxn(['New York', 'Columbus', 'Seattle']))
#print(objective_fxn(['New York', 'Seattle', 'Columbus']))

#Define the starting sequence and build a function to swap 2 cities at a time
initial_seq = list(distances.keys())
def swap(city_list, city_a, city_b):
    city_list[city_a], city_list[city_b] = city_list[city_b], city_list[city_a]
    return city_list

#print(swap(initial_seq, 0, 1))

#Simulated annealing optimization function
def simulated_anneal(city_list, T=1.0, T_min=0.000001, alpha=0.9):
    old_cost = objective_fxn(city_list)
    while T > T_min:
        for i in range(len(city_list)):
            for j in range(len(city_list)):
                new_seq = swap(city_list, i, j)
                new_cost = objective_fxn(new_seq)
                acceptance_prob = np.exp((old_cost-new_cost)/T)
                if acceptance_prob > random():
                    best_seq = new_seq
                    old_cost = new_cost
        T = T*alpha
    return best_seq, old_cost

print(initial_seq, objective_fxn(initial_seq))
print(simulated_anneal(initial_seq))

#-------------------------------------------------------------------------------------------------#
#-------------------------Particle Swarm Optimization of Flight Schedule--------------------------#
#-------------------------------------------------------------------------------------------------#

from random import randint, random

class flight(object):
    def __init__(self, origin, dest, depart, arrive, price):
        self.origin = origin
        self.dest = dest
        self.depart = depart
        self.arrive = arrive
        self.price = int(price)

flights = []
with open('flight_schedule.txt') as file:
    for line in file:
        origin, dest, depart, arrive, price = line.strip().split(',')
        flights.append(flight(origin, dest, depart, arrive, price))

#Replace : in times with . to make them easier to work with
for f in flights:
    f.depart = float(f.depart.replace(':', '.'))
    f.arrive = float(f.arrive.replace(':', '.'))
    
#Determine possible sequences of flights
flight_seqs = []
for f in flights:
    seq = []
    if f.origin != 'TPA':
        break
    seq.append(f)
    airport = f.dest
    time = float(f.arrive)
    #print(f.origin, airport, time)
    #Second flight
    for ff in flights:
        if ff.origin == airport and ff.depart > time and len(seq) < 2:
            #print(ff.origin, ff.dest, ff.arrive)
            seq.append(ff)
            airport = ff.dest
            time = float(ff.arrive)
            if airport == 'LGW':
                flight_seqs.append(seq)
                #print(len(flight_seqs))
            #Third flight
            for fff in flights:
                if fff.origin == airport and fff.depart > time and len(seq) < 3:
                    #print(fff.origin, fff.dest, fff.arrive)
                    seq.append(fff)
                    airport = fff.dest
                    time = float(fff.arrive)
                    flight_seqs.append(seq)
                    #print(len(flight_seqs))   
    
#Define the objective function to be minimized, given a flight schedule
def objective_fxn(flight_sch):
    total_price = 0
    layovers = 0
    lgw_arrival_time = 0.0
    for f in range(len(flight_sch)):
        total_price = total_price + flight_sch[f].price
        layovers = layovers + 1
    return total_price * layovers

#print(objective_fxn([flights[0], flights[1]]))
#print(objective_fxn(flight_seqs[0]))

class particle(object):
    def __init__(self, max_step, max_pos):
        self.vel = randint(0, max_step)  #Move at most x steps in any direction
        self.pos = randint(0, max_pos) #Random start poition in the sequence
        self.best_pos = 0

#Particle Swarm Optimization
#Note that the position is a index number for the flight_seqs
#flight_seqs lists all possible flight sequences
def particle_swarm_opt(d, swarm_size=min(len(flight_seqs), 20), best_swarm_pos=0):
    particles = []
    for p in range(swarm_size):
        particles.append(particle(max_step=2, max_pos=len(flight_seqs)-1) )
        particles[p].best_pos = particles[p].pos
        #print(particles[p].best_pos, best_swarm_pos)
        if objective_fxn(flight_seqs[particles[p].best_pos]) <= objective_fxn(flight_seqs[best_swarm_pos]):
            best_swarm_pos = particles[p].best_pos
    #Custom stop condition will be at 50 iterations
    i = 1
    while (i < 51):
        for p in range(swarm_size):
            particles[p].vel = particles[p].vel + int(round((0.5*random()*(particles[p].best_pos-particles[p].pos)) + (0.5*random()*(best_swarm_pos-particles[p].pos)), 0))
            particles[p].pos = particles[p].pos + particles[p].vel
            #The next if statement handles positions that stray outside the range of values
            #5 is hardcoded because there are only 6 possible sequences
            if particles[p].pos > 5:
                particles[p].pos = 5
            if objective_fxn(flight_seqs[particles[p].pos]) <= objective_fxn(flight_seqs[particles[p].best_pos]):
                particles[p].best_pos = particles[p].pos
                if objective_fxn(flight_seqs[particles[p].best_pos]) <= objective_fxn(flight_seqs[best_swarm_pos]):
                    best_swarm_pos = particles[p].best_pos
            i = i + 1
    return best_swarm_pos, objective_fxn(flight_seqs[best_swarm_pos])

#print(particle_swarm_opt(flight_seqs))

opt_flight_sch, lowest_cost = particle_swarm_opt(flight_seqs)
print('Best flight sequence: ', opt_flight_sch)
for f in flight_seqs[opt_flight_sch]:
    print(f.origin, f.dest)
print('The cost of this flight schedule is: ', lowest_cost/len(flight_seqs[opt_flight_sch]))
