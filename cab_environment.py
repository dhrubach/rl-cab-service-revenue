import numpy as np
import math
import random

from datetime import datetime
from itertools import product


class CabDriverEnvironment:
    def __init__(self, locations=5, cost=5, reward=9):
        self.hyperparameters = self.initialize_hyperparameters(locations, cost, reward)
        self.action_space = self.initialize_action_space()
        self.state_space = self.initialize_state_space()
        self.state_init = self.set_init_state()

        self.time_matrix = np.load("time_matrix.npy")

        self.reset_state()

    ## Initialize environment hyperparameters, total action space and total state space

    def initialize_hyperparameters(self, locations, cost, reward):
        return {
            "m": locations,  # number of cities, ranges from 1 ...... m (cannot start from 0 as (0,0) represents no ride)
            "t": 24,  # number of hours, ranges from 0 ....... t-1
            "d": 7,  # number of days, ranges from 0 ... d-1
            "C": cost,  # per hour fuel and other costs
            "R": reward,  # per hour revenue from a passanger
        }

    def initialize_action_space(self) -> list:
        """ An action is represented by a tuple (pick_up_location, drop_location).
        Depending on the current state of the cab represented by (current_location, current_time, day_of_week),
        driver will select the most appropriate action which maximizes reward.

        Given 'm' locations, action space : ( ( m - 1) * m ) + 1
            - action will never have same pick_up and drop location
            - (0, 0) represents no ride option --> hence '+1'
        """
        number_locations = self.hyperparameters["m"]
        total_action_space = [
            (i, j)
            for i in range(1, number_locations + 1)
            for j in range(1, number_locations + 1)
            if i != j
        ]

        total_action_space.append((0, 0))

        return total_action_space

    def initialize_state_space(self) -> list:
        """ Current state of a cab is represented by (current_location, current_time, day_of_week).
        Hence the complete state space is a product of number of locations, number of days in a week and number 
        of hours in a day.
        """

        number_locations = [i for i in range(1, self.hyperparameters["m"] + 1)]
        days_in_a_week = [i for i in range(0, self.hyperparameters["d"])]
        hours_in_a_day = [i for i in range(0, self.hyperparameters["t"])]

        total_state_space = product(number_locations, hours_in_a_day, days_in_a_week)

        # convert product object into a list
        return list(total_state_space)

    ## Set / Reset initial cab state

    def set_init_state(self):
        """ Select a random state for a cab to start
        """

        number_locations = [i for i in range(1, self.hyperparameters["m"] + 1)]
        days_in_a_week = [i for i in range(0, self.hyperparameters["d"])]
        hours_in_a_day = [i for i in range(0, self.hyperparameters["t"])]

        random_location = np.random.choice(number_locations)
        current_day = np.random.choice(days_in_a_week)
        current_hour = np.random.choice(hours_in_a_day)

        return (random_location, current_hour, current_day)

    def reset_state(self):
        return self.action_space, self.state_space, self.state_init

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """ Convert the state into a vector so that it can be fed to the NN. 
            This method converts a given state into a vector format. The vector is of size m + t + d.
        """

        all_locations = np.zeros(self.hyperparameters["m"])
        hours_in_a_day = np.zeros(self.hyperparameters["t"])
        days_in_a_week = np.zeros(self.hyperparameters["d"])

        current_location, current_hour, current_day = state

        all_locations[current_location - 1] = 1
        hours_in_a_day[current_hour] = 1
        days_in_a_week[current_day] = 1

        encoded_state = np.hstack(
            (all_locations, hours_in_a_day, days_in_a_week)
        ).reshape(
            1,
            self.hyperparameters["m"]
            + self.hyperparameters["t"]
            + self.hyperparameters["d"],
        )

        return encoded_state

    ## Action space at a given location

    def get_requests_per_location(self, current_state):
        """ Number of requests which a driver can receive at a particular location is 
        pre-defined by a Poisson Distribution. 
        
        Given a current state and distribution, calculate possible requests i.e. action_space
        of a cab
        """

        current_location = current_state[0]
        distribution_lambda = [2, 12, 4, 7, 8]

        total_possible_requests = np.random.poisson(
            distribution_lambda[current_location - 1]
        )

        # limit maximum possible requests routed to a cab to 15
        if total_possible_requests > 15:
            total_possible_requests = 15

        # select a random sample of requests from the total action space
        # remove no - ride option
        total_action_space = self.action_space[:-1]
        allowed_action_index = np.random.choice(
            range(len(total_action_space)), total_possible_requests
        )
        allowed_actions = [total_action_space[i] for i in allowed_action_index]

        allowed_actions.append((0, 0))

        return allowed_action_index, allowed_actions

    ## Execute Step

    def get_next_state(self, state, action):
        """ Calculate next state, reward and total ride time for a given
            state and action
        """
        current_location, current_hour, current_day = state
        location_from, location_to = action

        next_state = state

        total_rewards = self.get_rewards_per_ride(state, action)
        total_ride_time = 0

        if action == (0, 0):
            # no ride action
            # increase current hour by 1
            current_hour = int(current_hour + 1)
            current_hour, current_day = self.calc_revised_time_day(
                current_hour, current_day
            )

            total_ride_time = 1
            next_state = (current_location, current_hour, current_day)
        else:
            if current_location == location_from:
                total_trip_time, travel_time_to_customer = self.get_same_pickup_time(
                    state, action
                )

                # calculate time at the end of the trip
                time_at_trip_end = int(current_hour + total_trip_time)
                # factor next day if time exceeds 23:00 hours
                time_at_trip_end, day_at_trip_end = self.calc_revised_time_day(
                    time_at_trip_end, current_day
                )

                total_ride_time = total_trip_time
                next_state = (location_to, time_at_trip_end, day_at_trip_end)
            else:
                (
                    total_trip_time,
                    travel_time_to_customer,
                    time_at_customer_location,
                    day_at_customer_location,
                ) = self.get_different_pickup_time(state, action)

                # calculate time at the end of the trip
                # use the computed time at customer location instead of current hour
                time_at_trip_end = int(time_at_customer_location + total_trip_time)
                # factor next day if time exceeds 23:00 hours
                (time_at_trip_end, day_at_trip_end) = self.calc_revised_time_day(
                    time_at_trip_end, day_at_customer_location
                )

                total_ride_time = total_trip_time + travel_time_to_customer
                next_state = (location_to, time_at_trip_end, day_at_trip_end)

        return next_state, total_rewards, total_ride_time

    ## Reward Calculations

    def get_rewards_per_ride(self, state, action):
        """ Calculated Reward :
            (revenue earned from pickup point 𝑝 to drop point 𝑞) 
            - (Cost of battery used in moving from pickup point 𝑝 to drop point 𝑞) 
            - (Cost of battery used in moving from current point 𝑖 to pick-up point 𝑝)

            Assumptions :
                cost and revenue are purely functions of time i.e. for every hour of driving,
                cost and revenue is independent of the traffic conditions, speed, etc.
        """
        calculated_reward = 0
        total_trip_time = 0
        travel_time_to_customer = 0

        current_location, _, _ = state
        location_from, _ = action

        if action == (0, 0):
            calculated_reward = -self.hyperparameters["C"]
        else:
            if current_location == location_from:
                total_trip_time, travel_time_to_customer = self.get_same_pickup_time(
                    state, action
                )
            else:
                (
                    total_trip_time,
                    travel_time_to_customer,
                    _,
                    _,
                ) = self.get_different_pickup_time(state, action)

            # fmt:off
            calculated_reward = (
                self.hyperparameters['R'] * total_trip_time - 
                (self.hyperparameters['C'] * (total_trip_time + travel_time_to_customer))
                )
            # fmt:on

        return calculated_reward

    def get_same_pickup_time(self, state, action):
        """ Calculate total trip time when current location and pickup location are same
        """

        _, current_hour, current_day = state
        location_from, location_to = action

        # fmt:off
        total_trip_time = (self.time_matrix
            [int(location_from - 1)]
            [int(location_to - 1)]
            [int(current_hour)]
            [int(current_day)]
            )
        # fmt:on

        # set travel_time_to_customer as 0
        return total_trip_time, 0

    def get_different_pickup_time(self, state, action):
        """ Calculate total trip time and time taken to reach the customer when current location
            and pickup location are different
        """

        current_location, current_hour, current_day = state
        location_from, location_to = action

        # fmt:off
        travel_time_to_customer = (self.time_matrix
            [int(current_location - 1)]
            [int(location_from - 1)]
            [int(current_hour)]
            [int(current_day)]
            )
        # fmt:on

        time_at_customer_location = int(current_hour + travel_time_to_customer)
        day_at_customer_location = current_day

        (
            time_at_customer_location,
            day_at_customer_location,
        ) = self.calc_revised_time_day(
            time_at_customer_location, day_at_customer_location
        )

        # fmt:off
        total_trip_time = (self.time_matrix
            [int(location_from - 1)]
            [int(location_to - 1)]
            [int(time_at_customer_location)]
            [int(day_at_customer_location)]
            )
        # fmt:on

        return (
            total_trip_time,
            travel_time_to_customer,
            time_at_customer_location,
            day_at_customer_location,
        )

    ## Utility function

    def calc_revised_time_day(self, time_of_day, day_of_week):
        if time_of_day >= 24:
            time_of_day = int(time_of_day - 24)
            if day_of_week == 6:
                day_of_week = 0
            else:
                day_of_week = int(day_of_week + 1)

        return time_of_day, day_of_week

