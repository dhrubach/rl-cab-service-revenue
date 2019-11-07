import numpy as np
import math
import random

from datetime import datetime
from itertools import product


class CabDriverEnvironment:
    def __init__(self):
        self.hyperparameters = self.initialize_hyperparameters()
        self.action_space = self.initialize_action_space()
        self.state_space = self.initialize_state_space()
        self.state_init = self.set_init_state()

        self.reset()

    def initialize_hyperparameters(self):
        return {
            "m": 5,  # number of cities, ranges from 1 ...... m
            "t": 24,  # number of hours, ranges from 0 ....... t-1
            "d": 7,  # number of days, ranges from 0 ... d-1
            "C": 5,  # per hour fuel and other costs
            "R": 9,  # per hour revenue from a passanger
        }

    def initialize_action_space(self) -> list:
        """ An action is represented by a tuple (pick_up_location, drop_location).
        Depending on the current state of the cab represented by (current_location, day_of_week, current_time),
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
        """ Current state of a cab is represented by (current_location, day_of_week, current_time).
        Hence the complete state space is a product of number of locations, number of days in a week and number 
        of hours in a day.
        """

        number_locations = [i for i in range(1, self.hyperparameters["m"] + 1)]
        days_in_a_week = [i for i in range(0, self.hyperparameters["d"])]
        hours_in_a_day = [i for i in range(0, self.hyperparameters["t"])]

        total_state_space = product(number_locations, days_in_a_week, hours_in_a_day)

        # convert product object into a list
        return list(total_state_space)

    def set_init_state(self):
        """ Select a random state for a cab to start
        """

        number_locations = [i for i in range(1, self.hyperparameters["m"] + 1)]
        days_in_a_week = [i for i in range(0, self.hyperparameters["d"])]
        hours_in_a_day = [i for i in range(0, self.hyperparameters["t"])]

        random_location = np.random.choice(number_locations)
        current_day = np.random.choice(days_in_a_week)
        current_hour = np.random.choice(hours_in_a_day)

        return (random_location, current_day, current_hour)

    def reset(self):
        return self.action_space, self.state_space, self.state_init
