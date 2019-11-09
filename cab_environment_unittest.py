import logging
import numpy as np
import unittest

from cab_environment import CabDriverEnvironment

logger = logging.getLogger("__name__")


class CabDriverEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.cabDriverEnvironment = CabDriverEnvironment(locations=3)

    def test_action_space(self):
        estimated_action_space = self.cabDriverEnvironment.initialize_action_space()

        self.assertEqual(len(estimated_action_space), 7)

        index_terminal_action = estimated_action_space.index((0, 0))
        self.assertEqual(index_terminal_action, 6)

    def test_state_space(self):
        estimated_state_space = self.cabDriverEnvironment.initialize_state_space()

        self.assertEqual(len(estimated_state_space), 3 * 7 * 24)

    def test_requests_per_location(self):
        (
            allowed_action_index,
            allowed_actions,
        ) = self.cabDriverEnvironment.get_requests_per_location(
            self.cabDriverEnvironment.state_init
        )

        total_requests = len(allowed_action_index)
        self.assertEqual(total_requests, len(allowed_actions) - 1)

        self.assertEqual(allowed_actions[len(allowed_actions) - 1], (0, 0))

    def test_ride_time_same_pickup_location(self):
        current_day = 2  # Tuesday
        current_hour = 13  # 1 PM
        current_state = (1, 13, 2)

        # request to travel from location 1 to location 4
        current_action = (1, 4)

        # fmt:off
        # time matrix is zero indexed. 
        total_ride_time_from_matrix = (self.cabDriverEnvironment.time_matrix
            [current_action[0] - 1]
            [current_action[1] - 1]
            [current_hour]
            [current_day]
            )
        # fmt: on

        total_ride_time_from_env = self.cabDriverEnvironment.get_same_pickup_time(
            current_state, current_action
        )

        self.assertEqual(total_ride_time_from_matrix, total_ride_time_from_env[0])

    def test_ride_time_diff_pickup_location(self):
        current_day = 1  # Monday
        current_hour = 17  # 5 PM
        current_state = (3, 17, 1)

        # request to travel from location 1 to location 4
        current_action = (1, 4)

        (
            total_trip_time,
            travel_time_to_customer,
            _,
            _,
        ) = self.cabDriverEnvironment.get_different_pickup_time(
            current_state, current_action
        )

        # fmt:off
        # time matrix is zero indexed. 
        time_to_customer_from_matrix = (self.cabDriverEnvironment.time_matrix
            [current_state[0] - 1]
            [current_action[0] - 1]
            [current_hour]
            [current_day]
            )
        # fmt: on

        self.assertEqual(time_to_customer_from_matrix, travel_time_to_customer)

        # fmt:off
        # time matrix is zero indexed. 
        ride_time_from_matrix = (self.cabDriverEnvironment.time_matrix
            [current_action[0] - 1]
            [current_action[1] - 1]
            [current_hour]
            [current_day]
            )
        # fmt: on

        self.assertEqual(ride_time_from_matrix, total_trip_time)

    def test_calculate_rewards(self):
        # current location same as pickup location
        current_state = (1, 8, 1)
        current_action = (1, 3)

        # fmt:off
        ride_time_from_matrix = (self.cabDriverEnvironment.time_matrix
            [current_action[0] - 1]
            [current_action[1] - 1]
            [current_state[1]]
            [current_state[2]]
            )
        # fmt: on

        reward_factor = self.cabDriverEnvironment.hyperparameters["R"]
        cost_factor = self.cabDriverEnvironment.hyperparameters["C"]

        manual_reward = (reward_factor * ride_time_from_matrix) - (
            cost_factor * ride_time_from_matrix
        )

        self.assertEqual(
            manual_reward,
            self.cabDriverEnvironment.get_rewards_per_ride(
                current_state, current_action
            ),
        )

        # current location different from pickup location
        current_state = (1, 8, 1)
        current_action = (2, 3)

        # fmt:off
        time_to_customer_from_matrix = (self.cabDriverEnvironment.time_matrix
            [current_state[0] - 1]
            [current_action[0] - 1]
            [current_state[1]]
            [current_state[2]]
            )

        time_at_customer_location = int(current_state[1] + time_to_customer_from_matrix)
        day_at_customer_location = current_state[2]

        ride_time_from_matrix = (self.cabDriverEnvironment.time_matrix
            [current_action[0] - 1]
            [current_action[1] - 1]
            [time_at_customer_location]
            [day_at_customer_location]
            )
        # fmt: on

        manual_reward = (reward_factor * ride_time_from_matrix) - (
            cost_factor * (ride_time_from_matrix + time_to_customer_from_matrix)
        )

        self.assertEqual(
            manual_reward,
            self.cabDriverEnvironment.get_rewards_per_ride(
                current_state, current_action
            ),
        )

    def test_next_step(self):
        # scenario 1 : no ride
        current_state = (1, 8, 1)
        current_action = (0, 0)

        expected_next_state = (1, 9, 1)
        expected_reward = -5
        expected_total_trip_time = 1

        (
            next_state,
            total_rewards,
            total_ride_time,
        ) = self.cabDriverEnvironment.get_next_state(current_state, current_action)

        self.assertEqual(expected_next_state, next_state)
        self.assertEqual(expected_reward, total_rewards)
        self.assertEqual(expected_total_trip_time, total_ride_time)

        # scenario 2 : current location same as pickup location
        current_state = (1, 8, 1)
        current_action = (1, 3)

        expected_next_state = (3, 14, 1)
        expected_reward = 24
        expected_total_trip_time = 6

        (
            next_state,
            total_rewards,
            total_ride_time,
        ) = self.cabDriverEnvironment.get_next_state(current_state, current_action)

        self.assertEqual(expected_next_state, next_state)
        self.assertEqual(expected_reward, total_rewards)
        self.assertEqual(expected_total_trip_time, total_ride_time)

        # scenario 3 : current location different from pickup location
        current_state = (1, 19, 2)
        current_action = (2, 3)

        expected_next_state = (3, 8, 3)
        expected_reward = -2
        expected_total_trip_time = 13

        (
            next_state,
            total_rewards,
            total_ride_time,
        ) = self.cabDriverEnvironment.get_next_state(current_state, current_action)

        self.assertEqual(expected_next_state, next_state)
        self.assertEqual(expected_reward, total_rewards)
        self.assertEqual(expected_total_trip_time, total_ride_time)


if __name__ == "__main__":
    unittest.main()
