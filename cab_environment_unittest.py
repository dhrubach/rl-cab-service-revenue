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


if __name__ == "__main__":
    unittest.main()
