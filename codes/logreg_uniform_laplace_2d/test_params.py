

import unittest
import numpy as np

from main import run_program

from config import parse_args


def main():

    suite = unittest.TestSuite()
    suite.addTest(TestParams("test_weights_prior_params"))
    # suite.addTest(TestParams("test_data_x_type"))
    # suite.addTest(TestParams("test_num_data"))
    # suite.addTest(TestParams("test_num_samples"))
    
    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)


class TestParams(unittest.TestCase):
    """
    """
    def setUp(self):
        # default config
        self.config = parse_args()
        
    def run_test(self):
        num_runs = 4
        for i in range(num_runs):
            seed = i+1
            self.config['seed'] = seed
            run_program(self.config)

    def test_weights_prior_params(self):
        
        cases = [
            ('01', [[0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]]),
            ('02', [[0.0, 0.0], [[50.0, 0.0], [0.0, 50.0]]]),
            # ('03', [[0.0, 0.0], [[1.00, 0.8], [0.3, 1.00]]]),
        ]
        for args in cases:
            c = args[0]
            self.config['test_name'] = f"test_weights_prior_params_{c}"
            self.config['weights_prior_params'] = args[1]
            self.run_test()

    def test_data_x_type(self):
        cases = [
            ('01', "uniform"),
            ('02', "mixgauss"),
        ]
        for args in cases:
            c = args[0]
            self.config['test_name'] = f"test_data_x_type_{c}"
            self.config['data_x_type'] = args[1]
            self.run_test()
            
    def test_num_data(self):
        cases = [
            ('01', 2),
            ('02', 10),
            ('03', 50),
        ]
        for args in cases:
            c = args[0]
            self.config['test_name'] = f"test_num_data_{c}"
            self.config['num_data'] = args[1]
            self.config['num_data_half'] = args[1] // 2
            self.run_test()

    def test_num_samples(self):
        cases = [
            ('01', 1000),
            # ('02', 500),
            # ('03', 5000),
            # ('04', 10000),
        ]
        for args in cases:
            c = args[0]
            self.config['test_name'] = f"test_num_samples_{c}"
            self.config['num_samples'] = args[1]
            self.run_test()

    
            
    

            
if __name__ == '__main__':
    main()
    

    
