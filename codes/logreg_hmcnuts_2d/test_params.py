

import unittest

import numpy as np

from main import run_program

def main():

    suite = unittest.TestSuite()
    # suite.addTest(TestParams("test_num_data"))
    # suite.addTest(TestParams("test_num_samples"))
    suite.addTest(TestParams("test_weights_prior_params"))
    
    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)


class TestParams(unittest.TestCase):
    """
    """
    def setUp(self):
        # default config
        self.config = {}
        self.config["num_feats"] = 2
        self.config["num_data"] = 100
        self.config["num_data_half"] = self.config["num_data"] // 2
        self.config["num_samples"] = 1000
        
        self.config["data_x_marginal_params"] = [
            [[1.0, 5.0], [(1.0, 0.0), (0.0, 1.0)]],
            [[-5.0, 1.0], [(3.0, 0.0), (0.0, 3.0)]]]

        # weights prior distribution parameters
        self.config["weights_prior_params"] = \
            [[0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]]

        # init sigma used in numerical optimization for laplace approximation
        self.config["laplace_init_sigma"] = 0.01
        self.config["laplace_num_iters"] = 1000
    
        
    def run_test(self):
        num_runs = 1
        for i in range(num_runs):
            seed = i+1
            self.config['seed'] = seed
            run_program(self.config)

    def test_num_data(self):
        cases = [
            ('01', 20),
            ('02', 50),
            ('03', 100),
        ]
        for args in cases:
            c = args[0]
            self.config['test_name'] = f"test_num_data_{c}"
            self.config['num_data'] = args[1]
            self.config['num_data_half'] = args[1] // 2
            self.run_test()

    def test_num_samples(self):
        cases = [
            # ('01', 1000),
            # ('02', 500),
            ('03', 5000),
            # ('04', 10000),
        ]
        for args in cases:
            c = args[0]
            self.config['test_name'] = f"test_num_samples_{c}"
            self.config['num_samples'] = args[1]
            self.run_test()
    
    def test_weights_prior_params(self):
        
        cases = [
            # ('01', [[0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]]),
            # ('02', [[0.0, 0.0], [[3.0, 0.0], [0.0, 3.0]]]),
            ('03', [[0.0, 0.0], [[1.32, 0.48], [0.48, 0.23]]]),
        ]
        for args in cases:
            c = args[0]
            self.config['test_name'] = f"test_weights_prior_params_{c}"
            self.config['weights_prior_params'] = args[1]
            self.run_test()

            
if __name__ == '__main__':
    main()
    

    
