# -*- coding: utf-8 -*-
"""
__author__: bishwarup
"""
import os

config = {}
config["input_dir"] = "../input"
config["lgb_params"] = {
                    "objective": "regression",
                    "boosting_type": "gbdt",
                    "metric" : "rmse",
                    "learning_rate": 0.01,
                    "num_leaves": 128,
                    "max_depth" : 8,
                    "max_bin": 56,
                    "lambda_l1": 3,
                    "feature_fraction": 0.35,
                    "verbose" : -1,
                    "min_data_in_leaf" : 200,
                    "subsample": 1.,
                    "num_threads" : os.cpu_count()
                    }

config["xgb_params"] = {
            "objective": "reg:linear",
            "eta": 0.02,
            "max_depth": 10,
            "subsample" : 1.0,
            "colsample_bytree" : 0.35,
            "min_child_weight" : 1,
            "silent": 1,
            "gamma" : 3,
            "silent": 1,
            "nthread" : os.cpu_count()
            }

config["lgb_bag_1_seeds"] = [2019, 2031, 90, 192, 83123, 5601, 7313, 76, 9558, 916]
config["lgb_bag_2_seeds"] = [1256, 2015, 3190, 7192, 3123, 5611, 7013, 7672, 1013, 2154]
config["xgb_seeds"] = [2119, 1031, 190, 13192, 23123, 5603]