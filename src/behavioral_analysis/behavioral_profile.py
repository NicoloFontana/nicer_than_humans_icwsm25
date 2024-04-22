import json
from src.behavioral_analysis.main_behavioral_features import main_behavioral_features


class BehavioralProfile:
    def __init__(self, strategy_name, opponent_name):
        self.strategy_name = strategy_name
        self.opponent_name = opponent_name
        self.n_games = None
        self.features = {}

    def load_from_file(self, file_path, load_values=False):
        with open(file_path, "r") as profile_file:
            profile = json.load(profile_file)
            for feature_name in profile.keys():
                feature = main_behavioral_features[feature_name]()
                feature.load_from_file(file_path, load_values=load_values)
                self.features[feature_name] = feature
            self.n_games = max(feature.n_games for feature in self.features.values())
