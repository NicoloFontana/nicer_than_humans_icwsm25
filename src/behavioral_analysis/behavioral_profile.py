import json
from src.behavioral_analysis.main_behavioral_features import main_behavioral_features
from src.utils import OUT_BASE_PATH


class BehavioralProfile:
    def __init__(self, strategy_name, opponent_name):
        self.strategy_name = strategy_name
        self.opponent_name = opponent_name
        self.n_games = None
        self.features = {}

    def add_features(self, features):
        for feature_name in features.keys():
            self.features[feature_name] = features[feature_name]()

    def compute_features(self, main_history, opponent_history):
        for feature in self.features.values():
            feature.compute_feature(main_history, opponent_history)

    def load_from_file(self, file_path, load_values=False):
        with open(file_path, "r") as profile_file:
            profile = json.load(profile_file)
            for feature_name in profile.keys():
                feature = main_behavioral_features[feature_name]()
                feature.load_from_file(file_path, load_values=load_values)
                self.features[feature_name] = feature
            self.n_games = max(feature.n_games for feature in self.features.values())
            self.n_games = self.n_games if self.n_games != 0 else None

    def save_to_file(self, timestamp, subdir=None):
        behavioral_profiles_dir = OUT_BASE_PATH / str(timestamp) / "behavioral_profiles"
        dir_path = behavioral_profiles_dir if subdir is None else behavioral_profiles_dir / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        out_file_path = dir_path / f"behavioral_profile_{self.strategy_name}-{self.opponent_name}.json"
        features_to_json = {}
        for feature_name in self.features.keys():
            feature = self.features[feature_name]
            features_to_json[feature_name] = feature.to_json()
        with open(out_file_path, "w") as out_file:
            json_out = json.dumps(features_to_json, indent=4)
            out_file.write(json_out)
        return self.features
