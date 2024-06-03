import json
from src.behavioral_analysis.main_behavioral_dimensions import main_behavioral_dimensions
from src.utils import OUT_BASE_PATH


class BehavioralProfile:
    def __init__(self, strategy_name, opponent_name):
        self.strategy_name = strategy_name
        self.opponent_name = opponent_name
        self.n_games = None
        self.dimensions = {}

    def add_dimensions(self, dimensions_names):
        for dimension_name in dimensions_names:
            self.dimensions[dimension_name] = main_behavioral_dimensions[dimension_name]()

    def compute_dimensions(self, main_history, opponent_history):
        for dimension in self.dimensions.values():
            dimension.compute_dimension(main_history, opponent_history)

    def load_from_file(self, file_path, load_values=False):
        with open(file_path, "r") as profile_file:
            profile = json.load(profile_file)
            for dimension_name in profile.keys():
                dimension = main_behavioral_dimensions[dimension_name]()
                dimension.load_from_file(file_path, load_values=load_values)
                self.dimensions[dimension_name] = dimension
            self.n_games = max(feature.n_games for feature in self.dimensions.values())
            self.n_games = self.n_games if self.n_games != 0 else None

    def save_to_file(self, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file_path = out_dir / f"behavioral_profile_{self.strategy_name}-{self.opponent_name}.json"
        dimensions_to_json = {}
        for dimensions_name in self.dimensions.keys():
            dimension = self.dimensions[dimensions_name]
            dimensions_to_json[dimensions_name] = dimension.to_json()
        with open(out_file_path, "w") as out_file:
            json_out = json.dumps(dimensions_to_json, indent=4)
            out_file.write(json_out)
        return self.dimensions
