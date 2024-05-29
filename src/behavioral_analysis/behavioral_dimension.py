import json


class BehavioralDimension:
    def __init__(self, name: str):
        self.name = name
        self.values = []
        self.n_games = None
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_dimension(self, main_history: list, opponent_history: list) -> float:
        raise NotImplementedError()

    def update_aggregates(self):
        self.mean = sum(self.values) / len(self.values)
        self.variance = sum((x - self.mean) ** 2 for x in self.values) / len(self.values)
        self.std_dev = self.variance ** 0.5

    def load_from_file(self, file_path, load_values=False):
        with open(file_path, "r") as profile_file:
            profile = json.load(profile_file)
            feature = profile[self.name]
            self.values = feature["values"] if load_values else []
            self.n_games = len(feature["values"])
            self.mean = feature["mean"]
            self.variance = feature["variance"]
            self.std_dev = feature["std_dev"]

    def to_json(self):
        return {
            "values": self.values,
            "mean": self.mean,
            "variance": self.variance,
            "std_dev": self.std_dev
            # TODO confidence interval bounds
        }
