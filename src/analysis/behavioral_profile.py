def compute_niceness(main_history, opponent_history):
    n = len(main_history)
    is_nice = 1
    for i in range(n):
        if main_history[i] == 0:
            is_nice = 0
            break
        if opponent_history[i] == 0 and main_history[i] == 1:
            is_nice = 1
            break
    return is_nice


def compute_forgiveness(main_history, opponent_history):
    n = len(main_history)
    opponent_defection = 0  # for each opponent's defection, there is a chance to forgive
    penalties = 0
    forgiven = 0
    holding_grudge = False
    for i in range(n):
        if main_history[i] == 1 and holding_grudge:  # main's cooperation after defection
            forgiven += 1
            holding_grudge = False
        if i < n - 1 and opponent_history[i] == 1 and holding_grudge and main_history[i + 1] == 0:
            penalties += 1
        if opponent_history[i] == 0 and not holding_grudge:  # opponent's defection
            opponent_defection += 1
            holding_grudge = True
    forgiveness = forgiven / (opponent_defection + penalties) if opponent_defection + penalties > 0 else 0
    return forgiveness


def compute_retaliation(main_history, opponent_history):
    n = len(main_history)
    reactions = 0
    provocations = 0
    for i in range(n - 1):
        if opponent_history[i] == 0:  # opponent defection
            if i == 0:  # uncalled first defection
                reactions += 1 if main_history[i + 1] == 0 else 0
                provocations += 1
            else:
                if main_history[i - 1] == 1:  # uncalled defection
                    reactions += 1 if main_history[i + 1] == 0 else 0
                    provocations += 1
    provocability = reactions / provocations if provocations > 0 else 0
    return provocability


def compute_troublemaking(main_history, opponent_history):
    n = len(main_history)
    uncalled_defection = 1 if main_history[0] == 0 else 0  # first defection was uncalled
    occasions = 1
    for i in range(n - 1):
        if opponent_history[i] == 1:  # opponent's cooperation
            occasions += 1
            uncalled_defection += 1 if main_history[i + 1] == 0 else 0  # main's uncalled defection
    troublemaking = uncalled_defection / occasions
    return troublemaking


def compute_emulation(main_history, opponent_history):
    n = len(main_history)
    emulations = 0
    for i in range(n - 1):
        emulations += 1 if main_history[i + 1] == opponent_history[i] else 0
    emulation = emulations / (n - 1) if n > 1 else 0
    return emulation


behavioral_dimensions = {
    "nice": compute_niceness,
    "forgiving": compute_forgiveness,
    "retaliatory": compute_retaliation,
    "troublemaking": compute_troublemaking,
    "emulative": compute_emulation
}


class BehavioralProfile:
    def __init__(self, strategy_name, opponent_name):
        self.strategy_name = strategy_name
        self.opponent_name = opponent_name
        self.n_games = 0
        self.dimensions = {dim: [] for dim in behavioral_dimensions.keys()}

    def compute_dimensions(self, main_history, opponent_history):
        for dimension_name, dimension_function in behavioral_dimensions.items():
            self.dimensions[dimension_name].append(dimension_function(main_history, opponent_history))
        self.n_games += 1

    def __sub__(self, other):
        sub_profile = BehavioralProfile(f"{self.strategy_name}-{other.strategy_name}", self.opponent_name)
        n_games = min(self.n_games, other.n_games)
        sub_profile.n_games = n_games
        # Subtract the values of the dimensions
        for dimension_name in self.dimensions.keys():
            sub_profile.dimensions[dimension_name] = [self.dimensions[dimension_name][idx] - other.dimensions[dimension_name][idx] for idx in
                                                      range(n_games)] if dimension_name in other.dimensions else self.dimensions[dimension_name]
        for dimension_name in other.dimensions.keys():
            if dimension_name not in self.dimensions:
                sub_profile.dimensions[dimension_name] = [-other.dimensions[dimension_name][idx] for idx in range(n_games)]
        return sub_profile

    def __abs__(self):
        abs_profile = BehavioralProfile(f"abs({self.strategy_name})", self.opponent_name)
        abs_profile.n_games = self.n_games
        for dimension_name in self.dimensions.keys():
            abs_profile.dimensions[dimension_name] = [abs(val) for val in self.dimensions[dimension_name]]
        return abs_profile
