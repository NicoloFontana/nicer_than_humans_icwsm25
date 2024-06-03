from src.behavioral_analysis.behavioral_dimension import BehavioralDimension


class Niceness(BehavioralDimension):
    def __init__(self):
        super().__init__("niceness")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_dimension(self, main_history: list, opponent_history: list) -> float:
        n = len(main_history)
        is_nice = 1
        for i in range(n):
            if main_history[i] == 0:
                is_nice = 0
                break
            if opponent_history[i] == 0 and main_history[i] == 1:
                is_nice = 1
                break
        self.values.append(is_nice)
        self.update_aggregates()
        return is_nice


class Forgiveness(BehavioralDimension):
    def __init__(self):
        super().__init__("forgiveness")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_dimension(self, main_history: list, opponent_history: list) -> float:
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
        self.values.append(forgiveness)
        self.update_aggregates()
        return forgiveness


class Retaliation(BehavioralDimension):
    def __init__(self):
        super().__init__("provocability")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_dimension(self, main_history: list, opponent_history: list) -> float:
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
        self.values.append(provocability)
        self.update_aggregates()
        return provocability


class Emulation(BehavioralDimension):
    def __init__(self):
        super().__init__("emulation")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_dimension(self, main_history: list, opponent_history: list) -> float:
        n = len(main_history)
        emulations = 0
        for i in range(n - 1):
            emulations += 1 if main_history[i + 1] == opponent_history[i] else 0
        emulation = emulations / (n - 1) if n > 1 else 0
        self.values.append(emulation)
        self.update_aggregates()
        return emulation


class Troublemaking(BehavioralDimension):
    def __init__(self):
        super().__init__("troublemaking")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_dimension(self, main_history: list, opponent_history: list) -> float:
        n = len(main_history)
        uncalled_defection = 1 if main_history[0] == 0 else 0  # first defection was uncalled
        occasions = 1
        for i in range(n - 1):
            if opponent_history[i] == 1:  # opponent's cooperation
                occasions += 1
                uncalled_defection += 1 if main_history[i + 1] == 0 else 0  # main's uncalled defection
        troublemaking = uncalled_defection / occasions
        self.values.append(troublemaking)
        self.update_aggregates()
        return troublemaking


main_behavioral_dimensions = {
    "niceness": Niceness,
    "forgiveness": Forgiveness,
    "retaliation": Retaliation,
    "troublemaking": Troublemaking,
    "emulation": Emulation,
}


def dimensions_names_to_adjectives(dimensions_names):
    if isinstance(dimensions_names, str):
        dimensions_names = [dimensions_names]
    adjectives = []
    for dimension_name in dimensions_names:
        if dimension_name == "niceness":
            adjectives.append("nice")
        elif dimension_name == "forgiveness":
            adjectives.append("forgiving")
        elif dimension_name == "retaliation":
            adjectives.append("retaliatory")
        elif dimension_name == "troublemaking":
            adjectives.append("troublemaking")
        elif dimension_name == "emulation":
            adjectives.append("emulative")
    return adjectives
