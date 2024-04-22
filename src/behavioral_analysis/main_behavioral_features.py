from src.behavioral_analysis.behavioral_feature import BehavioralFeature


class Niceness(BehavioralFeature):
    def __init__(self):
        super().__init__("niceness")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_feature(self, main_history: list, opponent_history: list) -> float:
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


class Forgiveness1(BehavioralFeature):
    def __init__(self):
        super().__init__("forgiveness1")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_feature(self, main_history: list, opponent_history: list) -> float:
        n = len(main_history)
        total_unforgiveness = 0
        opponent_defection = 0  # for each opponent's defection, there is a chance to forgive
        for i in range(n - 1):
            if opponent_history[i] == 0:  # opponent's defection
                opponent_defection += 1
                start = i + 1
                forgiving_round = -1
                for j in range(start, n):
                    if main_history[j] == 1:  # main cooperates after opponent's defection
                        forgiving_round = j
                        break
                if forgiving_round == -1:
                    forgiving_round = n
                total_unforgiveness += (forgiving_round - start) / (n - start) if n > start else 0  # ratio between waited rounds to cooperate and remaining rounds
                # 0 if immediate cooperation, 1 if no cooperation
        relative_unforgiveness = total_unforgiveness / opponent_defection if opponent_defection > 0 else 0  # ratio between total unforgiveness and occasions to forgive
        # 0 if always forgives immediately, 1 if never forgives
        return 1 - relative_unforgiveness


class Forgiveness2(BehavioralFeature):
    def __init__(self):
        super().__init__("forgiveness2")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_feature(self, main_history: list, opponent_history: list) -> float:
        n = len(main_history)
        waited = 0  # how many rounds the main player waited to forgive in total
        remaining = 0  # how many rounds the main player had to forgive in total
        for i in range(n - 1):
            if opponent_history[i] == 0:  # opponent's defection
                start = i + 1
                forgiving_round = -1
                for j in range(start, n):
                    if main_history[j] == 1:  # main cooperates after opponent's defection
                        forgiving_round = j
                        break
                if forgiving_round == -1:
                    forgiving_round = n
                waited += forgiving_round - start
                remaining += n - start
        unforgiveness = waited / remaining if remaining > 0 else 0  # ratio between waited rounds to forgive and remaining rounds
        # 0 if always forgives immediately, 1 if never forgives
        return 1 - unforgiveness


class Provocability(BehavioralFeature):
    def __init__(self):
        super().__init__("provocability")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_feature(self, main_history: list, opponent_history: list) -> float:
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
        return reactions / provocations if provocations > 0 else 0


class Cooperativeness(BehavioralFeature):
    def __init__(self):
        super().__init__("cooperativeness")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_feature(self, main_history: list, opponent_history: list) -> float:
        n = len(main_history)
        cooperation = sum(main_history)
        return cooperation / n if n > 0 else 0



class Emulation(BehavioralFeature):
    def __init__(self):
        super().__init__("emulation")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_feature(self, main_history: list, opponent_history: list) -> float:
        n = len(main_history)
        emulations = 0
        for i in range(n - 1):
            emulations += 1 if main_history[i + 1] == opponent_history[i] else 0
        return emulations / (n - 1) if n > 1 else 0


class Troublemaking(BehavioralFeature):
    def __init__(self):
        super().__init__("troublemaking")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_feature(self, main_history: list, opponent_history: list) -> float:
        n = len(main_history)
        uncalled_defection = 1 if main_history[0] == 0 else 0  # first defection was uncalled
        occasions = 1
        for i in range(n - 1):
            if opponent_history[i] == 1:  # opponent's cooperation
                occasions += 1
                uncalled_defection += 1 if main_history[i + 1] == 0 else 0  # main's uncalled defection
        return uncalled_defection / occasions


class Naivety(BehavioralFeature):
    def __init__(self):
        super().__init__("naivety")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_feature(self, main_history: list, opponent_history: list) -> float:
        n = len(main_history)
        uncalled_cooperation = 1 if main_history[0] == 1 else 0  # first cooperation was uncalled
        occasions = 1
        for i in range(n - 1):
            if opponent_history[i] == 0:  # opponent's defection
                occasions += 1
                uncalled_cooperation += 1 if main_history[i + 1] == 1 else 0  # main's uncalled cooperation
        return uncalled_cooperation / occasions


class Consistency(BehavioralFeature):
    def __init__(self):
        super().__init__("consistency")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_feature(self, main_history: list, opponent_history: list) -> float:
        n = len(main_history)
        changes = 0
        for i in range(n - 1):
            changes += 1 if main_history[i] != main_history[i + 1] else 0
        return 1 - (changes / (n - 1)) if n > 1 else 0


main_behavioral_features = {
    "niceness": Niceness,
    "forgiveness1": Forgiveness1,
    "forgiveness2": Forgiveness2,
    "provocability": Provocability,
    # "provocability2": provocability2,
    "cooperativeness": Cooperativeness,
    "emulation": Emulation,
    "troublemaking": Troublemaking,
    "naivety": Naivety,
    "consistency": Consistency
}
