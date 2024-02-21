from src.checkers.checker import Checker
from src.strategies.strategy_utils import generate_game_rules_prompt
from src.utils import find_first_int


class AggregationChecker(Checker):
    def __init__(self, timestamp):
        questions = [
            "How many times did you choose {}?",
            "How many times did your opponent choose {}?",
            "What is your current total payoff?",
            "What is your opponent's current total payoff?",
        ]
        self.verbose = None
        super().__init__("aggregation_checker", questions, timestamp)

    def check_action_chosen(self, is_main_player, action, n_times):
        # Question 0: "How many times did you choose {}?"
        # Question 1: "How many times did your opponent choose {}?"
        question_idx = 0 if is_main_player else 1
        question = self.questions[question_idx]
        json_prompt = '\tRemember to use the following JSON format: {"answer": <N_TIMES>}<<SYS>>\n'
        question_prompt = f"Answer to the following question: {question.format(action)}"
        prompt = self.start_prompt + self.system_prompt + json_prompt + question_prompt + self.end_prompt
        correct_answer = str(n_times)
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        llm_answer = find_first_int(self.get_answer_from_llm(prompt, question))
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question)

    def check_total_payoff(self, is_main_player, payoff):
        # Question 2: "What is your current total payoff?"
        # Question 3: "What is your opponent's current total payoff?"
        question_idx = 2 if is_main_player else 3
        question = self.questions[question_idx]
        json_prompt = '\tRemember to use the following JSON format: {"answer": <TOTAL_PAYOFF>}<<SYS>>\n'
        question_prompt = f"Answer to the following question: {question}"
        prompt = self.start_prompt + self.system_prompt + json_prompt + question_prompt + self.end_prompt
        correct_answer = str(payoff)
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        llm_answer = find_first_int(self.get_answer_from_llm(prompt, question))
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question)

    def ask_questions(self, game, player_name="", verbose=False):
        self.verbose = verbose
        action_space = game.get_action_space()
        payoff_function = game.get_payoff_function()
        game_rules_prompt = generate_game_rules_prompt(action_space, payoff_function, game.get_iterations())
        self.system_prompt = game_rules_prompt
        own_history = game.get_actions_by_player(player_name)
        opponent_history = game.get_actions_by_player(game.get_opponent_name(player_name))
        own_payoff = game.get_total_payoff_by_player(player_name)
        opponent_payoff = game.get_total_payoff_by_player(game.get_opponent_name(player_name))
        for action in action_space:
            # Question 0: "How many times did you choose {}?"
            print(f"Question 0: {self.questions[0]}") if self.verbose else None
            n_times = own_history.count(action)
            self.check_action_chosen(True, action, n_times)
            # Question 1: "How many times did your opponent choose {}?"
            print(f"Question 1: {self.questions[1]}") if self.verbose else None
            n_times = opponent_history.count(action)
            self.check_action_chosen(False, action, n_times)
        # Question 2: "What is your current total payoff?"
        print(f"Question 2: {self.questions[2]}") if self.verbose else None
        self.check_total_payoff(True, own_payoff)
        # Question 3: "What is your opponent's current total payoff?"
        print(f"Question 3: {self.questions[3]}") if self.verbose else None
        self.check_total_payoff(False, opponent_payoff)

        if self.verbose:
            print("\n\n")
            for key in self.get_accuracy():
                print(f"{key}: {self.get_accuracy()[key]}")
