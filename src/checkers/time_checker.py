from src.checkers.checker import Checker
from src.games.two_players_pd_utils import to_nat_lang
from src.strategies.strategy_utils import generate_game_rules_prompt, generate_history_prompt
from src.utils import find_first_int, find_first_substring


class TimeChecker(Checker):
    def __init__(self, timestamp):
        questions = [
            "Which is the current round of the game?",
            "Which action did you play in round {}?",
            "Which action did your opponent play in round {}?",
            "How many points did you collect in round {}?",
            "How many points did your opponent collect in round {}?",
        ]
        self.verbose = None
        super().__init__("temporal_checker", questions, timestamp)

    def check_current_round(self, current_round):
        # Question 0: "Which is the current round of the game?"
        question = self.questions[0]
        json_prompt = '\tRemember to use the following JSON format: {"answer": <CURRENT_ROUND>}<<SYS>>\n'
        question_prompt = f"Answer to the following question: {question}"
        prompt = self.start_prompt + self.system_prompt + json_prompt + question_prompt + self.end_prompt
        correct_answer = str(current_round)
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        llm_answer = find_first_int(self.get_answer_from_llm(prompt, question))
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question)

    def check_action_played(self, is_main_player, inspected_round, action_played, action_space):
        # Question 1: "Which action did you play in round {}?"
        # Question 2: "Which action did your opponent play in round {}?"
        question_idx = 1 if is_main_player else 2
        question = self.questions[question_idx]
        json_prompt = '\tRemember to use the following JSON format: {"answer": <ACTION_PLAYED>}<<SYS>>\n'
        question_prompt = f"Answer to the following question: {question.format(inspected_round)}"
        prompt = self.start_prompt + self.system_prompt + json_prompt + question_prompt + self.end_prompt
        correct_answer = to_nat_lang(action_played)
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        nat_action_space = {to_nat_lang(action) for action in action_space}
        llm_answer = find_first_substring(self.get_answer_from_llm(prompt, question), nat_action_space)
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question)

    def check_points_collected(self, is_main_player, inspected_round, points_collected):
        # Question 3: "How many points did you collect in round {}?"
        # Question 4: "How many points did your opponent collect in round {}?"
        question_idx = 3 if is_main_player else 4
        question = self.questions[question_idx]
        json_prompt = '\tRemember to use the following JSON format: {"answer": <POINTS_COLLECTED>}<<SYS>>\n'
        question_prompt = f"Answer to the following question: {question.format(inspected_round)}"
        prompt = self.start_prompt + self.system_prompt + json_prompt + question_prompt + self.end_prompt
        correct_answer = str(points_collected)
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        llm_answer = find_first_int(self.get_answer_from_llm(prompt, question))
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question)

    def ask_questions(self, game, player_name="", verbose=False):
        self.verbose = verbose
        current_round = game.get_current_round()
        opponent_name = ""
        for name in game.get_players():
            if name != player_name:
                opponent_name = name
                break
        game_rules_prompt = generate_game_rules_prompt(game.get_action_space(), game.get_payoff_function(),
                                                       game.get_iterations())
        history_prompt = generate_history_prompt(game.get_actions_by_player(player_name),
                                                 game.get_actions_by_player(opponent_name))
        self.system_prompt = game_rules_prompt + history_prompt

        # Question 0: "Which is the current round of the game?"
        print(f"Question 0: {self.questions[0]}") if self.verbose else None
        self.check_current_round(current_round)
        # Question 1: "Which action did you play in round {}?"
        print(f"Question 1: {self.questions[1]}") if self.verbose else None
        for i in range(1, current_round):
            self.check_action_played(True, i, game.get_actions_by_iteration(i - 1)[player_name], game.get_action_space())
        # Question 2: "Which action did your opponent play in round {}?"
        print(f"Question 2: {self.questions[2]}") if self.verbose else None
        for i in range(1, current_round):
            self.check_action_played(False, i, game.get_actions_by_iteration(i - 1)[opponent_name], game.get_action_space())
        # Question 3: "How many points did you collect in round {}?"
        print(f"Question 3: {self.questions[3]}") if self.verbose else None
        for i in range(1, current_round):
            self.check_points_collected(True, i,
                                        game.get_payoff_function()(game.get_actions_by_iteration(i - 1)[player_name],
                                                                   game.get_actions_by_iteration(i - 1)[opponent_name]))
        # Question 4: "How many points did your opponent collect in round {}?"
        print(f"Question 4: {self.questions[4]}") if self.verbose else None
        for i in range(1, current_round):
            self.check_points_collected(False, i,
                                        game.get_payoff_function()(game.get_actions_by_iteration(i - 1)[opponent_name],
                                                                   game.get_actions_by_iteration(i - 1)[player_name]))

        if self.verbose:
            print("\n\n")
            for key in self.get_accuracy():
                print(f"{key}: {self.get_accuracy()[key]}")
