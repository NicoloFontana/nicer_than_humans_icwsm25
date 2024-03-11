from src.checkers.checker import Checker
from src.games.two_players_pd_utils import to_nat_lang
from src.llm_utils import generate_game_rules_prompt, generate_history_prompt, generate_prompt_from_sub_prompts
from src.utils import find_first_int, find_first_substring


class TimeChecker(Checker):
    def __init__(self, timestamp):
        questions = [
            "Which is the current round of the game?",
            "Which action did player A play in round {}?",
            "Which action did player B play in round {}?",
            "How many points did player A collect in round {}?",
            "How many points did player B collect in round {}?",
        ]
        questions_labels = [
            "current_round",
            "own_action",
            "opponent_action",
            "own_points",
            "opponent_points",
        ]
        self.verbose = None
        super().__init__("time_checker", questions, questions_labels, timestamp)

    def check_current_round(self, current_round, weight=1.0):
        # Question 0: "Which is the current round of the game?"
        question = self.questions[0]
        json_prompt = '\tRemember to use the following JSON format: {"answer": <CURRENT_ROUND>}\n'
        question_prompt = f"\tAnswer to the following question: {question}"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        correct_answer = str(current_round)
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        llm_answer = find_first_int(self.get_answer_from_llm(prompt, question))
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question, weight=weight)

    def check_action_played(self, is_main_player, inspected_round, action_played, action_space, weight=1.0):
        # Question 1: "Which action did you play in round {}?"
        # Question 2: "Which action did your opponent play in round {}?"
        question_idx = 1 if is_main_player else 2
        question = self.questions[question_idx]
        json_prompt = '\tRemember to use the following JSON format: {"answer": <ACTION_PLAYED>}\n'
        question_prompt = f"\tAnswer to the following question: {question.format(inspected_round)}"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        correct_answer = to_nat_lang(action_played)
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        nat_action_space = {to_nat_lang(action) for action in action_space}
        llm_answer = find_first_substring(self.get_answer_from_llm(prompt, question), nat_action_space)
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question, weight=weight)

    def check_points_collected(self, is_main_player, inspected_round, points_collected, weight=1.0):
        # Question 3: "How many points did you collect in round {}?"
        # Question 4: "How many points did your opponent collect in round {}?"
        question_idx = 3 if is_main_player else 4
        question = self.questions[question_idx]
        json_prompt = '\tRemember to use the following JSON format: {"answer": <POINTS_COLLECTED>}\n'
        question_prompt = f"\tAnswer to the following question: {question.format(inspected_round)}"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        correct_answer = str(points_collected)
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        llm_answer = find_first_int(self.get_answer_from_llm(prompt, question))
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question, weight=weight)

    def ask_questions(self, game, player_name="", verbose=False):
        self.verbose = verbose
        current_round = game.get_current_round()
        n_iterations = game.get_iterations()
        is_ended = game.is_ended
        opponent_name = ""
        for name in game.get_players():
            if name != player_name:
                opponent_name = name
                break
        action_space = game.get_action_space()
        payoff_function = game.get_payoff_function()
        game_rules_prompt = generate_game_rules_prompt(action_space, payoff_function, n_iterations)
        history_prompt = generate_history_prompt(game.get_actions_by_player(player_name),
                                                 game.get_actions_by_player(opponent_name), payoff_function, is_ended=is_ended)
        self.system_prompt = game_rules_prompt + history_prompt

        if not is_ended:
            # Question 0: "Which is the current round of the game?"
            print(f"Question 0: {self.questions[0]}") if self.verbose else None
            self.check_current_round(current_round)
        # Question 1: "Which action did you play in round {}?"
        print(f"Question 1: {self.questions[1]}") if self.verbose else None
        for i in range(1, current_round):
            self.check_action_played(True, i, game.get_actions_by_iteration(i - 1)[player_name],
                                     game.get_action_space())
        # Question 2: "Which action did your opponent play in round {}?"
        print(f"Question 2: {self.questions[2]}") if self.verbose else None
        for i in range(1, current_round):
            self.check_action_played(False, i, game.get_actions_by_iteration(i - 1)[opponent_name],
                                     game.get_action_space())
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
