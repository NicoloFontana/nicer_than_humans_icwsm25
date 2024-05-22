from src.checkers.checker import Checker
from src.games.two_players_pd_utils import to_nat_lang, player_1_, player_2_
from src.llm_utils import generate_game_rules_prompt, generate_history_prompt, generate_prompt_from_sub_prompts
from src.utils import find_first_int, find_first_substring


class TimeChecker(Checker):
    def __init__(self):
        questions = [
            "Which is the current round of the game?",
            f"Which action did player {player_1_} play in round {{}}?",
            f"Which action did player {player_2_} play in round {{}}?",
            f"How many points did player {player_1_} collect in round {{}}?",
            f"How many points did player {player_2_} collect in round {{}}?",
        ]
        questions_labels = [
            "current_round",
            f"action_{player_1_}",
            f"action_{player_2_}",
            f"points_{player_1_}",
            f"points_{player_2_}",
        ]
        super().__init__("time_checker", questions, questions_labels)

    def check_current_round(self, current_round, question_idx, weight=1.0):
        # Question 0: "Which is the current round of the game?"
        question = self.questions[question_idx]
        label = self.questions_labels[question_idx]
        json_prompt = 'Remember to use the following JSON format: {"answer": <CURRENT_ROUND>}\n'
        question_prompt = f"Answer to the following question: {question}\n"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        correct_answer = str(current_round)
        llm_answer = find_first_int(self.get_answer_from_llm(prompt, label))
        self.check_answer(llm_answer, correct_answer, label, weight=weight)

    def check_action_played(self, inspected_round, action_played, action_space, question_idx, weight=1.0):
        # Question 1: "Which action did you play in round {}?"
        # Question 2: "Which action did your opponent play in round {}?"
        question = self.questions[question_idx]
        label = self.questions_labels[question_idx]
        json_prompt = 'Remember to use the following JSON format: {"answer": <ACTION_PLAYED>}\n'
        question_prompt = f"Answer to the following question: {question.format(inspected_round)}\n"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        correct_answer = to_nat_lang(action_played, string_of_string=False)
        nat_action_space = {to_nat_lang(action, string_of_string=False) for action in action_space}
        llm_answer = find_first_substring(self.get_answer_from_llm(prompt, label), nat_action_space)
        self.check_answer(llm_answer, correct_answer, label, weight=weight)

    def check_points_collected(self, inspected_round, points_collected, question_idx, weight=1.0):
        # Question 3: "How many points did you collect in round {}?"
        # Question 4: "How many points did your opponent collect in round {}?"
        question = self.questions[question_idx]
        label = self.questions_labels[question_idx]
        json_prompt = 'Remember to use the following JSON format: {"answer": <POINTS_COLLECTED>}\n'
        question_prompt = f"Answer to the following question: {question.format(inspected_round)}\n"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        correct_answer = str(points_collected)
        llm_answer = find_first_int(self.get_answer_from_llm(prompt, label))
        self.check_answer(llm_answer, correct_answer, label, weight=weight)

    def ask_questions(self, game, player_name="", history_window_size=None):
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
                                                 game.get_actions_by_player(opponent_name), payoff_function, window_size=history_window_size, is_ended=is_ended)
        self.system_prompt = game_rules_prompt + history_prompt

        if not is_ended:
            question_idx = 0
            self.check_current_round(current_round, question_idx=question_idx)
        question_idx = 1
        if current_round % 10 == 0:
            for i in range(1, current_round):
                self.check_action_played(i, game.get_actions_by_iteration(i - 1)[player_name], game.get_action_space(), question_idx=question_idx)
            question_idx = 2
            for i in range(1, current_round):
                self.check_action_played(i, game.get_actions_by_iteration(i - 1)[opponent_name], game.get_action_space(), question_idx=question_idx)
            question_idx = 3
            for i in range(1, current_round):
                self.check_points_collected(i, game.get_payoff_function()(game.get_actions_by_iteration(i - 1)[player_name], game.get_actions_by_iteration(i - 1)[opponent_name]),
                                            question_idx=question_idx)
            question_idx = 4
            for i in range(1, current_round):
                self.check_points_collected(i, game.get_payoff_function()(game.get_actions_by_iteration(i - 1)[opponent_name], game.get_actions_by_iteration(i - 1)[player_name]),
                                        question_idx=question_idx)
