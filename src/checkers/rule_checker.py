from src.checkers.checker import Checker
from src.games.two_players_pd_utils import to_nat_lang
from src.llm_utils import generate_game_rules_prompt, generate_history_prompt, generate_prompt_from_sub_prompts, player_1_, player_2_
from src.utils import find_first_int, find_first_substring


class RuleChecker(Checker):
    def __init__(self, timestamp):
        questions = [
            # f"What is the highest payoff player {player_1_} can get in a single round?",
            # f"What is the lowest payoff player {player_1_} can get in a single round?",
            # f"Which actions is player {player_1_} allowed to play?",
            f"Which is player {player_1_}'s payoff in a single round if {player_1_} plays {{}} and {player_2_} plays {{}}?",
            f"Which is player {player_2_}'s payoff in a single round if {player_2_} plays {{}} and {player_1_} plays {{}}?",
            f"Which is player {player_2_}'s payoff in a single round if {player_1_} plays {{}} and {player_2_} plays {{}}?",  # Extra question for comparison
            f"Does there exist a combination of actions that gives a player a payoff of {{}} in a single round?",
            f"Does there exist a combination of actions that gives a payoff of {{}} to {player_1_} in a single round?",  # Extra question for comparison
            f"Does there exist a combination of moves from players {player_1_} and {player_2_} that would result in player {player_1_} receiving a payoff of {{}}?",  # ChatGPT suggestion
            f"Which actions should player {player_1_} and player {player_2_} play to give a payoff of {{}} to player {player_1_}?",
            f"Which combination of actions gives a payoff of {{}} to {player_1_} in a single round?",  # Extra question for comparison
            f"What combination of moves from players {player_1_} and {player_2_} would result in player {player_1_} receiving a payoff of {{}}?"  # ChatGPT suggestion
        ]
        questions_labels = [
            # "max_payoff",
            # "min_payoff",
            # "allowed_actions",
            "round_payoff",
            "opponent_round_payoff",
            "opponent_round_payoff_inverse",  # Extra
            "exists_combo",
            f"exists_combo_for_{player_1_}",  # Extra
            f"exists_combo_resulting_{player_1_}",  # ChatGPT
            "combo_for_payoff",
            f"combo_for_{player_1_}",  # Extra
            f"combo_resulting_{player_1_}"  # ChatGPT
        ]
        self.verbose = None
        super().__init__("rule_checker", questions, questions_labels, timestamp)

    def check_payoff_bounds(self, is_max, action_space, payoff_function):
        # Question 0: "What is the highest payoff you can get in a single round?"
        # Question 1: "What is the lowest payoff you can get in a single round?"
        min_payoff = None
        max_payoff = None
        for primary_action in action_space:
            for secondary_action in action_space:
                payoff = payoff_function(primary_action, secondary_action)
                if min_payoff is None or payoff < min_payoff:
                    min_payoff = payoff
                if max_payoff is None or payoff > max_payoff:
                    max_payoff = payoff
        if is_max:
            question = self.questions[0]
            correct_answer = str(max_payoff)
            json_prompt = 'Remember to use the following JSON format: {"answer": <MAX_PAYOFF>}\n'
        else:
            question = self.questions[1]
            correct_answer = str(min_payoff)
            json_prompt = 'Remember to use the following JSON format: {"answer": <MIN_PAYOFF>}\n'
        question_prompt = f"Answer to the following question: {question}"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        llm_answer = find_first_int(self.get_answer_from_llm(prompt, question))
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question)

    def check_allowed_actions(self, action_space):
        # Question 2: "Which actions are you allowed to play?"
        question = self.questions[2]
        correct_answer = {to_nat_lang(action) for action in action_space}
        json_prompt = 'Remember to use the following JSON format: {"answer": [<LIST_OF_AVAILABLE_ACTIONS>]}\n'
        # The answer is requested as a list for simplicity when finding the JSON. It is then converted to a set.
        question_prompt = f"Answer to the following question: {question}"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        llm_answer = set(self.get_answer_from_llm(prompt, question, need_str=False))
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question)

    def check_payoff_of_combo(self, primary_action, secondary_action, payoff_function, is_inverse=False, question_idx=None):
        # Question 3: "Which is your payoff if you play {} and your opponent plays {}?"
        # Question 4: "Which is your opponent's payoff if he plays {} and you play {}?"
        # Question 5: "Which is your opponent's payoff if you play {} and he plays {}?"
        question = self.questions[question_idx]
        correct_answer = str(payoff_function(primary_action, secondary_action))
        json_prompt = 'Remember to use the following JSON format: {"answer": <PAYOFF>}\n'
        if is_inverse:
            question_prompt = f"Answer to the following question: {question.format(to_nat_lang(secondary_action), to_nat_lang(primary_action))}"
        else:
            question_prompt = f"Answer to the following question: {question.format(to_nat_lang(primary_action), to_nat_lang(secondary_action))}"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        llm_answer = find_first_int(self.get_answer_from_llm(prompt, question))
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question)

    def check_exists_combo_for_payoff(self, action_space, payoff_function, given_payoff, question_idx=None):
        # Question 5: "Does exist a combination of actions that gives you a payoff of {} in a single round?"
        question = self.questions[question_idx]
        correct_answer = "Yes" if any(
            payoff_function(primary_action, secondary_action) == given_payoff for primary_action in action_space for
            secondary_action in action_space) else "No"
        json_prompt = 'Remember to use the following JSON format: {"answer": "Yes"} or {"answer": "No"}\n'
        question_prompt = f"Answer to the following question: {question.format(given_payoff)}"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        llm_answer = find_first_substring(self.get_answer_from_llm(prompt, question), {"Yes", "No"})
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question)

    def check_combo_for_payoff(self, action_space, payoff_function, given_payoff, question_idx=None):
        # Question 6: "Which is a combination of actions that gives you a payoff of {}?"
        question = self.questions[question_idx]
        correct_combos = []
        for primary_action in action_space:
            for secondary_action in action_space:
                payoff = payoff_function(primary_action, secondary_action)
                if payoff == given_payoff:
                    correct_combos.append([to_nat_lang(primary_action), to_nat_lang(secondary_action)])
        if len(correct_combos) == 0:
            correct_answer = "None"
        else:
            correct_answer = correct_combos[0]
        json_prompt = (
            'Remember to use the following JSON format: {"answer": [<FIRST_PLAYER_ACTION>, <SECOND_PLAYER_ACTION>]}.\n')
            # 'If the required combination does not exist, answer with None\n')
        question_prompt = f"Answer to the following question: {question.format(given_payoff)}"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        if correct_answer == "None":
            llm_answer = find_first_substring(self.get_answer_from_llm(prompt, question), {"None"})
        else:
            llm_answer = self.get_answer_from_llm(prompt, question, need_str=False)
        if not isinstance(llm_answer, list):
            llm_answer = [llm_answer]
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question)

    def ask_questions(self, game, player_name="", verbose=False):
        self.verbose = verbose
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
        # # Question 0: "What is the highest payoff you can get in a single round?"
        # print(f"Question 0: {self.questions[0]}") if self.verbose else None
        # self.check_payoff_bounds(True, action_space, payoff_function)
        # # Question 1: "What is the lowest payoff you can get in a single round?"
        # print(f"Question 1: {self.questions[1]}") if self.verbose else None
        # self.check_payoff_bounds(False, action_space, payoff_function)
        # # Question 2: "Which actions are you allowed to play?"
        # print(f"Question 2: {self.questions[2]}") if self.verbose else None
        # self.check_allowed_actions(action_space)
        for primary_action in action_space:
            for secondary_action in action_space:
                # Question 3: "Which is your payoff if you play {} and your opponent plays {}?"
                question_idx = 0
                print(f"Question {question_idx}: {self.questions[question_idx].format(primary_action, secondary_action)}") if self.verbose else None
                self.check_payoff_of_combo(primary_action, secondary_action, payoff_function, question_idx=question_idx)
                # Question 4: "Which is your opponent's payoff if he plays {} and you play {}?"
                question_idx = 1
                print(f"Question {question_idx}: {self.questions[question_idx].format(primary_action, secondary_action)}") if self.verbose else None
                self.check_payoff_of_combo(primary_action, secondary_action, payoff_function, question_idx=question_idx)
                # Question 5: "Which is your opponent's payoff if you play {} and he plays {}?"
                question_idx = 2
                print(f"Question {question_idx}: {self.questions[question_idx].format(primary_action, secondary_action)}") if self.verbose else None
                self.check_payoff_of_combo(primary_action, secondary_action, payoff_function, is_inverse=True, question_idx=question_idx)
        for payoff in {0, 1, 3, 5}:
            # Question 6: "Does exist a combination of actions that gives you a payoff of {} in a single round?"
            question_idx = 3
            print(f"Question {question_idx}: {self.questions[question_idx].format(payoff)}") if self.verbose else None
            self.check_exists_combo_for_payoff(action_space, payoff_function, payoff, question_idx=question_idx)
            question_idx = 4
            print(f"Question {question_idx}: {self.questions[question_idx].format(payoff)}") if self.verbose else None
            self.check_exists_combo_for_payoff(action_space, payoff_function, payoff, question_idx=question_idx)
            question_idx = 5
            print(f"Question {question_idx}: {self.questions[question_idx].format(payoff)}") if self.verbose else None
            self.check_exists_combo_for_payoff(action_space, payoff_function, payoff, question_idx=question_idx)
            # Question 7: "Which is a combination of actions that gives you a payoff of {}?"
            question_idx = 6
            print(f"Question {question_idx}: {self.questions[question_idx].format(payoff)}") if self.verbose else None
            self.check_combo_for_payoff(action_space, payoff_function, payoff, question_idx=question_idx)
            question_idx = 7
            print(f"Question {question_idx}: {self.questions[question_idx].format(payoff)}") if self.verbose else None
            self.check_combo_for_payoff(action_space, payoff_function, payoff, question_idx=question_idx)
            question_idx = 8
            print(f"Question {question_idx}: {self.questions[question_idx].format(payoff)}") if self.verbose else None
            self.check_combo_for_payoff(action_space, payoff_function, payoff, question_idx=question_idx)
