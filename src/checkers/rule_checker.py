from src.checkers.checker import Checker
from src.games.two_players_pd_utils import to_nat_lang
from src.llm_utils import generate_game_rules_prompt, generate_history_prompt, generate_prompt_from_sub_prompts
from src.utils import find_first_int, find_first_substring


class RuleChecker(Checker):
    def __init__(self, timestamp):
        questions = [
            "What is the highest payoff player A can get in a single round?",
            "What is the lowest payoff player A can get in a single round?",
            "Which actions is player A allowed to play?",
            "Which is player A's payoff in a single round if A plays {} and B plays {}?",
            "Which is player B's payoff in a single round if B plays {} and A plays {}?",
            "Which is player B's payoff in a single round if A plays {} and B plays {}?",  # Extra question for comparison
            "How many point does player A collect in a single round in A plays {} and B plays {}?",  # Extra question for comparison
            "How many point does player B collect in a single round in B plays {} and A plays {}?",  # Extra question for comparison
            "How many point does player B collect in a single round in A plays {} and B plays {}?",  # Extra question for comparison
            "Does exists a combination of actions that gives a player a payoff of {} in a single round?",
            "Can player A collect {} points in a single round?",  # Extra question for comparison
            "Can player B collect {} points in a single round?",  # Extra question for comparison
            "Which actions should player A and player B play to give a payoff of {} to player A?",
            "Which actions should player A and player B play to make player A collect {} points?",  # Extra question for comparison
        ]
        questions_labels = [
            "max_payoff",
            "min_payoff",
            "allowed_actions",
            "round_payoff",
            "opponent_round_payoff",
            "opponent_round_payoff_inverse",  # Extra
            "A_round_points",  # Extra
            "B_round_points",  # Extra
            "B_round_points_inverse",  # Extra
            "exists_combo",
            "can_A_collect",  # Extra
            "can_B_collect",  # Extra
            "combo_for_payoff",
            "combo_for_points",  # Extra
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
            'Remember to use the following JSON format: {"answer": [<FIRST_PLAYER_ACTION>, <SECOND_PLAYER_ACTION>]}.\n'
            'If the required combination does not exist, answer with None\n')
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
                print(f"Question 3: {self.questions[3].format(primary_action, secondary_action)}") if self.verbose else None
                self.check_payoff_of_combo(primary_action, secondary_action, payoff_function, question_idx=3)
                # Question 4: "Which is your opponent's payoff if he plays {} and you play {}?"
                print(f"Question 4: {self.questions[4].format(primary_action, secondary_action)}") if self.verbose else None
                self.check_payoff_of_combo(primary_action, secondary_action, payoff_function, question_idx=4)
                # Question 5: "Which is your opponent's payoff if you play {} and he plays {}?"
                print(f"Question 5: {self.questions[5].format(primary_action, secondary_action)}") if self.verbose else None
                self.check_payoff_of_combo(primary_action, secondary_action, payoff_function, is_inverse=True, question_idx=5)
                # Questions 6, 7, 8
                print(f"Question 6: {self.questions[6].format(primary_action, secondary_action)}") if self.verbose else None
                self.check_payoff_of_combo(primary_action, secondary_action, payoff_function, question_idx=6)
                print(f"Question 7: {self.questions[7].format(primary_action, secondary_action)}") if self.verbose else None
                self.check_payoff_of_combo(primary_action, secondary_action, payoff_function, question_idx=7)
                print(f"Question 8: {self.questions[8].format(primary_action, secondary_action)}") if self.verbose else None
                self.check_payoff_of_combo(primary_action, secondary_action, payoff_function, is_inverse=True, question_idx=8)
        # for payoff in range(0, 6):
        #     # Question 6: "Does exist a combination of actions that gives you a payoff of {} in a single round?"
        #     print(f"Question 9: {self.questions[9].format(payoff)}") if self.verbose else None
        #     self.check_exists_combo_for_payoff(action_space, payoff_function, payoff, question_idx=9)
        #     print(f"Question 10: {self.questions[10].format(payoff)}") if self.verbose else None
        #     self.check_exists_combo_for_payoff(action_space, payoff_function, payoff, question_idx=10)
        #     print(f"Question 11: {self.questions[11].format(payoff)}") if self.verbose else None
        #     self.check_exists_combo_for_payoff(action_space, payoff_function, payoff, question_idx=11)
        #     # Question 7: "Which is a combination of actions that gives you a payoff of {}?"
        #     print(f"Question 12: {self.questions[12].format(payoff)}") if self.verbose else None
        #     self.check_combo_for_payoff(action_space, payoff_function, payoff, question_idx=12)
        #     print(f"Question 13: {self.questions[13].format(payoff)}") if self.verbose else None
        #     self.check_combo_for_payoff(action_space, payoff_function, payoff, question_idx=13)
