from src.checkers.checker import Checker
from src.games.two_players_pd_utils import to_nat_lang
from src.llm_utils import generate_game_rules_prompt, generate_history_prompt, generate_prompt_from_sub_prompts
from src.utils import find_first_int, find_first_substring


class RuleChecker(Checker):
    def __init__(self, timestamp):
        questions = [
            "What is the highest payoff you can get in a single round?",
            "What is the lowest payoff you can get in a single round?",
            "Which actions are you allowed to play?",
            "Which is your payoff in a single round if you play {} and your opponent plays {}?",
            "Which is your opponent's payoff in a single round if he plays {} and you play {}?",
            "Does exists a combination of actions that gives you a payoff of {} in a single round?",
            "Which is a combination of actions that gives you a payoff of {} in a single round?",
        ]
        questions_labels = [
            "max_payoff",
            "min_payoff",
            "allowed_actions",
            "round_payoff",
            "opponent_round_payoff",
            "exists_combo",
            "combo_for_payoff",
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
            json_prompt = '\tRemember to use the following JSON format: {"answer": <MAX_PAYOFF>}<<SYS>>\n'
        else:
            question = self.questions[1]
            correct_answer = str(min_payoff)
            json_prompt = '\tRemember to use the following JSON format: {"answer": <MIN_PAYOFF>}<<SYS>>\n'
        question_prompt = f"\tAnswer to the following question: {question}"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        llm_answer = find_first_int(self.get_answer_from_llm(prompt, question))
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question)

    def check_allowed_actions(self, action_space):
        # Question 2: "Which actions are you allowed to play?"
        question = self.questions[2]
        correct_answer = {to_nat_lang(action) for action in action_space}
        json_prompt = '\tRemember to use the following JSON format: {"answer": [<LIST_OF_AVAILABLE_ACTIONS>]}<<SYS>>\n'
        # The answer is requested as a list for simplicity when finding the JSON. It is then converted to a set.
        question_prompt = f"\tAnswer to the following question: {question}"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        llm_answer = set(self.get_answer_from_llm(prompt, question, need_str=False))
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question)

    def check_payoff_of_combo(self, is_main_player, primary_action, secondary_action, payoff_function):
        # Question 3: "Which is your payoff if you play {} and your opponent plays {}?"
        # Question 4: "Which is your opponent's payoff if he plays {} and you play {}?"
        question_idx = 3 if is_main_player else 4
        question = self.questions[question_idx]
        correct_answer = str(payoff_function(primary_action, secondary_action))
        json_prompt = '\tRemember to use the following JSON format: {"answer": <PAYOFF>}<<SYS>>\n'
        question_prompt = f"\tAnswer to the following question: {question.format(to_nat_lang(primary_action), to_nat_lang(secondary_action))}"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        llm_answer = find_first_int(self.get_answer_from_llm(prompt, question))
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question)

    def check_exists_combo_for_payoff(self, action_space, payoff_function, given_payoff):
        # Question 5: "Does exist a combination of actions that gives you a payoff of {} in a single round?"
        question = self.questions[5]
        correct_answer = "Yes" if any(
            payoff_function(primary_action, secondary_action) == given_payoff for primary_action in action_space for
            secondary_action in action_space) else "No"
        json_prompt = '\tRemember to use the following JSON format: {"answer": "Yes"} or {"answer": "No"}<<SYS>>\n'
        question_prompt = f"\tAnswer to the following question: {question.format(given_payoff)}"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        llm_answer = find_first_substring(self.get_answer_from_llm(prompt, question), {"Yes", "No"})
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, question)

    def check_combo_for_payoff(self, action_space, payoff_function, given_payoff):
        # Question 6: "Which is a combination of actions that gives you a payoff of {}?"
        question = self.questions[6]
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
            '\tRemember to use the following JSON format: {"answer": [<FIRST_PLAYER_ACTION>, <SECOND_PLAYER_ACTION>]}. '
            '\nIf no combination of actions can give you a payoff of the required payoff, answer with {"answer": "None"}<<SYS>>\n')
        question_prompt = f"\tAnswer to the following question: {question.format(given_payoff)}"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
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
        # Question 0: "What is the highest payoff you can get in a single round?"
        print(f"Question 0: {self.questions[0]}") if self.verbose else None
        self.check_payoff_bounds(True, action_space, payoff_function)
        # Question 1: "What is the lowest payoff you can get in a single round?"
        print(f"Question 1: {self.questions[1]}") if self.verbose else None
        self.check_payoff_bounds(False, action_space, payoff_function)
        # Question 2: "Which actions are you allowed to play?"
        print(f"Question 2: {self.questions[2]}") if self.verbose else None
        self.check_allowed_actions(action_space)
        for primary_action in action_space:
            for secondary_action in action_space:
                pass
                # Question 3: "Which is your payoff if you play {} and your opponent plays {}?"
                print(
                    f"Question 3: {self.questions[3].format(primary_action, secondary_action)}") if self.verbose else None
                print(
                    f"Question 4: {self.questions[4].format(primary_action, secondary_action)}") if self.verbose else None
                self.check_payoff_of_combo(True, primary_action, secondary_action, payoff_function)
                # Question 4: "Which is your opponent's payoff if he plays {} and you play {}?"
                self.check_payoff_of_combo(False, primary_action, secondary_action, payoff_function)
        for payoff in range(0, 6):
            # Question 5: "Does exist a combination of actions that gives you a payoff of {} in a single round?"
            print(f"Question 5: {self.questions[5].format(payoff)}") if self.verbose else None
            self.check_exists_combo_for_payoff(action_space, payoff_function, payoff)
            # Question 6: "Which is a combination of actions that gives you a payoff of {}?"
            print(f"Question 6: {self.questions[6].format(payoff)}") if self.verbose else None
            self.check_combo_for_payoff(action_space, payoff_function, payoff)
