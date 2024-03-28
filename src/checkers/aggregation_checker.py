from src.checkers.checker import Checker
from src.games.two_players_pd_utils import to_nat_lang, player_1_, player_2_
from src.llm_utils import generate_game_rules_prompt, generate_history_prompt, generate_prompt_from_sub_prompts
from src.utils import find_first_int


class AggregationChecker(Checker):
    def __init__(self):
        questions = [
            f"How many times did player {player_1_} choose {{}}?",
            f"How many times did player {player_2_} choose {{}}?",
            f"What is player {player_1_}'s current total payoff?",
            f"What is player {player_2_}'s current total payoff?",
        ]
        questions_labels = [
            f"#actions_{player_1_}",
            f"#actions_{player_2_}",
            f"total_payoff_{player_1_}",
            f"total_payoff_{player_2_}",
        ]
        self.verbose = None
        super().__init__("aggregation_checker", questions, questions_labels)

    def check_action_chosen(self, action, n_times, question_idx, weight=1.0):
        # Question 0: "How many times did you choose {}?"
        # Question 1: "How many times did your opponent choose {}?"
        question = self.questions[question_idx]
        label = self.questions_labels[question_idx]
        json_prompt = 'Remember to use the following JSON format: {"answer": <N_TIMES>}\n'
        question_prompt = f"Answer to the following question: {question.format(to_nat_lang(action))}"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        correct_answer = str(n_times)
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        llm_answer = find_first_int(self.get_answer_from_llm(prompt, label))
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, label, weight=weight)

    def check_total_payoff(self, payoff, question_idx, weight=1.0):
        # Question 2: "What is your current total payoff?"
        # Question 3: "What is your opponent's current total payoff?"
        question = self.questions[question_idx]
        label = self.questions_labels[question_idx]
        json_prompt = 'Remember to use the following JSON format: {"answer": <TOTAL_PAYOFF>}\n'
        question_prompt = f"Answer to the following question: {question}"
        prompt = generate_prompt_from_sub_prompts([self.system_prompt, json_prompt, question_prompt])
        correct_answer = str(payoff)
        print(f"Correct: {correct_answer}", end=" ") if self.verbose else None
        llm_answer = find_first_int(self.get_answer_from_llm(prompt, label))
        print(f"LLM: {llm_answer}") if self.verbose else None
        self.check_answer(llm_answer, correct_answer, label, weight=weight)

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
        own_history = game.get_actions_by_player(player_name)
        opponent_history = game.get_actions_by_player(game.get_opponent_name(player_name))
        own_payoff = game.get_total_payoff_by_player(player_name)
        opponent_payoff = game.get_total_payoff_by_player(game.get_opponent_name(player_name))
        for action in action_space:
            question_idx = 0
            print(f"Question {question_idx}: {self.questions[question_idx]}") if self.verbose else None
            n_times = own_history.count(action)
            self.check_action_chosen(action, n_times, question_idx=question_idx)
            question_idx = 1
            print(f"Question {question_idx}: {self.questions[question_idx]}") if self.verbose else None
            n_times = opponent_history.count(action)
            self.check_action_chosen(action, n_times, question_idx=question_idx)
        question_idx = 2
        print(f"Question {question_idx}: {self.questions[question_idx]}") if self.verbose else None
        self.check_total_payoff(own_payoff, question_idx=question_idx)
        question_idx = 3
        print(f"Question {question_idx}: {self.questions[question_idx]}") if self.verbose else None
        self.check_total_payoff(opponent_payoff, question_idx=question_idx)
