import json
import warnings

from src.game.gt_game import GTGame
from src.game.two_players_pd_utils import from_nat_lang, player_1_
from src.model_client import ModelClient
from src.strategies.strategy import Strategy
from src.llm_utils import generate_game_rules_prompt, generate_history_prompt, generate_prompt_from_sub_prompts, MAX_NEW_TOKENS, TEMPERATURE
from src.utils import find_json_object, log


class OneVsOnePDLlmStrategy(Strategy):

    def __init__(self, game: GTGame, client: ModelClient, checkers=None, history_window_size=None):
        super().__init__("OneVsOnePDLlmStrategy")
        if client is None:
            raise ValueError("client cannot be None")
        try:
            client.generate_text("Quack", max_new_tokens=1)
        except Exception as e:
            raise Exception(f"Error {e} when creating ModelClient with model {client.model_url} and key {client.api_key}.") from e
        self.client = client
        self.game = game
        self.player_name = None
        self.action_answers = []
        self.max_new_tokens = MAX_NEW_TOKENS
        self.temperature = TEMPERATURE
        self.checkers = checkers
        self.history_window_size = history_window_size

        self.action_str = "action"
        self.reason_str = "reason"

    def get_client(self):
        return self.client

    def set_temperature(self, temperature):
        self.temperature = temperature

    def play(self) -> int:
        is_ended = self.game.is_ended
        action_space = self.game.get_action_space()
        payoff_function = self.game.get_payoff_function()
        n_iterations = self.game.get_iterations()
        own_history = self.game.get_actions_by_player(self.player_name)
        opponent_name = self.game.get_opponents_names(self.player_name)[0]
        opponent_history = self.game.get_actions_by_player(opponent_name)

        game_rules_prompt = generate_game_rules_prompt(action_space, payoff_function, n_iterations)
        history_prompt = generate_history_prompt(own_history, opponent_history, payoff_function, window_size=self.history_window_size, is_ended=is_ended)
        # TODO move all prompts parts in one place!
        json_prompt = f'Remember to use only the following JSON format: {{"{self.action_str}": <ACTION_of_{player_1_}>}}\n'  #, "{self.reason_str}": <YOUR_REASON>}}\n'
        next_action_prompt = f"Answer saying which action player {player_1_} should play."
        prompt = generate_prompt_from_sub_prompts([game_rules_prompt, history_prompt, json_prompt, next_action_prompt])
        generated_text = self.client.generate_text(prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
        # generated_text = generate_text(prompt, self.client, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
        action_answer = {
            "prompt": prompt,
            "generated_text": generated_text,
        }
        answer = find_json_object(generated_text)
        if answer is None:
            warnings.warn(
                f"No JSON parsable object found in generated text: {generated_text}. Returning 'Defect' action as 0.")
            action = 0
            # reason = "" ### REMOVE ###
        else:
            try:
                action = from_nat_lang(answer[self.action_str])
            except Exception as e:
                warnings.warn(f"{str(e)} in answer: {answer}. Returning 'Defect' action as 0.")
                action = 0
            ### REMOVE ###
            # try:
            #     reason = answer[self.reason_str]
            # except Exception as e:
            #     # warnings.warn(f"{str(e)} in answer: {answer}. Reason not found.")
            #     reason = ""
        action_answer[self.action_str] = action
        # action_answer[self.reason_str] = reason ### REMOVE ###
        self.action_answers.append(action_answer)
        return int(action)

    def wrap_up_round(self, out_dir=None, infix=None):
        self.ask_questions()
        if out_dir is not None:
            self.save_action_answers(out_dir=out_dir, infix=infix)
            self.save_checkers_results(out_dir=out_dir, infix=infix)

    def ask_questions(self):
        if self.checkers is not None:
            for checker in self.checkers:
                try:
                    checker.set_model_client(self.client)
                    checker.ask_checker_questions(self.game, self.player_name, history_window_size=self.history_window_size)
                except Exception as e:
                    warnings.warn(
                        f"Error {str(e)}. Checker {checker.get_name()} of type {type(checker)} failed to ask questions to the inference client.")

    def save_action_answers(self, out_dir, infix=None):
        out_dir_path = out_dir / "action_answers"
        out_dir_path.mkdir(parents=True, exist_ok=True)
        actions_per_round = {}
        for idx, action_answer in enumerate(self.action_answers):
            actions_per_round[idx] = action_answer
        json_action_answers = json.dumps(actions_per_round, indent=4)
        if infix is None:
            out_file_path = out_dir_path / f"{self.player_name}_action_answers.json"
        else:
            out_file_path = out_dir_path / f"{self.player_name}_action_answers_{infix}.json"
        with open(out_file_path, "w") as file:
            file.write(json_action_answers)
            log.info(f"{self.player_name} action answers saved.")

    def save_checkers_results(self, out_dir, infix=None):
        if self.checkers is not None:
            for checker in self.checkers:
                checker.save_results(out_dir, infix=infix)
                checker.save_complete_answers(out_dir, infix=infix)
