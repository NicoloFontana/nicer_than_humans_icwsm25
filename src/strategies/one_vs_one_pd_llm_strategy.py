import json
import time
import warnings

from huggingface_hub import InferenceClient
from openai import OpenAI

from src.games.gt_game import GTGame
from src.games.two_players_pd_utils import from_nat_lang, player_1_
from src.strategies.strategy import Strategy
from src.llm_utils import generate_game_rules_prompt, generate_history_prompt, generate_prompt_from_sub_prompts, \
    HF_API_TOKEN, MODEL, MAX_NEW_TOKENS, TEMPERATURE, generate_text, history_window_size, OPENAI_API_KEY
from src.utils import find_json_object, log, out_path, timestamp


class OneVsOnePDLlmStrategy(Strategy):

    def __init__(self, game: GTGame, player_name: str, client=None, checkers=None, history_window_size=None):
        super().__init__("OneVsOnePDLlmStrategy")
        try:
            if client is None:

                # TODO 3
                ### HuggingFace API ###
                self.client = InferenceClient(model=MODEL, token=HF_API_TOKEN)
                self.client.headers["x-use-cache"] = "0"

                ### OpenAI API ###
                # self.client = OpenAI(api_key=OPENAI_API_KEY)
            else:
                self.client = client
            # Check if the client is valid
            generate_text("Test", self.client, max_new_tokens=1)
        except Exception as e:
            raise Exception(f"Error in creating InferenceClient with model {MODEL} and token {HF_API_TOKEN}") from e
        self.game = game
        if player_name not in game.get_players():
            raise ValueError(f"player_name {player_name} not in game.get_players(): {game.get_players()}")
        self.player_name = player_name
        self.opponent_name = game.get_opponents_names(player_name)[0]
        self.action_answers = []
        self.max_new_tokens = MAX_NEW_TOKENS
        self.temperature = TEMPERATURE
        self.checkers = checkers
        self.history_window_size = history_window_size

        self.action_str = "action"
        self.reason_str = "reason"

    def get_client(self):
        return self.client

    def play(self) -> int:
        is_ended = self.game.is_ended
        action_space = self.game.get_action_space()
        payoff_function = self.game.get_payoff_function()
        n_iterations = self.game.get_iterations()
        own_history = self.game.get_actions_by_player(self.player_name)
        opponent_history = self.game.get_actions_by_player(self.opponent_name)

        game_rules_prompt = generate_game_rules_prompt(action_space, payoff_function, n_iterations)
        history_prompt = generate_history_prompt(own_history, opponent_history, payoff_function, window_size=self.history_window_size, is_ended=is_ended)
        json_prompt = f'Remember to use only the following JSON format: {{"{self.action_str}": <ACTION_of_{player_1_}>}}\n'  #, "{self.reason_str}": <YOUR_REASON>}}\n'
        next_action_prompt = f"Answer saying which action player {player_1_} should play."
        prompt = generate_prompt_from_sub_prompts([game_rules_prompt, history_prompt, json_prompt, next_action_prompt])
        generated_text = generate_text(prompt, self.client, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
        action_answer = {
            "prompt": prompt,
            "generated_text": generated_text,
        }
        answer = find_json_object(generated_text)
        if answer is None:
            warnings.warn(
                f"No JSON parsable object found in generated text: {generated_text}. Returning 'Defect' action as 0.")
            action = 0
            reason = ""
        else:
            try:
                action = from_nat_lang(answer[self.action_str])
            except Exception as e:
                warnings.warn(f"{str(e)} in answer: {answer}. Returning 'Defect' action as 0.")
                action = 0
            try:
                reason = answer[self.reason_str]
            except Exception as e:
                # warnings.warn(f"{str(e)} in answer: {answer}. Reason not found.")
                reason = ""
        action_answer[self.action_str] = action
        action_answer[self.reason_str] = reason
        self.action_answers.append(action_answer)
        return int(action)

    def wrap_up_round(self, save=False, infix=None):
        self.ask_questions()
        if save:
            self.save_action_answers(infix=infix)
            self.save_checkers_results(infix=infix)

    def ask_questions(self):
        if self.checkers is not None:
            for checker in self.checkers:
                try:
                    checker.set_inference_client(self.client)
                    checker.ask_questions(self.game, self.player_name, history_window_size=self.history_window_size)
                except Exception as e:
                    warnings.warn(
                        f"Error {str(e)}. Checker {checker.get_name()} of type {type(checker)} failed to ask questions to the inference client.")

    def save_action_answers(self, infix=None):
        out_dir_path = out_path / "action_answers"
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

    def save_checkers_results(self, infix=None):
        if self.checkers is not None:
            for checker in self.checkers:
                checker.save_results(infix=infix)
                checker.save_complete_answers(infix=infix)
        # plot_checkers_results(checkers_names, timestamp, curr_round, infix=curr_round)
