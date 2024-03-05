import time
import warnings

from huggingface_hub import InferenceClient

from src.games.gt_game import GTGame
from src.games.two_players_pd_utils import from_nat_lang
from src.strategies.strategy import Strategy
from src.llm_utils import generate_game_rules_prompt, generate_history_prompt, generate_prompt_from_sub_prompts, \
    HF_API_TOKEN, MODEL, MAX_NEW_TOKENS, TEMPERATURE, generate_text
from src.utils import find_json_object


class OneVsOnePDLlmStrategy(Strategy):

    def __init__(self, game: GTGame, player_name: str, opponent_name: str, client=None):
        super().__init__("OneVsOnePDLlmStrategy")
        try:
            if client is None:
                self.client = InferenceClient(model=MODEL, token=HF_API_TOKEN)
                self.client.headers["x-use-cache"] = "0"
            else:
                self.client = client
            # Check if the client is valid
            generate_text("Test", self.client, max_new_tokens=1)
        except Exception as e:
            raise Exception(f"Error in creating InferenceClient with model {MODEL} and token {HF_API_TOKEN}") from e
        if not isinstance(game, GTGame):
            raise TypeError(f"game must be of type GTGame, not {type(game)}")
        self.game = game
        if not isinstance(player_name, str):
            warnings.warn(f"player_name must be of type str, not {type(player_name)}. Converting to str.")
            player_name = str(player_name)
        if player_name not in game.get_players():
            raise ValueError(f"player_name {player_name} not in game.get_players(): {game.get_players()}")
        self.player_name = player_name
        if not isinstance(opponent_name, str):
            warnings.warn(f"opponent_name must be of type str, not {type(opponent_name)}. Converting to str.")
            opponent_name = str(opponent_name)
        if opponent_name not in game.get_players():
            raise ValueError(f"opponent_name {opponent_name} not in game.get_players(): {game.get_players()}")
        self.opponent_name = opponent_name
        self.max_new_tokens = MAX_NEW_TOKENS
        self.temperature = TEMPERATURE

    def get_client(self):
        return self.client

    def play(self, verbose=False) -> int:
        is_ended = self.game.is_ended()
        action_space = self.game.get_action_space()
        payoff_function = self.game.get_payoff_function()
        n_iterations = self.game.get_iterations()
        own_history = self.game.get_actions_by_player(self.player_name)
        opponent_history = self.game.get_actions_by_player(self.opponent_name)

        game_rules_prompt = generate_game_rules_prompt(action_space, payoff_function, n_iterations)
        history_prompt = generate_history_prompt(own_history, opponent_history, payoff_function, is_ended=is_ended)
        json_prompt = '\tRemember to use only the following JSON format: {"action": <YOUR_ACTION>, "reason": <YOUR_REASON>}<<SYS>>\n'
        next_action_prompt = f"\tAnswer saying which action you choose to play."
        prompt = generate_prompt_from_sub_prompts([game_rules_prompt, history_prompt, json_prompt, next_action_prompt])
        generated_text = generate_text(prompt, self.client, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
        answer = find_json_object(generated_text)
        if answer is None:
            warnings.warn(
                f"No JSON parsable object found in generated text: {generated_text}. Returning 'Defect' action as 0.")
            action = 0
        else:
            try:
                action = from_nat_lang(answer["action"])
            except Exception as e:
                warnings.warn(f"{str(e)} in answer: {answer}. Returning 'Defect' action as 0.")
                action = 0
        return int(action)

    def ask_questions(self, checkers, game, verbose=False):
        if checkers is not None:
            for checker in checkers:
                try:
                    checker.set_inference_client(self.client)
                    checker.ask_questions(game, self.player_name, verbose=verbose)
                except Exception as e:
                    warnings.warn(
                        f"Error {str(e)}. Checker {checker.get_name()} of type {type(checker)} failed to ask questions to the inference client.")
