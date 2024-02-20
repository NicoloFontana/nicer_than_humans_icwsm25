import warnings

from huggingface_hub import InferenceClient

from src.games.gt_game import GTGame
from src.games.two_players_pd_utils import from_nat_lang
from src.strategies.strategy import Strategy
from src.strategies.strategy_utils import generate_prompt
from src.utils import MODEL, HF_API_TOKEN, MAX_NEW_TOKENS, TEMPERATURE, find_json_object


class OneVsOnePDLlmStrategy(Strategy):

    def __init__(self, game: GTGame, player_name: str, opponent_name: str, model=MODEL, token=HF_API_TOKEN,
                 max_new_tokens=MAX_NEW_TOKENS,
                 temperature=TEMPERATURE, update_client=True):
        super().__init__("OneVsOnePDLlmStrategy")
        try:
            self.client = InferenceClient(model=model, token=token)
            # Check if the model and token are valid
            self.client.text_generation("Test", max_new_tokens=1, temperature=TEMPERATURE)
        except Exception as e:
            raise Exception(f"Error in creating InferenceClient with model {model} and token {token}") from e
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
        self.model = model
        self.token = token
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.update_client = update_client

    def get_client(self):
        return self.client

    def play(self, checkers=None, verbose=False):
        action_space = self.game.get_action_space()
        payoff_function = self.game.get_payoff_function()
        n_iterations = self.game.get_iterations()
        own_history = self.game.get_actions_by_player(self.player_name)
        opponent_history = self.game.get_actions_by_player(self.opponent_name)
        self.client = InferenceClient(model=self.model, token=self.token) if self.update_client else self.client
        prompt = generate_prompt(action_space, payoff_function, n_iterations, own_history, opponent_history)
        try:
            generated_text = self.client.text_generation(prompt, max_new_tokens=self.max_new_tokens,
                                                    temperature=self.temperature)
        except Exception as e:
            warnings.warn(f"Error {str(e)} in text generation with prompt: {prompt}. Substituting with empty string.")
            generated_text = ""
        answer = find_json_object(generated_text)
        if answer is None:
            warnings.warn(f"No JSON parsable object found in generated text: {generated_text}. Returning 'Defect' action as 0.")
            action = 0
        else:
            try:
                action = from_nat_lang(answer["action"])
            except Exception as e:
                warnings.warn(f"{str(e)} in answer: {answer}. Returning 'Defect' action as 0.")
                action = 0
        if checkers is not None:
            for checker in checkers:
                try:
                    checker.set_inference_client(self.client)
                    checker.ask_questions(self.game, self.player_name, verbose=verbose)
                except Exception as e:
                    warnings.warn(f"Error {str(e)}. Checker {checker} of type {type(checker)} failed to ask questions to the inference client.")
        return action
