import time
import warnings

from src.llm_utils import MAX_NEW_TOKENS, TEMPERATURE
from src.utils import log

huggingface_str = "huggingface"
openai_str = "openai"


class ModelClient:
    def __init__(self, model_name: str, model_url: str, api_key: str, provider: str = huggingface_str):
        self.model_name = model_name
        self.model_url = model_url
        self.api_key = api_key
        self.provider = provider
        if provider == huggingface_str:
            from huggingface_hub import InferenceClient
            self.api_client = InferenceClient(model=model_url, token=api_key)
            self.api_client.headers["x-use-cache"] = "0"
        elif provider == openai_str:
            from openai import OpenAI
            self.api_client = OpenAI(api_key=api_key)
            self.minute_requests = 0
            self.minute_requests_limit = 3500
            self.daily_requests = 0
            self.daily_requests_limit = 10000
        else:
            self.api_client = None

    def generate_text(self, prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
        generated_text = ""

        if self.provider == huggingface_str:
            # TODO 2/3 --> generate_rule_prompt
            ## HuggingFace API ###
            generated = False
            while not generated:
                try:
                    generated_text = self.api_client.text_generation(prompt, max_new_tokens=max_new_tokens,
                                                                  temperature=temperature)
                    generated = True
                except Exception as e:
                    if e.__class__.__name__ == "HfHubHTTPError" or e.__class__.__name__ == "OverloadedError":
                        warnings.warn("Model is overloaded. Waiting 2 seconds and retrying.")
                        time.sleep(2)
                    else:
                        warnings.warn(
                            f"Error {str(e)} in text generation with prompt: {prompt}. Substituting with empty string.")
                        generated_text = ""
                        generated = True
        elif self.provider == openai_str:
            ### OpenAI API ###
            ### HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
            if self.daily_requests > (self.daily_requests_limit - 100):
                # avoid daily requests limit
                current_time = time.time()
                local_time = time.localtime(current_time)
                next_midnight = time.struct_time((
                    local_time.tm_year, local_time.tm_mon, local_time.tm_mday + 1,
                    0, 0, 0, local_time.tm_wday, local_time.tm_yday + 1, local_time.tm_isdst
                ))
                next_midnight_seconds = time.mktime(next_midnight)
                seconds_until_midnight = (next_midnight_seconds - current_time) + 60
                log.info(f"Sleeping for {seconds_until_midnight} seconds to avoid daily limit.")
                time.sleep(seconds_until_midnight)
                self.daily_requests = 0
                self.minute_requests = 0
            if self.minute_requests > (self.minute_requests_limit - 100):
                # avoid minute requests limit
                log.info(f"Sleeping for 60 seconds to avoid minute limit.")
                time.sleep(60)
                self.minute_requests = 0
            response = self.api_client.chat.completions.with_raw_response.create(
                model=self.model_url,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_new_tokens
            )
            completion = response.parse()
            generated_text = completion.choices[0].message.content
            self.minute_requests += 1
            self.daily_requests += 1

        return generated_text
