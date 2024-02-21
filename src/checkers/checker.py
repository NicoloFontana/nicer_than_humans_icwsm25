import json
import os
import warnings

from huggingface_hub import InferenceClient

from src.utils import CHECKS_OUT_BASE_PATH, MAX_NEW_TOKENS, TEMPERATURE, MODEL, HF_API_TOKEN, find_json_object


class Checker:

    def __init__(self, checker_name, questions, timestamp):
        self.checker_name = checker_name
        dir_path = CHECKS_OUT_BASE_PATH + f"{checker_name}\\"
        os.makedirs(dir_path, exist_ok=True)
        self.out_file_name = dir_path + f"{timestamp}.txt"
        self.questions = questions
        self.generated_texts = {}
        for question in self.questions:
            self.generated_texts[question] = []
        self.answers = {}
        for question in self.questions:
            self.answers[question] = []
        self.scores = {}
        for question in self.questions:
            self.scores[question] = 0
        self.checks = {}
        for question in self.questions:
            self.checks[question] = 0

        self.results = {}
        self.start_prompt = "<s>[INST] "
        self.system_prompt = None
        self.end_prompt = "[/INST]"

        self.inference_client = None
        self.max_new_tokens = MAX_NEW_TOKENS
        self.temperature = TEMPERATURE

    def get_name(self):
        return self.checker_name

    def get_out_file_name(self):
        return self.out_file_name

    def get_answer_from_llm(self, prompt, question, need_str=True):
        """
        Get an answer from the LLM model given a prompt.
        :param prompt: prompt to be used for the LLM model.
        :param question: question which the prompt and the generated text are related to.
        :param need_str: requires the answer to be a string.
        :return: always a string if need_str is True, otherwise the type of the answer extracted from the JSON object.
        """
        if self.inference_client is None:
            warnings.warn("Inference client not set. Using default one.")
            self.inference_client = InferenceClient(model=MODEL, token=HF_API_TOKEN)
        try:
            generated_text = self.inference_client.text_generation(prompt, max_new_tokens=self.max_new_tokens,
                                                                   temperature=self.temperature)
        except Exception as e:
            warnings.warn(f"Error {str(e)} in text generation with prompt: {prompt}. Substituting with empty string.")
            generated_text = ""
        self.generated_texts[question].append(generated_text)
        json_object = find_json_object(generated_text)
        if json_object is not None:
            try:
                answer = json_object["answer"]
            except Exception as e:
                warnings.warn(f"Error {str(e)}. No key 'answer' in JSON: {json_object}")
                answer = ""
        else:
            warnings.warn(f"Could not find a valid JSON object in the generated text: {generated_text}")
            answer = ""
        if need_str and not isinstance(answer, str):
            answer = str(answer)
        return answer

    def check_answer(self, llm_answer, correct_answer, question):
        """
        Check if the LLM answer is correct and update the scores and checks accordingly.\n
        The correct answer can be a string, a set of strings or a list of strings.\n
        If the correct answer is a string, the LLM answer is correct if it is equal to it.\n
        If the correct answer is a set of strings, the LLM answer is correct if all its element are in the set and there are no extra ones.\n
        If the correct answer is a list of strings, the LLM answer is correct if all its elements are present in the same order as in the list.\n
        :param llm_answer: answer from the LLM to be checked.
        :param correct_answer: correct answer.
        :param question: question to which the correct answer is related to.
        :return: if the LLM answer is correct.
        """
        is_set = isinstance(correct_answer, set)
        is_list = isinstance(correct_answer, list)
        correct = False
        if is_set or is_list:
            if len(llm_answer) == len(correct_answer):
                correct = True
                if is_list:
                    for i in range(len(llm_answer)):
                        if isinstance(llm_answer[i], str) and isinstance(correct_answer[i], str):
                            if llm_answer[i].casefold() != correct_answer[i].casefold():
                                correct = False
                                break
                        else:
                            if llm_answer[i] != correct_answer[i]:
                                correct = False
                                break
                else:
                    insensitive_set = set()
                    for correct_ans in correct_answer:
                        if isinstance(correct_ans, str):
                            insensitive_set.add(correct_ans.casefold())
                        else:
                            insensitive_set.add(correct_ans)
                    for llm_ans in llm_answer:
                        if isinstance(llm_ans, str):
                            if llm_ans.casefold() not in insensitive_set:
                                correct = False
                                break
                        else:
                            if llm_ans not in insensitive_set:
                                correct = False
                                break
        else:
            if isinstance(llm_answer, str) and isinstance(correct_answer, str):
                correct = llm_answer.casefold() == correct_answer.casefold()
            else:
                correct = llm_answer == correct_answer
        self.answers[question].append(
            {"correct_answer": correct_answer, "llm_answer": llm_answer, "is_correct": correct})
        if correct:
            self.scores[question] += 1
        self.checks[question] += 1
        return correct

    def get_scores(self):
        return self.scores

    def get_checks(self):
        return self.checks

    def get_accuracy(self):
        accuracy = {}
        for question in self.questions:
            if self.checks[question] != 0:
                accuracy[question] = self.scores[question] / self.checks[question]
            else:
                accuracy[question] = 0
        return accuracy

    def set_inference_client(self, inference_client, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
        self.inference_client = inference_client
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def ask_questions(self, game):
        raise NotImplementedError

    def save_results(self):
        for question in self.questions:
            self.results[question] = {"generated_texts": self.generated_texts, "answers": self.answers[question],
                                      "score": self.scores[question], "checks": self.checks[question],
                                      "accuracy": self.scores[question] / self.checks[question] if self.checks[
                                                                                                       question] != 0 else 0}
        json_results = json.dumps(self.results, indent=4)
        out_file = open(self.out_file_name, "w")
        out_file.write(json_results)
