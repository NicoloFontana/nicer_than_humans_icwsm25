import json
import os
import time
import warnings
from pathlib import Path

from huggingface_hub import InferenceClient

from src.utils import find_json_object
from src.llm_utils import HF_API_TOKEN, OUT_BASE_PATH, MODEL, MAX_NEW_TOKENS, TEMPERATURE, generate_text


class Checker:

    def __init__(self, checker_name, questions, questions_labels, timestamp):
        self.name = checker_name
        self.dir_path = OUT_BASE_PATH / str(timestamp) / checker_name
        os.makedirs(self.dir_path, exist_ok=True)
        self.out_file_name = self.dir_path / checker_name
        self.out_file_name = self.out_file_name.with_suffix(".json")

        self.checker_str = "checker"
        self.label_str = "label"
        self.sample_mean_str = "sample_mean"
        self.sample_variance_str = "sample_variance"
        self.total_str = "total"
        self.positives_str = "positives"
        self.squared_diffs_sum_str = "squared_diffs_sum"
        self.prompt_str = "prompt"
        self.generated_text_str = "generated_text"
        self.answer_str = "answer"

        self.questions = questions
        self.questions_results = {}
        for question in questions:
            self.questions_results[question] = {
                self.checker_str: checker_name,
                self.label_str: questions_labels[questions.index(question)],
                self.sample_mean_str: 0,
                self.sample_variance_str: 0,
                self.total_str: 0,
                self.positives_str: 0,
                self.squared_diffs_sum_str: 0,
                self.prompt_str: [],
                self.generated_text_str: [],
                self.answer_str: [],
            }
        self.sample_mean = 0
        self.sample_variance = 0
        self.total = 0
        self.positives = 0
        self.squared_diffs_sum = 0
        self.system_prompt = None

        self.inference_client = None

    def get_name(self):
        return self.name

    def get_answer_from_llm(self, prompt, question, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, need_str=True):
        """
        Get an answer from the LLM model given a prompt.
        :param prompt: prompt to be used for the LLM model.
        :param question: question which the prompt and the generated text are related to.
        :param max_new_tokens: parameter for the InferenceClient
        :param temperature: parameter for the InferenceClient
        :param need_str: requires the answer to be a string.
        :return: always a string if need_str is True, otherwise the type of the answer extracted from the JSON object.
        """
        if self.inference_client is None:
            warnings.warn("Inference client not set. Using default one.")
            self.inference_client = InferenceClient(model=MODEL, token=HF_API_TOKEN)
            self.inference_client.headers["x-use-cache"] = "0"
        self.questions_results[question][self.prompt_str].append(prompt)
        generated_text = generate_text(prompt, self.inference_client, max_new_tokens=max_new_tokens,
                                       temperature=temperature)
        self.questions_results[question][self.generated_text_str].append(generated_text)
        json_object = find_json_object(generated_text)
        if json_object is not None:
            try:
                answer = json_object["answer"]
            except Exception as e:
                warnings.warn(f"Error {str(e)}. No key 'answer' in JSON: {json_object}. Returning entire generated text.")
                answer = generated_text
        else:
            warnings.warn(f"Could not find a valid JSON object in the generated text: {generated_text}. Returning entire generated text.")
            answer = generated_text
        if need_str:
            answer = str(answer)
        return answer

    def check_answer(self, llm_answer, correct_answer, question, weight=1.0):
        """
        Check if the LLM answer is correct and update the scores and checks accordingly.\n
        The correct answer can be a string, a set of strings or a list of strings.\n
        If the correct answer is a string, the LLM answer is correct if it is equal to it.\n
        If the correct answer is a set of strings, the LLM answer is correct if all its element are in the set and there are no extra ones.\n
        If the correct answer is a list of strings, the LLM answer is correct if all its elements are present in the same order as in the list.\n
        :param weight: how much the answer should be weighted.
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
        self.questions_results[question][self.answer_str].append(
            {"correct_answer": str(correct_answer), "llm_answer": str(llm_answer), "is_correct": correct})

        self.update_aggregates_for_question(question, int(correct), weight)
        self.update_aggregates_for_checker(correct, weight)
        return correct

    def update_aggregates_for_question(self, question, answer, weight):
        # Compute weighted sample mean
        self.questions_results[question][self.positives_str] += answer * weight
        positives = self.questions_results[question][self.positives_str]
        self.questions_results[question][self.total_str] += weight
        total = self.questions_results[question][self.total_str]
        sample_mean = positives / total if total > 0 else 0
        self.questions_results[question][self.sample_mean_str] = sample_mean
        # Compute weighted sample variance
        self.questions_results[question][self.squared_diffs_sum_str] += ((answer - sample_mean) ** 2) * weight
        squared_diffs_sum = self.questions_results[question][self.squared_diffs_sum_str]
        sample_variance = squared_diffs_sum / (total - 1) if total > 1 else 0
        self.questions_results[question][self.sample_variance_str] = sample_variance

    def update_aggregates_for_checker(self, answer, weight):
        # Compute weighted sample mean for the checker
        self.positives += answer * weight
        self.total += weight
        self.sample_mean = self.positives / self.total if self.total > 0 else 0
        # Compute the weighted sample variance for the checker
        self.squared_diffs_sum += ((answer - self.sample_mean) ** 2) * weight
        self.sample_variance = self.squared_diffs_sum / (self.total - 1) if self.total > 1 else 0

    def set_inference_client(self, inference_client):
        self.inference_client = inference_client

    def ask_questions(self, game):
        raise NotImplementedError

    def save_results(self, infix=None):
        results = {self.name: {
            self.checker_str: self.name,
            self.label_str: self.name,
            self.sample_mean_str: self.sample_mean,
            self.sample_variance_str: self.sample_variance,
            self.total_str: self.total,
            self.positives_str: self.positives,
            self.squared_diffs_sum_str: self.squared_diffs_sum,
        }}
        for question in self.questions_results:
            results[question] = {
                self.checker_str: self.name,
                self.label_str: self.questions_results[question][self.label_str],
                self.sample_mean_str: self.questions_results[question][self.sample_mean_str],
                self.sample_variance_str: self.questions_results[question][self.sample_variance_str],
                self.total_str: self.questions_results[question][self.total_str],
                self.positives_str: self.questions_results[question][self.positives_str],
                self.squared_diffs_sum_str: self.questions_results[question][self.squared_diffs_sum_str],
            }
        json_results = json.dumps(results, indent=4)
        if infix is None:
            with open(self.out_file_name, "w") as out_file:
                out_file.write(json_results)
                print(f"{self.name} results saved.")
        else:
            tmp_out_file_name = Path(str(self.out_file_name.with_suffix("")) + f"_{infix}.json")
            with open(tmp_out_file_name, "w") as out_file:
                out_file.write(json_results)
                print(f"{self.name} results saved.")

    def save_complete_answers(self):
        complete_answers = {}
        for question in self.questions_results:
            complete_answers[question] = {}
            label = self.questions_results[question][self.label_str]
            for idx in range(len(self.questions_results[question][self.prompt_str])):
                complete_answers[question][idx] = {
                    self.label_str: label,
                    self.prompt_str: self.questions_results[question][self.prompt_str][idx],
                    self.generated_text_str: self.questions_results[question][self.generated_text_str][idx],
                    self.answer_str: self.questions_results[question][self.answer_str][idx],
                }
        json_complete_answers = json.dumps(complete_answers, indent=4)
        tmp_out_file_name = Path(str(self.out_file_name.with_suffix("")) + "_complete_answers.json")
        with open(tmp_out_file_name, "w") as out_file:
            out_file.write(json_complete_answers)
            print(f"{self.name} answers saved.")
