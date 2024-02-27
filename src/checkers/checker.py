import json
import os
import warnings

from huggingface_hub import InferenceClient

from src.utils import CHECKS_OUT_BASE_PATH, MAX_NEW_TOKENS, TEMPERATURE, MODEL, HF_API_TOKEN, find_json_object


class Checker:

    def __init__(self, checker_name, questions, questions_labels, timestamp):
        self.name = checker_name
        self.dir_path = CHECKS_OUT_BASE_PATH + f"{timestamp}\\"
        os.makedirs(self.dir_path, exist_ok=True)
        self.out_file_name = self.dir_path + f"{checker_name}.json"

        self.checker_str = "checker"
        self.label_str = "label"
        self.sample_mean_str = "sample_mean"
        self.sample_variance_str = "sample_variance"
        self.n_samples_str = "n_samples"
        self.positives_str = "positives"
        self.generated_texts_str = "generated_texts"
        self.answers_str = "answers"

        self.questions = questions
        self.questions_results = {}
        for question in questions:
            self.questions_results[question] = {
                self.checker_str: checker_name,
                self.label_str: questions_labels[questions.index(question)],
                self.sample_mean_str: 0,
                self.sample_variance_str: 0,
                self.n_samples_str: 0,
                self.positives_str: 0,
                self.generated_texts_str: [],
                self.answers_str: [],
            }
        self.sample_mean = 0
        self.sample_variance = 0
        self.n_samples = 0
        self.positives = 0
        self.start_prompt = "<s>[INST] "
        self.system_prompt = None
        self.end_prompt = "[/INST]"

        self.inference_client = None
        self.max_new_tokens = MAX_NEW_TOKENS
        self.temperature = TEMPERATURE

    def get_name(self):
        return self.name

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
        self.questions_results[question][self.generated_texts_str].append(generated_text)
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
        if need_str:
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
        self.questions_results[question][self.answers_str].append(
            {"correct_answer": str(correct_answer), "llm_answer": str(llm_answer), "is_correct": correct})
        if correct:  # TODO weight answers(all_same, /curr_round,  1 quest per kind)
            self.questions_results[question][self.positives_str] += 1
            self.positives += 1
        self.questions_results[question][self.n_samples_str] += 1
        self.n_samples += 1

        return correct

    def compute_sample_mean_and_variance_per_question(self, question):
        # Compute the sample mean for the question
        quest_positives = self.questions_results[question][self.positives_str]
        quest_n_samples = self.questions_results[question][self.n_samples_str]
        quest_sample_mean = quest_positives / quest_n_samples if quest_n_samples > 0 else 0
        self.questions_results[question][self.sample_mean_str] = quest_sample_mean
        # Compute the sample variance for the question
        quest_positive_diffs = (1 - quest_sample_mean) ** 2 * quest_positives
        quest_negative_diffs = (-quest_sample_mean) ** 2 * (quest_n_samples - quest_positives)
        quest_sample_variance = (quest_positive_diffs + quest_negative_diffs) / (
                quest_n_samples - 1) if quest_n_samples > 1 else 0
        self.questions_results[question][self.sample_variance_str] = quest_sample_variance

    def compute_sample_mean_and_variance_of_checker(self):
        # Compute sample mean for the checker
        self.sample_mean = self.positives / self.n_samples if self.n_samples > 0 else 0
        # Compute the sample variance for the checker
        positive_diffs = (1 - self.sample_mean) ** 2 * self.positives
        negative_diffs = (-self.sample_mean) ** 2 * (self.n_samples - self.positives)
        self.sample_variance = (positive_diffs + negative_diffs) / (self.n_samples - 1) if self.n_samples > 1 else 0

    def set_inference_client(self, inference_client, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
        self.inference_client = inference_client
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def ask_questions(self, game):
        raise NotImplementedError

    def save_results(self, infix=None):
        results = {}
        self.compute_sample_mean_and_variance_of_checker()
        results[self.name] = {
            self.checker_str: self.name,
            self.label_str: self.name,
            self.sample_mean_str: self.sample_mean,
            self.sample_variance_str: self.sample_variance,
            self.n_samples_str: self.n_samples,
            self.positives_str: self.positives,
        }
        for question in self.questions_results:
            self.compute_sample_mean_and_variance_per_question(question)
            results[question] = {
                self.checker_str: self.name,
                self.label_str: self.questions_results[question][self.label_str],
                self.sample_mean_str: self.questions_results[question][self.sample_mean_str],
                self.sample_variance_str: self.questions_results[question][self.sample_variance_str],
                self.n_samples_str: self.questions_results[question][self.n_samples_str],
                self.positives_str: self.questions_results[question][self.positives_str],
            }
        json_results = json.dumps(results, indent=4)
        if infix is None:
            with open(self.out_file_name, "w") as out_file:
                out_file.write(json_results)
        else:
            with open(self.out_file_name.replace(".json", f"_{infix}.json"), "w") as out_file:
                out_file.write(json_results)

    def save_complete_answers(self):
        complete_answers = {}
        for question in self.questions_results:
            complete_answers[question] = {
                self.label_str: self.questions_results[question][self.label_str],
                self.generated_texts_str: self.questions_results[question][self.generated_texts_str],
                self.answers_str: self.questions_results[question][self.answers_str],
            }
        json_complete_answers = json.dumps(complete_answers, indent=4)
        with open(self.out_file_name.replace(".json", "_complete_answers.json"), "w") as out_file:
            out_file.write(json_complete_answers)
