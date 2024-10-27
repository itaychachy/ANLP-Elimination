# Advanced NLP Project - Elimination
# Omer Benishu, Itay Chachy, Matan Velner

from datasets import load_dataset
import transformers
from transformers import pipeline
import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from tqdm import tqdm
import re
from collections import defaultdict
import json
import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset
import matplotlib.gridspec as gridspec
import random

# Constants

DEBUG = True
# Note:
# 1. Fill the 'TOKEN' constant below with a valid token for 'HugginFace' (used for loading models, see [ClearablePipeline]).
# 2. Change the 'PATH' constant according to your drive path (for saving and loading files)
TOKEN = ''
PATH = '/content/gdrive/My Drive/Colab Notebooks/NLP Project/' if DEBUG else ''
DATASET_SIZE_LIMIT = 400
BASELINE_LABEL_KEY = 'answer_label'
BASELINE_ANSWER_KEY = 'answer_content'
ELIMINATION_LABEL_KEY = 'eliminated_label'
ELIMINATION_ANSWER_KEY = 'eliminated_answer'
# Note:
# Change the below constants to control your running options.
RUN_BASELINE = False
RUN_BASELINE_COT = False
RUN_BASELINE_STRATEGY = 'REGULAR'  # One of 'REGULAR', 'COT', 'ITERATIVE_COT', when COT = Chain of thought
RUN_ELIMINATION = False
RUN_ONE_SHOT_ELIMINATION = False
ANALYZE = True

if DEBUG:  # Full experiment runs on Cluster
    from google.colab import drive
    drive.mount('/content/gdrive')


def save_as_json(obj, name):
    with open(f'{PATH}Data/{name}.json', 'w') as file:
        json.dump(obj, file, indent=4)


def load_from_json(name):
    with open(f'{PATH}Data/{name}.json', 'r') as file:
        return json.load(file)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ClearableData:
    def __init__(self, load_dataset, transform, batch_size=1):
        self.dataset = None
        self.dataloader = None
        self.transform = transform
        self.load_dataset = load_dataset
        self.batch_size = batch_size

    def load(self):
        self.dataset = self.load_dataset()
        # Shuffle = False is needed, since data indecis are compared
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        return self.dataset

    def clear(self):
        del self.dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class ClearablePipeline:
    def __init__(self, name, device=device):
        self.name = name
        self.device = device
        self.p = None

    def load(self):
        if self.p is None:
            self.p = pipeline(
                task="text-generation",
                model=self.name,
                framework='pt',
                device=self.device,
                token=TOKEN,
                torch_dtype=torch.float16
            )
        return self.p

    def clear(self):
        del self.p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# Datasets Loader Functions
def load_openbookqa():
    ds = load_dataset("allenai/openbookqa", "additional")
    ds.set_format("torch")
    return ds['test'].select(range(DATASET_SIZE_LIMIT))


def load_race():
    ds = load_dataset("ehovy/race", "middle")
    ds.set_format("torch")
    return ds['test'].select(range(DATASET_SIZE_LIMIT))


def load_ai2_arc_easy():
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy")
    ds.set_format("torch")
    return ds['test'].select(range(DATASET_SIZE_LIMIT))


def load_ai2_arc_challenge():
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    ds.set_format("torch")
    return ds['test'].select(range(DATASET_SIZE_LIMIT))


def load_mmlu():
    ds = load_dataset("cais/mmlu", "all")
    ds.set_format("torch")
    return ds['validation']  # 1.5K Examples


# EasyAdditive
class MathWithDistractorDataset(Dataset):
    def __init__(self, num_samples=DATASET_SIZE_LIMIT):
        self.data = []
        items = ['lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'apple', 'banana', 'cherry', 'grape', 'kiwi']

        for _ in range(num_samples):
            a, b = random.randint(1, 20), random.randint(1, 20)
            correct = a + b
            o1 = correct + random.randint(1, 10)
            o2 = correct - random.randint(1, 10)
            random_answer = items[random.randint(0, len(items) - 1)]
            options = [o1, o2, random_answer, correct]
            random.shuffle(options)
            correct_index = options.index(correct)
            correct_label = chr(ord('A') + correct_index)
            random_index = options.index(random_answer)
            random_label = chr(ord('A') + random_index)
            self.data.append([f'{a} + {b} = ?', options, correct_label, random_label])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, options, correct_label, random_label = self.data[idx]
        return {
            'question': question,
            'answers': options,
            'correct_label': correct_label,
            'random_label': random_label
        }


def experiment_data_with_random_answer():
    return MathWithDistractorDataset()


baseline_datasets = [
    (ClearableData(load_ai2_arc_easy, lambda x: AI2ArcBaseline(x)), "AI2_ARC_EASY"),
    (ClearableData(load_ai2_arc_challenge, lambda x: AI2ArcBaseline(x)), "AI2_ARC_CHALLENGE"),
    (ClearableData(load_openbookqa, lambda x: OpenBookQABaseline(x)), "OpenBookQA"),
    (ClearableData(load_race, lambda x: RaceBaseline(x)), "RACE"),
    (ClearableData(load_mmlu, lambda x: MMLUBaseline(x)), "MMLU")
]

elimination_datasets = [
    (ClearableData(experiment_data_with_random_answer, lambda x: EasyAdditiveElimination(x)), "Easy_Additive"),
    (ClearableData(load_ai2_arc_easy, lambda x: AI2ArcElimination(x)), "AI2_ARC_EASY"),
    (ClearableData(load_ai2_arc_challenge, lambda x: AI2ArcElimination(x)), "AI2_ARC_CHALLENGE"),
    (ClearableData(load_openbookqa, lambda x: OpenBookQAElimination(x)), "OpenBookQA"),
    (ClearableData(load_race, lambda x: RaceElimination(x)), "RACE"),
    (ClearableData(load_mmlu, lambda x: MMLUElimination(x)), "MMLU")
]

one_shot_elimination_datasets = [
    (ClearableData(experiment_data_with_random_answer, lambda x: EasyAdditiveOneShotElimination(x)), "Easy_Additive"),
    (ClearableData(load_ai2_arc_easy, lambda x: AI2ArcOneShotElimination(x)), "AI2_ARC_EASY"),
    (ClearableData(load_ai2_arc_challenge, lambda x: AI2ArcOneShotElimination(x)), "AI2_ARC_CHALLENGE"),
    (ClearableData(load_openbookqa, lambda x: OpenBookQAOneShotElimination(x)), "OpenBookQA"),
    (ClearableData(load_race, lambda x: RaceOneShotElimination(x)), "RACE"),
    (ClearableData(load_mmlu, lambda x: MMLUOneShotElimination(x)), "MMLU")

]

pipelines = [
    (ClearablePipeline("Qwen/Qwen2-7B-Instruct"), "Qwen2-7B-Instruct"),
    (ClearablePipeline("meta-llama/Meta-Llama-3.1-8B-Instruct"), "Llama-3.1-8B-Instruct"),
    # (ClearablePipeline("Qwen/Qwen2-0.5B-Instruct"), "Qwen2-0.5B-Instruct") # Small model for sanity checks
]


class MultipleAnswersQuestionBaseline(ABC):
    def __init__(self, input):
        """
        :param input: question of any form
        """
        super().__init__()
        self.question = self._extract_question(input)
        self.answers = self._extract_answers(input)  # list of tuples (label, answer)
        self.correct_label = self._extract_correct_label(input)
        self.labels = [l for l, _ in self.answers]

    def get_full_question(self):
        answers = ""
        for l, answer in self.answers:
            answers += f'{l}. {answer}\n'
        labels_order_str = ', '.join(self.labels)
        if RUN_BASELINE_STRATEGY == 'REGULAR':
            user_content = f'Below is a multiple-choice question. Analyze the options and choose the right answer, explaining why it is correct. Your explanation should be clear, logical, and directly address the key concepts of the question.\n{self.question}\n{answers}\n\n Provide your response in the following JSON format:{{"explanation": "$$$EXPLANATION$$$", "{BASELINE_LABEL_KEY}": "$$$LABEL$$$","{BASELINE_ANSWER_KEY}": "$$$ANSWER$$$"}}\n Remember, "{BASELINE_LABEL_KEY}" is an enum of {labels_order_str}.'
        elif RUN_BASELINE_STRATEGY == 'ITERATIVE_COT':
            user_content = f'{self.question}\n{answers}\n\n1. Identify an incorrect answer and provide a reason for its elimination.\n2. Reiterate the question and remaining answers.\n3. Repeat steps 1 and 2 until only one answer remains.\n4. Present the final correct answer in the following JSON format:{{"{BASELINE_LABEL_KEY}": "$$$LABEL$$$","{BASELINE_ANSWER_KEY}": "$$$ANSWER$$$"}}'
        else:  # RUN_BASELINE_STRATEGY == 'COT'
            user_content = f'{self.question}\n{answers}\n\n1. Identify an incorrect answer and provide a reason for its elimination.\n2. Repeat step 1 until only one are left with only the right answer.\n3. Present the final correct answer in the following JSON format:{{"{BASELINE_LABEL_KEY}": "$$$LABEL$$$","{BASELINE_ANSWER_KEY}": "$$$ANSWER$$$"}}'
        return [
            {
                "role": "system",
                "content": self._get_system_content()
            },
            {
                "role": "user",
                "content": user_content
            },
        ]

    def _get_system_content(self):
        if RUN_BASELINE_STRATEGY == 'REGULAR':
            return "You are an intelligent assistant specializing in multiple-choice questions. Your task is to choose the correct answer by providing a clear, logical explanation of why it is right. Ensure your reasoning is detailed and precise."
        elif RUN_BASELINE_STRATEGY == 'ITERATIVE_COT':
            return 'You are an expert AI assistant tasked with solving multiple-choice questions by using a methodical elimination process. For each question, you should eliminate incorrect answers one by one, providing a clear explanation for each elimination. After each elimination, repeat the question without the eliminated option until you have identified the correct answer. Present the final answer in the specified JSON format.'
        else:  # RUN_BASELINE_STRATEGY == 'COT'
            return 'You are an expert AI assistant tasked with solving multiple-choice questions by using a methodical elimination process. For each question, you should eliminate incorrect answers one by one, providing a clear explanation for each elimination. Repeat this process until you have identified the correct answer. Present the final answer in the specified JSON format.'

    @abstractmethod
    def _extract_question(self, input):
        """Extract the question text from the input."""
        pass

    @abstractmethod
    def _extract_answers(self, input):
        """Extract the answers list from the input (list of tuples (label, answer))."""
        pass

    @abstractmethod
    def _extract_correct_label(self, input):
        """Extract the correct label from the input."""
        pass


class OpenBookQABaseline(MultipleAnswersQuestionBaseline):
    def __init__(self, input):
        super().__init__(input)

    def _extract_question(self, input):
        return input['fact1'][0] + '\n' + input['question_stem'][0]

    def _extract_answers(self, input):
        return [(l[0], a[0]) for l, a in zip(input['choices']['label'], input['choices']['text'])]

    def _extract_correct_label(self, input):
        return input['answerKey'][0]


class RaceBaseline(MultipleAnswersQuestionBaseline):
    def __init__(self, input):
        super().__init__(input)

    def _extract_question(self, input):
        return f'{input["article"][0]}\n{input["question"][0]}'

    def _extract_answers(self, input):
        return [(l[0], a[0]) for l, a in zip(['A', 'B', 'C', 'D'], input['options'])]

    def _extract_correct_label(self, input):
        return input['answer'][0]


class AI2ArcBaseline(MultipleAnswersQuestionBaseline):
    def __init__(self, input):
        super().__init__(input)

    def _extract_question(self, input):
        return input["question"][0]

    def _extract_answers(self, input):
        return [(l[0], a[0]) for l, a in zip(input['choices']['label'], input['choices']['text'])]

    def _extract_correct_label(self, input):
        return input['answerKey'][0]


class MMLUBaseline(MultipleAnswersQuestionBaseline):
    def __init__(self, input):
        super().__init__(input)

    def _extract_question(self, input):
        return input["question"][0]

    def _extract_answers(self, input):
        return [(l[0], a[0]) for l, a in zip(['A', 'B', 'C', 'D'], input['choices'])]

    def _extract_correct_label(self, input):
        return chr(65 + input['answer'][0])


class MultipleAnswersQuestionElimination(ABC):
    def __init__(self, input):
        """
        :param input: question of any form
        """
        super().__init__()
        self.question = self._extract_question(input)
        self.answers = self._extract_answers(input)  # list of tuples (label, answer)
        self.correct_label = self._extract_correct_label(input)
        self.labels = [l for l, _ in self.answers]

    def get_full_question(self):
        answers = ""
        for l, answer in self.answers:
            answers += f'{l}. {answer}\n'
        labels_order_str = ', '.join(self.labels)
        user_content = f'Below is a multiple-choice question. Analyze the options and eliminate one wrong answer, explaining why it is incorrect. Make sure not to choose the right answer, but rather choose a wrong option to eliminate and explain your choice. Your explanation for elimination should be clear, logical, and directly address the key concepts of the question.\n{self.question}\n{answers}\n\n Provide your response in the following JSON format:{{"explanation": "$$$EXPLANATION$$$", "{ELIMINATION_LABEL_KEY}": "$$$LABEL$$$","{ELIMINATION_ANSWER_KEY}": "$$$ANSWER$$$"}}\n Remember, "{ELIMINATION_LABEL_KEY}" is an enum of {labels_order_str}.'
        return [
            {
                "role": "system",
                "content": self._get_system_content()
            },
            {
                "role": "user",
                "content": user_content
            },
        ]

    def eliminate_answer(self, label):
        for l, a in self.answers:
            if l == label:
                self.answers.remove((l, a))
                break
        self.labels.pop()
        new_answers = []
        for (_, a), l in zip(self.answers, self.labels):
            new_answers.append((l, a))
        if ord(self.correct_label) > ord(label):
            self.correct_label = chr(ord(self.correct_label) - 1)
        self.answers = new_answers

    def _get_system_content(self):
        return "You are an intelligent assistant specializing in multiple-choice questions. Your task is to eliminate one incorrect answer by providing a clear, logical explanation of why it is wrong. Remember, there may be more than one incorrect option, but you should focus on eliminating just one. Ensure your reasoning is detailed and precise."

    @abstractmethod
    def _extract_question(self, input):
        """Extract the question text from the input."""
        pass

    @abstractmethod
    def _extract_answers(self, input):
        """Extract the answers list from the input (list of tuples (label, answer))."""
        pass

    @abstractmethod
    def _extract_correct_label(self, input):
        """Extract the correct label from the input."""
        pass


class OpenBookQAElimination(MultipleAnswersQuestionElimination):
    def __init__(self, input):
        super().__init__(input)

    def _extract_question(self, input):
        return input['fact1'][0] + '\n' + input['question_stem'][0]

    def _extract_answers(self, input):
        return [(l[0], a[0]) for l, a in zip(input['choices']['label'], input['choices']['text'])]

    def _extract_correct_label(self, input):
        return input['answerKey'][0]


class RaceElimination(MultipleAnswersQuestionElimination):
    def __init__(self, input):
        super().__init__(input)

    def _extract_question(self, input):
        return f'{input["article"][0]}\n{input["question"][0]}'

    def _extract_answers(self, input):
        return [(l[0], a[0]) for l, a in zip(['A', 'B', 'C', 'D'], input['options'])]

    def _extract_correct_label(self, input):
        return input['answer'][0]


class AI2ArcElimination(MultipleAnswersQuestionElimination):
    def __init__(self, input):
        super().__init__(input)

    def _extract_question(self, input):
        return input["question"][0]

    def _extract_answers(self, input):
        return [(l[0], a[0]) for l, a in zip(input['choices']['label'], input['choices']['text'])]

    def _extract_correct_label(self, input):
        return input['answerKey'][0]


class EasyAdditiveElimination(MultipleAnswersQuestionElimination):
    def __init__(self, input):
        super().__init__(input)

    def _extract_question(self, input):
        return input["question"][0]

    def _extract_answers(self, input):
        return [(l[0], a[0]) for l, a in zip(['A', 'B', 'C', 'D'], input['answers'])]

    def _extract_correct_label(self, input):
        return input['random_label'][0]


class MMLUElimination(MultipleAnswersQuestionElimination):
    def __init__(self, input):
        super().__init__(input)

    def _extract_question(self, input):
        return input["question"][0]

    def _extract_answers(self, input):
        return [(l[0], a[0]) for l, a in zip(['A', 'B', 'C', 'D'], input['choices'])]

    def _extract_correct_label(self, input):
        return chr(65 + input['answer'][0])


q0 = ['Which factor will most likely cause a person to develop a fever?',['a leg muscle relaxing after exercise','a bacterial population in the bloodstream','several viral particles on the skin','carbohydrates being digested in the stomach'], 1, 0, 'a leg muscle relaxing after exercise is relevant to sport and post training process, so it is unlikely to cause a person to develop fever']
q1 = ['as a source of light becomes closer, that source will appear brighter\nas a car approaches you in the night',['the headlights become more intense','the headlights recede into the dark','the headlights remain at a constant', 'the headlights turn off'], 0, 3, 'The distance from a car doesn\'t effect its light, so there is no reason for the lights to turn off']
q2 = ['Cars!!! Holidays! Thousands of prizes ! Hurry ! FREE with every packet of SPLASH! Your personal lucky number! Will be among the 500,000 Winners! Use SPLASH for the SOFTEST ... QIUCKEST...WHITEST WASH! DON\'T DELAY ... BUY A PACKET TODAY! \n This is _',['an introduction to some products', 'An advertisement for selling goods', 'a direction of a kind of washing machine', 'A notice about a football game'], 1, 3, 'Washing machine is very unrelated to cars, so it is very unlikely that this is the correct answer, so it is a good answer to eliminate.']
q3 = ['7 + 8 = ?',['3', '15', 'banana', '12'], 1, 2, 'banana is a fruit, and not a number, hence it is irrelevant to the question and a good elimination choice.']
q4 = ['Which expression is equivalent to 5 x 9?',[ "(5 x 4) x (6 x 5)", "(5 x 5) + (5 x 4)", "(5 x 5) + (5 x 9)", "(5 x 9) x (6 x 9)" ],1 ,0, '5 x 4 is 20 and 6 x 5 is 30 so their multiplication is greater than 5 x 9 which is 45']


def adjust_example_to_labels(n_labels, question, answers, correct_ind, eliminated_ind, explanation):
    if n_labels == 4:
        return f'{question}\nA. {answers[0]}\nB. {answers[1]}\nC. {answers[2]}\nD. {answers[3]}\n{{"explanation": "{explanation}", "{ELIMINATION_LABEL_KEY}": "{chr(65 + eliminated_ind)}","{ELIMINATION_ANSWER_KEY}": "{answers[eliminated_ind]}"}}'
    elif n_labels == 3:
        l = [0, 1, 2, 3]
        l.remove(correct_ind)
        l.remove(eliminated_ind)
        random.shuffle(l)
        keep = l[0]
        t = [correct_ind, eliminated_ind, keep]
        random.shuffle(t)
        answers = [answers[i] for i in t]
        eliminated_ind = t.index(eliminated_ind)
        return f'{question}\nA. {answers[0]}\nB. {answers[1]}\nC. {answers[2]}\n{{"explanation": "{explanation}", "{ELIMINATION_LABEL_KEY}": "{chr(65 + eliminated_ind)}","{ELIMINATION_ANSWER_KEY}": "{answers[eliminated_ind]}"}}'
    else:
        t = [correct_ind, eliminated_ind]
        random.shuffle(t)
        answers = [answers[i] for i in t]
        eliminated_ind = t.index(eliminated_ind)
        return f'{question}\nA. {answers[0]}\nB. {answers[1]}\n{{"explanation": "{explanation}", "{ELIMINATION_LABEL_KEY}": "{chr(65 + eliminated_ind)}","{ELIMINATION_ANSWER_KEY}": "{answers[eliminated_ind]}"}}'


examples = {
     'arc': q0,
     'openbookqa': q1,
     'race': q2,
     'easy_additive': q3,
     'mmlu': q4
}


class MultipleAnswersQuestionOneShotElimination(ABC):
    def __init__(self, input):
        """
        :param input: question of any form
        """
        super().__init__()
        self.question = self._extract_question(input)
        self.answers = self._extract_answers(input)  # list of tuples (label, answer)
        self.correct_label = self._extract_correct_label(input)
        self.labels = [l for l, _ in self.answers]

    def get_full_question(self):
        answers = ""
        for l, answer in self.answers:
            answers += f'{l}. {answer}\n'
        labels_order_str = ', '.join(self.labels)
        example = examples[self._get_name()]
        example = adjust_example_to_labels(len(self.labels), example[0], example[1], example[2], example[3], example[4])
        user_content = f'Below is a multiple-choice question. Analyze the options and eliminate one wrong answer, explaining why it is incorrect. Make sure not to choose the right answer, but rather choose a wrong option to eliminate and explain your choice. Your explanation for elimination should be clear, logical, and directly address the key concepts of the question.\n For example:\n{example}\n\n{self.question}\n{answers}\n\n Provide your response in the following JSON format:{{"explanation": "$$$EXPLANATION$$$", "{ELIMINATION_LABEL_KEY}": "$$$LABEL$$$","{ELIMINATION_ANSWER_KEY}": "$$$ANSWER$$$"}}\n Remember, "{ELIMINATION_LABEL_KEY}" is an enum of {labels_order_str}.'
        return [
            {
                "role": "system",
                "content": self._get_system_content()
            },
            {
                "role": "user",
                "content": user_content
            },
        ]

    def eliminate_answer(self, label):
        for l, a in self.answers:
            if l == label:
                self.answers.remove((l, a))
                break
        self.labels.pop()
        new_answers = []
        for (_, a), l in zip(self.answers, self.labels):
            new_answers.append((l, a))
        if ord(self.correct_label) > ord(label):
            self.correct_label = chr(ord(self.correct_label) - 1)
        self.answers = new_answers

    def _get_system_content(self):
        return "You are an intelligent assistant specializing in multiple-choice questions. Your task is to eliminate one incorrect answer by providing a clear, logical explanation of why it is wrong. Remember, there may be more than one incorrect option, but you should focus on eliminating just one. Ensure your reasoning is detailed and precise."

    @abstractmethod
    def _get_name(self):
        """Name of dataset (to extract the correct example)"""
        pass

    @abstractmethod
    def _extract_question(self, input):
        """Extract the question text from the input."""
        pass

    @abstractmethod
    def _extract_answers(self, input):
        """Extract the answers list from the input (list of tuples (label, answer))."""
        pass

    @abstractmethod
    def _extract_correct_label(self, input):
        """Extract the correct label from the input."""
        pass


class OpenBookQAOneShotElimination(MultipleAnswersQuestionOneShotElimination):
    def __init__(self, input):
        super().__init__(input)

    def _extract_question(self, input):
        return input['fact1'][0] + '\n' + input['question_stem'][0]

    def _extract_answers(self, input):
        return [(l[0], a[0]) for l, a in zip(input['choices']['label'], input['choices']['text'])]

    def _extract_correct_label(self, input):
        return input['answerKey'][0]

    def _get_name(self):
        return 'openbookqa'


class RaceOneShotElimination(MultipleAnswersQuestionOneShotElimination):
    def __init__(self, input):
        super().__init__(input)

    def _extract_question(self, input):
        return f'{input["article"][0]}\n{input["question"][0]}'

    def _extract_answers(self, input):
        return [(l[0], a[0]) for l, a in zip(['A', 'B', 'C', 'D'], input['options'])]

    def _extract_correct_label(self, input):
        return input['answer'][0]

    def _get_name(self):
        return 'race'


class AI2ArcOneShotElimination(MultipleAnswersQuestionOneShotElimination):
    def __init__(self, input):
        super().__init__(input)

    def _extract_question(self, input):
        return input["question"][0]

    def _extract_answers(self, input):
        return [(l[0], a[0]) for l, a in zip(input['choices']['label'], input['choices']['text'])]

    def _extract_correct_label(self, input):
        return input['answerKey'][0]

    def _get_name(self):
        return 'arc'


class EasyAdditiveOneShotElimination(MultipleAnswersQuestionOneShotElimination):
    def __init__(self, input):
        super().__init__(input)

    def _extract_question(self, input):
        return input["question"][0]

    def _extract_answers(self, input):
        return [(l[0], a[0]) for l, a in zip(['A', 'B', 'C', 'D'], input['answers'])]

    def _extract_correct_label(self, input):
        return input['random_label'][0]

    def _get_name(self):
        return 'easy_additive'


class MMLUOneShotElimination(MultipleAnswersQuestionOneShotElimination):
    def __init__(self, input):
        super().__init__(input)

    def _extract_question(self, input):
        return input["question"][0]

    def _extract_answers(self, input):
        return [(l[0], a[0]) for l, a in zip(['A', 'B', 'C', 'D'], input['choices'])]

    def _extract_correct_label(self, input):
        return chr(65 + input['answer'][0])

    def _get_name(self):
        return 'mmlu'


def extract_label(output, answers, label_key, answer_key):
    """
    Extracts the label that the model chose to eliminate from its output.
    @param output: The output of the model, JSON (should be).
    @param answers: List[Tuple]: (label, answer)
    @param label_key: The key of the label to eliminate.
    @param answer_key: The key of the answer to eliminate.
    @return: The label that the model chose to eliminate, or None if such a label wasn't found.
    """
    try:
        l, r = output.find('{'), output.rfind('}')
        output_dict = json.loads(output[l: r + 1])
        label = output_dict[label_key]
        for l, _ in answers:
            if l == label:
                return l
        answer = output_dict[answer_key]
        for l, a in answers:
            if a in answer:
                return l
        return None
    except Exception as e:
        label_pattern = re.compile(fr'(?<={label_key}":\s")[^"]*')
        answer_pattern = re.compile(fr'(?<={answer_key}":\s")[^"]*')

        potential_label = label_pattern.search(output)
        if potential_label:
            potential_label = potential_label.group().strip()
            for l, _ in answers:
                if l == potential_label:
                    return l
        potential_answer = answer_pattern.search(output)
        if potential_answer:
            potential_answer = potential_answer.group().strip()
            for l, a in answers:
                if a == potential_answer:
                    return l

        pattern = re.compile(r'\b(' + '|'.join(re.escape(l) for l, _ in answers) + r')\b')
        match = pattern.search(output)
        return match.group(0) if match else None


def _answer_dist_defaultdict_for_baseline():
    return {'model_n': 0, 'actual_n': 0}


def create_data_dict_for_baseline(name):
    d = {}
    d['name'] = name
    d['n_labels'] = 0
    d['failed_to_answer'] = 0
    d['right_guess'] = 0
    d['wrong_guess'] = 0
    d['wrong_indices'] = []
    d['answers_distribution'] = defaultdict(_answer_dist_defaultdict_for_baseline)
    return d


def run_baseline(prefix):
    baseline_results = []
    for p_ind, (p, model_name) in tqdm(enumerate(pipelines), desc="Iterating Models"):
        model_dict_results = {}
        model_dict_results['name'] = model_name
        gen_pipeline = p.load()
        model_dict_results['datasets'] = []
        for data_ind, (data, data_name) in tqdm(enumerate(baseline_datasets), desc="Iterating Datasets"):
            data_dict_results = create_data_dict_for_baseline(data_name)
            data.load()
            for i, input in tqdm(enumerate(data.dataloader), desc="Iterating data"):
                qa = data.transform(input)
                message = qa.get_full_question()
                output = gen_pipeline(message, max_new_tokens=1200, return_full_text=False)[0]['generated_text']
                label = extract_label(output, qa.answers, BASELINE_LABEL_KEY, BASELINE_ANSWER_KEY)
                data_dict_results['answers_distribution'][qa.correct_label]['actual_n'] += 1
                data_dict_results['answers_distribution'][label]['model_n'] += 1
                data_dict_results['n_labels'] = len(qa.labels)

                if DEBUG:
                    print(message)
                    print(output)
                    print(label)
                    print(qa.correct_label)

                if label is None:
                    data_dict_results['failed_to_answer'] += 1
                    data_dict_results['wrong_indices'].append(i)
                elif label != qa.correct_label:
                    data_dict_results['wrong_guess'] += 1
                    data_dict_results['wrong_indices'].append(i)
                else:
                    data_dict_results['right_guess'] += 1
                if DEBUG and i > 2:
                    break
            data_dict_results['n'] = data_dict_results['right_guess'] + data_dict_results['wrong_guess'] + data_dict_results['failed_to_answer']
            data_dict_results['accuracy'] = data_dict_results['right_guess'] / data_dict_results['n']
            data.clear()
            model_dict_results['datasets'].append(data_dict_results)
            save_as_json(model_dict_results, f'{prefix}baseline_results_{p_ind}_{data_ind}')

        # p.clear()
        baseline_results.append(model_dict_results)
        save_as_json(baseline_results, f'{prefix}baseline_results_{p_ind}')

    save_as_json(baseline_results, f'{prefix}baseline_results')
    if DEBUG:
        print(baseline_results)


def _answer_dist_defaultdict_for_elimination():
    return {
        'model_n': defaultdict(int),
        'actual_n': 0
    }


def create_data_dict_for_elimination(name):
    d = {}
    d['name'] = name
    d['n_labels'] = 0
    d['failed_to_answer'] = 0
    d['right_guess'] = 0
    d['wrong_guess'] = 0
    d['wrong_indices_and_step_of_mistake'] = []
    d['answers_distribution'] = defaultdict(_answer_dist_defaultdict_for_elimination)
    return d


def run_elimination(datasets, prefix=''):
    elimination_results = []
    for p_ind, (p, model_name) in tqdm(enumerate(pipelines), desc="Iterating Models"):
        model_dict_results = {}
        model_dict_results['name'] = model_name
        gen_pipeline = p.load()
        model_dict_results['datasets'] = []
        for data_ind, (data, data_name) in tqdm(enumerate(datasets), desc="Iterating Datasets"):
            data_dict_results = create_data_dict_for_elimination(data_name)
            data.load()
            for i, input in tqdm(enumerate(data.dataloader), desc="Iterating data"):
                qa = data.transform(input)
                data_dict_results['answers_distribution'][qa.correct_label]['actual_n'] += 1
                data_dict_results['n_labels'] = len(qa.answers)
                n_of_eliminations = 0
                while len(qa.labels) > 1:
                    message = qa.get_full_question()
                    output = gen_pipeline(message, max_new_tokens=300, return_full_text=False)[0]['generated_text']
                    elimination_label_guess = extract_label(output, qa.answers, ELIMINATION_LABEL_KEY, ELIMINATION_ANSWER_KEY)
                    data_dict_results['answers_distribution'][elimination_label_guess]['model_n'][n_of_eliminations] += 1
                    if DEBUG:
                        print(message)
                        print(output)
                        print(elimination_label_guess)
                        print(qa.correct_label)

                    if elimination_label_guess is None:
                        data_dict_results['failed_to_answer'] += 1
                        data_dict_results['wrong_indices_and_step_of_mistake'].append((i, n_of_eliminations))
                        break
                    elif elimination_label_guess == qa.correct_label:
                        data_dict_results['wrong_guess'] += 1
                        data_dict_results['wrong_indices_and_step_of_mistake'].append((i, n_of_eliminations))
                        break
                    else:
                        qa.eliminate_answer(elimination_label_guess)
                        n_of_eliminations += 1
                if len(qa.labels) == 1:
                    data_dict_results['right_guess'] += 1
                if DEBUG and i > 2:
                    break
            data_dict_results['n'] = data_dict_results['right_guess'] + data_dict_results['wrong_guess'] + data_dict_results['failed_to_answer']
            data_dict_results['accuracy'] = data_dict_results['right_guess'] / data_dict_results['n']
            data.clear()
            model_dict_results['datasets'].append(data_dict_results)
            save_as_json(model_dict_results, f'{prefix}elimination_results_{p_ind}_{data_ind}')

        # p.clear()
        elimination_results.append(model_dict_results)
        save_as_json(elimination_results, f'{prefix}elimination_results_{p_ind}')

    save_as_json(elimination_results, f'{prefix}elimination_results')
    if DEBUG:
        print(elimination_results)


if RUN_BASELINE:
    if RUN_BASELINE_COT:
       RUN_BASELINE_STRATEGY = 'COT'
       run_baseline('cot_')
       RUN_BASELINE_STRATEGY = 'ITERATIVE_COT'
       run_baseline('iterative_cot_')
    else:
       RUN_BASELINE_STRATEGY = 'REGULAR'
       run_baseline()

if RUN_ELIMINATION:
    run_elimination(elimination_datasets)

if RUN_ONE_SHOT_ELIMINATION:
    run_elimination(one_shot_elimination_datasets, 'one_shot_')


# Results Analysis
def plot_accuracies_one_experiment(baseline, experiment, label):
    terms = [d['name'].replace("_", " ") for d in baseline['datasets']]
    baseline_accuracy = [d['accuracy'] for d in baseline['datasets']]
    # Easy_Additive will be analyzed seperatly
    experiment_accuracy = [d['accuracy'] for d in experiment['datasets'] if d['name'] != 'Easy_Additive']
    fig = plt.figure(figsize = (10, 5))
    barWidth = 0.25
    br1 = np.arange(len(baseline_accuracy))
    br2 = [x + barWidth for x in br1]
    plt.bar(br1, baseline_accuracy, color='#D32F2F', width=barWidth, edgecolor='white', label='Baseline')
    plt.bar(br2, experiment_accuracy, color='#D95F02', width=barWidth, edgecolor='white', label=label)
    plt.xlabel("Datasets")
    plt.ylabel("Accuracy")
    plt.xticks([r + barWidth / 2 for r in range(len(baseline_accuracy))], terms)
    model_name = baseline['name'].removesuffix('-Instruct')
    title = f'{model_name} - Accuracy ({label})'
    # plt.title(title)
    plt.legend(loc='lower right')
    fig_title = title.lower().replace(' ', '_')
    plt.savefig(f'{PATH}Plots/{fig_title}.png', dpi=fig.dpi)
    plt.show()


def plot_accuracies_two_experiments(baseline, experiment1, experiment2, label1, label2):
    terms = [d['name'].replace("_", " ") for d in baseline['datasets']]
    baseline_accuracy = [d['accuracy'] for d in baseline['datasets']]
    # Easy_Additive will be analyzed seperatly
    experiment1_accuracy = [d['accuracy'] for d in experiment1['datasets'] if d['name'] != 'Easy_Additive']
    experiment2_accuracy = [d['accuracy'] for d in experiment2['datasets'] if d['name'] != 'Easy_Additive']

    fig = plt.figure(figsize = (10, 5))
    barWidth = 0.2

    br1 = np.arange(len(baseline_accuracy))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.bar(br1, baseline_accuracy, color='#D32F2F', width=barWidth, edgecolor='white', label='Baseline')
    plt.bar(br2, experiment1_accuracy, color='#D95F02', width=barWidth, edgecolor='white', label=label1)
    plt.bar(br3, experiment2_accuracy, color='#FDBF6F', width=barWidth, edgecolor='white', label=label2)
    plt.xlabel("Datasets")
    plt.ylabel("Accuracy")
    plt.xticks([r + barWidth for r in range(len(baseline_accuracy))], terms)
    model_name = baseline['name'].removesuffix('-Instruct')
    title = f'{model_name} - Accuracy Multiple Experiments ({label1}, {label2})'
    # plt.title(title)
    plt.legend(loc='lower right')
    fig_title = title.lower().replace(' ', '_')
    plt.savefig(f'{PATH}Plots/{fig_title}.png', dpi=fig.dpi)
    plt.show()


def plot_step_of_mistake(data, title):
    def count_mistakes(data):
        step_counts = defaultdict(int)
        for _, step in data['wrong_indices_and_step_of_mistake']:
            step_counts[step] += 1
        return step_counts

    # Easy_Additive will be analyzed seperatly
    if data['datasets'][-1]['name'] == 'Easy_Additive':
        datasets = data['datasets'][:-1]
    else:
        datasets = data['datasets']

    colors = ListedColormap(plt.cm.Spectral(np.linspace(0, 1, 10))).colors
    if len(datasets) == 5:
        gs = gridspec.GridSpec(2, 6, figure=plt.figure(figsize=(15, 10)))
        ax1 = plt.subplot(gs[0, 0:2])
        ax2 = plt.subplot(gs[0, 2:4])
        ax3 = plt.subplot(gs[0, 4:])
        ax4 = plt.subplot(gs[1, 1:3])
        ax5 = plt.subplot(gs[1, 3:5])
        fig = plt.gcf()
        gs.tight_layout(fig)
        ax_lst = [ax1, ax2, ax3, ax4, ax5]
    else:
        gs = gridspec.GridSpec(2, 2, figure=plt.figure(figsize=(10, 10)))
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 0])
        ax4 = plt.subplot(gs[1, 1])
        fig = plt.gcf()
        gs.tight_layout(fig)
        ax_lst = [ax1, ax2, ax3, ax4]

    for idx, dataset in enumerate(datasets):
        step_counts = count_mistakes(dataset)
        sorted_keys = sorted(step_counts.keys())
        labels = [f'Step {step + 1}' for step in sorted_keys]
        sizes = [step_counts[step] for step in sorted_keys]

        ax_lst[idx].pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90)
        ax_lst[idx].set_title(dataset['name'].replace("_", " "), fontsize=12, fontweight='bold')

    model_name = data['name'].removesuffix('-Instruct')
    title = f'{model_name} - Step Of Elimination Mistake ({title})'
    # plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_title = title.lower().replace(' ', '_')
    plt.savefig(f'{PATH}Plots/{fig_title}.png', dpi=fig.dpi)
    plt.show()


def plot_comparison_to_subset_questions_in_baseline(baseline_data, elimination_data, title, subset):
    def compare_mistakes(baseline, elimination):
        wrong_indices_in_elimination = set(index for index, _ in elimination['wrong_indices_and_step_of_mistake'])
        results = defaultdict(int)
        if subset == "All":
            indecis_to_compare = set(range(baseline['n']))
        elif subset == "Right":
            indecis_to_compare = set([i for i in range(baseline['n'])]) - set(baseline['wrong_indices'])
        else:
            # subset == "Wrong"
            indecis_to_compare = set(baseline['wrong_indices'])
        for index in indecis_to_compare:
            if index in wrong_indices_in_elimination:
                step = next(step for idx, step in elimination['wrong_indices_and_step_of_mistake'] if idx == index)
                results[f"Step {step + 1}"] += 1
            else:
                results["Success"] += 1
        return results

    colors = ListedColormap(plt.cm.Spectral(np.linspace(0, 1, 10))).colors
    # Easy_Additive will be analyzed seperatly
    if elimination_data['datasets'][-1]['name'] == 'Easy_Additive':
        datasets = elimination_data['datasets'][:-1]
    else:
        datasets = elimination_data['datasets']

    if len(datasets) == 5:
        gs = gridspec.GridSpec(2, 6, figure=plt.figure(figsize=(15, 10)))
        ax1 = plt.subplot(gs[0, 0:2])
        ax2 = plt.subplot(gs[0, 2:4])
        ax3 = plt.subplot(gs[0, 4:])
        ax4 = plt.subplot(gs[1, 1:3])
        ax5 = plt.subplot(gs[1, 3:5])
        fig = plt.gcf()
        gs.tight_layout(fig)
        ax_lst = [ax1, ax2, ax3, ax4, ax5]
    else:
        gs = gridspec.GridSpec(2, 2, figure=plt.figure(figsize=(10, 10)))
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 0])
        ax4 = plt.subplot(gs[1, 1])
        fig = plt.gcf()
        gs.tight_layout(fig)
        ax_lst = [ax1, ax2, ax3, ax4]

    for idx, dataset in enumerate(baseline_data['datasets']):
        elimination_dataset = next(item for item in datasets if item['name'] == dataset['name'])
        step_counts = compare_mistakes(dataset, elimination_dataset)
        sorted_keys = sorted(step_counts.keys())
        labels = sorted_keys
        sizes = [step_counts[key] for key in sorted_keys]

        ax_lst[idx].pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90)
        ax_lst[idx].set_title(dataset['name'].replace("_", " "), fontsize=12, fontweight='bold')

    model_name = baseline_data['name'].removesuffix('-Instruct')
    title = f'{model_name} - Step Of Elimination Mistake ({title}_{subset})'
    # plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_title = title.lower().replace(' ', '_')
    plt.savefig(f'{PATH}Plots/{fig_title}.png', dpi=fig.dpi)
    plt.show()

def compute_experiment_partial_accuracy(data):
    def count_mistakes(d):
        step_counts = defaultdict(int)
        for (_, step) in d['wrong_indices_and_step_of_mistake']:
            step_counts[step] += 1
        return step_counts

    results = {}
    name = data['name'].removesuffix('-Instruct')
    results[name] = {}
    for dataset in data['datasets']:
        steps_per_question = (dataset['n_labels'] - 1)
        total_n = dataset['n'] * steps_per_question
        accuracy = dataset['right_guess'] * steps_per_question
        mistakes_per_step = count_mistakes(dataset)
        for step, n in mistakes_per_step.items():
            accuracy += (step * n)
        results[name][dataset['name']] = accuracy / total_n
    return results


def analyze_easy_additive_experiment(data1_path, data2_path, title1, title2):
    data1 = load_from_json(data1_path)
    data2 = load_from_json(data2_path)

    def extract_results(data):
        information = data['datasets'][-1]  # Easy_Additive
        total = information['n']
        right_eliminations_top1 = len([step for _, step in information['wrong_indices_and_step_of_mistake'] if step == 0])
        right_eliminations_top2 = len([step for _, step in information['wrong_indices_and_step_of_mistake'] if step == 0 or step == 1])
        return right_eliminations_top1 / total, right_eliminations_top2 / total

    result1 = extract_results(data1)
    result2 = extract_results(data2)
    results =  {
        title1: {'top1' :result1[0], 'top2': result1[1]},
        title2: {'top1' :result2[0], 'top2': result2[1]},
    }
    print(results)
    return results


def analyze_results_one_experiment(baseline_path, experiment_path, title):
    baseline_data = load_from_json(baseline_path)
    experiment_data = load_from_json(experiment_path)
    plot_accuracies_one_experiment(baseline_data, experiment_data, title)
    plot_step_of_mistake(experiment_data, title)
    plot_comparison_to_subset_questions_in_baseline(baseline_data, experiment_data, title, "All")
    plot_comparison_to_subset_questions_in_baseline(baseline_data, experiment_data, title, "Right")
    plot_comparison_to_subset_questions_in_baseline(baseline_data, experiment_data, title, "Wrong")
    partial_accuracy = compute_experiment_partial_accuracy(experiment_data)
    print(f'partial_accuracy: {partial_accuracy}')


def analyze_results_two_experiments(baseline_path, experiment1_path, experiment2_path, title1, title2):
    baseline_data = load_from_json(baseline_path)
    experiment1_data = load_from_json(experiment1_path)
    experiment2_data = load_from_json(experiment2_path)
    plot_accuracies_two_experiments(baseline_data, experiment1_data, experiment2_data, title1, title2)


def analyze_results_accuracies(
        baseline_path,
        elimination_path,
        one_shot_elimination_path,
        cot_baseline_path,
        iterative_cot_baseline_path,
    ):
    baseline = load_from_json(baseline_path)
    elimination = load_from_json(elimination_path)
    one_shot_elimination = load_from_json(one_shot_elimination_path)
    cot_baseline = load_from_json(cot_baseline_path)
    iterative_cot_baseline = load_from_json(iterative_cot_baseline_path)
    terms = [d['name'].replace("_", " ") for d in baseline['datasets']]
    baseline_accuracy = [d['accuracy'] for d in baseline['datasets']]
    # Easy_Additive will be analyzed seperatly
    elimination_accuracy = [d['accuracy'] for d in elimination['datasets'] if d['name'] != 'Easy_Additive']
    one_shot_elimination_accuracy = [d['accuracy'] for d in one_shot_elimination['datasets'] if d['name'] != 'Easy_Additive']
    cot_baseline_accuracy = [d['accuracy'] for d in cot_baseline['datasets'] if d['name'] != 'Easy_Additive']
    iterative_cot_baseline_accuracy = [d['accuracy'] for d in iterative_cot_baseline['datasets'] if d['name'] != 'Easy_Additive']

    fig = plt.figure(figsize = (12, 5))
    barWidth = 0.15

    br1 = np.arange(len(baseline_accuracy))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    colors = ListedColormap(plt.cm.Spectral(np.linspace(0.3, 1, 5))).colors
    plt.bar(br1, baseline_accuracy, color=colors[0], width=barWidth, edgecolor='white', label='Baseline')
    plt.bar(br2, cot_baseline_accuracy, color=colors[1], width=barWidth, edgecolor='white', label='COT Elimination')
    plt.bar(br3, iterative_cot_baseline_accuracy, color=colors[2], width=barWidth, edgecolor='white', label='Iterative COT Elimination')
    plt.bar(br4, elimination_accuracy, color=colors[3], width=barWidth, edgecolor='white', label='Zero-Shot Elimination')
    plt.bar(br5, one_shot_elimination_accuracy, color=colors[4], width=barWidth, edgecolor='white', label='One-Shot Elimination')

    plt.xlabel("Datasets")
    plt.ylabel("Accuracy")
    plt.xticks([r + 2 * barWidth for r in range(len(baseline_accuracy))], terms)
    model_name = baseline['name'].removesuffix('-Instruct')
    title = f'{model_name} - multiple accuracies)'
    # plt.title(title)
    plt.legend(loc='lower right')
    fig_title = title.lower().replace(' ', '_')
    plt.savefig(f'{PATH}Plots/{fig_title}.png', dpi=fig.dpi)
    plt.show()


if ANALYZE:
    ############ QWEN ############
    analyze_results_one_experiment(
        'qwen/baseline',
        'qwen/elimination',
        'Zero-Shot Elimination'
    )

    ############ LLAMA ############
    # Baseline vs Zero-Shot
    analyze_results_one_experiment(
        'llama/baseline',
        'llama/elimination',
        'Zero-Shot Elimination'
    )

    # Baseline vs One-Shot
    analyze_results_one_experiment(
        'llama/baseline',
        'llama/one_shot_elimination',
        'One-Shot Elimination'
    )

    # Baseline vs Zero-Shot vs One-Shot
    analyze_results_two_experiments(
        'llama/baseline',
        'llama/elimination',
        'llama/one_shot_elimination',
        'Zero-Shot Elimination',
        'One-Shot Elimination'
    )

    # Baseline vs COT vs Iterative COT
    analyze_results_two_experiments(
        'llama/baseline',
        'llama/cot_baseline',
        'llama/iterative_cot_baseline',
        'Chain-Of-Thought',
        'Iterative Chain-Of-Thought'
    )

    # Easy-Additive
    analyze_easy_additive_experiment(
        'llama/elimination',
        'llama/one_shot_elimination',
        'Zero-Shot Elimination',
        'One-Shot Elimination'
    )

    # Accuracies comparison
    analyze_results_accuracies(
        'llama/baseline',
        'llama/elimination',
        'llama/one_shot_elimination',
        'llama/cot_baseline',
        'llama/iterative_cot_baseline'
    )
