import os
import re
import sys
import time
import json
import math
import queue
import random
import logging
import argparse
import threading
import subprocess
import tempfile
import shutil
import asyncio
from collections import OrderedDict
from typing import List, Tuple, Dict, Any

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import yaml
import numpy as np
import torch
from transformers import set_seed
import pandas as pd
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

from imagination_aimo2.aggregators import *

## ---- Setup logging utils ---- ##
LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(name)-16s %(levelname)7s: %(message)s"
logging.basicConfig(
    stream=sys.stdout, level=LEVEL, format=LOG_FORMAT, datefmt="%m/%d %I:%M:%S %p"
)
root_logger = logging.getLogger()


def addFile(self, filename):
    handler = logging.FileHandler(filename)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    self.addHandler(handler)


logging.Logger.addFile = addFile


def getLogger(name):
    return root_logger.getChild(name)


LOGGER = getLogger("local_eval")

## ---- Setup environment variables and setups of libararies ---- ##
# Configure environment and settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"
pd.set_option("display.max_colwidth", None)


class AnswerExtractor:
    @staticmethod
    def extract_python_code(text: str) -> List[str]:
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [matches[-1]] if matches else []

    @staticmethod
    def process_python_code(query: str) -> Tuple[str, int]:
        query = "import math\nimport numpy as np\nimport sympy as sp\n" + query
        current_rows = query.strip().split("\n")
        new_rows = []
        new_rows_codes = []

        for row in current_rows:
            stripped_row = row.strip()
            new_rows.append(row)
            if stripped_row and not stripped_row.startswith("#"):
                new_rows_codes.append(stripped_row)

        ans = "\n".join(new_rows)
        return ans

    @staticmethod
    def extract_boxed_text(text: str) -> int:
        pattern = r"oxed{(.*?)}"
        matches = re.findall(pattern, text)
        if not matches:
            return None
        return AnswerExtractor.canonicalize_number(matches[-1])

    @staticmethod
    def canonicalize_number(content: str) -> int:
        if content.isdigit():
            num = int(content)
            if math.isinf(num):
                LOGGER.warn(f"Parsed infinite value from {content}.")
        else:
            nums = re.findall(r"the final answer is.*?(\d+)", content)
            if not nums:
                return None
            num = int(nums[-1])

        return num % 1000


class PythonREPL:
    """Python code execution environment"""

    def __init__(self, timeout=30):
        self.timeout = timeout

    def __call__(self, query: str) -> Tuple[bool, str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "tmp.py")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(query)

            try:
                result = subprocess.run(
                    ["python3", temp_file_path],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return False, f"Execution timed out after {self.timeout} seconds."

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode == 0:
                return True, stdout
            else:
                # Process the error message to remove the temporary file path
                error_lines = stderr.split("\n")
                cleaned_errors = []
                for line in error_lines:
                    if temp_file_path in line:
                        # Remove the path from the error line
                        line = line.replace(temp_file_path, "<temporary_file>")
                    cleaned_errors.append(line)
                cleaned_error_msg = "\n".join(cleaned_errors)
                # Include stdout in the error case
                combined_output = (
                    f"{stdout}\n{cleaned_error_msg}" if stdout else cleaned_error_msg
                )
                return False, combined_output


class PipedModel:
    def __init__(self, model_cfg, inference_cfg):
        # TODO: Currently use all GPUs for one model. Ignore the `model_cfg['gpu_indices']` configuration
        # If need to orchestrate multiple model, this could be changed
        num_gpus = torch.cuda.device_count()

        self.model_path = model_cfg["model_path"]
        self.backend = "turbomind"
        self.backend_config = TurbomindEngineConfig(
            tp=num_gpus,
            session_len=inference_cfg.pop("gen_max_new_tokens")
            + inference_cfg["max_prefill_token_num"]
            + 128,
            **inference_cfg,
        )
        self.pipe = pipeline(self.model_path, self.backend_config)

    def __getattr__(self, name):
        return getattr(self.pipe, name)

    async def _stop_sessions(self, pipe, size):
        """Helper method to stop model sessions"""
        for i in range(0, size):
            await pipe.stop_session(i + 1)

    async def _stop_one_session(self, pipe, session_id):
        """Helper method to stop a single model session"""
        await pipe.stop_session(session_id + 1)


class BasicActor:
    def __init__(
        self,
        cfg,
        model_dict,
        gen_cfg,
        prompt_list,
        prompt_list_combine="concat",
        common_prompt=None,
        answer_extractor=None,
        answer_aggretor=None,
        callback_every_fast_forward=False,
    ):
        # Prepare models
        # `main_model` need to be configured
        assert ("main_model",) == tuple(model_dict.keys())
        self._init_models(model_dict, cfg)
        LOGGER.info("Models initialized successfully.")
        self._model_dict_cfg = model_dict

        # Prepare prompt list
        self.prompt_list = self._init_prompts(
            prompt_list, prompt_list_combine, common_prompt
        )
        self.num_samples = len(self.prompt_list)
        # use "number" in each item in the `prompt_list` configuration to configure number of samples w. that prompt

        # Prepare generation config
        # TODO: we can design strategies to dynamically adjust generation config based on the left time or other information
        self.gen_config = GenerationConfig(**gen_cfg)

        # Initialize Python executor, answer's extractor and aggregator
        # TODO: let's see what we need to configure for these
        self.python_executor = PythonREPL()
        self.answer_extractor = AnswerExtractor()
        self.answer_aggregator = AllVoteMajorityAggregator()

        # Prepare callbacks for `stream_generate`
        self.callbacks = [
            {"upon": "finish stop", "function": self._callback_extract_answer},
            # for testing `callback_every_fast_forward`
            # {"upon": "every 100", "function": self._callback_log_index_and_response}
        ]
        self.callback_every_fast_forward = callback_every_fast_forward

        # Maintain time
        self.question_times = {}
        self.total_time = 0.0
        self.initialized_time = (
            time.time()
        )  # record the current time, for potential early stop strategy based on cutoff time

    @staticmethod
    def _init_prompts(prompt_list, prompt_list_combine, common_prompt):
        assert prompt_list_combine in {"concat", "random", "interleave"}
        common_prompt = common_prompt or {
            "system": "",
            "user_prefix": "",
            "user_suffix": "",
        }
        prompt_expand_lists = [
            [
                (
                    common_prompt.get("system", "") + prompt_choice.get("system", ""),
                    common_prompt.get("user_prefix", "")
                    + prompt_choice.get("user_prefix", ""),
                    prompt_choice.get("user_suffix", "")
                    + common_prompt.get("user_suffix", ""),
                )
            ]
            * prompt_choice.get("number", 1)
            for prompt_choice in prompt_list
        ]
        if prompt_list_combine == "concat":
            prompt_list = sum(prompt_expand_lists, [])
        elif prompt_list_combine == "random":
            prompt_list = sum(prompt_expand_lists, [])
            random.shuffle(prompt_list)
        else:  # interleave
            prompt_list = sum(prompt_expand_lists, [])
            _max_len = max([len(single_list) for single_list in prompt_expand_lists])
            prompt_list = sum(
                [
                    sum(
                        [
                            single_list[ind : ind + 1]
                            for single_list in prompt_expand_lists
                        ],
                        [],
                    )
                    for ind in range(_max_len)
                ],
                [],
            )
        return prompt_list

    def _init_models(self, model_dict, cfg):
        for attr_name, cfg_key_name in model_dict.items():
            model_pipe = PipedModel(**cfg[cfg_key_name])
            setattr(self, attr_name, model_pipe)

    def predict_for_question(self, question: str, id_=None) -> int:
        """Predict answer for a single question"""
        # Start timing this question
        question_start_time = time.time()

        prompts = [
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prefix + question + user_suffix},
            ]
            for system, user_prefix, user_suffix in self.prompt_list
        ]

        # Generate and process model outputs
        (
            cot_answers,
            code_answers,
            out_lens,
            python_code_map_list,
            code_exec_error_map_list,
            outputs,
        ) = self.stream_generate(prompts, self.gen_config, self.callbacks)

        # Combine and select final answer
        aggregated_answer = self.answer_aggregator.aggregate_answer(
            cot_answers, code_answers
        )

        # Print debugging information
        LOGGER.info(f"CoT answers: {cot_answers}")
        LOGGER.info(f"Code answers: {code_answers}")
        LOGGER.info(f"Aggregated answer: {aggregated_answer}")

        # Calculate and store timing information
        question_end_time = time.time()
        question_duration = question_end_time - question_start_time
        self.question_times[id_] = question_duration
        self.total_time += question_duration

        # Print timing information
        LOGGER.info(f"Question {id_} solving time: {question_duration:.2f} seconds")
        LOGGER.info(f"Total solving time so far: {self.total_time:.2f} seconds")

        return (
            aggregated_answer,
            cot_answers,
            code_answers,
            out_lens,
            python_code_map_list,
            code_exec_error_map_list,
            outputs,
            question_duration,
        )

    def stream_generate(self, prompts, gen_config, callbacks):
        # Setup answer's store
        num_prompts = len(prompts)

        outputs = [""] * num_prompts  # Store complete output for each prompt
        token_counts = [0] * num_prompts  # Store token count for each prompt
        yield_counts = [0] * num_prompts  # Store yield count for each prompt
        completed_status = [
            False
        ] * num_prompts  # Flag to mark if each prompt is completed

        # The answer got by match \\bbox.
        # The dict is a mapping FROM the token count when parsing TO the last parsed boxed answer
        cot_answers = [OrderedDict() for _ in range(num_prompts)]
        # The answer got by parsing and executing code.
        # The dict is a mapping FROM the token count when parsing TO the last parsed python code
        code_answers = [OrderedDict() for _ in range(num_prompts)]
        # The Python code parsed.
        # The dict is a mapping FROM the token count when parsing TO the last parsed python code
        python_code_map_list = [OrderedDict() for _ in range(num_prompts)]
        # Python code execution errors.
        # The dict is a mapping FROM token count when parsing TO the execution error string
        code_exec_error_map_list = [OrderedDict() for _ in range(num_prompts)]

        # Group callback functions by criteria
        callbacks_on_finish = [
            (
                re.match("finish (.+)", callback_item["upon"]).group(1),
                callback_item["function"],
            )
            for callback_item in self.callbacks
            if callback_item["upon"].startswith("finish")
        ]
        callbacks_every_n = [
            (
                int(re.match(r"every (\d+)", callback_item["upon"]).group(1)),
                callback_item["function"],
            )
            for callback_item in self.callbacks
            if callback_item["upon"].startswith("every")
        ]
        if callbacks_every_n and self.callback_every_fast_forward:
            last_callback_yield_counts = [
                [0] * num_prompts for _ in range(len(callbacks_every_n))
            ]

        iterator = self.main_model.stream_infer(prompts, gen_config)

        resp_queue = queue.Queue()
        _END = object()

        def put_into_queue(_queue):
            try:
                for item in iterator:
                    _queue.put(item)
            finally:
                _queue.put(_END)

        thread = threading.Thread(target=put_into_queue, args=(resp_queue,))
        thread.start()
        last_response_list = [None] * num_prompts
        while 1:
            try:
                response = resp_queue.get_nowait()
                if response is _END:
                    break
                index = response.index
                token_counts[index] = response.generate_token_len
                yield_counts[index] += 1
                if response.text is not None:
                    outputs[index] += response.text
                if not self.callback_every_fast_forward:
                    # Call callback functions
                    for _every_n, callback_func in callbacks_every_n:
                        if yield_counts[index] % _every_n == 0:
                            callback_func(
                                index,
                                outputs,
                                cot_answers,
                                code_answers,
                                token_counts,
                                python_code_map_list,
                                code_exec_error_map_list,
                                self.main_model,
                                response,
                            )
                else:
                    last_response_list[index] = response
                if response.finish_reason is not None:
                    completed_status[index] = True
                    for callback_finish_reason, callback_func in callbacks_on_finish:
                        if callback_finish_reason == response.finish_reason:
                            callback_func(
                                index,
                                outputs,
                                cot_answers,
                                code_answers,
                                token_counts,
                                python_code_map_list,
                                code_exec_error_map_list,
                                self.main_model,
                                response,
                            )
            except queue.Empty:
                if self.callback_every_fast_forward:
                    for index in range(num_prompts):
                        if completed_status[index] or last_response_list[index] is None:
                            continue
                        for callback_ind, (_every_n, callback_func) in enumerate(
                            callbacks_every_n
                        ):
                            if (
                                yield_counts[index]
                                - last_callback_yield_counts[callback_ind][index]
                                >= _every_n
                            ):
                                LOGGER.info(
                                    f"{index}, {yield_counts[index]},"
                                    f" {last_callback_yield_counts[callback_ind][index]}"
                                )
                                last_callback_yield_counts[callback_ind][index] = (
                                    yield_counts[index]
                                )
                                callback_func(
                                    index,
                                    outputs,
                                    cot_answers,
                                    code_answers,
                                    token_counts,
                                    python_code_map_list,
                                    code_exec_error_map_list,
                                    self.main_model,
                                    last_response_list[index],
                                )

        # TODO: if a callback flag stop. just rush to the final generated token, and try one parse.

        return (
            cot_answers,
            code_answers,
            token_counts,
            python_code_map_list,
            code_exec_error_map_list,
            outputs,
        )

    def _callback_log_index_and_response(
        self,
        index,
        outputs,
        cot_answers,
        code_answers,
        token_counts,
        python_code_map_list,
        code_exec_error_map_list,
        model,
        response,
    ):
        LOGGER.info(f"{index}, {response}")

    ## ----- The answer parsing callback ----
    def _callback_extract_answer(
        self,
        index,
        outputs,
        cot_answers,
        code_answers,
        token_counts,
        python_code_map_list,
        code_exec_error_map_list,
        model,
        response,
    ):
        """Try to parse the cot & code answers and populate into the lists."""
        cur_token_count = token_counts[index]
        # Try to process cot output
        cot_answer = self._try_parse_boxed_answer(outputs[index])
        if cot_answer:
            self._update_map_when_different_from_the_last(
                cot_answers[index], cur_token_count, cot_answer
            )

        # Try to process code output
        code_answer, code_exec_error, python_code = self._try_parse_code_answer(
            outputs[index]
        )
        if code_answer:
            self._update_map_when_different_from_the_last(
                code_answers[index], cur_token_count, code_answer
            )
        if python_code:
            if (
                not python_code_map_list[index]
                or list(python_code_map_list[index].values())[-1] != python_code
            ):
                python_code_map_list[index][cur_token_count] = python_code
                code_exec_error_map_list[index][cur_token_count] = code_exec_error

    def _try_parse_boxed_answer(self, text):
        return self.answer_extractor.extract_boxed_text(text)

    def _try_parse_code_answer(self, text):
        answer, code_exec_error, python_code = None, None, None

        # Try to extract a full Python code
        python_codes = self.answer_extractor.extract_python_code(text)
        if python_codes:
            # Only execute the last one
            # currently, the answer extractor actually only return the last matched code
            python_code = self.answer_extractor.process_python_code(python_codes[-1])
            exec_success, exec_output = self.python_executor(python_code)
            if exec_success:
                pattern = r"(\d+)(?:\.\d+)?"  # Matches integers or decimals like 468.0
                matches = re.findall(pattern, exec_output)
                if matches:
                    answer = self.answer_extractor.canonicalize_number(matches[-1])
            else:
                code_exec_error = exec_output

        return answer, code_exec_error, python_code

    @staticmethod
    def _update_map_when_different_from_the_last(map_, cur_token_count, new_value):
        if not map_ or list(map_.values())[-1] != new_value:
            map_[cur_token_count] = new_value


class EarlyStopActor(BasicActor):
    def __init__(
        self,
        cfg,
        model_list,
        prompt_list,
        common_prompt,
        early_stop_strategy,
        answer_extractor,
        answer_aggretor,
    ):
        super().__init__(
            cfg,
            model_list,
            prompt_list,
            common_prompt,
            answer_extractor,
            answer_aggretor,
        )
        self.early_stop_stategy = early_stop_strategy
        self.callbacks.append(
            {
                "upon": f"every {self.early_stop_stategy['check_every']}",
                "function": self._callback_check_early_stop,
            }
        )

    def _callback_check_early_stop(self):
        # TODO
        pass


def _report_statistics(results, log_each_question=False):
    no_code_ratio_list = []
    code_exec_error_ratio_list = []
    answer_wrong_fail_parseint_ratio_list = []
    answer_wrong_number_ratio_list = []

    correct_cot_ratio_list = []
    correct_code_ratio_list = []
    correct_cot_priority_ratio_list = []
    correct_code_priority_ratio_list = []

    avg_out_len_list = []

    aggregated_correct = 0.0

    num_result = len(results)
    num_sample_list = []
    for result_item in results:
        (
            id_,
            question,
            answer,
            cot_answers,
            code_answers,
            out_lens,
            correct_answer,
            python_code_map_list,
            code_exec_error_map_list,
        ) = [
            result_item[key]
            for key in [
                "id",
                "question",
                "answer",
                "cot_answers",
                "code_answers",
                "out_lens",
                "correct_answer",
                "python_code_map_list",
                "code_exec_error_map_list",
            ]
        ]

        final_code_answers = [
            list(dct.values())[-1] if dct else None for dct in code_answers
        ]
        final_cot_answers = [
            list(dct.values())[-1] if dct else None for dct in cot_answers
        ]

        # Pick the corresponding python code of the final code answer
        # if there are no code answer, then try to see if there are any code
        corresponding_python_codes = []
        corresponding_code_exec_errors = []
        for code_answer_dct, python_code_dct, code_err_dct in zip(
            code_answers, python_code_map_list, code_exec_error_map_list
        ):
            if code_answer_dct:
                token_id = list(code_answer_dct.keys())[-1]
                corresponding_python_codes.append(python_code_dct[token_id])
                corresponding_code_exec_errors.append(code_err_dct[token_id])
            elif python_code_dct:
                corresponding_python_codes.append(list(python_code_dct.values())[-1])
                corresponding_code_exec_errors.append(list(code_err_dct.values())[-1])
            else:
                corresponding_python_codes.append(None)
                corresponding_code_exec_errors.append(None)

        final_code_or_cot_answers = [
            cot_a if code_a is None else code_a
            for cot_a, code_a in zip(final_cot_answers, final_code_answers)
        ]
        final_cot_or_code_answers = [
            code_a if cot_a is None else cot_a
            for cot_a, code_a in zip(final_cot_answers, final_code_answers)
        ]
        num_samples = len(final_code_answers)
        num_sample_list.append(num_samples)

        # break down error to no code, or code wrong execution, or answer wrong
        num_no_code = 0.0
        num_code_exec_error = 0.0
        num_answer_wrong_fail_parseint = 0.0
        num_answer_wrong_number = 0.0
        for code_answer, corr_python_code, corr_code_exec_error in zip(
            final_code_answers,
            corresponding_python_codes,
            corresponding_code_exec_errors,
        ):
            if code_answer == correct_answer:
                continue
            if corr_python_code is None:
                num_no_code += 1
            elif corr_code_exec_error:
                num_code_exec_error += 1
            else:
                if code_answer is None:
                    # fail to parse a valid integer from the output
                    num_answer_wrong_fail_parseint += 1
                else:
                    assert isinstance(code_answer, int)
                    num_answer_wrong_number += 1

        no_code_ratio_list.append(num_no_code / float(num_samples))
        code_exec_error_ratio_list.append(num_code_exec_error / float(num_samples))
        answer_wrong_fail_parseint_ratio_list.append(
            num_answer_wrong_fail_parseint / float(num_samples)
        )
        answer_wrong_number_ratio_list.append(
            num_answer_wrong_number / float(num_samples)
        )

        single_correct_cot_ratio = sum(
            np.array(final_cot_answers) == correct_answer
        ) / float(num_samples)
        correct_cot_ratio_list.append(single_correct_cot_ratio)
        single_correct_code_ratio = sum(
            np.array(final_code_answers) == correct_answer
        ) / float(num_samples)
        correct_code_ratio_list.append(single_correct_code_ratio)
        single_correct_cot_priority_ratio = sum(
            np.array(final_cot_or_code_answers) == correct_answer
        ) / float(num_samples)
        correct_cot_priority_ratio_list.append(single_correct_cot_priority_ratio)
        single_correct_code_priority_ratio = sum(
            np.array(final_code_or_cot_answers) == correct_answer
        ) / float(num_samples)
        correct_code_priority_ratio_list.append(single_correct_code_priority_ratio)

        single_avg_out_len = np.mean(out_lens)
        avg_out_len_list.append(single_avg_out_len)

        aggregated_correct += answer == correct_answer

        if log_each_question:
            code_error_breakdown_log_string = (
                f"Total {num_samples}\n\t- No code: {num_no_code}"
                f"\n\t- Exec error: {num_code_exec_error}"
                f"\n\t- Fail to parse int: {num_answer_wrong_fail_parseint}"
                f"\n\t- Wrong number: {num_answer_wrong_number}"
            )
            LOGGER.info(
                f"[Question id]: {id_}\n[Question]: {question}\n[Correct Answer]:"
                f" {correct_answer}\n[Aggregated Answer]: {answer}\n[CoT correct"
                f" ratio]: {single_correct_cot_ratio}; [CoT with Code-results as aux"
                f" correct ratio]: {single_correct_cot_priority_ratio}\n[Code correct"
                f" ratio]: {single_correct_code_ratio}; [Code with CoT-results as aux"
                f" correct ratio]: {single_correct_code_priority_ratio}\n[Average out"
                f" len]: {single_avg_out_len}\n[Code error break down]"
                f" {code_error_breakdown_log_string}"
            )

    aggregated_correct_ratio = aggregated_correct / num_result
    avg_single_correct_cot_ratio = np.mean(correct_cot_ratio_list)
    avg_single_correct_code_ratio = np.mean(correct_code_ratio_list)
    avg_single_correct_cot_priority_ratio = np.mean(correct_cot_priority_ratio_list)
    avg_single_correct_code_priority_ratio = np.mean(correct_code_priority_ratio_list)
    avg_single_avg_out_len = np.mean(avg_out_len_list)

    if "question_duration" in results[0]:
        question_duration_list = [result["question_duration"] for result in results]
        total_question_duration = np.sum(question_duration_list)
    else:
        # To be compatible with the dumped results.json by the old code
        question_duration_list = None
        total_question_duration = None

    avg_no_code_ratio = np.mean(no_code_ratio_list)
    avg_code_exec_error_ratio = np.mean(code_exec_error_ratio_list)
    avg_answer_wrong_fail_parseint_ratio = np.mean(
        answer_wrong_fail_parseint_ratio_list
    )
    avg_answer_wrong_number_ratio = np.mean(answer_wrong_number_ratio_list)

    code_error_breakdown_log_string = (
        f"\n\t- No code: {avg_no_code_ratio}"
        f"\n\t- Exec error: {avg_code_exec_error_ratio}"
        f"\n\t- Fail to parse int: {avg_answer_wrong_fail_parseint_ratio}"
        f"\n\t- Wrong number: {avg_answer_wrong_number_ratio}"
    )
    LOGGER.info(
        f"[Aggregated answer's correct ratio]: {aggregated_correct_ratio}\n[Average"
        f" single correct CoT ratio]: {avg_single_correct_cot_ratio}\n[Average"
        f" single correct Code ratio]: {avg_single_correct_code_ratio}\n[Average"
        " single correct CoT (with code aux) ratio]:"
        f" {avg_single_correct_cot_priority_ratio}\n[Average single correct Code"
        f" (with cot aux) ratio]: {avg_single_correct_code_priority_ratio}\n[Average"
        f" out len]: {avg_single_avg_out_len}\n"
        f"[Code error break down - ratio] {code_error_breakdown_log_string}"
    )

    # Store statistics for easy checking
    statistics_dct = {
        "no_code_ratio_list": no_code_ratio_list,
        "code_exec_error_ratio_list": code_exec_error_ratio_list,
        "answer_wrong_fail_parseint_ratio_list": answer_wrong_fail_parseint_ratio_list,
        "answer_wrong_number_ratio_list": answer_wrong_number_ratio_list,
        "correct_cot_ratio_list": correct_cot_ratio_list,
        "correct_code_ratio_list": correct_code_ratio_list,
        "correct_cot_priority_ratio_list": correct_cot_priority_ratio_list,
        "correct_code_priority_ratio_list": correct_code_priority_ratio_list,
        "avg_out_len_list": avg_out_len_list,
        "aggregated_correct": aggregated_correct,
        "num_sample_list": num_sample_list,
        "num_result": num_result,
        "total_question_duration": total_question_duration,
        "question_duration_list": question_duration_list,
    }
    return statistics_dct


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file", type=str, help="Path to the config file")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--exam-dataset-files", type=str, required=True)
    parser.add_argument("--seed", default=None, type=int)
    args = parser.parse_args()

    if args.seed is not None:
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        set_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    exam_dataset_files = args.exam_dataset_files.split(",")

    with open(args.cfg_file, "r") as rf:
        cfg = yaml.safe_load(rf)

    # Prepare output directory, dump the config, open the log file
    LOGGER.info(f"Preapring the output directory {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    outputs_perq_path = os.path.join(args.output_path, "outputs_per_question")
    os.makedirs(outputs_perq_path, exist_ok=True)

    cfg["seed"] = args.seed
    cfg["exam_dataset_files"] = exam_dataset_files
    with open(os.path.join(args.output_path, "config.yaml"), "w") as wf:
        yaml.safe_dump(cfg, wf)
    LOGGER.addFile(os.path.join(args.output_path, "eval.log"))
    shutil.copy(
        os.path.abspath(__file__), os.path.join(args.output_path, "local_eval.py")
    )

    LOGGER.info("%s", cfg)

    actor_cls_name = cfg["actor"].pop("actor_cls", "BasicActor")
    actor = globals()[actor_cls_name](cfg, **cfg["actor"])

    # Create list to store results and outputs
    results = []
    outputs_list = []

    for exam_dataset_file in exam_dataset_files:
        LOGGER.info(f"Processing file: {exam_dataset_file}")
        # Read CSV file using pandas
        df = pd.read_csv(exam_dataset_file)

        # Process each row
        for question_index, row in df.iterrows():
            id_ = row["id"]
            question = row["question"]
            correct_answer = row.get(
                "answer", None
            )  # Get answer if available, otherwise None

            LOGGER.info(
                f"Prediction for question {id_}: {question}."
                + (
                    f" Correct answer: {correct_answer}."
                    if correct_answer is not None
                    else ""
                )
            )

            # Predict answer
            (
                answer,
                cot_answers,
                code_answers,
                out_lens,
                python_code_map_list,
                code_exec_error_map_list,
                outputs,
                question_duration,
            ) = actor.predict_for_question(question, id_)

            # Store result
            results.append(
                {
                    "id": id_,
                    "question": question,
                    "answer": answer,
                    "cot_answers": cot_answers,
                    "code_answers": code_answers,
                    "out_lens": out_lens,
                    "python_code_map_list": python_code_map_list,
                    "code_exec_error_map_list": code_exec_error_map_list,
                    "correct_answer": correct_answer,
                    "question_duration": question_duration,
                }
            )
            with open(
                os.path.join(outputs_perq_path, f"{question_index}.json"), "w"
            ) as wf:
                json.dump(outputs, wf)

    LOGGER.info("Processing complete.")

    statistics = _report_statistics(results, log_each_question=True)
    result_path = args.output_path
    LOGGER.info(f"Dumping results to {result_path}   ...")
    with open(os.path.join(result_path, "results.json"), "w") as wf:
        json.dump(results, wf)
    with open(os.path.join(result_path, "statistics.json"), "w") as wf:
        json.dump(statistics, wf)
