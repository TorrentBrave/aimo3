import os
import re
import sys
from collections import defaultdict

import json
import yaml
from termcolor import cprint
from transformers import AutoTokenizer
from imagination_aimo2.local_eval import PythonREPL, AnswerExtractor, PipedModel
from lmdeploy import GenerationConfig

answer_extractor = AnswerExtractor()
python_executor = PythonREPL()
tokenizer = AutoTokenizer.from_pretrained("models/dpsk-qwen-14b-finetune-v1-epoch4-awq")


def read_file(filename, raw=False):
    if raw:
        with open(filename, "r") as rf:
            data = rf.read()
        return data

    if filename.endswith(".yaml"):
        with open(filename, "r") as rf:
            data = yaml.safe_load(rf)
    elif filename.endswith(".json"):
        with open(filename, "r") as rf:
            data = json.load(rf)
    else:
        with open(filename, "r") as rf:
            data = rf.read()
        return data
    return data


def _get_code_answers(code_matches, code_extraction_dir, sample_ind):
    code_answers = []
    for code_ind, code_match in enumerate(code_matches):
        python_code_file = os.path.join(
            code_extraction_dir, f"sample{sample_ind}-code{code_ind}.py"
        )
        exec_output_file = os.path.join(
            code_extraction_dir, f"sample{sample_ind}-code{code_ind}.out"
        )
        if os.path.exists(exec_output_file) and os.path.exists(python_code_file):
            # read cached output from the disk file
            with open(python_code_file, "r") as rf:
                python_code = rf.read()
            exec_success_match = re.search(r"# Execution success: (0|1)", python_code)
            assert exec_success_match
            exec_success = int(exec_success_match.group(1))
            with open(exec_output_file, "r") as rf:
                exec_output = rf.read()
        else:
            # execute the python code
            python_code = code_match.group(1)
            exec_success, exec_output = python_executor(python_code)
            # save to cache
            with open(python_code_file, "w") as wf:
                wf.write(
                    f"# Matched token index: {code_match.start(1)} -"
                    f" {code_match.end(1)}\n"
                    f"# Execution success: {1 if exec_success else 0}"
                    + python_code
                )
            with open(exec_output_file, "w") as wf:
                wf.write(exec_output)

        if exec_success:
            pattern = r"(\d+)(?:\.\d+)?"  # Matches integers or decimals like 468.0
            matches = re.findall(pattern, exec_output)
            if matches:
                code_answer = answer_extractor.canonicalize_number(matches[-1])
                if code_answer is not None:
                    code_answers.append((code_match.end(0), code_answer))

    return code_answers


def extract_solutions_from_result_file(
    ques_ind,
    output_dir,
    early_stop=True,
    early_stop_extract_n=2000,
    first_n_answer_vote=None,
    first_n_stop_sample=None,
):
    output_file = os.path.join(output_dir, "outputs_per_question", f"{ques_ind}.json")
    samples = read_file(output_file)

    code_extraction_dir = os.path.join(
        output_dir, "outputs_per_question", "code_extraction", f"{ques_ind}"
    )
    os.makedirs(code_extraction_dir, exist_ok=True)

    answer_to_solution_map = defaultdict(lambda: [])
    pos_answer_solution_for_question = []
    for sample_ind, sample in enumerate(samples):
        cot_answer_matches = [
            (match.end(0), answer_extractor.canonicalize_number(match.group(1)))
            for match in list(re.finditer(r"oxed{(.*?)}", sample))
            if answer_extractor.canonicalize_number(match.group(1)) is not None
        ]
        code_answer_matches = _get_code_answers(
            list(re.finditer(r"```python\s*(.*?)\s*```", sample, re.DOTALL)),
            code_extraction_dir,
            sample_ind,
        )
        all_answer_matches = sorted(
            [(item[0], item[1], "cot") for item in cot_answer_matches]
            + [(item[0], item[1], "code") for item in code_answer_matches]
        )

        if not all_answer_matches:
            # no answer for this sample
            continue

        if not early_stop:
            # If `early_stop`==False, extract all strings after "</think>" as the candidate solution
            close_think_match = re.search(r"<\/think>", sample)
            if not close_think_match:
                print(
                    f"Don't find </think> in question {ques_ind} sample {sample_ind}!"
                    " Skip this sample."
                )
                continue
            solution = sample[close_think_match.end(0) :]
            answer = all_answer_matches[-1][1]
            if all_answer_matches[-1][2] == "code":
                solution += (
                    f" The execution result of the Python code is \\bboxed{answer}."
                )
            answer_to_solution_map[answer].append(solution)
        else:
            # If `early_stop`==True, extract `early_stop_extract_n` chars before the first answer match position as the candidate solution
            match_pos, answer = all_answer_matches[0][:2]
            solution = sample[max(match_pos - early_stop_extract_n, 0) : match_pos]
            if all_answer_matches[0][2] == "code":
                solution += (
                    f" The execution result of the Python code is \\bboxed{answer}."
                )
            pos_answer_solution_for_question.append(
                (match_pos, sample_ind, answer, solution)
            )

    if early_stop:
        # Only consider the first `first_n_stop_sample` samples that stop.
        if first_n_stop_sample:
            pos_answer_solution_for_question = sorted(pos_answer_solution_for_question)[
                :first_n_stop_sample
            ]
        for _, _, answer, solution in pos_answer_solution_for_question:
            answer_to_solution_map[answer].append(solution)

    print(
        f"Question {ques_ind}: {len(answer_to_solution_map)} possible answers: "
        + ", ".join(
            [
                f"{answer} ({len(solutions)} sols)"
                for answer, solutions in answer_to_solution_map.items()
            ]
        )
    )
    if first_n_answer_vote is not None:
        final_answer_solution_list = sorted(
            list(answer_to_solution_map.items()), key=lambda item: -len(item[1])
        )[:first_n_answer_vote]
        final_answer_solution_list = [
            (answer, sols[0], len(sols)) for answer, sols in final_answer_solution_list
        ]  # let's just choose the first one from the solution list
    else:
        final_answer_solution_list = [
            (answer, sols[0], len(sols))
            for answer, sols in answer_to_solution_map.items()
        ]
    return final_answer_solution_list


def prepare_llm_aggregation_prompt(ques_ind, question, output_dir, save_dir, **kwargs):
    extracted_solution_list = extract_solutions_from_result_file(
        ques_ind, output_dir, **kwargs
    )

    json_save_file = os.path.join(save_dir, f"cand_solutions_{ques_ind}.json")
    with open(json_save_file, "w") as wf:
        json.dump(extracted_solution_list, wf)
    save_file = os.path.join(save_dir, f"prompt_{ques_ind}.txt")

    aggregation_prompt = (
        "Here are multiple candidate solutions for the following math problem: "
        + question
        + "\nPlease analyze what is the error in each solution, and report the final"
        " answer in \\bbox{}. The number of votes enclosed between ** is worth taking"
        " as a reference for the solution's reliability. Candidate solutions:\n"
        + "\n\n".join(
            [
                f"[[Solution {sol_ind}. Answer: {answer}. **{num_vote} votes**]] {sol}"
                for sol_ind, (answer, sol, num_vote) in enumerate(
                    extracted_solution_list
                )
            ]
        )
    )
    with open(save_file, "w") as wf:
        wf.write(aggregation_prompt)
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": aggregation_prompt},
    ], extracted_solution_list


def extract_choices_and_test_llm_aggregation_for_output_dir(
    output_dir, save_suffix, a_model_cfg=None, **kwargs
):
    if not a_model_cfg:
        cfg_file = os.path.join(output_dir, "config.yaml")
    else:
        cfg_file = a_model_cfg
    cfg = read_file(cfg_file)
    ori_results = read_file(os.path.join(output_dir, "results.json"))
    save_dir = os.path.join(output_dir, f"llm_aggregation_{save_suffix}")
    os.makedirs(save_dir, exist_ok=True)

    num_questions = len(ori_results)

    if os.path.exists(os.path.join(save_dir, "output_0.txt")):
        # Load and print
        solution_lists = [
            read_file(os.path.join(save_dir, f"cand_solutions_{ques_ind}.json"))
            for ques_ind in range(num_questions)
        ]
        outputs = [
            read_file(os.path.join(save_dir, f"output_{ques_ind}.txt"))
            for ques_ind in range(num_questions)
        ]
        token_counts = [len(tokenizer(output)["input_ids"]) for output in outputs]
    else:
        # Initialize model, prepare prompts and run the model
        model = PipedModel(**cfg["main_model"])
        # cfg["actor"]["gen_cfg"].update(temperature=0.8)
        cfg["actor"]["gen_cfg"].update(do_sample=False)
        gen_config = GenerationConfig(**cfg["actor"]["gen_cfg"])

        prompts, solution_lists = list(
            zip(
                *[
                    prepare_llm_aggregation_prompt(
                        ques_ind,
                        ori_results[ques_ind]["question"],
                        output_dir,
                        save_dir,
                        **kwargs,
                    )
                    for ques_ind in range(num_questions)
                ]
            )
        )

        outputs = [""] * num_questions
        token_counts = [0] * num_questions
        for response in model.stream_infer(list(prompts), gen_config):
            index = response.index
            token_counts[index] = response.generate_token_len
            outputs[index] += response.text
            if response.finish_reason is not None:
                print(
                    f"Aggregation generation for question {index} ended. Finish reason:"
                    f" {response.finish_reason}."
                )

        for ques_ind in range(num_questions):
            output_path = os.path.join(save_dir, f"output_{ques_ind}.txt")
            with open(output_path, "w") as wf:
                wf.write(outputs[ques_ind])

    # do not use multiple sample now
    aggregated_correct = 0
    change_correct = 0
    for ques_ind, output in enumerate(outputs):
        ori_answer = ori_results[ques_ind]["answer"]
        ori_correct = ori_results[ques_ind]["correct_answer"] == ori_answer

        pattern = r"oxed{(.*?)}"
        matches = re.findall(pattern, output)
        if not matches:
            print(
                f"Question {ques_ind}: Cannot matched answer surrounded in bbox. Format"
                " error? Fallback to use the original aggregated answer."
            )
            aggregated_answer = ori_answer
        else:
            aggregated_answer = answer_extractor.canonicalize_number(matches[-1])

        correct = aggregated_answer == ori_results[ques_ind]["correct_answer"]
        aggregated_correct += correct

        string = (
            f"{ques_ind}: {aggregated_answer} ({token_counts[ques_ind]} tokens). Ori"
            f" aggregated: {ori_answer}. Correct answer:"
            f" {ori_results[ques_ind]['correct_answer']}."
        )

        if not ori_correct and correct:
            color = "green"
            change_correct += 1
        elif ori_correct and not correct:
            color = "red"
            change_correct -= 1
        elif ori_answer != aggregated_answer:
            color = "yellow"
        else:
            color = None
        if color:
            cprint(string, color, end=" ")
        else:
            print(string, end=" ")

        print(
            "Choices: "
            + ", ".join(
                [
                    f"{cand_answer} ({num_vote} votes)"
                    for cand_answer, _, num_vote in solution_lists[ques_ind]
                ]
            )
        )

    print(
        f"Correct num: {aggregated_correct} "
        + (f"(+{change_correct})" if change_correct >= 0 else f"({change_correct})")
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("suffix")
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-extract-n", default=2000, type=int)
    parser.add_argument("--first-n-answer-vote", default=5, type=int)
    parser.add_argument("--first-n-stop-sample", default=None, type=int)
    parser.add_argument(
        "--a-model-cfg",
        default=None,
        help=(
            "The aggregation model's configuration file. `main_model` and"
            " `actor.gen_cfg` fields needed."
        ),
    )
    args = parser.parse_args()
    if not args.early_stop and args.first_n_stop_sample:
        print(
            "WARN: `--early-stop` is not set, `--first_n_stop_sample"
            f" {args.first_n_stop_saple}` will be ignored."
        )

    extract_choices_and_test_llm_aggregation_for_output_dir(
        args.output_dir,
        args.suffix,
        a_model_cfg=args.a_model_cfg,
        early_stop=args.early_stop,
        early_stop_extract_n=args.early_stop_extract_n,
        first_n_answer_vote=args.first_n_answer_vote,
        first_n_stop_sample=args.first_n_stop_sample,
    )
