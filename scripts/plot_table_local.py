import os
import sys
import copy
import random
from collections import Counter

from tomark import Tomark
import numpy as np
import json
import yaml

from imagination_aimo2.local_eval import _report_statistics
from imagination_aimo2.local_eval import (
    AllVoteMajorityAggregator,
    AnswerPriorityVoteMajorityAggregator,
)

random.seed(123)

all_voter_1 = AllVoteMajorityAggregator()
all_voter_2 = AnswerPriorityVoteMajorityAggregator("cot", 1, 1, 1)
codep_voter = AnswerPriorityVoteMajorityAggregator("code", 1, 1, 0)
cotp_voter = AnswerPriorityVoteMajorityAggregator("cot", 1, 1, 0)

# NOTE: we assume interleave choose, as we default to interleave code-oriented and cot-oriented prompts
cot_prompt_index_list = list(range(0, 32, 2))
code_prompt_index_list = list(range(1, 32, 2))


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
        raise
    return data


def add_fields_compare_aggregation(
    dct, cfg, stats, results, prompt_index_list=None, field_name_suffix=""
):
    # Compare all-vote V.S. code-priority-vote V.S. cot-priority-vote
    correct_times = np.array([0.0, 0.0, 0.0])
    for result in results:
        cot_answers, code_answers = result["cot_answers"], result["code_answers"]
        if prompt_index_list is not None:
            cot_answers = [cot_answers[ind] for ind in prompt_index_list]
            code_answers = [code_answers[ind] for ind in prompt_index_list]
        a_answer_all1, a_answer_all2, a_answer_codep, a_answer_cotp = [
            voter.aggregate_answer(cot_answers, code_answers)
            for voter in [all_voter_1, all_voter_2, codep_voter, cotp_voter]
        ]
        correct_times += [
            a_answer == result["correct_answer"]
            for a_answer in [a_answer_all1, a_answer_codep, a_answer_cotp]
        ]
    dct["Aggregation All-Vote V.S. CodeP-Vote V.S. CoTP-Vote" + field_name_suffix] = (
        " V.S. ".join([f"{c_t:.1f}" for c_t in correct_times])
    )


def calculate_avg_single_ratios_(_results, prompt_ind_list):
    results = copy.deepcopy(_results)
    for result in results:
        result["cot_answers"] = [result["cot_answers"][ind] for ind in prompt_ind_list]
        result["code_answers"] = [
            result["code_answers"][ind] for ind in prompt_ind_list
        ]
        result["python_code_map_list"] = [
            result["python_code_map_list"][ind] for ind in prompt_ind_list
        ]
        result["code_exec_error_map_list"] = [
            result["code_exec_error_map_list"][ind] for ind in prompt_ind_list
        ]
        result["out_lens"] = [result["out_lens"][ind] for ind in prompt_ind_list]
    sub_stats = _report_statistics(results)
    return [
        np.mean(sub_stats["correct_cot_ratio_list"]),
        np.mean(sub_stats["correct_code_ratio_list"]),
        np.mean(sub_stats["correct_cot_priority_ratio_list"]),
        np.mean(sub_stats["correct_code_priority_ratio_list"]),
    ]


def add_fields_compare_promptlist(
    dct,
    cfg,
    stats,
    results,
    aggregator=all_voter_1,
    field_name_suffix="",
    mix_bootstrap_sample_time=40,
):
    # Using 16 code/cot prompts V.S. 8 code prompts + 8 cot prompts
    correct_times = np.array([0.0, 0.0, 0.0])
    for result in results:
        cot_answers, code_answers = result["cot_answers"], result["code_answers"]
        cot_prompt_cot_answers = [cot_answers[ind] for ind in cot_prompt_index_list]
        cot_prompt_code_answers = [code_answers[ind] for ind in cot_prompt_index_list]
        code_prompt_cot_answers = [cot_answers[ind] for ind in code_prompt_index_list]
        code_prompt_code_answers = [code_answers[ind] for ind in code_prompt_index_list]

        code_prompt_a_answer = aggregator.aggregate_answer(
            code_prompt_cot_answers, code_prompt_code_answers
        )
        cot_prompt_a_answer = aggregator.aggregate_answer(
            cot_prompt_cot_answers, cot_prompt_code_answers
        )
        mix_prompt_a_answers = []
        for _ in range(mix_bootstrap_sample_time):
            sampled_cot_answers, sampled_code_answers = zip(
                *(
                    random.sample(
                        list(zip(cot_prompt_cot_answers, cot_prompt_code_answers)), 8
                    )
                    + random.sample(
                        list(zip(code_prompt_cot_answers, code_prompt_code_answers)), 8
                    )
                )
            )
            mix_prompt_a_answers.append(
                aggregator.aggregate_answer(sampled_cot_answers, sampled_code_answers)
            )
        print(mix_prompt_a_answers)
        correct_answer = result["correct_answer"]
        correct_times += [
            code_prompt_a_answer == correct_answer,
            cot_prompt_a_answer == correct_answer,
            np.mean(np.array(mix_prompt_a_answers) == correct_answer),
        ]

    dct[
        "Aggregated correct number [16 code prompts V.S. 16 cot prompts V.S. 8 cot +"
        " code prompts]"
        + field_name_suffix
    ] = " V.S. ".join([f"{c_t:.1f}" for c_t in correct_times])


def add_fields_statistics(dct, cfg, stats, results):
    # Len and time
    dct["Total solving time"] = stats["total_question_duration"]
    dct["Avg outlen"] = np.mean(stats["avg_out_len_list"])
    # Correct ratios
    _num_result = stats["num_result"]
    _num_sample_list = stats["num_sample_list"]
    if not len(Counter(_num_sample_list)) == 1:
        # number of sample vary. don't support now
        raise
    _num_sample = _num_sample_list[0]
    dct[f"Aggregated correct questions (/{_num_result})"] = stats["aggregated_correct"]
    dct[f"CoT Avg correct samples (/{_num_sample})"] = np.mean(
        np.array(stats["correct_cot_ratio_list"]) * _num_sample
    )
    dct[f"Code Avg correct samples (/{_num_sample})"] = np.mean(
        np.array(stats["correct_code_ratio_list"]) * _num_sample
    )
    dct[f"CoT (Code aux) Avg correct samples (/{_num_sample})"] = np.mean(
        np.array(stats["correct_cot_priority_ratio_list"]) * _num_sample
    )
    dct[f"Code (CoT aux) Avg correct samples (/{_num_sample})"] = np.mean(
        np.array(stats["correct_code_priority_ratio_list"]) * _num_sample
    )
    # Code error breakdown
    avg_no_code_num = np.mean(np.array(stats["no_code_ratio_list"]) * _num_sample)
    avg_code_exec_error_num = np.mean(
        np.array(stats["code_exec_error_ratio_list"]) * _num_sample
    )
    avg_fail_parseint_num = np.mean(
        np.array(stats["answer_wrong_fail_parseint_ratio_list"]) * _num_sample
    )
    avg_wrong_number_num = np.mean(
        np.array(stats["answer_wrong_number_ratio_list"]) * _num_sample
    )
    dct[f"Code error break down (/{_num_sample})"] = (
        f"No code: {avg_no_code_num:.2f}; "
        f"Exec error: {avg_code_exec_error_num:.2f}; "
        f"Fail parseint: {avg_fail_parseint_num:.2f}; "
        f"Wrong number: {avg_wrong_number_num:.2f}"
    )


def to_markdown(output_dirs):
    """
    column_names are like
        "Model", "Quantization", "Gen cfg", "Total solving time",
        "Avg outlen", "Aggregated correct questions (/30)",
        "CoT Avg correct samples (/32)", "Code Avg correct samples (/32)",
        "CoT (code aux) Avg correct samples (/32)", "Code (CoT aux) Avg correct samples (/32)", "Code error break down (/16)"
    ]
    """
    gen_cfg_fields = ["max_new_tokens", "temperature", "repetition_penalty"]
    table = []

    for output_dir in output_dirs:
        if not os.path.exists(os.path.join(output_dir, "statistics.json")):
            print(f"Skip {output_dir} as no `statistics.json` is under this dir")
            continue

        cfg = read_file(os.path.join(output_dir, "config.yaml"))
        stats = read_file(os.path.join(output_dir, "statistics.json"))
        results = read_file(os.path.join(output_dir, "results.json"))
        dct = {}

        # Model
        dct["Model"] = os.path.basename(cfg["main_model"]["model_cfg"]["model_path"])
        # Output dir
        dct["output dir"] = output_dir
        # Quantization. NOTE: not robust, use convention to include "awq" in the model name
        quant_str = "AWQ4 " if "awq" in dct["Model"] else ""
        _quant_policy = cfg["main_model"]["inference_cfg"]["quant_policy"]
        quant_str += (
            "KV16"
            if _quant_policy is None or _quant_policy == 0
            else f"KV{_quant_policy}"
        )
        dct["Quantization"] = quant_str
        # Gen cfg
        dct["Gen cfg"] = "; ".join(
            [
                f"{field_name}: {cfg['actor']['gen_cfg'][field_name]}"
                for field_name in gen_cfg_fields
            ]
        )

        # Stats
        add_fields_statistics(dct, cfg, stats, results)

        # Aggregation ablation
        # add_fields_compare_aggregation(
        #     dct, cfg, stats, results, prompt_index_list=None, field_name_suffix=""
        # )

        # Prompt list ablation (Aggregation acc comparison): 16 Code V.S. 16 CoT V.S. 8 Code + 8 CoT
        # mix_bootstrap_sample_time = 40
        # add_fields_compare_promptlist(
        #     dct,
        #     cfg,
        #     stats,
        #     results,
        #     aggregator=all_voter_1,
        #     mix_bootstrap_sample_time=mix_bootstrap_sample_time,
        # )
        # # Prompt list ablation (Avg single ratio comparison): 16 Code V.S. 16 CoT V.S. 8 Code + 8 CoT
        # avg_single_ratio_16cot = calculate_avg_single_ratios_(
        #     results, cot_prompt_index_list
        # )  # return 4 number
        # avg_single_ratio_16code = calculate_avg_single_ratios_(
        #     results, code_prompt_index_list
        # )
        # # return mix_bootstrap_sample_timex4 numbers
        # avg_single_ratio_8cot8code = np.array(
        #     [
        #         calculate_avg_single_ratios_(
        #             results,
        #             random.sample(cot_prompt_index_list, 8)
        #             + random.sample(code_prompt_index_list, 8),
        #         )
        #         for _ in range(mix_bootstrap_sample_time)
        #     ]
        # )
        # field_prefixes = ["only cot", "only code", "cot (code aux)", "code (cot aux)"]
        # for which_s_ratio in range(4):
        #     field_prefix = field_prefixes[which_s_ratio]
        #     dct.update(
        #         {
        #             f"{field_prefix} 16CoT prompts": (
        #                 f"{avg_single_ratio_16cot[which_s_ratio]:.2f}"
        #             ),
        #             f"{field_prefix} 16Code prompts": (
        #                 f"{avg_single_ratio_16code[which_s_ratio]:.2f}"
        #             ),
        #             f"{field_prefix} 8CoT+8Code prompts (mean)": "{:.2f}".format(
        #                 np.mean(avg_single_ratio_8cot8code[:, which_s_ratio])
        #             ),
        #             f"{field_prefix} 8CoT+8Code prompts (min, 1/4 quantile, 3/4 quantile, max)": (
        #                 "{:.2f} {:.2f} {:.2f} {:.2f}".format(
        #                     np.min(avg_single_ratio_8cot8code[:, which_s_ratio]),
        #                     np.quantile(
        #                         avg_single_ratio_8cot8code[:, which_s_ratio], 0.25
        #                     ),
        #                     np.quantile(
        #                         avg_single_ratio_8cot8code[:, which_s_ratio], 0.75
        #                     ),
        #                     np.max(avg_single_ratio_8cot8code[:, which_s_ratio]),
        #                 )
        #             ),
        #         }
        #     )

        for key in dct:
            if isinstance(dct[key], float):
                dct[key] = f"{dct[key]:.2f}"
        table.append(dct)

    print(f"Processed {len(table)} records.")

    if not table:
        return ""

    markdown = Tomark.table(table)
    return markdown


if __name__ == "__main__":
    print(to_markdown(sys.argv[1:]))
