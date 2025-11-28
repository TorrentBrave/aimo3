import os
import re
import sys
import textwrap
from collections import Counter

import json
import yaml
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
import seaborn as sn
from transformers import AutoTokenizer
from imagination_aimo2.local_eval import PythonREPL, AnswerExtractor

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
        raise
    return data


NO_VALUE = 0
WRONG_VALUE = 1
CORRECT = 2
# EXEC_ERROR = 3
# OTHER_ERROR = 4

IS_CODE_ANSWER = "code"
IS_COT_ANSWER = "cot"

item_to_cmapind = {
    (NO_VALUE, None): 0,
    (WRONG_VALUE, IS_COT_ANSWER): 1,  # dark green
    (CORRECT, IS_COT_ANSWER): 2,  # light green
    (WRONG_VALUE, IS_CODE_ANSWER): 3,  # dark blue
    (CORRECT, IS_CODE_ANSWER): 4,  # light blue
}
cmap = ["#000000", "#124f34", "#38e899", "#132147", "#3c6be8"]


def _pad_value(list_of_list, pad_value):
    num_answers_list = [len(ans_status_list) for ans_status_list in list_of_list]
    max_ans_num = max(num_answers_list)
    return [
        ans_status_list + [pad_value] * (max_ans_num - len(ans_status_list))
        for ans_status_list in list_of_list
    ], num_answers_list


def plot_answer_ratio(answer_stats_for_exp, save_path):
    """Create a stack bar plot."""
    # red: first correct -> final wrong. green: first wrong -> final correct
    fig, ax = plt.subplots()
    colors = ["#000000", "#444444", "#AAAAAA", "#cf1313", "#13cf45", "#EEEEEE"]
    sample_one_answer_q, fc_lc_q, fw_lw_q, fc_lw_q, fw_lc_q, sample_no_answer_q = zip(
        *answer_stats_for_exp
    )

    num_questions = len(sample_one_answer_q)
    xs = range(0, num_questions)
    ax.bar(xs, sample_one_answer_q, color=colors[0])
    bottom = np.array(sample_one_answer_q)

    for data, color in zip(
        [fc_lc_q, fw_lw_q, fc_lw_q, fw_lc_q, sample_no_answer_q], colors[1:]
    ):
        ax.bar(xs, data, color=color, bottom=bottom)
        bottom = bottom + np.array(data)

    ax.set_xticks(range(0, num_questions))
    ax.set_xticklabels([f"q{ind}" for ind in range(0, num_questions)], fontsize=6)

    ax.get_figure().savefig(save_path)
    plt.close()


def get_answers_status_for_question(output_dir, ques_ind, correct_answer):
    answers_status_for_question = []
    earlystop_lengths_for_question = []
    samples = read_file(
        os.path.join(output_dir, "outputs_per_question", f"{ques_ind}.json")
    )

    # save python code and execution output for easy checking
    code_extraction_dir = os.path.join(
        output_dir, "outputs_per_question", "code_extraction", f"{ques_ind}"
    )
    os.makedirs(code_extraction_dir, exist_ok=True)

    for sample_ind, sample in enumerate(samples):
        sample_token_len = len(tokenizer(sample)["input_ids"])

        cot_answers = [
            (
                (
                    match.start(1),
                    match.end(1),
                    len(tokenizer(sample[: match.end(0)])["input_ids"]),
                ),
                answer_extractor.canonicalize_number(match.group(1)),
            )
            for match in list(re.finditer(r"oxed{(.*?)}", sample))
        ]
        # cot_answers_status_for_sample = [OTHER_ERROR if ans is None else int(ans ==  taa    correct_answer) + 1 for ans in cot_answers]
        cot_answers_status_for_sample = [
            (
                ans_item[0],
                int(ans_item[1] == correct_answer) + 1,
                IS_COT_ANSWER,
                ans_item[1],
            )
            for ans_item in cot_answers
            if ans_item[1] is not None
        ]

        code_answers_status_for_sample = []
        for code_ind, code_match in enumerate(
            re.finditer(r"```python\s*(.*?)\s*```", sample, re.DOTALL)
        ):
            python_code = code_match.group(1)
            exec_success, exec_output = python_executor(python_code)

            # Save the python code and exec output
            python_code_file = os.path.join(
                code_extraction_dir, f"sample{sample_ind}-code{code_ind}.py"
            )
            exec_output_file = os.path.join(
                code_extraction_dir, f"sample{sample_ind}-code{code_ind}.out"
            )
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
                    answer = answer_extractor.canonicalize_number(matches[-1])
                    # if answer is None:
                    #     code_answers_status_for_sample.append(OTHER_ERROR)
                    code_answers_status_for_sample.append(
                        (
                            (
                                code_match.start(1),
                                code_match.end(1),
                                len(
                                    tokenizer(sample[: code_match.end(0)])["input_ids"]
                                ),
                            ),
                            int(answer == correct_answer) + 1,
                            IS_CODE_ANSWER,
                            answer,
                        )
                    )
            #     else:
            #         code_answers_status_for_sample.append(OTHER_ERROR)
            # else:
            #     code_answers_status_for_sample.append(EXEC_ERROR)
        answers_status_for_sample = sorted(
            cot_answers_status_for_sample + code_answers_status_for_sample
        )

        # Calculate how much earlier (in token length) the early-stop version get the answer
        # print the token position of the first answer match's ending, the last answer match's ending,
        # and the total token length
        if answers_status_for_sample:
            print(
                answers_status_for_sample[0][0][2],
                answers_status_for_sample[-1][0][2],
                sample_token_len,
            )
            earlystop_lengths_for_question.append(
                (
                    answers_status_for_sample[0][0][2],
                    answers_status_for_sample[-1][0][2],
                    sample_token_len,
                )
            )
        else:
            # no answer, so no early stop
            earlystop_lengths_for_question.append(
                (sample_token_len, sample_token_len, sample_token_len)
            )

        # match start (for answer sorting), WRONG_VALUE/CORRECT, IS_CODE_ANSWER/IS_COT_ANSWER, answer (for heatmap annotation)
        answers_status_for_sample = [
            (item[0][0], item[1], item[2], item[3])
            for item in answers_status_for_sample
        ]
        answers_status_for_question.append(answers_status_for_sample)

    return answers_status_for_question, earlystop_lengths_for_question


def _add_legend(colors, labels, ax, **kwargs):
    ax.legend(
        handles=[
            Patch(facecolor=color, label=label) for color, label in zip(colors, labels)
        ],
        **kwargs,
    )


def plot_answer_token_length(
    answers_status_for_exp, earlystop_lengths_for_exp, c_a_answers_for_exp, save_path
):
    num_samples = len(earlystop_lengths_for_exp[0])
    num_questions = len(earlystop_lengths_for_exp)
    plt.figure()
    df_earlystop_lengths_for_exp = pd.DataFrame(
        [
            {
                "ques_ind": ques_ind,
                "sample_ind": sample_ind,
                "sample_length": lengths_for_sample[2],
                "first_answer_length": lengths_for_sample[0],
                "last_answer_length": lengths_for_sample[1],
            }
            for ques_ind, lengths_for_question in enumerate(earlystop_lengths_for_exp)
            for sample_ind, lengths_for_sample in enumerate(lengths_for_question)
        ]
    )

    # Plot overall sample length's bar
    ax = sn.barplot(
        df_earlystop_lengths_for_exp,
        x="ques_ind",
        y="sample_length",
        hue="sample_ind",
        palette=["#EEEEEE"] * num_samples,
        legend=None,
    )
    # set xticklabels on the first barplot for showing the aggregated and correct answers
    ax.set_xticks(range(num_questions))
    ax.set_xticklabels(
        [
            f"{ques_ind}\n({c_answer},\n{a_answer})"
            for ques_ind, (c_answer, a_answer) in enumerate(c_a_answers_for_exp)
        ],
        fontsize=4,
    )
    for xtick, (c_answer, a_answer) in zip(ax.get_xticklabels(), c_a_answers_for_exp):
        xtick.set_color("g" if c_answer == a_answer else "r")

    # Plot last answer's length's bar; set color based on correctness of the answer
    # light green & light red
    colors = [
        [
            (
                "#EEEEEE"
                if len(answer_status) == 0
                else ("#afeba0" if answer_status[-1][1] == CORRECT else "#ed928c")
            )
            for answer_status in answer_status_for_question
        ]
        for answer_status_for_question in answers_status_for_exp
    ]  # num_questions * num_samples
    colors = list(zip(*colors))  # num_samples * num_questions
    ax = sn.barplot(
        df_earlystop_lengths_for_exp,
        x="ques_ind",
        y="last_answer_length",
        hue="sample_ind",
        palette=["#ADD8E6"] * num_samples,
        legend=None,
    )
    for bar_question, colors_for_a_sample_index in zip(
        ax.containers[-num_samples:], colors
    ):
        # ax.containers will be of length (2*num_samples), each have num_questions bars
        for bar_sample, color_sample in zip(bar_question, colors_for_a_sample_index):
            bar_sample.set_facecolor(color_sample)

    # Plot first answer's length's bar; set color based on correctness of the answer
    # dark green & dark red
    colors = [
        [
            (
                "#EEEEEE"
                if len(answer_status) == 0
                else ("#327521" if answer_status[0][1] == CORRECT else "#b04735")
            )
            for answer_status in answer_status_for_question
        ]
        for answer_status_for_question in answers_status_for_exp
    ]
    colors = list(zip(*colors))  # num_samples * num_questions
    ax = sn.barplot(
        df_earlystop_lengths_for_exp,
        x="ques_ind",
        y="first_answer_length",
        hue="sample_ind",
        palette=["#00008B"] * num_samples,
        legend=None,
    )
    for bar_question, colors_question in zip(ax.containers[-num_samples:], colors):
        # ax.containers will be of length (3*num_samples), each have num_questions bars
        for bar_sample, color_sample in zip(bar_question, colors_question):
            bar_sample.set_facecolor(color_sample)

    # Add customized legend to indicate the meaning of four bar color
    _add_legend(
        ["#327521", "#b04735", "#afeba0", "#ed928c"],
        ["First Correct", "First Wrong", "Last Correct", "Last Wrong"],
        ax,
    )

    plt.savefig(save_path)
    plt.close()


def save_correct_heatmap(output_dirs, plot_every_question=True, reuse_cache=True):
    for output_dir in output_dirs:
        print(f"Handling {output_dir} ...")
        cache_dir = os.path.join(output_dir, "outputs_per_question", "cache_plot")
        os.makedirs(cache_dir, exist_ok=True)

        results = read_file(os.path.join(output_dir, "results.json"))
        num_questions = len(results)

        answer_stats_for_exp = []
        answers_status_for_exp = []
        earlystop_lengths_for_exp = []

        for ques_ind in range(num_questions):
            correct_answer = results[ques_ind]["correct_answer"]
            aggregated_answer = results[ques_ind]["answer"]

            # Some code run for a long time. Caching results for quicker plotting for multiple times
            cache_file_for_question = os.path.join(cache_dir, f"{ques_ind}.json")
            if reuse_cache and os.path.exists(cache_file_for_question):
                loaded = read_file(cache_file_for_question)
                answers_status_for_question, earlystop_lengths_for_question = (
                    loaded["answers_status"],
                    loaded["earlystop_lengths"],
                )
            else:
                answers_status_for_question, earlystop_lengths_for_question = (
                    get_answers_status_for_question(
                        output_dir, ques_ind, correct_answer
                    )
                )
                with open(cache_file_for_question, "w") as wf:
                    json.dump(
                        {
                            "answers_status": answers_status_for_question,
                            "earlystop_lengths": earlystop_lengths_for_question,
                        },
                        wf,
                    )
            answers_status_for_exp.append(answers_status_for_question)

            num_samples = len(answers_status_for_question)

            # Plot answer statusheatmap for this question; record number of valid answers
            plot_answers_status_for_question, _ = _pad_value(
                answers_status_for_question, (None, NO_VALUE, None, None)
            )

            if plot_every_question:
                print(f"Plotting {output_dir}, #{ques_ind} question ...")
                plt.figure()
                ax = sn.heatmap(
                    [
                        [item_to_cmapind[tuple(item[1:3])] for item in list_]
                        for list_ in plot_answers_status_for_question
                    ],
                    # print the answer on each cell, print emtpy string for padding cells
                    annot=[
                        ["" if item[3] is None else str(item[3]) for item in list_]
                        for list_ in plot_answers_status_for_question
                    ],
                    fmt="s",
                    cmap=cmap,
                    vmin=0,
                    vmax=4,
                )
                ax.set_title(
                    f"Question {ques_ind}. Correct: {correct_answer}; Aggregated:"
                    f" {aggregated_answer}\nAll answers: "
                    + "\n".join(
                        textwrap.wrap(
                            ", ".join(
                                [
                                    f"{cand_answer} ({num_vote} vs)"
                                    for cand_answer, num_vote in sorted(
                                        Counter(
                                            [
                                                list_[-1][3]
                                                for list_ in answers_status_for_question
                                                if list_
                                            ]
                                        ).items(),
                                        key=lambda item: -item[1],
                                    )
                                ]
                            ),
                            100,
                        )
                    ),
                    fontdict={"fontsize": 6},
                )
                ax.set_yticks(0.5 + np.arange(0, num_samples))
                ax.tick_params(axis="both", which="minor", labelsize=8)
                ax.set_yticklabels(
                    [str(ind) for ind in range(0, num_samples)], fontsize=6
                )

                ax.set_xticks(
                    0.5 + np.arange(0, len(plot_answers_status_for_question[0]))
                )
                ax.set_xticklabels(
                    [
                        str(num)
                        for num in range(
                            1, len(plot_answers_status_for_question[0]) + 1
                        )
                    ]
                )

                c_bar = ax.collections[0].colorbar
                c_bar.set_ticks([0, 1, 2, 3, 4])
                c_bar.set_ticklabels(
                    ["", "CoT wrong", "CoT correct", "Code wrong", "Code correct"]
                )
                save_path = os.path.join(
                    output_dir, "outputs_per_question", f"{ques_ind}_answer_status.pdf"
                )
                ax.get_figure().savefig(save_path)
                plt.close()

            # stat the number of no answer or one answer
            # for sample that have more than one answer: stat the number of the first answer to be correct; the ratio of the final answer to be correct; the ratio of first correct and final wrong; the ratio of first wrong and final correct
            # the average ratio of all answers to be correct
            sample_no_answer = 0
            sample_one_answer = 0
            fc_lc, fw_lw, fc_lw, fw_lc = 0, 0, 0, 0
            for answers_status_for_sample in answers_status_for_question:
                if len(answers_status_for_sample) == 0:
                    sample_no_answer += 1
                elif len(answers_status_for_sample) == 1:
                    sample_one_answer += 1
                else:
                    fc_lc += int(
                        answers_status_for_sample[0][1] == CORRECT
                        and answers_status_for_sample[-1][1] == CORRECT
                    )
                    fw_lw += int(
                        answers_status_for_sample[0][1] == WRONG_VALUE
                        and answers_status_for_sample[-1][1] == WRONG_VALUE
                    )
                    fc_lw += int(
                        answers_status_for_sample[0][1] == CORRECT
                        and answers_status_for_sample[-1][1] == WRONG_VALUE
                    )
                    fw_lc += int(
                        answers_status_for_sample[0][1] == WRONG_VALUE
                        and answers_status_for_sample[-1][1] == CORRECT
                    )
            answer_stats_for_exp.append(
                [sample_one_answer, fc_lc, fw_lw, fc_lw, fw_lc, sample_no_answer]
            )
            earlystop_lengths_for_exp.append(earlystop_lengths_for_question)

        save_path = os.path.join(output_dir, "answer_refine_vis.pdf")
        print(f"Start plotting answer ratio to {save_path} ...")
        plot_answer_ratio(answer_stats_for_exp, save_path)

        save_path = os.path.join(output_dir, "answer_token_length.pdf")
        print(f"Start plotting answer ratio to {save_path} ...")
        plot_answer_token_length(
            answers_status_for_exp,
            earlystop_lengths_for_exp,
            [(result["correct_answer"], result["answer"]) for result in results],
            save_path,
        )


if __name__ == "__main__":
    save_correct_heatmap(sys.argv[1:])
