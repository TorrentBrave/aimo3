from collections import Counter, defaultdict


class AnswerPriorityVoteMajorityAggregator:
    def __init__(
        self, priority="code", highp_weight=1, lowp_pos_weight=1, lowp_neg_weight=0
    ):
        self.priority = priority
        assert self.priority in {"code", "cot"}
        self.highp_weight = highp_weight
        # the weight of the lower-priority answer when the higher-priority answer doesn't exist
        self.lowp_pos_weight = lowp_pos_weight
        # the weight of the lower-priority answer when the higher-priority answer exists
        self.lowp_neg_weight = lowp_neg_weight

    def aggregate_answer(self, cot_answers, code_answers) -> int:
        final_code_answers = [
            list(dct.values())[-1] if dct else None for dct in code_answers
        ]
        final_cot_answers = [
            list(dct.values())[-1] if dct else None for dct in cot_answers
        ]
        answers_and_weights = []
        for code_ans, cot_ans in zip(final_code_answers, final_cot_answers):
            if self.priority == "code":
                if code_ans is not None:
                    answers_and_weights += [
                        (code_ans, self.highp_weight),
                        (cot_ans, self.lowp_neg_weight),
                    ]
                else:
                    answers_and_weights += [(cot_ans, self.lowp_pos_weight)]
            else:  # cot priority
                if cot_ans is not None:
                    answers_and_weights += [
                        (cot_ans, self.highp_weight),
                        (code_ans, self.lowp_neg_weight),
                    ]
                else:
                    answers_and_weights += [(code_ans, self.lowp_pos_weight)]
        aggregated_answers_and_weights = defaultdict(lambda: 0.0)
        for ans, weight in answers_and_weights:
            if ans is None:
                continue
            aggregated_answers_and_weights[ans] += weight

        if not aggregated_answers_and_weights:
            return -1

        # Get the answer with the highest weight
        aggregated_answer = sorted(
            aggregated_answers_and_weights.items(), key=lambda item: -item[1]
        )[0][0]
        return aggregated_answer % 1000


class AllVoteMajorityAggregator:
    @staticmethod
    def aggregate_answer(cot_answers, code_answers) -> int:
        final_code_answers = [
            list(dct.values())[-1] if dct else None for dct in code_answers
        ]
        final_cot_answers = [
            list(dct.values())[-1] if dct else None for dct in cot_answers
        ]
        # Just use all answers to do the vote
        # TODO: if code/cot & outputs length is related to the overall correct ratio.
        # can add some weighting strategy here
        valid_answers = [
            int(ans)
            for ans in final_code_answers + final_cot_answers
            if ans is not None and int(ans) >= 0
        ]
        if not valid_answers:
            return 49
        # New weighting strategy
        weighted_answers = defaultdict(float)
        
        for answer in valid_answers:
            # Default weight is 1.0
            weight = 1.0
            
            # Apply 0.6 weight for answers < 20 or answers that are multiples of 100
            if answer <= 20 or answer % 100 == 0:
                weight = 0.6
                
            weighted_answers[answer] += weight
        
        # Get the answer with the highest weight
        # When weights are equal, prefer the larger answer
        aggregated_answer = sorted(
            weighted_answers.items(), key=lambda item: (-item[1], -item[0])
        )[0][0]
        
        return aggregated_answer % 1000