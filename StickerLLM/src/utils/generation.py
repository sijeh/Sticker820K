import torch
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.stopping_criteria import StoppingCriteria


class InvalidScoreLogitsProcessor(LogitsProcessor):

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 20005] = 5e4
        return scores


class RetrievalProbLogitsProcessor(LogitsProcessor):

    def __init__(self, ret_token_id, prob_scale) -> None:
        super().__init__()
        self.ret_token_id = ret_token_id
        self.prob_scale = prob_scale

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[..., self.ret_token_id] = self.prob_scale * scores[..., self.ret_token_id]
        return scores
    

class InvalidIdProcessor(LogitsProcessor):

    def __init__(self, max_id) -> None:
        super().__init__()
        self.max_id = max_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[..., self.max_id+1: ] = 0.0
        return scores


class RetrievalEndLogitsProcessor(LogitsProcessor):

    def __init__(self, ret_token_id, retend_token_id) -> None:
        super().__init__()
        self.ret_token_id = ret_token_id
        self.retend_token_id = retend_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids[0, -1] == self.ret_token_id:
            scores[..., self.retend_token_id] = scores.max() + 10.0
        return scores



class RetrievalStoppingCriteria(StoppingCriteria):

    def __init__(self, ret_token_id) -> None:
        super().__init__()
        self.ret_token_id = ret_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0, -2] == self.ret_token_id