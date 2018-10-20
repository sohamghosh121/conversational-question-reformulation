import torch
import torch.nn.functional as F
from allennlp.modules.encoder_base import _EncoderBase
from allennlp.common import Registrable
from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.nn import util
from allennlp.modules import TimeDistributed
import numpy as np
from torch.autograd import Variable

class ContextualizedQuestionEncoder(_EncoderBase, Registrable):
    pass

"""
    To calculate whether a QA pair n turns ago was a followup:
    - this pair was a followup AND the next pair was a followup
"""

def get_masked_past_qa_pairs(question,
                             question_mask,
                             answer,
                             answer_mask,
                             followup_list,
                             followup_yes_label,
                             batch_size,
                             max_qa_count,
                             max_q_len,
                             max_a_len,
                             n=1):
    emb_sz = question.size(-1)

    question_mask = question_mask.view(batch_size, max_qa_count, max_q_len)
    answer_mask = answer_mask.view(batch_size, max_qa_count, max_a_len)

    total_qa_count = batch_size * max_qa_count
    past_question_mask = F.pad(question_mask, (0, 0, n, 0, 0, 0))[:, :-n, :] \
        .contiguous() \
        .view(total_qa_count, max_q_len)
    past_answer_mask = F.pad(answer_mask, (0, 0, n, 0, 0, 0))[:, :-n, :]\
        .contiguous()\
        .view(total_qa_count, max_a_len)

    followup_list_dialog_wise = followup_list.view(batch_size, max_qa_count, -1)
    followup_mask = (followup_list_dialog_wise == followup_yes_label).byte()

    is_followup_mask = F.pad(followup_mask, (0, 0, 1, 0, 0, 0), value=False)[:, :-1, :]
    for i in range(n - 1):
        # this says that this turn said "SHOULD FOLLOW UP"
        is_followup_mask_ = F.pad(followup_mask, (0, 0, 2 + i, 0, 0, 0), value=False)[:, :-(2 + i), :]
        # this encodes the recurrence to encode a continuous sequence of followups
        is_followup_mask = is_followup_mask & is_followup_mask_  # has to be contiguous
    is_followup_mask = is_followup_mask.float()


    past_answer = answer.view(batch_size, max_qa_count, max_a_len, emb_sz)
    past_answer = F.pad(past_answer, (0, 0, 0, 0, n, 0, 0, 0))
    past_answer = past_answer[:, :-n, :]  # remove the last one
    past_answer = is_followup_mask.unsqueeze(2).expand_as(past_answer) * past_answer
    past_answer = past_answer.view(total_qa_count, max_a_len, emb_sz)

    past_question = question.view(batch_size, max_qa_count, max_q_len, emb_sz)
    past_question = F.pad(past_question, (0, 0, 0, 0, n, 0, 0, 0))
    past_question = past_question[:, :-n, :]
    # zero out turns which were not a followup
    past_question = is_followup_mask.unsqueeze(2).expand_as(
        past_question) * past_question
    past_question = past_question.view(total_qa_count, max_q_len, emb_sz)

    is_followup_mask = is_followup_mask.view(is_followup_mask.size(0) * is_followup_mask.size(1), 1)

    return past_question, past_question_mask, past_answer, past_answer_mask, is_followup_mask

def get_weights_per_turn(method, batch_size, qa_len, num_turn, entropy=None, turn_mask=None):
    if method == 'uniform':
        weights = torch.ones(batch_size, num_turn, 1, 1) / num_turn
        return Variable(weights)
    elif method == 'exponential':
        # for us qa pairs are appended in increasing number of turns away from
        # curr question!
        turn = np.arange(num_turn, 0, -1)
        weight = np.exp(turn) / np.sum(np.exp(turn))
        weight_np = np.tile(weight[None, None, :], (batch_size, qa_len, 1))
        weights = torch.from_numpy(weight_np)
        weights = weights.view(batch_size * qa_len, num_turn, 1, 1)
        return weights
    elif method == 'entropy':
        assert entropy is not None
        neg_entropy = - entropy
        turn_mask = turn_mask[:, :, None].expand_as(entropy)
        weights = util.masked_softmax(neg_entropy, turn_mask)  # softmax across turns
        # remember entropy weighting is at the word level!
        return Variable(weights.view(batch_size * qa_len, num_turn, qa_len, 1))

def bidaf(q: torch.Tensor,
          c: torch.Tensor,
          masks: torch.Tensor,
          att: MatrixAttention,
          mention_scorer
          ):
    """

    :param q: batch of queries (B, M)
    :param c: batch of contexts (B, N)
    :masks c: batch of masks (B, N)
    :param att: attention layer
    :return: tuple of
        q_hat - context aware query representation
        entropy - entropy of attention scores
    """
    # (B, N, M)

    q_c_att = att(q, c) # (B, M, N)
    if mention_scorer is not None:
        mention_scores = mention_scorer(c)  # (B, N)
        mention_scores_ex = mention_scores.squeeze(-1).unsqueeze(1).expand_as(q_c_att)
        q_c_att = q_c_att + mention_scores_ex
    sm_att = util.masked_softmax(q_c_att, masks) # (B, M, N)
    log_sm_att = util.masked_log_softmax(q_c_att, masks) # (B, M, N)
    entropy = torch.sum(- sm_att * log_sm_att, 2) # (B, M)
    q_hat = util.weighted_sum(c, sm_att) # (B, M)
    return q_hat, entropy

class BiAttContext_MultiTurn(ContextualizedQuestionEncoder):
    def __init__(self,
                 num_turns: int,
                 combination: str,
                 input_dim: int,
                 qq_attention: MatrixAttention,
                 qa_attention: MatrixAttention,
                 use_mention_score=False,
                 use_antecedent_score=False):
        super(BiAttContext_MultiTurn, self).__init__()
        self.num_turns = num_turns
        self.combination = combination
        self.qq_attention = qq_attention
        self.qa_attention = qa_attention

        if use_mention_score:
            self.mention_score = TimeDistributed(torch.nn.Linear(input_dim, 1))
        else:
            self.mention_score = None


        if use_antecedent_score:
            self.antecedent_score = TimeDistributed(torch.nn.Sequential(
                torch.nn.Linear(input_dim, 1), torch.nn.Sigmoid()))
        else:
            self.antecedent_score = None

        if self.combination in ['entropy+exponential', 'exponential+entropy']:
            self.entropy_combination_weight = Variable(0, requires_grad=True)

        self.q_hat_enc = TimeDistributed(torch.nn.Linear(input_dim * 3, input_dim))

    # These are now lists -> past_questions, past_answers, ....
    def forward(self, curr_question,
                past_questions,
                past_answers,
                past_q_masks,
                past_ans_masks,
                qa_masks,
                curr_q_mask=None):

        qq_hats, qa_hats, qq_entropies, qa_entropies = [], [], [], []
        i = 1
        for past_question, past_q_mask, past_answer, past_ans_mask in zip(past_questions, past_q_masks, past_answers, past_ans_masks):
            qq_hat, qq_entropy = bidaf(curr_question, past_question, past_q_mask, self.qq_attention, self.mention_score)
            qa_hat, qa_entropy = bidaf(curr_question, past_answer, past_ans_mask, self.qa_attention, self.mention_score)
            qq_hats.append(qq_hat.unsqueeze(1))
            qa_hats.append(qa_hat.unsqueeze(1))
            qq_entropies.append(qq_entropy.unsqueeze(1))
            qa_entropies.append(qa_entropy.unsqueeze(1))
            i += 1

        # each qq_hat, qa_hat -> (B, N_QA, 1, M, EMB_SZ)
        # concatenated, along dim=2 it becomes (B, N_QA, N_TURN, M, EMB_SZ)
        # we need to get weights of size (B, N_QA, N_TURN)
        # multiply , sum along dim=2

        qq_hat_multiturn = torch.cat(qq_hats, dim=1)
        qa_hat_multiturn = torch.cat(qa_hats, dim=1)
        qq_entropy_multiturn = torch.cat(qq_entropies, dim=1)
        qa_entropy_multiturn = torch.cat(qa_entropies, dim=1)

        turn_mask = torch.cat([qa_mask for qa_mask in qa_masks], dim=1)


        if self.combination == 'entropy':
            weights_qq = get_weights_per_turn(self.combination, curr_question.size(0), curr_question.size(1),
                                              self.num_turns, qq_entropy_multiturn, turn_mask)
            weights_qa = get_weights_per_turn(self.combination, curr_question.size(0), curr_question.size(1),
                                              self.num_turns, qa_entropy_multiturn, turn_mask)
        elif self.combination == 'entropy+exponential':
            weights_exp = get_weights_per_turn('exponential', curr_question.size(0), curr_question.size(1),
                                           self.num_turns)
            weights_qq = get_weights_per_turn(self.combination, curr_question.size(0), curr_question.size(1),
                                              self.num_turns, qq_entropy_multiturn, turn_mask)
            weights_qa = get_weights_per_turn(self.combination, curr_question.size(0), curr_question.size(1),
                                              self.num_turns, qa_entropy_multiturn, turn_mask)
            w_entropy = F.sigmoid(self.entropy_combination_weight)  # [0, 1] weighting
            weights_qa = w_entropy * weights_qa + (1 - w_entropy) * weights_exp
            weights_qq = weights_qq + weights_exp
        else:
            weights = get_weights_per_turn(self.combination, curr_question.size(0), curr_question.size(1),
                                           self.num_turns)
            weights_qq = weights
            weights_qa = weights
        if qq_hat_multiturn.is_cuda:
            weights_qq = weights_qq.cuda()
            weights_qa = weights_qa.cuda()

        qq_hat_multiturn_weighted = weights_qq * qq_hat_multiturn
        qa_hat_multiturn_weighted = weights_qa * qa_hat_multiturn

        qq_hat = torch.sum(qq_hat_multiturn_weighted, 1)
        qa_hat = torch.sum(qa_hat_multiturn_weighted, 1)

        q_final = torch.cat([curr_question, qq_hat, qa_hat], dim=2)
        q_final_enc = self.q_hat_enc(q_final)

        if self.antecedent_score is not None:
            a_score = self.antecedent_score(curr_question)
            a_score = a_score.expand_as(q_final_enc)  # (B, 1)
            q_final_enc = a_score * q_final_enc + (1 - a_score) * curr_question  # gated

        return q_final_enc

class BiAttContext_SingleTurn(ContextualizedQuestionEncoder):
    def __init__(self,
                 input_dim: int,
                 qq_attention: MatrixAttention,
                 qa_attention: MatrixAttention,
                 use_mention_score=False,
                 use_antecedent_score=False):
        super(BiAttContext_SingleTurn, self).__init__()
        self.qq_attention = qq_attention
        self.qa_attention = qa_attention
        self.q_hat_enc = TimeDistributed(torch.nn.Linear(input_dim * 3, input_dim))
        if use_mention_score:
            self.mention_score = TimeDistributed(torch.nn.Linear(input_dim, 1))
        else:
            self.mention_score = None

        if use_antecedent_score:
            self.antecedent_score = TimeDistributed(torch.nn.Sequential(
                torch.nn.Linear(input_dim, 1), torch.nn.Sigmoid()))
        else:
            self.antecedent_score = None

    def forward(self, curr_question,
                past_question,
                past_answer,
                past_q_mask,
                past_ans_mask,
                curr_q_mask=None):
        qq_hat, _ = bidaf(curr_question, past_question, past_q_mask, self.qq_attention)
        qa_hat, _ = bidaf(curr_question, past_answer, past_ans_mask, self.qa_attention)
        q_final = torch.cat([curr_question, qq_hat, qa_hat], dim=2)
        q_final_enc = self.q_hat_enc(q_final)

        if self.antecedent_score is not None:
            a_score = self.antecedent_score(curr_question)
            a_score = a_score.expand_as(q_final_enc)
            q_final_enc = a_score * q_final_enc + (1 - a_score) * curr_question  # gated
        return q_final_enc

ContextualizedQuestionEncoder.register('biatt_ctx_single')(BiAttContext_SingleTurn)
ContextualizedQuestionEncoder.register('biatt_ctx_multi')(BiAttContext_MultiTurn)
