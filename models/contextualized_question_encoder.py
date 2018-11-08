import torch
import torch.nn.functional as F
from allennlp.modules.encoder_base import _EncoderBase
from allennlp.common import Registrable
from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn import util
from allennlp.modules import TimeDistributed
import numpy as np
from torch.autograd import Variable

from allennlp.modules.seq2seq_encoders import StackedBidirectionalLstm

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
        weights = torch.ones(batch_size, num_turn, qa_len, 1) / num_turn
        return Variable(weights)
    elif method == 'exponential':
        # for us qa pairs are appended in increasing number of turns away from
        # curr question!
        turn = np.arange(num_turn, 0, -1)
        weight = np.exp(turn) / np.sum(np.exp(turn))
        weight_np = np.tile(weight[None, :, None], (batch_size, 1, qa_len))
        weights = torch.from_numpy(weight_np.astype(np.float32))
        weights = weights.unsqueeze(-1)
        return weights
    elif method == 'entropy':
        assert entropy is not None
        neg_entropy = - entropy
        turn_mask = turn_mask[:, :, None].expand_as(entropy)
        weights = util.masked_softmax(neg_entropy, turn_mask, 1)  # softmax across turns
        # remember entropy weighting is at the word level!
        return Variable(weights.view(batch_size, num_turn, qa_len, 1))

def bidaf(curr_q: torch.Tensor,
          past_ctx_enc: torch.Tensor,
          past_ctx_emb: torch.Tensor,
          masks: torch.Tensor,
          att: MatrixAttention,
          ant_scorer
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

    q_c_att = att(curr_q, past_ctx_enc) # (B, M, N)
    if ant_scorer is not None:
        ant_scores = ant_scorer(past_ctx_enc)  # (B, N)
        ant_scores_ex = ant_scores.squeeze(-1).unsqueeze(1).expand_as(q_c_att)
        q_c_att = q_c_att + ant_scores_ex
    sm_att = util.masked_softmax(q_c_att, masks)  # (B, M, N)
    log_sm_att = util.masked_log_softmax(q_c_att, masks) # (B, M, N)
    entropy = torch.sum(- sm_att * log_sm_att, 2)  # (B, M)
    # print(past_ctx_emb.size(), sm_att.size())
    q_hat = util.weighted_sum(past_ctx_emb, sm_att)  # (B, M)
    return q_hat, entropy, sm_att

class BiAttContext_MultiTurn(ContextualizedQuestionEncoder):
    def __init__(self,
                 num_turns: int,
                 combination: str,
                 qq_attention: MatrixAttention,
                 qa_attention: MatrixAttention,
                 coref_layer: Seq2SeqEncoder,
                 use_mention_score=False,
                 use_antecedent_score=False):
        super(BiAttContext_MultiTurn, self).__init__()
        self.num_turns = num_turns
        self.combination = combination
        self.qq_attention = qq_attention
        self.qa_attention = qa_attention
        self._coref_layer = coref_layer

        coref_output_dim = self._coref_layer.get_output_dim()
        coref_input_dim = self._coref_layer.get_input_dim()

        if use_mention_score:
            self.mention_score = TimeDistributed(torch.nn.Linear(coref_output_dim, 1))
        else:
            self.mention_score = None


        if use_antecedent_score:
            self.antecedent_score = TimeDistributed(torch.nn.Sequential(
                torch.nn.Linear(coref_output_dim, 1), torch.nn.Sigmoid()))
        else:
            self.antecedent_score = None

        if self.combination == 'entropy+exponential':
            if torch.cuda.is_available():
                self.entropy_combination_weight = torch.nn.Parameter(torch.cuda.FloatTensor(1), requires_grad=True)
            else:
                self.entropy_combination_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.q_hat_enc = TimeDistributed(torch.nn.Linear(coref_input_dim * 3, coref_input_dim))

    # These are now lists -> past_questions, past_answers, ....
    def forward(self, curr_question,
                past_questions,
                past_answers,
                past_q_masks,
                past_ans_masks,
                qa_masks,
                curr_q_mask):
        qq_hats, qa_hats, qq_entropies, qa_entropies, sm_att_as, sm_att_qs = [], [], [], [], [], []
        i = 1

        curr_question_enc = self._coref_layer(curr_question, curr_q_mask)

        for past_question, past_q_mask, past_answer, past_ans_mask in zip(past_questions, past_q_masks, past_answers, past_ans_masks):
            past_question_enc = self._coref_layer(past_question, past_q_mask)
            past_answer_enc = self._coref_layer(past_answer, past_ans_mask)

            qq_hat, qq_entropy, sm_att_q = bidaf(curr_question_enc,
                                                 past_question_enc,
                                                 past_question,
                                                 past_q_mask,
                                                 self.qq_attention,
                                                 self.antecedent_score)
            qa_hat, qa_entropy, sm_att_a = bidaf(curr_question_enc,
                                                 past_answer_enc,
                                                 past_answer,
                                                 past_ans_mask,
                                                 self.qa_attention,
                                                 self.antecedent_score)

            qq_hats.append(qq_hat.unsqueeze(1))
            qa_hats.append(qa_hat.unsqueeze(1))
            qq_entropies.append(qq_entropy.unsqueeze(1))
            qa_entropies.append(qa_entropy.unsqueeze(1))
            sm_att_qs.append(sm_att_q)
            sm_att_as.append(sm_att_a)
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
            weights_qq = get_weights_per_turn('entropy', curr_question.size(0), curr_question.size(1),
                                              self.num_turns, qq_entropy_multiturn, turn_mask)
            weights_qa = get_weights_per_turn('entropy', curr_question.size(0), curr_question.size(1),
                                              self.num_turns, qa_entropy_multiturn, turn_mask)
            w_entropy = torch.sigmoid(self.entropy_combination_weight)  # [0, 1] weighting

            weights_qa = w_entropy * weights_qa + (1 - w_entropy) * weights_exp
            weights_qq = w_entropy * weights_qq + (1 - w_entropy) * weights_exp
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
            a_score = self.antecedent_score(curr_question_enc)
            a_score = a_score.expand_as(q_final_enc)  # (B, 1)
            q_final_enc = a_score * q_final_enc + (1 - a_score) * curr_question  # gated

        return q_final_enc, weights_qq, weights_qa, sm_att_qs, sm_att_as

ContextualizedQuestionEncoder.register('biatt_ctx_multi')(BiAttContext_MultiTurn)
