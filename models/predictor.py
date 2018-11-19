import json
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model


@Predictor.register('dialog_qa_ctx')
class DialogQAPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm')

    def predict(self, jsonline: str) -> JsonDict:
        out = self.predict_json(json.loads(jsonline))
        print('OUT:')
        print(out)
        return out

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects json that looks like the original quac data file.
        """
        paragraph_json = json_dict[0]['paragraphs'][0]
        paragraph = paragraph_json['context']
        paragraph_pos = paragraph_json['context_pos']
        tokenized_paragraph = self._tokenizer.split_words(paragraph)
        qas = paragraph_json['qas']
        metadata = {}
        metadata["instance_id"] = [qa['id'] for qa in qas]
        question_text_list = [qa["question"].strip().replace("\n", "") for qa in qas]
        question_pos_list = [qa["question_pos"] for qa in qas]
        answer_texts_list = [[qa['answer']] for qa in qas]
        answer_pos_list = [paragraph_pos[qa['answer_start']:qa['answer_start'] + len(qa['answer'])]
                           for qa in qas]
        metadata["answer_texts_list"] = answer_texts_list
        metadata["question_tokens"] = [self._tokenizer.split_words(q) for q in question_text_list]
        metadata["question_pos"] = question_pos_list
        metadata["answer_pos"] = answer_pos_list
        metadata["passage_tokens"] = [tokenized_paragraph]
        metadata["predict"] = True
        span_starts_list = [[qa['answer_start']] for qa in qas]
        span_ends_list = []
        for st_list, an_list in zip(span_starts_list, answer_texts_list):
            span_ends = [start + len(answer) for start, answer in zip(st_list, an_list)]
            span_ends_list.append(span_ends)
        yesno_list = [str(qa['yesno']) for qa in qas]
        followup_list = [str(qa['followup']) for qa in qas]
        instance = self._dataset_reader.text_to_instance(question_text_list,
                                                         paragraph,
                                                         span_starts_list,
                                                         span_ends_list,
                                                         tokenized_paragraph,
                                                         yesno_list,
                                                         followup_list,
                                                         metadata)
        return instance
