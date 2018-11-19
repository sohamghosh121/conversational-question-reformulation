import spacy
import json
import sys

nlp = spacy.load('en_core_web_sm')

if __name__ == '__main__':
    f = sys.argv[1]
    save_name = sys.argv[1].split('.')[0] + '_ling' + '.json'
    j = json.load(open(f))
    for ix in range(len(j['data'])):
        p_tokens = j['data'][ix]['paragraphs'][0]['context']
        p_tokens = ' '.join(p_tokens.split())
        passage = nlp(p_tokens)
        passage_pos = [p.pos_ for p in passage]
        for q_ix in range(len(j['data'][ix]['paragraphs'][0]['qas'])):
            q_tokens = j['data'][ix]['paragraphs'][0]['qas'][q_ix]['question']
            q_tokens = ' '.join(q_tokens.split())
            question = nlp(q_tokens)
            question_pos = [q.pos_ for q in question]
            j['data'][ix]['paragraphs'][0]['qas'][q_ix]['question_pos'] = question_pos
        j['data'][ix]['paragraphs'][0]['context_pos'] = passage_pos
    json.dump(j, open(save_name, 'w'))

