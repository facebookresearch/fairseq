# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
import json


def convert_sentence_to_json(sentence):
    if '_' in sentence:
        prefix, rest = sentence.split('_', 1)
        query, rest = rest.split('_', 1)
        query_index = len(prefix.rstrip().split(' '))
    else:
        query, query_index = None, None

    prefix, rest = sentence.split('[', 1)
    pronoun, rest = rest.split(']', 1)
    pronoun_index = len(prefix.rstrip().split(' '))

    sentence = sentence.replace('_', '').replace('[', '').replace(']', '')

    return {
        'idx': 0,
        'text': sentence,
        'target': {
            'span1_index': query_index,
            'span1_text': query,
            'span2_index': pronoun_index,
            'span2_text': pronoun,
        },
    }


def extended_noun_chunks(sentence):
    noun_chunks = {(np.start, np.end) for np in sentence.noun_chunks}
    np_start, cur_np = 0, 'NONE'
    for i, token in enumerate(sentence):
        np_type = token.pos_ if token.pos_ in {'NOUN', 'PROPN'} else 'NONE'
        if np_type != cur_np:
            if cur_np != 'NONE':
                noun_chunks.add((np_start, i))
            if np_type != 'NONE':
                np_start = i
            cur_np = np_type
    if cur_np != 'NONE':
        noun_chunks.add((np_start, len(sentence)))
    return [sentence[s:e] for (s, e) in sorted(noun_chunks)]


def find_token(sentence, start_pos):
    found_tok = None
    for tok in sentence:
        if tok.idx == start_pos:
            found_tok = tok
            break
    return found_tok


def find_span(sentence, search_text, start=0):
    search_text = search_text.lower()
    for tok in sentence[start:]:
        remainder = sentence[tok.i:].text.lower()
        if remainder.startswith(search_text):
            len_to_consume = len(search_text)
            start_idx = tok.idx
            for next_tok in sentence[tok.i:]:
                end_idx = next_tok.idx + len(next_tok.text)
                if end_idx - start_idx == len_to_consume:
                    span = sentence[tok.i:next_tok.i + 1]
                    return span
    return None


@lru_cache(maxsize=1)
def get_detokenizer():
    from sacremoses import MosesDetokenizer
    detok = MosesDetokenizer(lang='en')
    return detok


@lru_cache(maxsize=1)
def get_spacy_nlp():
    import en_core_web_lg
    nlp = en_core_web_lg.load()
    return nlp


def jsonl_iterator(input_fname, positive_only=False, ngram_order=3, eval=False):
    detok = get_detokenizer()
    nlp = get_spacy_nlp()

    with open(input_fname) as fin:
        for line in fin:
            sample = json.loads(line.strip())

            if positive_only and 'label' in sample and not sample['label']:
                # only consider examples where the query is correct
                continue

            target = sample['target']

            # clean up the query
            query = target['span1_text']
            if query is not None:
                if '\n' in query:
                    continue
                if query.endswith('.') or query.endswith(','):
                    query = query[:-1]

            # split tokens
            tokens = sample['text'].split(' ')

            def strip_pronoun(x):
                return x.rstrip('.,"')

            # find the pronoun
            pronoun_idx = target['span2_index']
            pronoun = strip_pronoun(target['span2_text'])
            if strip_pronoun(tokens[pronoun_idx]) != pronoun:
                # hack: sometimes the index is misaligned
                if strip_pronoun(tokens[pronoun_idx + 1]) == pronoun:
                    pronoun_idx += 1
                else:
                    raise Exception('Misaligned pronoun!')
            assert strip_pronoun(tokens[pronoun_idx]) == pronoun

            # split tokens before and after the pronoun
            before = tokens[:pronoun_idx]
            after = tokens[pronoun_idx + 1:]

            # the GPT BPE attaches leading spaces to tokens, so we keep track
            # of whether we need spaces before or after the pronoun
            leading_space = ' ' if pronoun_idx > 0 else ''
            trailing_space = ' ' if len(after) > 0 else ''

            # detokenize
            before = detok.detokenize(before, return_str=True)
            pronoun = detok.detokenize([pronoun], return_str=True)
            after = detok.detokenize(after, return_str=True)

            # hack: when the pronoun ends in a period (or comma), move the
            # punctuation to the "after" part
            if pronoun.endswith('.') or pronoun.endswith(','):
                after = pronoun[-1] + trailing_space + after
                pronoun = pronoun[:-1]

            # hack: when the "after" part begins with a comma or period, remove
            # the trailing space
            if after.startswith('.') or after.startswith(','):
                trailing_space = ''

            # parse sentence with spacy
            sentence = nlp(before + leading_space + pronoun + trailing_space + after)

            # find pronoun span
            start = len(before + leading_space)
            first_pronoun_tok = find_token(sentence, start_pos=start)
            pronoun_span = find_span(sentence, pronoun, start=first_pronoun_tok.i)
            assert pronoun_span.text == pronoun

            if eval:
                # convert to format where pronoun is surrounded by "[]" and
                # query is surrounded by "_"
                query_span = find_span(sentence, query)
                query_with_ws = '_{}_{}'.format(
                    query_span.text,
                    (' ' if query_span.text_with_ws.endswith(' ') else '')
                )
                pronoun_with_ws = '[{}]{}'.format(
                    pronoun_span.text,
                    (' ' if pronoun_span.text_with_ws.endswith(' ') else '')
                )
                if query_span.start < pronoun_span.start:
                    first = (query_span, query_with_ws)
                    second = (pronoun_span, pronoun_with_ws)
                else:
                    first = (pronoun_span, pronoun_with_ws)
                    second = (query_span, query_with_ws)
                sentence = (
                    sentence[:first[0].start].text_with_ws
                    + first[1]
                    + sentence[first[0].end:second[0].start].text_with_ws
                    + second[1]
                    + sentence[second[0].end:].text
                )
                yield sentence, sample.get('label', None)
            else:
                yield sentence, pronoun_span, query, sample.get('label', None)


def filter_noun_chunks(chunks, exclude_pronouns=False, exclude_query=None, exact_match=False):
    if exclude_pronouns:
        chunks = [
            np for np in chunks if (
                np.lemma_ != '-PRON-'
                and not all(tok.pos_ == 'PRON' for tok in np)
            )
        ]

    if exclude_query is not None:
        excl_txt = [exclude_query.lower()]
        filtered_chunks = []
        for chunk in chunks:
            lower_chunk = chunk.text.lower()
            found = False
            for excl in excl_txt:
                if (
                    (not exact_match and (lower_chunk in excl or excl in lower_chunk))
                    or lower_chunk == excl
                ):
                    found = True
                    break
            if not found:
                filtered_chunks.append(chunk)
        chunks = filtered_chunks

    return chunks
