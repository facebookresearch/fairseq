""" from https://github.com/keithito/tacotron """
import numpy as np
import re
from . import cleaners
from .symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

# Special symbols
SOS_TOK = '<s>'
EOS_TOK = '</s>'

def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    if not m:
      sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
    sequence += _arpabet_to_sequence(m.group(2))
    text = m.group(3)

  return sequence


def sample_code_chunk(code, size):
    assert(size > 0 and size <= len(code))
    start = np.random.randint(len(code) - size + 1)
    end = start + size
    return code[start:end], start, end


def code_to_sequence(code, code_dict, collapse_code):
    if collapse_code:
        prev_c = None
        sequence = []
        for c in code:
            if c in code_dict and c != prev_c:
                sequence.append(code_dict[c])
                prev_c = c
    else:
        sequence = [code_dict[c] for c in code if c in code_dict]
        if len(sequence) < 0.95 * len(code):
            print('WARNING : over 5%% codes are OOV')

    return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def sequence_to_code(sequence, code_dict):
    '''Analogous to sequence_to_text'''
    id_to_code = {i: c for c, i in code_dict.items()}
    return ' '.join([id_to_code[i] for i in sequence])


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s != '_' and s != '~'
