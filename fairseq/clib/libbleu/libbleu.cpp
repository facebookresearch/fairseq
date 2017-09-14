/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <map>
#include <array>
#include <cstring>
#include <cstdio>

typedef struct
{
    size_t reflen;
    size_t predlen;
    size_t match1;
    size_t count1;
    size_t match2;
    size_t count2;
    size_t match3;
    size_t count3;
    size_t match4;
    size_t count4;
} bleu_stat;

// left trim (remove pad)
void bleu_ltrim(size_t* len, int** sent, int pad) {
  size_t start = 0;
  while(start < *len) {
    if (*(*sent + start) != pad) { break; }
    start++;
  }
  *sent += start;
  *len -= start;
}

// right trim remove (eos)
void bleu_rtrim(size_t* len, int** sent, int pad, int eos) {
  size_t end = *len - 1;
  while (end > 0) {
    if (*(*sent + end) != eos && *(*sent + end) != pad) { break; }
    end--;
  }
  *len = end + 1;
}

// left and right trim
void bleu_trim(size_t* len, int** sent, int pad, int eos) {
  bleu_ltrim(len, sent, pad);
  bleu_rtrim(len, sent, pad, eos);
}

size_t bleu_hash(int len, int* data) {
  size_t h     = 14695981039346656037ul;
  size_t prime = 0x100000001b3;
  char* b      = (char*) data;
  size_t blen  = sizeof(int) * len;

  while (blen-- > 0) {
    h ^= *b++;
    h *= prime;
  }

  return h;
}

void bleu_addngram(
    size_t *ntotal, size_t *nmatch, size_t n,
    size_t reflen, int* ref, size_t predlen, int* pred) {

  if (predlen < n) { return; }

  predlen = predlen - n + 1;
  (*ntotal) += predlen;

  if (reflen < n) { return; }

  reflen = reflen - n + 1;

  std::map<size_t, size_t> count;
  while (predlen > 0) {
    size_t w = bleu_hash(n, pred++);
    count[w]++;
    predlen--;
  }

  while (reflen > 0) {
    size_t w = bleu_hash(n, ref++);
    if (count[w] > 0) {
      (*nmatch)++;
      count[w] -=1;
    }
    reflen--;
  }
}

extern "C" {

void bleu_zero_init(bleu_stat* stat) {
  std::memset(stat, 0, sizeof(bleu_stat));
}

void bleu_one_init(bleu_stat* stat) {
  bleu_zero_init(stat);
  stat->count1 = 1;
  stat->count2 = 1;
  stat->count3 = 1;
  stat->count4 = 1;
  stat->match1 = 1;
  stat->match2 = 1;
  stat->match3 = 1;
  stat->match4 = 1;
}

void bleu_add(
    bleu_stat* stat,
    size_t reflen, int* ref, size_t predlen, int* pred, int pad, int eos) {

  bleu_trim(&reflen, &ref, pad, eos);
  bleu_trim(&predlen, &pred, pad, eos);
  stat->reflen += reflen;
  stat->predlen += predlen;

  bleu_addngram(&stat->count1, &stat->match1, 1, reflen, ref, predlen, pred);
  bleu_addngram(&stat->count2, &stat->match2, 2, reflen, ref, predlen, pred);
  bleu_addngram(&stat->count3, &stat->match3, 3, reflen, ref, predlen, pred);
  bleu_addngram(&stat->count4, &stat->match4, 4, reflen, ref, predlen, pred);
}

}
