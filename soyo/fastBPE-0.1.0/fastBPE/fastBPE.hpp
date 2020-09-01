#pragma once

#include <algorithm>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <set>
#include <stdio.h>
#include <string>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h> // ftruncate
#include <unordered_map>
#include <unordered_set>
#include <vector>


namespace fastBPE {

using namespace std;

const size_t kMaxPairs = 1000 * 1000 * 1000;
const size_t kThreads = max(1, min(10, int(thread::hardware_concurrency())));
const char *kEndWord = "</w>";
const size_t kEndWordLength = 4;
const char *kTokenDelim = "@@";
const size_t kTokenDelimLength = 2;

int safeOpen(const char *file_path, int flags, mode_t mode = 0) {
  int fd = open(file_path, flags, mode);
  if (fd < 0) {
    fprintf(stderr, "Cannot open text file %s\n", file_path);
    exit(EXIT_FAILURE);
  }
  return fd;
}

void readText(const char *fp, unordered_map<string, uint32_t> &word_count) {
  string cur_word;
  uint64_t total = 0;
  auto deal_with_char = [&](char cur_char){
    if (cur_char == ' ' || cur_char == '\n') {
      if (cur_word.size() == 0)
        return;
      // end of word
      auto it = word_count.find(cur_word);
      int count = it != word_count.end() ? it->second : 0;
      word_count[cur_word] = count + 1;
      total++;
      cur_word.clear();
    } else {
      cur_word.push_back(cur_char);
    }
  };

  if (string(fp).compare("-") == 0) {
    for (std::string line; std::getline(std::cin, line);) {
      for(char c: line){
        deal_with_char(c);
      }
      deal_with_char('\n');
    }
  }
  else {
    int fd = safeOpen(fp, O_RDONLY);

    struct stat s;
    fstat(fd, &s);
    fprintf(stderr, "Loading vocabulary from %s ...\n", fp);

    size_t size = s.st_size;
    char *f = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);

    for (size_t i = 0; i < size; i++) {
      deal_with_char(f[i]);
    }
  }
  fprintf(stderr, "Read %lu words (%lu unique) from text file.\n", total,
          word_count.size());
}

std::pair<size_t, uint64_t> output_or_count(
  unordered_map<string, string> &bpe, size_t size, char *f, char *fo
) {
  string cur_word;
  size_t charOut = 0;
  uint64_t total = 0;
  for (size_t i = 0; i < size; i++) {
    auto &cur_char = f[i];
    if (cur_char == ' ' || cur_char == '\n') {
      if (cur_word.size() == 0) {
        if (fo != nullptr) fo[charOut] = cur_char;
        charOut++;
        continue;
      }
      // end of word : write bpe to output
      auto it = bpe.find(cur_word);
      assert(it != bpe.end());
      for (auto x : it->second) {
        if (fo != nullptr) fo[charOut] = x;
        charOut++;
      }
      if (fo != nullptr) fo[charOut] = cur_char;
      charOut++;

      total++;
      cur_word.clear();
    } else {
      cur_word.push_back(cur_char);
    }
  }
  return std::make_pair(charOut, total);
}

void outputText(const char *fpo, const char *fp,
                unordered_map<string, string> &bpe) {

  int fd = safeOpen(fp, O_RDONLY);
  auto fdOut = safeOpen(fpo, O_RDWR | O_CREAT | O_TRUNC, 0666);

  struct stat s;
  fstat(fd, &s);

  fprintf(stderr, "Applying BPE to %s ...\n", fp);
  auto size = s.st_size;
  char *f = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);

  auto p = output_or_count(bpe, size, f, nullptr);
  size_t out_size = p.first;

  if (ftruncate(fdOut, out_size) < 0) {
    fprintf(stderr, "Couldn't truncate output file %s to size %lu\n", fpo,
            out_size);
    exit(EXIT_FAILURE);
  }


  char *fo = (char *)mmap(NULL, out_size, PROT_WRITE, MAP_SHARED, fdOut, 0);
  if (fo == MAP_FAILED) {
    fprintf(stderr, "Output memory map failed : %d.\n", errno);
    exit(EXIT_FAILURE);
  }
  p = output_or_count(bpe, size, f, fo);
  fprintf(stderr, "Modified %lu words from text file.\n", p.second);
  munmap(fo, out_size);
  munmap(f, size);
  close(fdOut);
  close(fd);
}

struct pair_hash {
  template <class T1, class T2> size_t operator()(const pair<T1, T2> &p) const {
    auto h1 = hash<T1>{}(p.first);
    auto h2 = hash<T2>{}(p.second);
    size_t seed = h1;
    // boost::hash_combine
    return h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
};

void tokenize(const unordered_map<string, uint32_t> &word_count,
              unordered_map<string, uint32_t> &token_to_int,
              vector<string> &int_to_token, vector<list<uint32_t>> &words,
              vector<int32_t> &counts) {

  for (auto &x : word_count) {
    auto &word = x.first;

    words.push_back(list<uint32_t>());
    auto &current_word = words.back();
    counts.push_back(x.second);

    int pos = 0, realLength = 0;
    int lastStart = 0;
    while (word[pos]) {
      bool newChar = (word[pos] & 0xc0) != 0x80; // not a continuation byte
      realLength += newChar;
      // new token
      if (newChar && pos > 0) {
        auto new_token = word.substr(lastStart, pos - lastStart);
        if (token_to_int.count(new_token) == 0) {
          int_to_token.push_back(new_token);
          token_to_int[new_token] = int_to_token.size() - 1;
        }
        current_word.push_back(token_to_int[new_token]);
        lastStart = pos;
      }
      pos++;
    }
    auto new_token = word.substr(lastStart, string::npos) + kEndWord;
    if (token_to_int.count(new_token) == 0) {
      int_to_token.push_back(new_token);
      token_to_int[new_token] = int_to_token.size() - 1;
    }
    current_word.push_back(token_to_int[new_token]);
  }
}

void tokenize_str(const unordered_map<string, uint32_t> &word_count,
                  unordered_map<string, vector<string>> &words) {

  for (auto &x : word_count) {
    auto &word = x.first;
    words[word] = vector<string>();

    int pos = 0, realLength = 0;
    int lastStart = 0;
    while (word[pos]) {
      bool newChar = (word[pos] & 0xc0) != 0x80; // not a continuation byte
      realLength += newChar;
      // new token
      if (newChar && pos > 0) {
        auto new_token = word.substr(lastStart, pos - lastStart);
        words[word].push_back(new_token);
        lastStart = pos;
      }
      pos++;
    }
    auto new_token = word.substr(lastStart, string::npos) + kEndWord;
    words[word].push_back(new_token);
  }
}

using tp = pair<uint32_t, uint32_t>;
using tps = pair<string, string>;
using pc = unordered_map<tp, pair<int32_t, tp> *, pair_hash>;

void count_in_word(
    list<uint32_t> &word, uint32_t wi, uint32_t count, pc &pair_counts,
    vector<pair<int32_t, tp>> &contiguous_counts,
    unordered_map<tp, unordered_set<uint32_t>, pair_hash> &where) {
  bool second = false;
  tp cur_pair;
  for (uint32_t token : word) {
    if (second) {
      cur_pair.first = cur_pair.second;
    }
    cur_pair.second = token;
    if (second) {
      auto it = pair_counts.find(cur_pair);
      if (it == pair_counts.end()) {
        contiguous_counts.emplace_back(0, cur_pair);
        auto *added = &contiguous_counts.back();
        pair_counts.emplace(piecewise_construct, forward_as_tuple(cur_pair),
                            forward_as_tuple(added));
        where[cur_pair].emplace();
      }
      if (count > 0) {
        where[cur_pair].insert(wi);
      } else {
        where[cur_pair].erase(wi);
      }
      pair_counts[cur_pair]->first += count;
    } else {
      second = true;
    }
  }
}

void find_maxp(vector<pair<int32_t, tp>> &contiguous_counts, tp &maxp,
               int32_t &max_c) {
  max_c = 0;
  for (auto &x : contiguous_counts) {
    if (x.first > max_c) {
      max_c = x.first;
      maxp = x.second;
    } else if (x.first == max_c and x.second < maxp) {
      maxp = x.second;
    }
  }
}

void getvocab(const char *inputFile1, const char *inputFile2) {
  // get vocab
  unordered_map<string, uint32_t> word_count;
  readText(inputFile1, word_count);
  if (strcmp(inputFile2, "") != 0) {
    readText(inputFile2, word_count);
  }

  // sort vocab
  auto compFunctor = [](pair<string, int> elem1, pair<string, int> elem2) {
    return elem1.second > elem2.second ||
           (elem1.second == elem2.second && elem1.first < elem2.first);
  };
  set<pair<string, int>, decltype(compFunctor)> sorted_vocab(
      word_count.begin(), word_count.end(), compFunctor);
  assert(word_count.size() == sorted_vocab.size());

  // print sorted vocab
  for (auto element : sorted_vocab)
    cout << element.first << " " << element.second << endl;
}

void learnbpe(const uint32_t kNPairs, const char *inputFile1,
              const char *inputFile2) {
  // get vocab
  unordered_map<string, uint32_t> word_count;
  readText(inputFile1, word_count);
  if (strcmp(inputFile2, "") != 0) {
    readText(inputFile2, word_count);
  }

  // a token is an int, it represents a string
  unordered_map<string, uint32_t> token_to_int;
  vector<string> int_to_token;

  vector<list<uint32_t>> words;
  vector<int32_t> counts;

  tokenize(word_count, token_to_int, int_to_token, words, counts);

  vector<pair<int32_t, tp>> contiguous_counts;
  contiguous_counts.reserve(kMaxPairs);

  pc pair_counts;
  unordered_map<tp, unordered_set<uint32_t>, pair_hash> where_to_update;

  tp cur_pair;
  int32_t max_c = 0;
  tp max_p;
  for (uint32_t wi = 0; wi < words.size(); wi++) {
    count_in_word(words[wi], wi, counts[wi], pair_counts, contiguous_counts,
                  where_to_update);
  }
  find_maxp(contiguous_counts, max_p, max_c);
  for (size_t i = 0; i < kNPairs; i++) {
    // create new token for pair. replace
    auto new_token = int_to_token[max_p.first] + int_to_token[max_p.second];
    cout << int_to_token[max_p.first] << " " << int_to_token[max_p.second]
         << " " << max_c << endl;

    uint32_t new_token_id = int_to_token.size();
    int_to_token.push_back(new_token);
    token_to_int[new_token] = new_token_id;
    max_c = 0;
    auto change_count = [&](tp pair, int32_t v, uint32_t wi) {
      auto it = pair_counts.find(pair);
      if (it != pair_counts.end()) {
        // assert(it->second + v >= 0);
        it->second->first += v;
      } else {
        if (v > 0) {
          contiguous_counts.emplace_back(v, pair);
          pair_counts.emplace(piecewise_construct, forward_as_tuple(pair),
                              forward_as_tuple(&(contiguous_counts.back())));
          where_to_update[pair] = unordered_set<uint32_t>();
        }
      }
      if (v > 0)
        where_to_update[pair].insert(wi);
    };

    for (auto wi : where_to_update[max_p]) {
      auto &cur_word = words[wi];
      auto it = cur_word.begin();
      bool second = false;
      while (it != cur_word.end()) {
        if (second) {
          cur_pair.first = cur_pair.second;
        }
        cur_pair.second = *it;

        if (second) {
          // found the pair
          if (cur_pair == max_p) {
            it--; // points to first element of pair
            // if there is a token before us
            if (it != cur_word.begin()) {
              it--;
              change_count(make_pair(*it, cur_pair.first), -counts[wi], wi);
              change_count(make_pair(*it, new_token_id), counts[wi], wi);
              it++;
            }

            it = cur_word.insert(it, new_token_id); // it points to new token
            it++;                    // it points to first element of pair
            it = cur_word.erase(it); // it points to second element of pair
            it = cur_word.erase(it); // it points to next value

            // if there is a token after the one we inserted
            if (it != cur_word.end()) {
              change_count(make_pair(cur_pair.second, *it), -counts[wi], wi);
              change_count(make_pair(new_token_id, *it), counts[wi], wi);
            }
            cur_pair.second = new_token_id;
          } else {
            it++;
          }
        } else {
          second = true;
          it++;
        }
      }
    }

    if (pair_counts.find(max_p) != pair_counts.end()){
      pair_counts[max_p]->first = 0;
    }
    find_maxp(contiguous_counts, max_p, max_c);
  }
}

void split(vector<string> &splits, const string &text, char sep) {
  size_t start = 0, end = 0;
  while ((end = text.find(sep, start)) != string::npos) {
    if (end != start)
      splits.push_back(text.substr(start, end - start));
    start = end + 1;
  }
  if (end != start && start < text.size())
    splits.push_back(text.substr(start));
}

void readVocab(const char *fp, unordered_map<string, uint32_t> &vocab) {
  ifstream file(fp);
  if (!file) {
    fprintf(stderr, "Cannot open vocabulary file %s\n", fp);
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "Loading vocabulary from %s ...\n", fp);
  string line;
  uint64_t total = 0;
  while (getline(file, line)) {
    vector<string> splits;
    split(splits, line, ' ');
    assert(splits.size() == 2);
    assert(vocab.find(splits[0]) == vocab.end());
    int count = stoi(splits[1]);
    vocab[splits[0]] = count;
    total += count;
  }
  fprintf(stderr, "Read %lu words (%lu unique) from vocabulary file.\n", total,
          vocab.size());
}

void readCodes(const char *fp, unordered_map<tps, uint32_t, pair_hash> &codes,
               unordered_map<string, tps> &reversed_codes) {
  ifstream file(fp);
  if (!file) {
    fprintf(stderr, "Cannot open codes file %s\n", fp);
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "Loading codes from %s ...\n", fp);
  string line;
  while (getline(file, line)) {
    vector<string> splits;
    split(splits, line, ' ');
    assert(splits.size() == 3);
    auto pair = make_pair(splits[0], splits[1]);
    string concat = splits[0] + splits[1];
    assert(codes.find(pair) == codes.end());
    assert(reversed_codes.find(concat) == reversed_codes.end());
    codes[pair] = codes.size();
    reversed_codes[concat] = pair;
  }
  fprintf(stderr, "Read %lu codes from the codes file.\n", codes.size());
}

void decompose(const string s, vector<string> &newSubwords,
               const unordered_map<string, tps> &reversed_codes,
               const unordered_map<string, uint32_t> &vocab, bool isFinal) {
  auto it = reversed_codes.find(s);
  if (it == reversed_codes.end()) {
    // TODO this whole block below is just some sanity check
    // if we cannot un-merge a subword, it has to be a char
    string s2 = isFinal ? s.substr(0, s.size() - kEndWordLength) : s;
    int count = 0;
    for (size_t j = 0; j < s2.size(); j++) {
      if ((s2[j] & 0xc0) != 0x80) {
        count++;
      }
    }
    assert(count == 1);
    newSubwords.push_back(s);
    return;
  }
  assert(it != reversed_codes.end());
  string token1 = it->second.first;
  if (vocab.find(token1 + kTokenDelim) == vocab.end()) {
    decompose(token1, newSubwords, reversed_codes, vocab, false);
  } else {
    newSubwords.push_back(token1);
  }
  string token2 = it->second.second;
  auto query = token2 + kTokenDelim;
  if (isFinal) {
    query = token2.substr(0, token2.size() - kEndWordLength);
  }
  if (vocab.find(query) == vocab.end()) {
    decompose(token2, newSubwords, reversed_codes, vocab, isFinal);
  } else {
    newSubwords.push_back(token2);
  }
}

void limitVocab(const vector<string> &subwords, vector<string> &newSubwords,
                const unordered_map<string, tps> &reversed_codes,
                const unordered_map<string, uint32_t> &vocab) {
  string query;
  for (size_t i = 0; i < subwords.size(); i++) {
    bool isFinal = i == subwords.size() - 1;
    auto &subword = subwords[i];
    if (isFinal) {
      query = subword.substr(0, subword.size() - kEndWordLength);
    } else {
      query = subword + kTokenDelim;
    }
    if (vocab.find(query) == vocab.end()) {
      decompose(subword, newSubwords, reversed_codes, vocab, isFinal);
    } else {
      newSubwords.push_back(subword);
    }
  }
}

string process_bpe(vector<string> &subwords,
                   unordered_map<tps, uint32_t, pair_hash> &codes,
                   unordered_map<string, tps> &reversed_codes,
                   unordered_map<string, uint32_t> &vocab) {
  // merge subWords as much as possible
  vector<string> newSubwords;
  while (subwords.size() > 1) {
    // find the best pair
    int bestPairId = -1;
    auto bestPair = codes.end(); // TODO ugly hack that works
    for (size_t i = 0; i < subwords.size() - 1; i++) {
      auto pair = make_pair(subwords[i], subwords[i + 1]);
      auto it = codes.find(pair);
      int pairRank = it == codes.end() ? -1 : it->second;
      if (pairRank >= 0 && (bestPairId == -1 || int(bestPair->second) > pairRank)) {
        bestPair = it;
        bestPairId = i;
      }
    }
    // if we cannot merge anything, stop
    if (bestPairId == -1) {
      break;
    }
    // otherwise, merge subWords
    bool justMerged = false;
    newSubwords = vector<string>();
    for (size_t i = 0; i < subwords.size(); i++) {
      if ((i + 1 < subwords.size()) && (not justMerged) &&
          subwords[i] == bestPair->first.first &&
          subwords[i + 1] == bestPair->first.second) {
        newSubwords.push_back(subwords[i] + subwords[i + 1]);
        justMerged = true;
      } else {
        if (not justMerged) {
          newSubwords.push_back(subwords[i]);
        }
        justMerged = false;
      }
    }
    subwords = newSubwords;
  }
  // check that we are only using words in the dictionary
  if (vocab.size() > 0) {
    vector<string> newSubwords;
    limitVocab(subwords, newSubwords, reversed_codes, vocab);
    subwords = newSubwords;
  }
  // concat subWords
  string result;
  for (auto x : subwords) {
    result = result + x + kTokenDelim + " ";
  }
  return result.substr(
    0,
    result.size() - kEndWordLength - kTokenDelimLength - 1 // "</w>@@ "
  );
}

void applybpe(const char *outputFile, const char *inputFile,
              const char *codesPath, const char *vocabPath) {
  // read vocabulary (to which we want to limit the output file)
  unordered_map<string, uint32_t> vocab;
  if (strcmp(vocabPath, "") != 0) {
    readVocab(vocabPath, vocab);
  }

  // read codes
  unordered_map<tps, uint32_t, pair_hash> codes;
  unordered_map<string, tps> reversed_codes;
  readCodes(codesPath, codes, reversed_codes);

  // read input file words
  unordered_map<string, uint32_t> word_count;
  readText(inputFile, word_count);

  // tokenize
  unordered_map<string, vector<string>> bpeTok;
  tokenize_str(word_count, bpeTok);

  vector<pair<string, vector<string>>> bpeTokVec;
  for (auto x : bpeTok) {
    bpeTokVec.push_back(x);
  }

  // apply BPE codes to each word
  unordered_map<string, string> bpe[kThreads];
  vector<thread> threads;
  for (size_t i = 0; i < kThreads; i++) {
    threads.emplace_back(
      [&](size_t this_thread) {
        for (size_t w = this_thread; w < bpeTokVec.size(); w += kThreads) {
          auto &x = bpeTokVec[w];
          bpe[this_thread][x.first] = process_bpe(x.second, codes, reversed_codes, vocab);
        }
      },
      i
    );
  }

  unordered_map<string, string> final_bpe;
  for (size_t i = 0; i < kThreads; i++) {
    threads[i].join();
    for (auto x : bpe[i]) {
      final_bpe[x.first] = x.second;
    }
  }
  // output
  outputText(outputFile, inputFile, final_bpe);
}


class BPEApplyer {
private:
  unordered_map<string, uint32_t> vocab;
  unordered_map<tps, uint32_t, pair_hash> codes;
  unordered_map<string, tps> reversed_codes;

public:
  BPEApplyer(const string& codesPath, const string& vocabPath) {
    if (vocabPath.size() > 0) readVocab(vocabPath.c_str(), vocab);
    readCodes(codesPath.c_str(), codes, reversed_codes);
  }

  vector<string> apply(vector<string>& sentences) {
    vector<string> res;
    for(auto &s: sentences) {
      res.emplace_back("");
      string& cur = res.back();
      vector<string> words;
      split(words, s, ' ');
      for (size_t i = 0; i < words.size(); i++) {
        auto word = words[i];
        vector<string> word_bpes;
        int pos = 0, realLength = 0;
        int lastStart = 0;
        while (word[pos]) {
          bool newChar = (word[pos] & 0xc0) != 0x80; // not a continuation byte
          realLength += newChar;
          if (newChar && pos > 0) {
            auto new_token = word.substr(lastStart, pos - lastStart);
            word_bpes.push_back(new_token);
            lastStart = pos;
          }
          pos++;
        }
        auto bpe = word.substr(lastStart, string::npos) + kEndWord;
        word_bpes.push_back(bpe);
        cur += process_bpe(word_bpes, codes, reversed_codes, vocab);
        if (i < words.size() - 1) cur += " ";
      }
    }
    return res;
  }

};


void applybpe_stream(const char *codesPath, const char *vocabPath) {
  BPEApplyer applyer(codesPath, vocabPath);
  std::string line;
  while(std::getline(std::cin, line)) {
    vector<string> tmp;
    tmp.push_back(line);
    for(auto& l : applyer.apply(tmp)){
      std::cout << l << std::endl;
    }
  }
}

};