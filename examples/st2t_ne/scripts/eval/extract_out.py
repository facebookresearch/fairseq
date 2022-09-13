#!/usr/bin/python3
import sys

if __name__ == '__main__':
  sents = {}
  for line in sys.stdin:
    if line.startswith('D-'):
      id = line.split('\t')[0].split('-')[1]
      sents[id] = line.split('\t')[2]

  for i in range(len(sents)):
    try:
      sys.stdout.write(sents[str(i)])
    except KeyError:
      print("Error in", i)
      continue
