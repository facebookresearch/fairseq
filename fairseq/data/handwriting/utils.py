# -*- coding: utf8 -*-
#   Copyright 2019 JSALT2019 Distant Supervision Team
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import logging
import os
import sys
import json

'''
 buildAlphabetMode should be false in test/validation mode, and true in training

 mode 1 means: remove all spaces in input text, and just put 1 blank at SENT START/END
 mode 2 is roughly "remove all spaces in input text, and use blanks BETWEEN all characters as targets"
 mode 3: no blanks, no spaces, no word boundaries encoded - plain character seq
 mode 4 keep word boundaries (spaces) and add an extra blank before and after the sentence
 mode 5 as mode 4, keep word boundaries, but no extra blanks

 Default is mode 2, but mode 4 or 5 is nice for user.
 Alphabet is a reference to dictionary i.e. just manipulate, and do not assign

 Special symbol:
    '*' id==0 which is the 'blank' in CTC
    '_' id==1 which is the 'SPACE' between words in output strings
    Unknown symbols are mapped to blanks if necessary.
'''


def processTranscriptLine(line_, alphabet_, mode_ = 2, buildAlphabetMode_= True, verbose_=0):
    # count lines
    idx = 0

    # Build a python list of encoded characters, via alphabet map/dict
    tmp = []

    logging.debug("processTranscriptLine() alphabet = " + str(alphabet_.chars) +
                  " mode_= " + str(mode_) +
                  " buildAlphabetMode= " + str(buildAlphabetMode_))

    # Step #0 : deal with alphabet from training text
    if buildAlphabetMode_ and len(alphabet_) == 0:
        alphabet_.insertDict('*',0) # symbol and idx in Neural Network
        if mode_ == 4 or mode_ == 5:
            alphabet_.insertDict(' ',1)

    # Step #1: make input characters from line_ to a python lists tmp with encoded chars
    # Note: the "+1" to address the fact that class 0 is gap/silence in CTC
    i=0
    for c in line_:
        if mode_ < 4:
            if c != ' ' and c != '.' and c!= '\'': # Skip these noisy ones
                if buildAlphabetMode_ and not alphabet_.existDict(c):
                    alphabet_.insertDict(c, len(alphabet_))

                # If validation/recognition/test mode, ignore unknown symbols we do not know. Replace with blank.
                if buildAlphabetMode_ == False and not alphabet_.existDict(c):
                    c= "*"
                tmp.append( alphabet_.ch2idx(c) )

                i = i + 1

        else: # mode 4 or 5 , keep word boundaries
            if c != '.' and c!= '\'': # Skip these noisy ones
                if not alphabet_.existDict(c):
                    if buildAlphabetMode_:
                        alphabet_.insertDict(c, len(alphabet_))
                    else:
                        # Not building a dictionary here. We are in recognition/eval/test mode. But, we have an unknown.
                        logging.warn("processTranscriptLine() the symbol :" + str(c) +
                                     ": is an UNK in this mode and alphabet. Ignore.",
                                      once=True)

                        c = '*' # Replace unknown symbol with existing symbol in alphabet. Use a blank '*'
                tmp.append( alphabet_.ch2idx(c) )

                i = i + 1

    if verbose_ > 0:
        print ("mode = " + str(mode_) + ", tmp list =" + str(tmp))
        print ("tmp list len =" + str(len(tmp)))

    # Step #2: transfer encoded lists 'tmp' to 'result' plus SENTENCE START/END symbols
    result = []

    if mode_ == 1: # mode (a) '01230'
         result.append(0)
         for i in tmp:
             result.append(i)
         result.append(0)

    if mode_ == 2: # mode (b) '0102030' *a*m*o*r*d*e*d*e*x*
        result.append(0)
        for i in tmp:
            result.append( i)
            result.append(0)

    if mode_ == 3: # mode (c), no '0' labels, '123' amordede...
        result = [ i for i in tmp]

    # mode (d) '012130' i.e. insert space/underscore id==1  at word boundaries in transcripts *amor_de_dexar_las_cos*
    if mode_ == 4: 
        result.append(0)
        for i in tmp:
            result.append( i)
        result.append(0) # add extra 0 at word end

    # mode (e) '1213' i.e. insert word boundaries id=1=='_' at word boundaries in transcripts amor_de_dexar_las_cos  
    # and no trailing/leading blank
    if mode_ == 5: 
        for i in tmp:
            result.append( i)

    idx = idx  + 1

    assert idx == 1 # 1 line only

    if verbose_ > 0:
        print ( "processTranscriptLine() mode_ = " + str(mode_) + " alphabet = " +str( alphabet_.content() ))
        print ( "processTranscriptLine() alphabet len = " + str(len(alphabet_)))
        print ( "processTranscriptLine() mode=" + str(mode_) + " result (encoded)   = " + str(result) )
        print ( "processTranscriptLine() mode=" + str(mode_) + " result (no blanks) = " + str(alphabet_.idx2str(result, noBlanks=True)) )
        print ( "processTranscriptLine() mode=" + str(mode_) + " result (raw      ) = " + str(alphabet_.idx2str(result )) )

    return result
# End of processTranscriptLine()

def num_between(num, start, end):
    return num >= start and num <= end