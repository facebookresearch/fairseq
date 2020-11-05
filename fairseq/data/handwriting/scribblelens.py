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

import copy
import io
import logging
import os
import sys
import re
import zipfile

import torch.utils.data
import PIL.Image
import torchvision
import numpy as np
import pandas as pd

# Future FIXME: extract Path from Aligner into its own class and just import class Path
from . import aligner
from . import utils
from .alphabet import Alphabet
from .handwriting_dictionary import HandwritingDictionary


SCRIBE_RULES = '''writer-name	ID		directory
brouwer.chili       	        0		scribblelens.corpus.v1/nl/brouwer.chili
craen.de.vos.ijszee 	        1		scribblelens.corpus.v1/nl/craen.de.vos.ijszee
hamel.p1156         	        2		scribblelens.corpus.v1/nl/hamel.p1156/
kieft                          	3		scribblelens.corpus.v1/nl/kieft
tasman             		        4		scribblelens.corpus.v1/nl/tasman
van.neck.tweede     	        5		scribblelens.corpus.v1/nl/van.neck.tweede
van.neck.vierde     	        6		scribblelens.corpus.v1/nl/van.neck.vierde
zeewijck 				        7		scribblelens.corpus.v1/nl/zeewijck
craen.de.vos.ijszee		        1   	scribblelens.corpus.v1/nl/unsupervised/craen.de.vos.ijszee
tasman					        4   	scribblelens.corpus.v1/nl/unsupervised/tasman
van.neck.tweede			        5   	scribblelens.corpus.v1/nl/unsupervised/van.neck.tweede
van.neck.vierde			        6   	scribblelens.corpus.v1/nl/unsupervised/van.neck.vierde
zeewijck				        7   	scribblelens.corpus.v1/nl/unsupervised/zeewijck
5084.p184				        10  	scribblelens.corpus.v1/nl/unsupervised/5084.p184
cambodia.36.64			        11  	scribblelens.corpus.v1/nl/unsupervised/cambodia.36.64
neptunus				        12  	scribblelens.corpus.v1/nl/unsupervised/neptunus
cambodia.missive.1.13	        13  	scribblelens.corpus.v1/nl/unsupervised/cambodia.missive.1.13
ridderschap				        14  	scribblelens.corpus.v1/nl/unsupervised/ridderschap
sparendam				        15  	scribblelens.corpus.v1/nl/unsupervised/sparendam
speelman				        16  	scribblelens.corpus.v1/nl/unsupervised/speelman
voc.octrooi				        17  	scribblelens.corpus.v1/nl/unsupervised/voc.octrooi
owic.brieven.11/17[2346]	    20  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.11/17[2346]
owic.brieven.11/280		        21  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.11/280
owic.brieven.11/38[2356]	    22  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.11/38[2356]
owic.brieven.11/121[68]		    23  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.11/121[68]
owic.brieven.46/(3-14)	        30  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.46/([3-9]|10|11|12|13|14) # CHECK
owic.brieven.46/5[012]		    31  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.46/5[012]
owic.brieven.46/100		        32  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.46/100
owic.brieven.46/35[01]		    33  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.46/35[01]
owic.brieven.49/1.[89]		    40  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/1.[89]
owic.brieven.49/2.[1345678]	    41  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/2.[1345678]
owic.brieven.49/3.[356-11]	    42      scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/3.
owic.brieven.49/4.[124567]	    43      scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/4.[124567]
owic.brieven.49/14.[1234]	    44      scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/14.[1234]
owic.brieven.49/17.[3457]	    45      scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/17.[3457]
owic.brieven.49/20.[1]		    46      scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/20.1
owic.brieven.49/21.[1368-13]    47      scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/21.
owic.brieven.49/24.[2]		    48      scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/24.2
owic.brieven.49/26.[249]      	49	    scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/26.
owic.brieven.49/27.[34]		    50	    scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/27.
owic.brieven.49/27A.[45]	    51  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/27A
owic.brieven.49/31.3		    52  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/31.
owic.brieven.49/32.5		    53  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/32.
owic.brieven.49/33.9		    54  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/33.
owic.brieven.49/37.[8,11]	    55  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/37.
owic.brieven.49/41.2		    56  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/41.
owic.brieven.49/43.6		    57  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/43.
owic.brieven.49/45.4		    58  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/45.
owic.brieven.49/48.[34]		    59  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/48.
owic.brieven.49/49.2		    60  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/49.
owic.brieven.49/54.6		    61  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/54.
owic.brieven.49/55.[8,12]	    62  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/55.
owic.brieven.49/61.4		    63  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/61.
owic.brieven.49/65.1		    64  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/65.
owic.brieven.49/66.2		    65  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/66.
owic.brieven.49/68.9		    66  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/68
owic.brieven.49/75.2		    67  	scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/75
owic.brieven.49/16.3	        68      scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/16.
owic.brieven.49/18.[34]	        69      scribblelens.corpus.v1/nl/unsupervised/owic.brieven.49/18.
owic.brieven.46/300 	        70      scribblelens.corpus.v1/nl/unsupervised/owic.brieven.46/300
roggeveen					    80  	scribblelens.corpus.v1/nl/unsupervised/roggeveen/[6.0-25.0]
roggeveen					    81  	scribblelens.corpus.v1/nl/unsupervised/roggeveen/[25.1-39.0]
roggeveen					    82  	scribblelens.corpus.v1/nl/unsupervised/roggeveen/[39.1-47.0]
roggeveen					    83  	scribblelens.corpus.v1/nl/unsupervised/roggeveen/[47.1-60.0]
roggeveen					    84	    scribblelens.corpus.v1/nl/unsupervised/roggeveen/[61.1-79.1]
'''

SCRIBE_RULES_ROGGEVEEN = '''directory   ID
6.0			80
6.1			80
7.0			80
7.1			80
8.0			80
8.1			80
9.0			80
9.1			80
10.0		80
10.1		80
11.0		80
11.1		80
12.0		80
12.1		80
13.0		80
13.1		80
14.0		80
14.1		80
15.0		80
15.1		80
16.0		80
16.1		80
17.0		80
17.1		80
18.0		80
18.1		80
19.0		80
19.1		80
20.0		80
20.1		80
21.0		80
21.1		80
22.0		80
22.1		80
23.0		80
23.1		80
24.0		80
24.1		80
25.0		80
25.1		81
26.0		81
26.1		81
27.0		81
27.1		81
28.0		81
28.1		81
29.0		81
29.1		81
30.0		81
30.1		81
31.0		81
31.1		81
32.0		81
32.1		81
33.0		81
33.1		81
34.0		81
34.1		81
35.0		81
35.1		81
36.0		81
36.1		81
37.0		81
37.1		81
38.0		81
38.1		81
39.0		81
39.1		82
40.0		82
40.1		82
41.0		82
41.1		82
42.0		82
42.1		82
43.0		82
43.1		82
44.0		82
44.1		82
45.0		82
45.1		82
46.0		82
46.1		82
47.0		82
47.1		83
48.0		83
48.1		83
49.0		83
49.1		83
50.0		83
50.1		83
51.0		83
51.1		83
52.0		83
52.1		83
53.0		83
53.1		83
54.0		83
54.1		83
55.1		83
56.0		83
56.1		83
57.0		83
57.1		83
58.1		83
59.0		83
59.1		83
60.0		83
60.1		84
61.0		84
61.1		84
62.0		84
62.1		84
63.0		84
63.1		84
64.0		84
64.1		84
65.0		84
65.1		84
66.0		84
66.1		84
67.0		84
67.1		84
68.0		84
68.1		84
69.0		84
69.1		84
70.0		84
70.1		84
71.0		84
71.1		84
72.0		84
72.1		84
73.0		84
73.1		84
74.0		84
74.1		84
75.0		84
75.1		84
76.0		84
76.1		84
77.0		84
77.1		84
78.0		84
78.1		84
79.0		84
79.1		84
'''


class ScribbleLensDataset(torch.utils.data.Dataset):
    """Scribble Lens dataset."""

    def __init__(self,
                 root='data/scribblelens.corpus.v1.2.zip',
                 alignment_root="", # Default empty i.e. unused
                 split='supervised',
                 slice='empty', # tasman, kieft, brouwers
                 slice_filename=None,
                 colormode='bw',
                 vocabulary="", # The alphabet filename in json format
                 transcript_mode=2,
                 target_height=32,
                 target_width=-1,
                 # transform=None
                 ):
        """
        Args:
            root (string): Root directory of the dataset.
            alignmentRoot (string): Root directory of the path alignments. There should be one .ali file per image.
            split (string): The subset of data to provide.
                Choices are: train, test, supervised, unsupervised.
            slice_filename (string): Don't use existing slice and use a custom slice from a filename. The file
                should use the same format as in the dataset.
            colormode (string): The color of data to provide.
                Choices are: bw, color, gray.
            alphabet (dictionary): Pass in a pre-build alphabet from external source, or build during training if empty
            transcript_mode(int): Defines how we process space in target text, and blanks in targets [1..5]
            target_height (int, None): The height in pixels to which to resize the images.
                Use None for the original size, -1 for proportional scaling
            target_width (int, None): The width in pixels to which to resize the images.
                Use None for the original size, -1 for proportional scaling
            transform (callable, optional): Optional transform to be applied
                on a sample.
        Note:
            The alphabet or vocabulary needs to be generated with an extra tool like
                     generateAlphabet.py egs/scribblelens/yamls/tasman.yaml
        """
        transform=None
        self.root = root

        self.file = zipfile.ZipFile(root)
        root = 'scribblelens.corpus.v1'

        self.target_width = target_width
        self.target_height = target_height
        # if transform:
        #     self.pre_transform = torchvision.transforms.Compose([
        #                         torchvision.transforms.ToTensor(),
        #                         construct_from_kwargs(transform, additional_parameters={'scribblelens': True}),
        #                         torchvision.transforms.ToPILImage(),
        #                         ])
        # else:
        self.pre_transform = None

        self.transform=torchvision.transforms.Compose([
                 torchvision.transforms.Grayscale(),
                 torchvision.transforms.ToTensor(),
                 ])
        self.scribes = []

        self.trainingMode = (split == 'train')

        logging.debug(f"ScribbleLensDataset() constructor for split = {split}")

        # 'vocabulary' Filename from .yaml. alphabet has the vocabulary as a dictionary for CTC output targets.
        self.transcriptMode = transcript_mode
        assert ( self.transcriptMode >= 1 ) and ( self.transcriptMode <= 5 )

        self.vocabulary = vocabulary
        if self.vocabulary != "" and not os.path.isfile(self.vocabulary):
            print ("ERROR: You specified a vocabulary that does not exist: " + str(self.vocabulary))
            sys.exit(4)
        
        # [!] changed    
        self.alphabet = HandwritingDictionary(self.vocabulary)
        
        self.vocab_size = len(self.alphabet)
        self.must_create_alphabet = (self.vocabulary == '')
        self.nLines = 0

        self.alignmentFile = None # Optional
        if alignment_root != "":
            self.alignmentFile = zipfile.ZipFile(alignment_root)
            alignment_root = os.path.basename(alignment_root)[:-4]  # Remove .zip ext
            self.pathAligner = aligner.Aligner("none", self.alphabet) # Path I/O

        assert(colormode in {'bw', 'color', 'gray'})
        assert(split in {'train', 'test', 'supervised', 'unsupervised'})
        assert(slice in {'empty', 'custom', 'tasman', 'zeewijck', 'brouwer.chili', 'craen.de.vos.ijszee',
                         'van.neck.tweede', 'van.neck.vierde', 'kieft'})
        assert target_height != -1 or target_width != -1

        fnm_pattern = r'(?P<item_dir>(.*?)/'\
                      r'(?P<scribe>[^/]*)/'\
                      r'((?P<page>\d+)(\.(?P<page_side>\d+))?/)?)'\
                      r'line(\.\D+)?(?P<line_index>\d+)\.jpg'
        fnm_matcher = re.compile(fnm_pattern)

        if slice_filename is not None and slice != 'custom':
            logging.error('If you want a custom slice_filename you should use "custom" as slice.')
            sys.exit(1)

        corpora_filenames = []
        if split == 'unsupervised':
            corpora_filenames = [os.path.join(root, 'corpora', f'all.nl.{colormode}.lines.{split}.dat')]
        elif split == 'supervised':
            for mode in ['train', 'test']:
                corpora_filenames.append(os.path.join(root, 'corpora', f'nl.{colormode}.lines.{mode}.dat'))
        elif slice in {'tasman', 'kieft', 'zeewijck','brouwer.chili', 'craen.de.vos.ijszee', 'van.neck.tweede','van.neck.vierde' }:
            corpora_filenames.append(os.path.join(root, 'corpora', f'nl.{colormode}.lines.{slice}.{split}.dat'))
        elif slice != 'custom':
            corpora_filenames = [os.path.join(root, 'corpora', f'nl.{colormode}.lines.{split}.dat')]

        custom_corpora_filenames = []
        if slice == 'custom':
            custom_corpora_filenames.append(slice_filename)

        initialAlphabetSize = len(self.alphabet)

        self.data = []
        for corpora_filename in corpora_filenames:
            szBefore = len(self.alphabet)
            with self.file.open(corpora_filename) as f:
                self._read_corpora_filename(f, fnm_matcher, root, initialAlphabetSize, split)
                print ("ScribbleLensDataset()  datafile: " +
                        str(corpora_filename) + " and alphabet sizes before and after reading are "
                        + str(szBefore) + " and " + str(len(self.alphabet)) )

        # FIXME: Ugly solution to using custom slices from external file
        for corpora_filename in custom_corpora_filenames:
            with open(corpora_filename,'rb') as f:
                szBefore = len(self.alphabet)
                self._read_corpora_filename(f, fnm_matcher, "", initialAlphabetSize, split)
                print ("ScribbleLensDataset() custom datafile: " + str(corpora_filename) + " and alphabet sizes before and after reading are " + str(szBefore) + " and " + str(len(self.alphabet)) )

        if self.vocabulary and self.must_create_alphabet:
            logging.warning(f'Vocabulary {self.vocabulary} not found. Serializing the newly generated alphabet...')
            utils.writeDictionary(self.vocabulary)

        self.data_frame = pd.DataFrame(self.data)
        self.data_frame['scribe_id'] = np.nan
        self.data_frame['scribe'] = np.nan

        # Obtain the scribe ID
        scribe_pats = pd.read_csv(io.StringIO(SCRIBE_RULES),
                                  delimiter=None, encoding='utf8', sep=r'\s+', comment='#')
        for index, row in scribe_pats.iterrows():
            selection = self.data_frame['img_filename'].str.match(row['directory'])
            self.data_frame.loc[selection, 'scribe_id'] = row['ID']
            self.data_frame.loc[selection, 'scribe'] = row['writer-name']

        # Special treatment of Roggeveen
        roggeveen_mapping = {}
        for l in SCRIBE_RULES_ROGGEVEEN.split('\n')[1:]:
            if not l.strip():
                continue

            directory, scribe_id = l.split()
            roggeveen_mapping.setdefault(scribe_id, []).append(directory)

        for k, v in roggeveen_mapping.items():
            roggeveen_selection = self.data_frame['item_dir'].str\
                .startswith('scribblelens.corpus.v1/nl/unsupervised/roggeveen/')
            selection = self.data_frame['item_dir'].apply(lambda x: x.rsplit('/')[-2]).isin(v)

            self.data_frame.loc[roggeveen_selection & selection, 'scribe_id'] = int(k)
            self.data_frame.loc[roggeveen_selection & selection, 'scribe'] = f'roggeveen.{k}'

        self.data_frame['scribe_id'] = self.data_frame['scribe_id'].astype(int)

        """
        df2 = self.data_frame.copy()
        df2['scribe_dir'] = df2['item_dir'].apply(lambda x: x.rsplit('/', 1)[-2])
        df2[['scribe', 'scribe_id', 'item_dir']].drop_duplicates().sort_values(['scribe_id', 'item_dir'])\
            .to_csv('scribblelens.scribes', index=False, sep='\t')
        """

        self.metadata = {
            'alignment': {
                'type': 'categorical',
                'num_categories': len(self.alphabet)},
            'text': {
                'type': 'categorical',
                'num_categories': len(self.alphabet)},
        }

        self.file.close()
        self.file = None
        if self.alignmentFile is not None:
            self.alignmentFile.close()
            self.alignmentFile = None

    def _read_corpora_filename(self, f, fnm_matcher, root, initialAlphabetSize, split):
        for l in f:
            l = l.decode('utf8')
            if not l.strip():
                continue

            tokens = l.split()
            img_filename = os.path.join(root, tokens[0])

            match = re.match(fnm_matcher, img_filename)
            if not match:
                raise ValueError(f'Filename {img_filename} did not match the expected pattern.')

            item = match.groupdict()

            item['img_filename'] = img_filename
            if item['scribe'] in self.scribes:
                item['scribe_id'] = self.scribes.index(item['scribe'])
            else:
                item['scribe_id'] = len(self.scribes)
                self.scribes.append(item['scribe'])

            # Set alignment field in item[]
            if self.alignmentFile is not None:
                aliFilename = img_filename.replace(".jpg",".ali")
                classIndexAlignmentList, myTargetStr = self.pathAligner.readAlignmentFileFromZip(aliFilename, self.alignmentFile)
                item['alignment'] = torch.IntTensor(classIndexAlignmentList)
                item['alignment_text'] = myTargetStr
                # item['alignment_rle'], _ = distsup.utils.rleEncode(item['alignment'])

            # Read transcription, and format accodring to mode
            if len(tokens) > 1:
                item['text_filename'] = os.path.join(root, tokens[1])

                # If necessary, build alphabet while loading train corpus. This is where we parse the transcript files.
                # In yaml, look at  -> vocabulary: myalphabet.json
                # For test corpus, assume same vocabulary: myalphabet.json  and self.trainingMode = False
                #
                buildNewAlphabetFlag = (self.vocabulary == "") or ((self.vocabulary != "") and not os.path.isfile(self.vocabulary))
                self.readTranscript(item['text_filename'], split, buildNewAlphabetFlag)

            self.data.append(item)

    '''
        Input:
            Aligner class instance. This could be a "recognition path aligner" or a "forced aligner"

            if targets_ == None (and targets_len_ == None) then we are in recognition/test/forward-only mode.
    '''
    def decode(self, aligner_, \
        decodesWithBlanks_, decodes_, \
        log_probs_, log_prob_lens_, \
        targets_, targets_len_, \
        batch_, \
        verbose_=0 ):

        featuresLen = batch_['features_len']
        imageFilenameList_ = batch_['img_filename']
        try:
            # Expect original image size of [ nRows x nColumns ] == [orgHeight x orgWidth]
            orgImageSizes_ = batch_['img_size']
        except:
            orgImageSizes_ = None

        batchSize = log_probs_.shape[1]

        assert len(log_probs_.shape) == 3
        assert len(decodesWithBlanks_) == len(imageFilenameList_), "ERROR: batchSize for decoded path should be same a list of filenames!"

        # Decode the batch
        for i in range(0,batchSize):
            sz = len(decodesWithBlanks_[i])
            assert log_prob_lens_[i] == sz

            currentPathWithBlanks = decodesWithBlanks_[i]
            currentLogProbs = log_probs_[:sz,i,:]
            currentPathNoBlanks = torch.IntTensor(decodes_[i])
            if targets_ is not None:
                currentTargets = targets_[i]
                currentTargetLen = targets_len_[i]
            else:
                currentTargets = None
                currentTargetLen = None

            if verbose_ > 0:
                # processPaths() knows how to handle empty targets/targetlen
                self.processPaths( currentPathWithBlanks, currentPathNoBlanks, \
                    currentTargets, currentTargetLen, \
                    i, self.alphabet, verbose_ )

            # Write one path to file, either via forced alignment or based on this recognized alignment.
            if aligner_ != None:
                orgHeight = orgImageSizes_[i][0].item()
                orgWidth = orgImageSizes_[i][1].item()
                resizedHeight = batch_['features'].shape[2]
                nFeatures = resizedWidth = featuresLen[i].item() # FIX --> This was wrong earlier: batch_['features'].shape[1]

                assert orgHeight > 0
                assert resizedHeight > 0

                '''
                Note: we have 3 (stretch) factors, and call makePath in Aligner or ForcedAligner
                (1) the org vs resized (32 pixels) e.g. factor 5.75 (stretchFactor1)
                (2) the compression of the CNN encoder e.g. a factor of 6 or 8 (stretchFactor2)
                (3) The difference between nFeature on input and logProbs CTC outputs

                stretchFactor1 = (orgHeight * 1.0) / (resizedHeight * 1.0)
                stretchFactor2 = orgWidth / (len(currentLogProbs)  * 1.0)
                stretchFactor3 = resizedWidth / (len(currentLogProbs) * 1.0)

                FIXME: add an assert that when (currentTargets == None) we are working with an Aligner, and not an ForcedAligner!
                '''
                pathStr = aligner_.makePath(currentPathWithBlanks, currentLogProbs,
                    currentTargets, currentTargetLen,
                    self.alphabet,
                    imageFilenameList_[i], orgImageSizes_[i],
                    nFeatures ) # Target length for path == nFeatures. Guaranteed the same length

    '''
        We get 1 path which has no symbols removed, one path with no reps, no blanks, and a target path
        All are tensors [ 1 x something]
        verbose is set via .yaml, model
        If targets_ == None then we are in recognition forward() mode, also assume that targetLen_ == None
        and just process the recognized strings, not the targets
    '''
    def processPaths(self, pathWithBlanks_, pathClean_, targets_, targetLen_, idx_, alphabet_, verbose_ = 0):

        if targets_ is not None:
            mytarget = targets_[:targetLen_]
            assert mytarget.argmin() >= 0
            assert mytarget[ mytarget.argmax() ] < len(alphabet_)

        prefix = "ProcessPaths(" + str(idx_) + ")"
        if verbose_ > 1:
            print ( prefix + " : path with all blanks " + str(pathWithBlanks_))
        if verbose_ > 0:
            print ( prefix + " : path with  no blanks, no repetitions " + str(pathClean_))

        pathStrWithBlanks = alphabet_.idx2str(pathWithBlanks_)
        pathStrClean      = alphabet_.idx2str(pathClean_)
        if targets_ is not None:
            targetStr         = alphabet_.idx2str(mytarget)
            targetStrClean    = alphabet_.idx2str(mytarget, noBlanks=True)
            if verbose_ > 0:
                print ( prefix + " : targets_ " + str(mytarget))

        if verbose_ > 1:
            print ( prefix + " : pathStrWithBlanks :" + str(pathStrWithBlanks))
        if verbose_ > 0:
            print ( prefix + " : pathStrClean      :" + str(pathStrClean))
        if verbose_ > 1 and targets_ is not None:
            print ( prefix + " : targetStr         :" + str(targetStr))
        if verbose_ > 0 and targets_ is not None:
            print ( prefix + " : targetStrClean    :" + str(targetStrClean))

    # Read a .txt transcript file with > one <  line of text
    # Assume one line of text in transcript file.
    def readTranscript(self, filename_, split_, buildNewAlphabetFlag_):
        verbose = 0

        if verbose > 0:
            print ( "ScribbleLensDataset:readTranscript(" + str(self.nLines) + ") filename " + str(filename_) + \
                    " split_ = " + str(split_) + \
                    " training mode = " + str(self.trainingMode) + \
                    " alphabet size = " + str(len(self.alphabet)) + \
                    " transcriptMode = " + str(self.transcriptMode) + \
                    " buildNewAlphabetFlag_ = " + str(buildNewAlphabetFlag_) )

        lineCount = 0

        result = []
        with self.file.open( filename_ ) as f:
            rawText = f.read().decode('utf8').strip()  # Read Unicode characters
            logging.debug(f"ScribbleLensDataset:readTranscript() rawText ::{rawText}::")

            # self.Alphabet is a ref arg if you like, lives inside this data class
            # It is either 0 length if new, or filled earlier from vocabulary file.
            result = utils.processTranscriptLine(rawText,
                                self.alphabet,
                                mode_= self.transcriptMode,
                                buildAlphabetMode_= buildNewAlphabetFlag_ ,
                                verbose_=verbose)
            assert len(result) > 0
            lineCount += 1

        assert lineCount == 1
        logging.debug(f"ScribbleLensDataset:readTranscript() result ::{result}::")

        self.nLines += 1

        return result

    # Return a flag 'True' when processing a training dataset for a training run,
    # and False when doing recognitions, path alignments.
    def getTrainingMode(self):
        return self.trainingMode

    def __del__(self):
        if self.file:
            self.file.close()

    @staticmethod
    def tokenize_text(text):
        return torch.tensor(np.frombuffer(text.encode('utf-8'), dtype=np.uint8))

    @staticmethod
    def detokenize_text(tensor):
        return tensor.cpu().detach().numpy().tobytes().decode('utf-8')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.file:  # Reopen to work with multiprocessing
            self.file = zipfile.ZipFile(self.root)
        item = self.data[idx]

        df_item = self.data_frame.iloc[idx]
        item['scribe'] = df_item['scribe']
        item['scribe_id'] = df_item['scribe_id']

        new_item = copy.deepcopy(item)

        # Load and transform the image. Q: Why not mode 'L' for Gray level?
        with self.file.open(item['img_filename']) as img_f:
            image = PIL.Image.open(img_f).convert('RGB')

        if self.pre_transform:
            image = self.pre_transform(image)

        # Pass down the original image size, e.g., to stretch the forced alignement path later
        new_item['img_size'] = [ image.size[1], image.size[0] ]

        target_width = self.target_width or image.size[0]
        target_height = self.target_height or image.size[1]
        if target_width == -1:
            target_width = int(
                0.5 + 1.0 * target_height / image.height * image.width)
        elif target_height == -1:
            target_height = int(
                0.5 + 1.0 * target_width / image.height * image.height)

        image = image.resize((target_width, target_height),
                             PIL.Image.BICUBIC)

        # From PIL image to Torch tensor [ 32 x w] in this transform
        if self.transform:
            image = self.transform(image)

        # Images will be in W x H x C
        # FIXME: We should not change the layout here, should be handled by the user throught Transforms
        new_item['image'] = image.permute(2, 1, 0)

        # Load the transcription if it exists
        if 'text_filename' in item:
            with self.file.open(item['text_filename']) as f:
                rawText = f.read().decode('utf8').strip() # FIXME: For 2 or 4 byte Unicode, remove decode()
                assert(rawText == self.detokenize_text(self.tokenize_text(rawText)))

                # 'text' is raw text from transcription file
                # self.tokenize_text(text) was used to map it to plain ASCII values (but fails for unicode chars > 256)
                # self.transcriptMode = e.g. 2 default
                # This code here is called when building batches for train & test.
                # Assume we did load OR built the alphabet earlier.

                # self.Alphabet is a ref arg if you like, lives inside this data class
                result = utils.processTranscriptLine(rawText,
                                    self.alphabet,
                                    mode_= self.transcriptMode,
                                    buildAlphabetMode_= False, #self.trainingMode,
                                    verbose_=0)
                assert len(result) > 0
                new_item['text'] = torch.tensor(result)
                new_item['text_available'] = True
        else:
            new_item['text'] = torch.tensor([])
            new_item['text_available'] = False
            new_item['text_filename'] = ''

        if 'alignment' in new_item:
            if new_item['image'].size(0) != new_item['alignment'].size(0):
                print(f"{new_item['img_filename']} has bad alignment length: "
                      f"image len: {new_item['image'].size(0)} "
                      f"alignment len: {new_item['alignment'].size(0)}")
                raise Exception("Bad alignment length")
        return new_item


def main():
    import torchvision
    import matplotlib.pyplot as plt
    import pprint

    # We should update this test
    dataset = ScribbleLensDataset('data/scribblelens.corpus.v1.zip',
                                  transform=torchvision.transforms.Compose([
                                      torchvision.transforms.Grayscale(),
                                      torchvision.transforms.ToTensor()
                                  ]),
                                  split='supervised',
                                  colormode='bw')

    item = dataset[0]
    pprint.pprint(item)

    plt.imshow(item['image'][:, :, 0].t(),
               cmap=plt.cm.Greys)
    plt.xticks([])
    plt.yticks([])
    if 'text' in item:
        plt.xlabel(dataset.alphabet.idx2str(item['text'], noBlanks=True))
    plt.show()


if __name__ == '__main__':
    main()
