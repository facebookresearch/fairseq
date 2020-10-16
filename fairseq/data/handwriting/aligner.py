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

'''
    Two classes here:
        Aligner, to get path from a model during recognition, and visualize
        ForcedAligner, to get a path from the ground truth
'''
import os
import sys
import io
import torch
import cv2
import numpy as np

class Aligner:
    def __init__(self, outFile_, alphabet_):
        self.fpOutputPath = None
        self.filename = outFile_
        self.alphabet = alphabet_

        if outFile_ != "" and outFile_ != "none":
            self.fpOutputPath = open(outFile_, "w")
            print ("Aligner: __init__ opened path to write:" + str(outFile_) )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fpOutputPath != None:
            self.fpOutputPath.close()
            print ("Aligner: __exit__ closed file: " + str(self.filename) )
            self.fpOutputPath = None

    def __del__(self):
        if self.fpOutputPath != None:
            self.fpOutputPath.close()
            print ("Aligner: __del__ closed file: " + str(self.filename) )
            self.fpOutputPath = None

    def writePathToFile(self, imageFilename_, path_, targetStr_):
        # Build path string - Header line
        pathStr = imageFilename_ + " "  + str(len(path_)) + " targetStr:" + str(targetStr_) + "\n"

        # Build path string, per time stamp
        for t in range(0, len(path_)):
            label = path_[t][0]
            score = path_[t][1]
            scoreStr = "{0:.6f}".format(score)
            outLine = str(t) + " " + str(t+1) + " " + str(label) + " " + str(scoreStr) + "\n"
            pathStr = pathStr + outLine

        # Write path string to output file
        self.append(pathStr)

        return pathStr

    def pathFromStringAndLogprobs(self, pathStrWithBlanks_, pathLogProbs_):
        sz = len(pathLogProbs_)
        path = []
        for t in range(0,sz):
            label = pathStrWithBlanks_[t]
            if label == " ":
                label = "_" # SPACE, typically idx in alphabet == 1
            #score = pathLogProbs_[t].item()
            score = pathLogProbs_[t]

            path.append( [ label, score] )
        return path

    '''
        Input: a test string which is a path, to be appended to an output file.
    '''
    def append(self, text_):
        self.fpOutputPath.write(text_)

    '''
        Input : logProbs_.shape is 2D  [ seqLen, nColumns]
        Output: tensor with all probabilities from the (best) path
    '''
    def getPathLogProbs(self, logProbs_, path_):
        sz = logProbs_.shape[0]
        assert path_.shape[0] == sz

        result = []
        for i in range(0, sz):
            result.append( logProbs_[i, path_[i] ])

        return torch.FloatTensor(result)

    '''
        Called from 'model' sequential.py
        Input:
            pathWithBlanks_ a [ 1 x N] tensor which includes blanks and has the indices in the alphabet
            pathLogProbs_   a [ 1 x N] tensor which includes blanks, the scores for the path.
            targets         a [ 1 x K] tensor which includes blanks, the groundtruth
            alphabet_ : the dictionary with idx to character mapping

        Note:
            To debug and visualize:
            print ( self.prettyPrintList(path) ) #DBG

            if targets_ == None (and targetLen_) then we are in forward-only mode.
            Just look at the recognition results
    '''
    def makePath(self, pathWithBlanks_, log_probs_,
        targets_, targetLen_,
        alphabet_,
        imageFilename_, orgImageSize_,
        nFeatures_,
        verbose_ = 0):

        # Extract path from 2D matrix of logProbs_
        pathLogProbs_ = self.getPathLogProbs(log_probs_, pathWithBlanks_)

        assert pathWithBlanks_.shape == pathLogProbs_.shape
        if targets_ is not None:
            assert targets_.shape[0] == targetLen_
        assert len(alphabet_) > 0
        assert not torch.isnan(pathLogProbs_).any(), "ERROR: Aligner.makePath() the input has NaNs in log probs!"

        sz = len(pathLogProbs_)
        assert sz > 0

        pathStrWithBlanks = alphabet_.idx2str(pathWithBlanks_,noDuplicates=False,noBlanks=False)
        pathStr = ""
        filename = imageFilename_

        # targetStr
        if targets_ is not None:
            mytarget = targets_[:targetLen_]
            assert mytarget.argmin() >= 0
            assert mytarget[ mytarget.argmax() ] < len(alphabet_)
            assert len(alphabet_) > 0

            targetStr = alphabet_.idx2str(mytarget)
        else:
            # Insert the recognized, cleaned string here ...
            targetStr = alphabet_.idx2str(pathWithBlanks_, noDuplicates=True, noBlanks=True)

        if verbose_ > 0:
            print ("makePath: pathWithBlanks_ =  " + str(pathWithBlanks_))
            print ("makePath: pathLogProbs_ =  " + str(pathLogProbs_))
            print ("makePath: pathStrWithBlanks =  " + str(pathStrWithBlanks))
            print ("makePath: len( pathStrWithBlanks) =  " + str(len(pathStrWithBlanks)))
            print ("makePath: targetStr =  " + str(targetStr))
            print ("makePath: targetLen_ =  " + str(sz))
            print ("makePath: pathSz =  " + str(sz))
            print ("makePath: filename =  " + str(filename))
            print ("makePath: nFeatures_ =  " + str(nFeatures_))

        assert len(pathStrWithBlanks) == sz, "ERROR: The transcript of this path is shorter than pathWithBlanks_"

        # First build the path from the recognition data we got.
        pathLogProbsList = [element.item() for element in pathLogProbs_.flatten()]
        path = self.pathFromStringAndLogprobs(pathStrWithBlanks, pathLogProbs_)

        # Stretch the path to the width of the input features.
        path = self.stretchToLength(path, nFeatures_, insertSeparators_=False)

        # Convert CTC center path to a Kaldi-like path. The other way round, from ' * * e *' to 'e e e e'
        # (from mostly CTC centers to repeating symbols to indicate spans of characters)
        if 0:
            path = self.convertCTC2Kaldi(path)

        # If necessary stretch the patch
        # NOTE: we stretch a factor due to the CNN stack compression.
        #       Actually, if we the orginal image size then we could stretch by that factor
        # QUESTION: Do we do the stretching in visualization only?
        # QUESTION: Probably should not generate repeating labels e.g. we have a 'e' somewhere. then the streching would put four 'e's in path when factor == 4
        #
        if 0: # stretchFactor_ != 1:
            path = self.stretch(path, stretchFactor_)

        # Write path to file
        pathStr = self.writePathToFile(imageFilename_, path, targetStr)

        return pathStr

    def prettyPrintList(self, list_):
        resultStr = ""
        for i in range(0, len(list_)):
            resultStr = resultStr + str(i) + " " + str(list_[i]) + "\n"
        return resultStr

    def convertCTC2Kaldi(self, path_):
        # a) Remove blanks (i.e. the result is much shorter)
        path2 = self.removeBlanksFromPath(path_)

        # b) Get the  delta +/- around the centers
        pathPlusDelta = self.convertCentersToArea(path2, len(path_) )

        # c) make a new path, same length original, but with area marked as characters
        path4 = self.applyDeltas(path_, pathPlusDelta)
        assert len(path4) == len(path_)

        return path4

    '''
    '''
    def applyDeltas(self, path_, pathPlusDelta_ ):
        newPath = path_.copy()

        for sample in pathPlusDelta_:
            # list [start time, end time, symbo, ,score]
            start = sample[0]
            end = sample[1]
            ch = sample[2]
            sc = sample[3]

            t = start
            while t < end:
                #print ( "newpath[" +str(t) + "]=" + str(newPath[t]) )
                assert t < len(newPath)
                assert len(newPath[t]) == 2
                #assert newPath[t][0] == "*", "ERROR: cannot override timestamp %d as that is not a blank!" % t # Only override blanks
                if newPath[t][0] == "*":
                    newPath[t][0] =  ch
                    newPath[t][1] =  sc
                t += 1
        return newPath

    '''
        When visualizing, we want to seperate 'oo' properly. If we have
        a blank at boundary then that is enough.
        Detect repeating symbols (non blanks) and insert blank there
        symbolBoundaries_[i] = pair of [ idx in path, symbol]
    '''
    def insertBlanksAsSymbolSeperators(self, newPath_, symbolBoundaries_):
        idx = 0
        for pair in symbolBoundaries_:
            if idx > 0:
                if symbolBoundaries_[idx-1][1] == symbolBoundaries_[idx][1]:
                    # repeat detected : put a blank on previous right boundary
                    targetIdx = symbolBoundaries_[idx-1][0]
                    newPath_[targetIdx][0] = '*'
            idx += 1

    '''
        Input:
        - a list of lists [character label, score], length N
        - a stretchFactor N
        Output:
        - similar list but length 4 * N i.e. duplicate the intermediate ones
    '''
    def stretch(self, path_, stretchFactor_):
        orgSz = len(path_)
        newSz = int(orgSz * stretchFactor_)
        newPath = [None] * newSz

        for t in range( len(newPath)):
            orgIdx = int(t / (stretchFactor_ * 1.0))
            newPath[t] = path_[orgIdx].copy()

        return newPath

    '''
        symbolBoundaries_ is a optional list of lists (of length 2) with pairs [boundary index, symbols]
    '''
    def stretchToLength(self, path_, newSz_, insertSeparators_=False, symbolBoundaries_=None):
        orgSz = len(path_)
        newPath = [None] * newSz_
        stretchFactor_ = orgSz / (newSz_ * 1.0)

        for t in range( newSz_):
            orgIdx = int(t * stretchFactor_ + 0.5 )
            if orgIdx >= orgSz:
                orgIdx = orgSz -1
            newPath[t] = path_[orgIdx].copy()

        # Scale the symbolBoundaries_ list
        if symbolBoundaries_ is not None:
            for t in range(len(symbolBoundaries_)):
                oldIx = symbolBoundaries_[t][0]
                newIdx = int(oldIx * (1.0 / stretchFactor_) + 0.5)
                if newIdx > newSz_:
                    newIdx = newSz_ -1
                symbolBoundaries_[t] = [ newIdx, symbolBoundaries_[t][1] ]

        # Optional insert of blanks between repeats like 'oo'
        if (symbolBoundaries_ is not None ) and insertSeparators_:
            self.insertBlanksAsSymbolSeperators(newPath, symbolBoundaries_)

        return newPath

    '''
        keepCentersOnly() ~~ removeDuplicates in path but keep best probability
        We want one 'e' if we see an 'e', no a set of repetitions

        In : a path of pairs (list) [ character, score]
        Out: another path, same length
		
		<fixme, name of image>.jpg 108 targetStr:*merckelijcke schade daer weder affgeraeckt ende hebben ons tot*
		0 1  * -4.38690185546875e-05
		1 2  * -0.0009212493896484375
		2 3  * -0.00018405914306640625
		3 4  * -0.00017261505126953125
		4 5  m -0.1310567855834961
		5 6  * -0.24684619903564453
		6 7  * -1.33514404296875e-05
		7 8  e -0.06367206573486328		<-- Look, we have 2 e's here. Keep the one with highest prob, and put in middle of seq
		8 9  e -0.01826000213623047
		9 10  r -0.0006761550903320312
		10 11  * -2.288818359375e-05
		11 12  c -0.0002651214599609375
		12 13  k -0.03693866729736328
		13 14  * -0.12099361419677734
		14 15  e -0.0007238388061523438
		15 16  * -0.7822256088256836
		16 17  l -0.0044269561767578125
		17 18  i -0.3120155334472656
		18 19  i -0.6767368316650391
		19 20  j -0.0020036697387695312
		20 21  c -0.0012416839599609375
		21 22  k -0.05278205871582031
		22 23  k -0.061593055725097656
		23 24  e -0.035645484924316406
		
    '''
    def keepCentersOnly(self, path_):
        assert len(path_) > 0
        newPath = [None] * len(path_)
		
        idx = 0
        prev = '*'
        blankScore = -1.0 # Maybe 0.0 is too good

        t = 0
        while t < len(path_):
            ch = path_[t][0]
            newPath[t] = path_[t].copy()

            duration = 1
            #if not (ch in {'*', ' ', '_' } ):
            if not (ch in {'*'} ):
                # we have a new character to handle
                # duration is 1 or more repetitions of this character
                myCharCenterScore, duration = self.getCenterProb(path_, t) # scan ahead
                assert duration > 0

                # Clean area in duration
                for i in range(duration):
                    newPath[t + i] = [ '*', blankScore ]

                # Set the center entry
                centerIdx = int(t + (duration / 2))
                newPath[centerIdx] = [ ch, myCharCenterScore ]

            #Shift
            prev = ch
            t += duration

        return newPath

    def getCenterProb(self, path_, startTime_):
        t = startTime_
        ch = path_[startTime_][0]

        bestScore = -1000000000.0
        while t < len(path_) and ch == path_[t][0]:
            if path_[t][1] > bestScore:
                bestScore = path_[t][1]
            t += 1
        duration = t - startTime_

        assert duration > 0
        return bestScore, duration

    '''
        Input:
        - a path list of lists [character label, score], length N
        Output:
        - similar list but length 4 * N i.e. duplicate the intermediate ones
    '''
    def removeBlanksFromPath(self, path_):
        t = 0
        centerData = []
        for t in range(0, len(path_)):
            label = path_[t][0]
            score = path_[t][1]

            if label != "*":
                centerData.append( [ t, label, score] )
        return centerData

    def resizeImage(self, image_, scaleFactor):
        nNewRows = int(image_.shape[0] * scaleFactor)
        nNewCols = int(image_.shape[1] * scaleFactor)
        resizedImage = cv2.resize(image_, (nNewCols, nNewRows))  #width, height
        return resizedImage

    def saveImage(self, filename_, image_):
        print ( "Saving to file: " + str(filename_))

        # Make sure horizontal length (nColumns < 2^16 and nRows < 2^16 for JPEG specs)
        if image_.shape[1] > 65000:
            nColumns = image_.shape[1]
            scaleFactor = 65000.0 / (image_.shape[1] * 1.0)

            nNewRows = int(image_.shape[0] * scaleFactor)
            nNewCols = int(image_.shape[1] * scaleFactor)

            resizedImage = cv2.resize(image_, (nNewCols, nNewRows))  #width, height

            print ("saveImage() RESIZE IMAGE " + str(image_.shape) + " before save as nCols > 65000 (not possible for JPG) nColumns= " + \
                str(nColumns)  + " scaleFactor = " + str(scaleFactor) + \
                " NEW dimensions[" + str(resizedImage.shape) +  "]" )

            cv2.imwrite(filename_, resizedImage)
        else:
            # Normal action
            cv2.imwrite(filename_, image_)

    def scaleData(self, data_, scaleFactor):
        result = []
        for d in data_:
            leftCol  = d[0]
            rightCol = d[1]
            sym = d[2]
            score = d[3]

            newLeft  = int(int(leftCol ) * 1.0 * scaleFactor)
            newRight = int(int(rightCol) * 1.0 * scaleFactor)

            result.append( [ newLeft, newRight, sym, score] )
        return result

    def scaleCenterData(self, data_, scaleFactor):
        result = []
        for d in data_:
            newT  = int(int(d[0]) * 1.0 * scaleFactor)
            result.append( [ newT, d[1],  d[2]] )

        return result

    '''
        data_ is a python list of columns where to draw lines. format 3-tuple (list) [ timestamp, character, score]
    '''
    def drawLinesInImage(self, image_, data_, draw_):
        imageCopy = np.copy(image_)

        nRows = imageCopy.shape[0]
        nCols = imageCopy.shape[1]

        cnt = 0
        for d in data_:
            assert len(d) == 3

            t     = d[0]
            sym   = d[1]
            score = d[2]

            cv2.line(imageCopy, (t,0), (t, nRows), color=255, thickness=1)
            if 1:
                cv2.putText(imageCopy, sym, (t-5, 20), cv2.FONT_HERSHEY_SIMPLEX, 1., 255 )
            cnt += 1
        if draw_:
            cv2.imshow("drawLinesInImage AFTER", imageCopy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return imageCopy

    # Input: data_ is a list of lists of 4 [leftcol, right col, sym, score]
    #https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
    def drawRectanglesInImage(self, image_, data_, draw_):
        print ("drawRectanglesInImage")

        imageCopy = np.copy(image_)
        overlay   = np.copy(image_)

        nRows = image_.shape[0]
        nCols = image_.shape[1]

        cnt = 0
        for d in data_:
            #print ("[" + str(cnt) + "]" + str(d) )

            leftCol  = d[0]
            rightCol = d[1]
            sym   = d[2]
            score = d[3]

            if (cnt % 2) == 0:
                color = 128
            else:
                color = 200

            # A filled, NON tranparanet rectangle  (which deletes the digits)
            #cv2.rectangle(imageCopy, (leftCol,0), (rightCol, nRows), color=color, thickness=-1)

            # Rectangle with fat borders
            #cv2.rectangle(imageCopy, (leftCol,0), (rightCol, nRows), color=color, thickness=3)

            # Transparent
            alpha = 0.5
            cv2.rectangle(overlay, (leftCol,0), (rightCol, nRows), color=color, thickness=-1)
            cv2.putText(overlay, sym, (leftCol,20), cv2.FONT_HERSHEY_SIMPLEX, 1., 255 )

            cnt += 1

        # Now that we have all filled rectangles in 'overlay' combine images.
        cv2.addWeighted(overlay, alpha, imageCopy, 1 - alpha, 0, imageCopy)

        if draw_:
            img2 = self.resizeImage(imageCopy, 1.0)
            self.saveImage("a.jpg", img2)

            cv2.imshow("drawRectanglesInImage AFTER", img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return imageCopy

    '''
        This convert a list like
        [[48, '3', -0.000533],
        [80, '1', -0.000559],
        [112, '4', -0.018588],
        [144, '1', -0.00038]
        ...

        To

        [[48 -delta, 48+delta, '3', -0.000533],
        [80-delta, 80+delta, '1', -0.000559],
        [112-delta, 112+delta,'4', -0.018588],
        [144-delta, 144+delta, '1', -0.00038]
        ...

    '''
    def convertCentersToArea(self, centerData_, width_ ):
        #print (" width_ = " + str(width_))

        # 1) Make delta list
        sz = len(centerData_)
        delta = []
        for i in range(1, sz):
            assert centerData_[i] > centerData_[i-1]
            diff = ( centerData_[i][0] - centerData_[i-1][0] ) / 2
            delta.append(diff)

        print (len(delta) )

        # 2) Make new tuples
        if 0:
            #ORG
            minLeftCol = centerData_[0][0] - delta[0]
            if minLeftCol < 0:
                minLeftCol = 0
            maxRightCol = centerData_[sz-1][0] + delta[sz-2]
            if maxRightCol > (width_-1):
                maxRightCol = width_-1
            result = [  [ int(minLeftCol), int(centerData_[0][0] + delta[0]), centerData_[0][1] ,centerData_[0][2] ] ]
            for i in range(1, len(delta)):
                leftCol  = int(centerData_[i][0] - delta[i-1])
                rightCol = int(centerData_[i][0] + delta[i] )
                result.append( [ leftCol, rightCol, centerData_[i][1] ,centerData_[i][2] ] )
            result.append ( [ int(centerData_[sz-1][0] - delta[sz-2]), int(maxRightCol), centerData_[sz-1][1] ,centerData_[sz-1][2] ] )
        else:
            # New
            # Assume the 'center' is actally the left of the character where the CTC has emitted it.
            minLeftCol = centerData_[0][0]
            if minLeftCol < 0:
                minLeftCol = 0
            maxRightCol = centerData_[sz-1][0] + delta[ len(delta) -1 ] * 2
            if maxRightCol > (width_-1):
                maxRightCol = width_-1

            result = [  [ int(minLeftCol), int(centerData_[0][0] + delta[0]*2), centerData_[0][1] ,centerData_[0][2] ] ]
            for i in range(1, len(delta)):
                leftCol  = int(centerData_[i][0] )
                rightCol = int(centerData_[i][0] + delta[i]*2 )
                result.append( [ leftCol, rightCol, centerData_[i][1] ,centerData_[i][2] ] )

            result.append ( [ centerData_[sz-1][0], int(maxRightCol), centerData_[sz-1][1] ,centerData_[sz-1][2] ] )

        # print ( self.prettyPrintList(result)) #DBG

        return result

    def loadAndResizeImage(self, filename_, height_ = -1):
        verbose = 1

        assert os.path.isfile(filename_), "ERROR: loadAndResizeImage() File image .jpg does not exist"
        image = cv2.imread(filename_) # Numpy uint8 or so

        img0 = image[:,:,0]

        # Resize optional, via numpy/cv2
        if verbose > 0:
            print ("loadAndResizeImage() org size  " + str(filename_) + " " +  str(img0.shape))
        assert height_ > 0

        nRows = img0.shape[0]
        scaleFactor = (height_ * 1.0) / nRows
        nNewRows = int(img0.shape[0] * scaleFactor + 0.5)
        nNewCols = int(img0.shape[1] * scaleFactor + 0.5)
        if verbose > 0:
            print ("loadAndResizeImage() new size [" + str(nNewRows) + " x " + str(nNewCols) + "]" )

        img1 = cv2.resize(img0, (nNewCols, nNewRows))  #width, height

        #self.saveImage("test.jpg",img1)
        #sys.exit(3)

        # Return new image AND orginal height
        return img1, img0, nRows

    def readAlignmentFile(self, aliFilename_):
        assert os.path.isfile(aliFilename_), "ERROR: readAlignmentFile() --path File does not exist"

        fp = open(aliFilename_,"r",encoding='utf-8')
        scores, syms, imageFilename, header = self.readOnePath(fp)
        fp.close()

        targetStrIdx = header.index("targetStr:")
        assert targetStrIdx > 0,"ERROR: no target string in header of .ali file"
        targetStr = header[targetStrIdx + 10:].rstrip().replace("*","")

        assert len(self.alphabet) > 0, "ERROR: readAlignmentFile() has an empty alphabet"
        result = self.alphabet.symList2idxList(syms)

        return result, targetStr

    def readAlignmentFileFromZip(self, aliFilename_, zipFile_):
        assert zipFile_ is not None, "ERROR: readAlignmentFileFromZip() zip file fp is empty."

        fp = io.TextIOWrapper(zipFile_.open(aliFilename_), 'utf8')
        scores, syms, imageFilename, header = self.readOnePath(fp)
        fp.close()

        targetStrIdx = header.index("targetStr:")
        assert targetStrIdx > 0,"ERROR: no target string in header of .ali file"
        targetStr = header[targetStrIdx + 10:].rstrip().replace("*","")

        assert len(self.alphabet) > 0, "ERROR: readAlignmentFileFromZip() has an empty alphabet"
        result = self.alphabet.symList2idxList(syms)

        return result, targetStr

    def writePathString(self, aliFilename_, pathStr_):
        print ("ALigner::writePathString() to file = " + str(aliFilename_))

        fp = open(aliFilename_,"w+")
        assert fp
        fp.write(pathStr_)
        fp.close()

    def makePathString(self,header_,syms_,scores_):
        out = header_ + "\n"
        for idx in range(len(scores_)):
            scoreStr = "{0:.6f}".format(scores_[idx])
            out = out + str(idx) + " " + str(idx+1) + " " + str(syms_[idx]) + " " + str(scoreStr) + "\n"
        return out

    def readOnePath(self, fp_, headerLine_=""):
        assert fp_, "ERROR: readOnePath() fp is not valid"

        if headerLine_ !="":
            rawText = headerLine_.strip()
        else:
            rawText= fp_.readline()

        words = rawText.split()
        assert len(words) >= 2, "ERROR: The header line in path does not have >= 2 entries (image name, path size, extra annotations)"

        imageFilename = words[0]
        pathLength = int(words[1])

        mymin = +100000.0
        mymax = -100000.0
        scores = []
        syms = []
        for l in range(0, pathLength):
            l = fp_.readline().strip()
            words = l.split()
            assert len(words) == 4, "ERROR: The timestamp line in path does not have 4 entries"

            t = int(words[0])
            sym = words[2]
            if sym == "SPACE":
                sym = "_"

            score= float(words[3]) # prob or score
            if score > mymax:
                mymax = score
            if score < mymin:
                mymin = score

            scores.append(score)
            syms.append(sym)
        # end for() time line in path

        # Distinguish whether we have probs or scores\
        # FIXME: reformat probs to scores.
        if mymax <= 0.0:
            scoreFlag = True
        else:
            scoreFlag = False

        assert len(scores) > 0 # Sometimes we generated empty paths
        assert len(scores) == pathLength
        assert len(scores) == len(syms)

        return scores, syms, imageFilename, rawText

    '''
        This file has a collection of one or more paths
        Paths could have Unicode symbols
        headerline (filename, width) others
        seq (t   t+1     symbol  score or prob)

    '''
    def readPath(self, filename_):
        print ("Aligner.readPath() " + str(filename_))
        assert os.path.isfile(filename_), "ERROR: Aligner.readPath() File does not exist"

        fp = open(filename_,"r",encoding='utf-8')

        # get > one < path only. We get 2 Python lists now with scores and symbols
        scores, syms, imageFilename, header = self.readOnePath(fp)
        assert len(scores) == len(syms)

        # Build path as list of [syms,scores]
        combinedSymsAndScores = []
        for idx in range(len(scores)):
            combinedSymsAndScores.append( [ syms[idx], scores[idx]] )

        # Resize image (we got the image name from path)
        resizedImg, orgImg, orgHeight = self.loadAndResizeImage(imageFilename, 32)

        # stretchedPath will have same width as org image e.g. 2277 entries for 2277 pixel columns in line
        orgImageWidth = orgImg.shape[1]
        strechedPath = self.stretchToLength(combinedSymsAndScores, orgImageWidth)

        # Clean the path. This is for easy visualization and not necessary otherwise. Example 'e e e e' becomes ' * * e *'
		# Input is a list of pairs of [ characters, score ]
        strechedPath = self.keepCentersOnly(strechedPath)

        # Grab the centers of symbols and build tuple list (time, sym, score)
        t = 0
        centerData = []
        centerCount = 0
        for t in range(0, len(strechedPath)):
            if strechedPath[t][0] != "*":
                centerData.append( [ t, strechedPath[t][0], strechedPath[t][1] ] )
                centerCount += 1

        self.drawLinesInImage(resizedImg, centerData, False)

        # areaData has the left, right columns for an image of [32 x width]. If we want to use this
        # for the orginal image then we need to scale the areadata with (orgwidth/ 32)

        # Draw segmentation path in resized image
        areaData = self.convertCentersToArea(centerData, orgImg.shape[1])

        showImage = True
        self.drawRectanglesInImage(orgImg, areaData, showImage)

        fp.close()

# --------------------------------------
# End of class Aligner, start of ForcedAligner
# --------------------------------------
class ForcedAligner(Aligner):
    def __init__(self, outFile_, alphabet_, maxFeatures_=5000, maxTargets_=500):
        super(ForcedAligner, self).__init__(outFile_, alphabet_)
        print ("ForcedAligner: __init__ ")

        self.alphabet = alphabet_
        assert len(alphabet_) > 0, "ERROR: ForcedAligner() got an empty alphabet"

        self.inf = -10000000.0
        self.no_path = -1

        # Simplified stack decoder for forced align. We look at the frontier i.e. all tokens at
        # time 't' which corrensponds to going through 'score' array row-by-row.
        # You can recombine neighboring tokens but not other paths.
        #
        # Search through [seqlen x targetLength] and store in arrays of [ maxWidth x maxTargets] i.e. maxWidth rows of [1 x maxTargets]
        self.height = maxFeatures_
        self.width  = maxTargets_
        assert self.width > 0
        assert self.height > 0

        self.score  = torch.zeros( (self.height, self.width), dtype=torch.float32)
        self.tb     = torch.zeros( (self.height, self.width), dtype=torch.int32)

        # init
        self.clean()

    def clean(self):
        self.score.fill_(self.inf)
        self.tb.fill_(self.no_path)

        self.nTargets = -1
        self.nFeatureVectors = -1

    def forceLeadingAndTrailingBlank(self, targets_):
        '''Path alignment. Force a blank at start and stop to force align to space/SIL.
        Convert input (Torch int tensor) to list, check, and convert and return.
        '''
        blankId = 0 # alphabet['*'] == 0 typically
        tmp = targets_.tolist()

        if int(tmp[0]) != blankId:
            tmp.insert(0,0)
        if int(tmp[len(tmp)-1]) != blankId:
            tmp.append(0)

        return  torch.IntTensor(tmp), len(tmp)

    '''
        Called via sequential.py (model) -> scribblelens/data.py decode()

        Same interface as in class Aligner.

        For the forced aligner, which is here, we just ignore the recognized path ( pathWithBlanks_, pathLogProbs_,)
        and instead will do a forced alignment.
    '''
    def makePath(self, pathWithBlanks_, log_probs_, \
        targets_, targetLen_, \
        alphabet_, \
        imageFilename_, orgImageSize_, \
        nFeatures_ ,
        verbose_ = 0):

        # Now apply targetLen to targets_ as targetLen_ is real length, and targets_ is batch with maxSize.
        assert targetLen_ > 0
        targets_ = targets_[:targetLen_]

        assert targets_.shape[0] == targetLen_
        assert len(alphabet_) > 0
        assert not torch.isnan(log_probs_).any(), "ERROR: ForcedAligner.makePath() the input has NaNs in log probs!"

        # Init score & tb array
        self.clean()

        # Make sure we have a 'blank' to align to, at start and stop of path.
        targets_, targetLen_ = self.forceLeadingAndTrailingBlank(targets_)

        # The return path is Kaldi-like with repeating sequence like 't t h h h e e e e' for 'the'
        # Optionally, we can split repeating symbols like 'oo' in 'door' based on symbolBoundaries
        path, symbolBoundaries, oversampleFactor = self.forceAlign3(targets_, log_probs_, imageFilename_, orgImageSize_, verbose_)

        # Stretch the path after force align to the width of the input features.
        # Use insertSeparators_ to split double 'oo' and 'ee' IFF you want that.
        path = self.stretchToLength(path, nFeatures_, insertSeparators_=False, symbolBoundaries_=symbolBoundaries)

        # Output forced align path to file
        targetStr = alphabet_.idx2str(targets_)
        pathStr = self.writePathToFile(imageFilename_, path, targetStr)

        return pathStr

    '''
        Should be optimized for use-case and model
    '''
    def updateTransitionCost(self, symIndex_, spaceIdx_ = 1):
        transitionSelf = -2 # loop, for all character which are non-blank -1.8 or -4.0  when 1000:1 then we see path diffs.
        transitionNext = -1 # go to next character in sequence -1.2 or -1.0

        blankSelf = transitionSelf * 2 # -1.5
        blankNext = transitionNext * 2 # -1.1
    	
        selfTransitionCost = transitionSelf
        nextTransitionCost = transitionNext

        if symIndex_ == 0: # is a blank?
            selfTransitionCost = blankSelf
            nextTransitionCost = blankNext

        if symIndex_ == spaceIdx_: # is a space? Cannot stay in SPACE, and expensive to go through
            selfTransitionCost = -1000.0 # -1000.0
            nextTransitionCost = -100.0  # -100.0

        return [selfTransitionCost, nextTransitionCost]

    '''
        When CNN stack produce nFeatureVectors < nTargets, we replicate logprobs
        in effect oversampling when generating a path.
    '''
    def replicateRows(self, data_, factor_ = 2 ):
        assert len(data_.shape ) == 2
        assert factor_ > 1

        newData = data_.repeat_interleave(factor_,dim=0)

        assert factor_ * data_.shape[0] == newData.shape[0] # new height
        assert data_.shape[1] == newData.shape[1] # same width
        assert data_[0,0]  == newData[0,0] # 1st row check

        if factor_ >= 2:
            assert data_[0,0]  == newData[1,0] # 2nd row check

        assert len(newData.shape ) == 2

        return newData

    '''
        Visualize this as rows of length [ 1 x nTargets] of the sequence,
        and number of rows == sequenceLength

        We process the frontier of tokens at every time 't'
        and we can only recombine tokens from previous and current target index,
        which is what we need for forced align

        CHECK:
        - make sure we use the correct seqLen length when handling batches > 1
    '''
    def forceAlign3(self, targets_, logprobs_, imageFilename_, orgImageSize_, verbose_ = 0):

        nFeatureVectors = logprobs_.shape[0]
        nOrgFeatureVectors = orgImageSize_[1].item()
        nClasses = logprobs_.shape[1]
        nTargets = len(targets_)

        if verbose_ > 0:
            print ("forceAlign3() nFeatureVectors = " + str(nFeatureVectors) + " nTargets = " +str(nTargets) + \
                 " nClasses =  " + str(nClasses) + " orgImageSize_ =" + str(orgImageSize_))
            print ("forceAlign3() logprobs_.shape row[0]= " + str(logprobs_[0,:].shape))
            print ("forceAlign3() logprobs_.shape = " + str(logprobs_.shape))
            print ("forceAlign3() targets_.shape  = " + str(targets_.shape))
            print ("forceAlign3() targets_  = " + str(targets_))
            print ("forceAlign3() score.shape = " + str(self.score.shape))
            print ("forceAlign3() score.shape row[0]= " + str(self.score[0,:].shape))
            print ("forceAlign3() nTargets  = " + str(nTargets) + " nClasses = " + str(nClasses) + " nFeatureVectors=seqLen = "  + str(nFeatureVectors)  )

        assert len(logprobs_.shape ) == 2 # 2D array

        # Copy for safe keeping
        self.nTargets = nTargets
        self.nFeatureVectors = nFeatureVectors

        # Assure fit in score and traceback
        assert self.height >= nFeatureVectors
        assert self.width  >= nTargets

        # Assure more feature vectors than targets. We need to handle this, maybe via duplicating inputs
        # We use factor to indicate oversampling. We actually duplicate logprobs[] by the factor and let it ripple through
        factor = 1 # Normal case, 95%+ of inputs
        if nFeatureVectors <  nTargets:
            factor = int((nTargets * 1.0) /nFeatureVectors ) + 1
            print ("WARNING: imageFilename_ = " + str(imageFilename_) + " oversampled path as not enough feature vectors. nFeatureVectors= " + str(nFeatureVectors) + ", nTargets= " + str(nTargets) )
            logprobs_ = self.replicateRows(logprobs_, factor)
            nFeatureVectors = logprobs_.shape[0] # New size

            assert nFeatureVectors >= nTargets
            assert nClasses == logprobs_.shape[1]

        classIndexOfSpace = -1
        spaceSym = " "
        if self.alphabet.existDict(spaceSym):
            classIndexOfSpace = self.alphabet.ch2idx(spaceSym)
            assert classIndexOfSpace == 1

        # Init search
        self.score[0][0] = logprobs_[0][0]
        self.tb   [0][0] = 0 # self.no_path

        if verbose_ > 0:
            print ( "\n-------t= row= " + str(0) +"----------- factor = " + str(factor) + "-------")
            print ("\nAFTER : row[0]=" + str(self.score[0,:1]))
            print ("AFTER :  tb[0]=" + str(self.tb[0,:1]))

        for row in range(1, nFeatureVectors):
            lastCol = min(row, nTargets-1)

            if verbose_ > 0:
                print ( "\n-------t= row= " + str(row) +" of " +str(nFeatureVectors) + "-----------")
                print ("BEFORE: lastCol = " + str(lastCol) )
                torch.set_printoptions(sci_mode=False)
                print ("BEFORE: row[t=" + str(row) + "]=" + str(self.score[row,:lastCol+1]))
                print ("BEFORE:  tb[t=" + str(row) + "]=" + str(self.tb[row,:lastCol+1]))
                print ("forceAlign3() nTargets  = " + str(nTargets) + " nClasses = " + str(nClasses) + " nFeatureVectors=seqLen = "  + str(nFeatureVectors)  )

            for col in range(lastCol+1):
            #for col in range(lastCol):
                '''
                    a) The only way to get to targetIdx==0 is via SELF transition. For all time stamps (rows)

                    b) For the final targetIdx 'lastCol', at time==row==0, you can only do a NEXT transition.
                       If time > 0 then NEXT and SELF are possible. This is important at end of path.

                    c) Choose the best of SELF and NEXT (from previous target symbol)  transitions

                    TODO:
                    - handle '_' space alignment with high transitions i.e. force SPACE to be at correct place.
                '''
                selfTransitionCost, nextTransitionCost = self.updateTransitionCost(targets_[col], classIndexOfSpace)

                if col == 0 or (col == lastCol and row == 0):
                    if col == 0:
                        # a) The only way to get to targetIdx==0 is via SELF transition. For all time stamps (rows)
                        prevScore    = self.score[row-1][col].item()
                        prevTB       = col
                        targetCol    = targets_[col]
                        currentScore = logprobs_[row][targetCol].item()

                        assert prevScore > self.inf, "ERROR: you are referring to a score that is not initialized!"

                        self.score[row][col] = prevScore  + selfTransitionCost + currentScore
                        self.tb   [row][col] = prevTB
                        newScore = self.score[row][col].item()

                        if verbose_ > 0:
                            print ("SELF: score[t=" + str(row) + "][class=" + str(col) + "]= " + str(newScore) + " computed as " + \
                                "self.score[" + str(row - 1) + "][" + str(col) +"] " + str(prevScore) + \
                                " + selfTransitionCost " + str(selfTransitionCost) + \
                                " + logprobs_["+str(row)+ "][" + str(targetCol) + "]= " + str(currentScore)  )
                            print ("SELF:    tb[t=" + str(row) + "][class=" + str(col) + "]= prevClass = " + str(prevTB) )

                    else:
                        # CHECK: this might be unnecessary.
                        # b) For the final targetIdx 'lastCol', at time==row==0, you can only do a NEXT transition.
                        #    If time > 0 then NEXT and SELF are possible. This is important at end of path.
                        prevScore    = self.score[row-1][col-1].item()
                        prevTB       = col - 1
                        targetCol    = targets_[col]
                        currentScore = logprobs_[row][targetCol].item()
                        assert prevScore > self.inf, "ERROR: you are referring to a score that is not initialized!"

                        self.score[row][col] = prevScore  + nextTransitionCost + currentScore
                        self.tb   [row][col] = prevTB
                        newScore = self.score[row][col].item()

                        if verbose_ > 0:
                            print ("NEXT: score[t=" + str(row) + "][class=" + str(col) + "]= " + str(newScore) + " computed as " + \
                                "self.score[" + str(row - 1) + "][" + str(col - 1) +"] " + str(prevScore) + \
                                " + nextTransitionCost " + str(nextTransitionCost) + \
                                " + logprobs_["+str(row)+ "][" + str(targetCol) + "]= " + str(currentScore)  )
                            print ("NEXT:    tb[t=" + str(row) + "][class=" + str(col) + "]= prevClass = " + str(prevTB) )
                else:
                    # c) Choose the best of SELF and NEXT (from previous target symbol)  transitions
                    #    Compare based on sum of path score PLUS transition costs
                    targetCol     = targets_[col] # Vital. Indirect. 'col' pulls indices out of the target vector to retrieve log_probs from.
                    currentScore  = logprobs_[row][targetCol].item()

                    prevSelfScore = self.score[row-1][col].item()   + selfTransitionCost
                    prevNextScore = self.score[row-1][col-1].item() + nextTransitionCost

                    if prevSelfScore > prevNextScore:
                        # self transition is best
                        prevTB    = col
                        prevScore = prevSelfScore
                        transition = "self"
                    else:
                        # next transition is best
                        prevTB    = col - 1
                        prevScore = prevNextScore
                        transition = "next"

                    self.score[row][col] = prevScore  + currentScore
                    self.tb   [row][col] = prevTB
                    newScore = self.score[row][col].item()

                    if verbose_ > 0:
                        print ( "col= " + str(col) + " (" +str(transition) + "): score[t=" + str(row) + "][class=" + str(col) + "]= " + str(newScore) + " computed as " + \
                            "self.score[" + str(row - 1) + "][" + str(col - 1) +"] " + str(prevScore) + \
                            " + logprobs_["+str(row)+ "][" + str(targetCol) + "]= " + str(currentScore)  )
                        print ( "col= " + str(col) + " (" + str(transition) + "):    tb[t=" + str(row) + "][class=" + str(col) + "]= prevClass = " + str(prevTB) )

            if verbose_ > 0:
                torch.set_printoptions(sci_mode=False)
                print ("\nAFTER : row[t=" + str(row) + "]=" + str(self.score[row,:lastCol+1]))
                print ("AFTER :  tb[t=" + str(row) + "]=" + str(self.tb[row,:lastCol+1]))

        # Finish up, retrieve best path via TraceBack array 'tb'
        path, symbolBoundaries = self.bestPath(targets_, logprobs_, nFeatureVectors, verbose_)

        return path, symbolBoundaries, factor

    def bestPath(self, targets_, logprobs_, nFeatureVecs_, verbose_ = 0):
        assert nFeatureVecs_ > 0
        if verbose_ > 0:
            print ("bestPath() tb.shape  = " + str(self.tb.shape))
            print ("bestPath() self.nTargets =  " + str(self.nTargets ) + " self.nFeatureVectors =" + str(nFeatureVecs_))

        row = nFeatureVecs_ - 1
        targetIdx = self.nTargets - 1

        # 1) get the traceback of symbols
        pathsSyms  = []
        pathScores = []

        symbolBoundaries = [ ] # Such that we can seperate 'oo' and 'ee'
        prevTargetIdx = -1

        while row >= 0:
            # Get symbol & score
            chIdxInAlphabet = targets_[targetIdx].item()

            #myscore = self.score[row][chIdxInAlphabet].item() # Path score (incl transition), or do we just want the instantaneous logProbs from LSTM?
            myscore = logprobs_[row][chIdxInAlphabet].item()

            # Whenever we change the targetIdx (compared to prevTargetIdx) that is the begin of a new symbol
            if verbose_ > 1:
                print ("class[t=" + str(row) + "]= " + str(targetIdx) + " chIdxInAlphabet=" +str(chIdxInAlphabet) )

            # Save data
            pathsSyms.insert (0, chIdxInAlphabet)
            pathScores.insert(0, myscore)

            if prevTargetIdx != targetIdx:
                symbolBoundaries.insert(0, [row, self.alphabet.idx2ch(chIdxInAlphabet) ] )

            # Shift
            prevTargetIdx = targetIdx
            targetIdx = self.tb[row][targetIdx].item()
            row -= 1

        assert len(symbolBoundaries) > 0

        # 2) Generate a string from result
        # We have the path in result. An alignment for every character to every feature vector
        alignedStr = self.alphabet.idx2str(pathsSyms, noDuplicates=False, noBlanks=False)

        assert nFeatureVecs_ == len(alignedStr), "ERROR: The generated forced align string should have same length as the number of feature vector==seqLen"

        # 3) Generate a path incl. score. This is a Python list of lists [label, score]
        path = self.pathFromStringAndLogprobs(alignedStr,  pathScores)

        return path, symbolBoundaries

# --------------------------------------
'''
    log probs shape = torch.Size([78, 68])

     where 68 is alphabet size, so 78 time steps
    echo "*door dien het soo mistich was Doch de zee begon te slechten,*" | wc
      1      12      63

    Fun, not ideal, SOMETIMES, we compressed too much. We could have 49 feature vectors but 63 targets.
    If we have more targets than feature vectors, then we duplicate/n-plicate each input/feature vector
    and we go from 49 to 98 feature vectors to align vs 63 targets.
'''

# Class #0 is the blank,
mylogprobs = torch.tensor([[    -0.0000,    -13.9620,    -23.1881,    -18.6309,    -21.9276,
            -21.4505,    -15.4502,    -21.7958,    -17.4019,    -20.4657,
            -21.2391,    -18.8014,    -25.8747,    -20.6013,    -29.6001,
            -22.9801,    -17.6486,    -20.2540,    -20.9769,    -21.6086,
            -17.7701,    -14.7663,    -18.6713,    -18.0158,    -18.4053,
            -24.8438,    -36.9585,    -14.0385,    -20.4455,    -20.6708,
            -15.1912,    -19.2747,    -22.5260,    -20.1671,    -21.7179,
            -19.6354,    -22.8083,    -27.2353,    -18.7717,    -20.1562,
            -24.7157,    -18.7732,    -23.9670,    -19.5851,    -21.0046,
            -29.2015,    -14.8001,    -26.2165,    -19.1668,    -19.2888,
            -21.0671,    -21.8788,    -26.5340,    -23.7708,    -25.6324,
            -16.1936,    -25.9904,    -24.0358,    -20.3967,    -15.2838,
            -29.5235,    -20.0191,    -20.4623,    -25.8248,    -20.7026,
            -17.5870,    -26.7759,    -24.8957],
        [    -0.0001,    -11.4726,    -15.8401,    -17.2140,    -16.4461,
            -18.1099,    -11.4369,    -16.1479,    -11.7512,    -12.9534,
            -15.3252,    -12.8600,    -19.6133,    -15.6562,    -23.5516,
            -18.1197,    -11.9423,    -12.3900,    -18.1929,    -15.7402,
            -12.0273,    -12.1579,    -13.6992,    -13.0294,    -11.9877,
            -16.0261,    -26.9402,    -11.9732,    -15.6247,    -14.0538,
            -11.8318,    -15.5793,    -16.8586,    -15.8972,    -16.7129,
            -13.0778,    -16.9743,    -20.0113,    -14.9004,    -15.9663,
            -18.4604,    -12.4369,    -20.2782,    -15.6113,    -14.5692,
            -23.1558,    -10.8254,    -20.8961,    -15.6453,    -15.7780,
            -18.7081,    -15.6663,    -21.2306,    -17.7183,    -21.3342,
            -12.4544,    -17.0469,    -19.5063,    -16.8705,    -11.1556,
            -23.2844,    -16.3018,    -13.9419,    -18.4861,    -14.3801,
            -13.9396,    -20.7307,    -18.9278],
        [    -0.4946,    -10.9890,     -9.6145,    -14.4803,    -11.6633,
            -14.9429,    -10.5413,     -8.8707,    -10.6186,     -5.6293,
             -8.1630,     -7.2985,    -14.2162,     -7.9735,    -15.6695,
            -11.9873,     -4.4205,     -8.7767,    -18.4433,    -14.3625,
             -4.3399,    -15.4782,     -8.1939,     -3.8534,     -6.4781,
            -11.0122,    -17.2646,    -13.0473,    -10.3627,     -7.5607,
             -8.1184,    -12.9345,    -11.0987,    -11.2792,    -18.3102,
             -1.1110,    -10.7940,    -18.3144,    -12.5329,     -7.8087,
            -12.9726,     -6.7561,    -16.2870,    -12.1989,     -7.9480,
            -19.1146,     -8.3319,    -13.3906,    -14.7972,    -12.2693,
            -16.1299,    -11.4283,    -16.3586,    -15.7596,    -20.6447,
             -7.9520,     -8.5888,    -16.9559,    -16.4546,     -5.9243,
            -17.7056,    -17.8127,     -9.6809,    -10.0776,     -6.9463,
             -9.0122,    -16.8663,    -14.4187],
        [    -6.2827,    -10.1348,    -16.0833,     -4.8323,    -10.1315,
            -11.7466,    -15.4324,    -11.9646,    -16.0185,    -14.8263,
             -7.2877,     -7.9080,    -10.0412,     -0.0539,     -8.9935,
             -7.9229,     -7.4328,    -16.1447,    -15.4627,    -19.3244,
            -18.2096,     -9.7765,    -11.3556,     -8.0125,    -11.5944,
            -19.3104,    -13.9115,     -9.1632,     -3.2749,    -15.5419,
             -7.4921,    -13.6233,     -8.2047,    -13.3247,    -10.8381,
             -9.7789,    -10.3945,    -13.5411,    -15.5035,    -11.2470,
             -9.4846,    -20.5172,     -8.4498,    -10.4502,    -15.0523,
             -8.8800,    -15.6665,     -8.4255,    -11.6221,     -9.3056,
            -10.0819,    -10.6422,    -10.6564,    -17.4891,    -14.0052,
            -14.7638,    -17.0827,    -11.8645,    -14.1961,    -10.1833,
            -12.2612,    -13.6331,    -10.8048,    -15.0639,     -8.9064,
            -10.6173,    -15.2501,    -15.2483],
        [   -15.6361,    -10.5647,    -15.0561,     -6.9288,    -10.4569,
             -9.5331,    -19.9755,    -13.3040,    -16.4132,    -12.4006,
             -7.2408,    -15.3834,     -7.9693,     -8.0795,     -0.0181,
             -4.9266,    -17.6903,    -19.9832,    -12.4510,    -13.3125,
            -18.5628,    -16.0449,    -14.1777,    -12.5487,    -19.6402,
            -17.3650,     -6.4454,    -15.9815,    -11.2758,    -16.9176,
            -17.7083,    -14.1667,     -9.4183,    -14.5153,    -10.9065,
            -17.6018,    -11.7358,     -8.9915,    -19.0328,    -11.5257,
            -12.5652,    -23.4341,     -9.6887,    -13.4318,    -17.6025,
             -8.6890,    -15.0005,     -5.4967,    -18.0330,    -15.8637,
             -9.5577,    -14.1192,     -6.2994,    -11.9831,    -11.2011,
            -17.9400,    -15.4261,    -11.8279,    -13.8567,    -17.0951,
             -9.3317,    -15.1781,    -13.3076,    -15.7813,    -12.5788,
            -14.4926,    -11.6221,    -14.2279],
        [   -13.4288,    -14.2686,    -13.5572,    -15.4986,    -15.2427,
            -13.2128,    -21.5313,    -17.6004,    -19.9259,    -10.4971,
             -6.8827,    -13.1532,    -11.6676,    -10.9067,     -6.8297,
             -0.0061,    -20.1170,    -14.7514,    -21.1209,    -10.6545,
            -15.6043,    -23.0945,    -19.5655,    -12.9961,    -19.4681,
            -13.5224,     -5.8492,    -18.3368,    -12.5394,    -15.3374,
            -20.7299,    -16.4863,     -7.3363,    -16.5417,    -11.5917,
            -17.7114,    -14.9105,    -10.4403,    -24.6723,     -8.6923,
            -17.2287,    -20.6917,    -19.3398,    -18.0148,    -17.8481,
            -13.3084,    -17.7747,    -11.3117,    -22.6014,    -20.4626,
            -18.7715,    -10.3260,     -9.1453,    -14.6719,    -19.3054,
            -19.3772,    -12.9897,    -18.1605,    -19.9882,    -15.0028,
            -11.3880,    -19.6706,    -11.8692,    -13.5127,    -11.7671,
            -14.6744,    -14.9828,    -16.1172],
        [    -0.1324,     -4.8511,    -24.6525,     -2.1615,    -18.3575,
             -8.3724,    -15.3594,    -12.1787,    -12.6762,    -17.5954,
            -14.5456,    -17.2110,    -13.6157,     -7.6002,    -11.6953,
             -8.3318,    -15.7798,    -27.1874,     -9.7153,    -12.5975,
            -14.8952,    -13.8032,    -18.0219,    -13.2494,    -21.6650,
            -26.0789,    -24.2324,    -13.7641,    -16.1092,    -22.5320,
            -19.4452,    -16.0383,    -19.0671,    -15.6665,    -13.2506,
            -21.1427,    -16.8551,    -20.1829,    -20.1859,    -16.3328,
            -21.7043,    -25.9120,    -14.2631,    -16.8418,    -24.3504,
            -19.2556,    -13.1559,    -14.8045,    -19.2278,    -20.1909,
            -12.1050,    -22.9101,    -17.7996,    -17.9139,    -20.3568,
            -17.0323,    -28.0614,    -17.1933,    -16.7506,    -16.7502,
            -17.0497,    -21.0720,    -20.8708,    -28.6884,    -18.7893,
            -14.9290,    -21.9944,    -22.1102],
        [    -7.5949,     -6.9717,    -20.4578,     -0.0047,    -17.1972,
            -15.1984,    -19.1429,    -12.1264,    -14.3156,    -16.5230,
            -15.6340,    -24.6156,     -9.6382,     -7.2974,     -6.9368,
            -10.1337,    -13.3587,    -24.1779,     -8.1345,    -10.9298,
            -21.2738,     -9.6228,    -13.6491,    -12.9340,    -24.8387,
            -27.8287,    -21.8744,    -19.2303,    -15.2508,    -22.5907,
            -20.8450,    -19.1065,    -16.6523,    -21.3369,    -11.6004,
            -23.8109,    -14.7156,    -12.4496,    -19.4288,    -18.5934,
            -16.6047,    -29.9225,     -8.1760,    -17.0307,    -21.2093,
            -14.5482,    -14.3342,    -10.0334,    -19.8275,    -17.5253,
             -7.1346,    -21.9107,    -13.0562,    -13.6357,    -18.9734,
            -17.0682,    -25.9709,    -12.6886,    -14.6402,    -23.0029,
            -19.0416,    -19.6301,    -21.2553,    -28.1669,    -18.0178,
            -20.3169,    -19.9099,    -20.5499],
        [    -0.0004,     -9.4316,    -29.4794,     -8.1347,    -16.9531,
            -15.2945,    -12.5565,    -15.2880,    -10.9378,    -17.0578,
            -18.6114,    -19.2904,    -22.3876,    -14.3084,    -17.9206,
            -19.0847,    -17.3466,    -25.4043,    -17.7635,    -16.4040,
            -16.4408,    -14.1893,    -22.0548,    -21.3925,    -27.4252,
            -23.5834,    -31.0259,    -22.5184,    -25.4093,    -29.9841,
            -24.8228,    -24.5202,    -29.0336,    -18.9434,    -18.3274,
            -23.1171,    -25.9857,    -22.4207,    -23.4583,    -26.9300,
            -29.5777,    -30.2463,    -23.0079,    -23.3361,    -28.4776,
            -30.2140,    -15.5809,    -26.2277,    -25.5164,    -27.0989,
            -21.3204,    -28.6737,    -27.1978,    -22.9120,    -26.4182,
            -23.5499,    -32.2680,    -26.1991,    -24.1505,    -20.6374,
            -27.1646,    -27.9199,    -24.6510,    -37.1778,    -25.3554,
            -21.6086,    -29.8170,    -30.7009],
        [    -3.9961,     -8.3438,    -26.0515,     -0.1343,     -7.6161,
             -2.3468,     -6.5790,     -9.5163,     -8.1329,    -18.3629,
            -11.5450,     -9.5137,    -15.5586,     -5.0435,     -7.4068,
            -11.6135,    -15.1750,    -22.6142,    -10.2436,    -13.4975,
            -16.7040,     -7.5507,    -17.8237,    -20.5354,    -21.5875,
            -16.5190,    -16.1588,    -12.7034,    -14.7810,    -27.6047,
            -19.2365,    -18.2300,    -20.8913,    -11.0051,     -6.6282,
            -20.7415,    -20.3777,    -12.9108,    -15.7513,    -23.9942,
            -21.6297,    -28.5265,    -12.4118,    -15.8646,    -26.4825,
            -14.7726,    -15.0135,    -17.0458,    -14.5146,    -18.5614,
            -14.7170,    -21.1839,    -18.1320,    -16.5503,    -10.4135,
            -23.2293,    -28.7316,    -17.0910,    -14.9053,    -14.3156,
            -11.9599,    -16.1616,    -17.5687,    -29.9907,    -21.0201,
            -14.6905,    -19.0646,    -22.3658],
        [    -5.3402,    -10.3217,    -17.6965,    -11.3692,    -12.7159,
             -9.6010,    -13.3772,    -10.7736,     -7.3395,    -10.1046,
             -5.9859,     -0.4383,     -8.4269,     -1.0827,    -14.0432,
             -6.6766,     -5.5813,    -11.7480,    -17.1838,    -13.5729,
            -13.6456,    -11.6680,    -19.0926,     -9.5344,     -8.6822,
            -12.9622,    -10.5896,     -9.3912,     -6.1663,    -16.1084,
            -14.2465,    -17.1981,    -12.4087,    -14.4326,     -9.5986,
            -13.3819,    -12.2022,    -14.9239,    -21.3186,    -19.7328,
            -16.5564,    -19.3686,    -19.4955,    -15.3480,    -20.3536,
            -12.3935,    -18.3175,    -17.8502,    -18.5417,    -19.7710,
            -22.4060,    -13.1004,    -21.3210,    -19.0785,    -19.3805,
            -22.8529,    -15.8072,    -21.7838,    -20.6478,    -10.0628,
            -12.1529,    -18.9956,     -8.7709,    -21.6944,    -10.1350,
            -14.7098,    -19.2770,    -18.2895],
        [    -9.8252,     -7.8621,    -19.9293,    -15.5386,    -15.0172,
            -10.6343,     -8.5348,    -11.3716,     -0.0095,     -9.1240,
            -12.6243,     -4.8011,    -10.8811,     -8.9035,    -17.5596,
            -16.3650,     -9.0474,    -13.7190,    -12.9874,    -10.7939,
            -11.5569,     -9.4320,    -21.9260,    -16.4849,    -10.5473,
            -10.3611,    -14.7782,    -12.9952,    -15.1265,    -18.0877,
            -19.2918,    -17.6943,    -22.3664,    -12.7193,    -11.8329,
            -15.8678,    -15.4012,    -15.5590,    -19.1427,    -28.1225,
            -18.9412,    -17.1356,    -22.6317,    -16.6830,    -21.5677,
            -18.0920,    -15.2177,    -24.2262,    -18.6743,    -23.4307,
            -24.2330,    -18.8409,    -26.4314,    -16.4399,    -19.0051,
            -23.1401,    -16.6433,    -21.4905,    -20.2080,    -15.6295,
            -15.3201,    -20.4272,    -12.4130,    -26.7949,    -15.1888,
            -17.7490,    -20.9683,    -20.4042],
        [    -9.5136,     -8.6398,    -24.8047,    -16.6156,    -15.1191,
            -10.5206,     -0.5299,    -11.5824,     -0.8905,    -16.2451,
            -22.1144,     -8.5709,    -22.0522,    -15.0371,    -25.1624,
            -26.9797,    -12.0482,    -17.3405,    -13.1108,    -14.0256,
             -9.1848,     -8.6311,    -22.9108,    -24.1546,    -13.5084,
            -10.8440,    -24.2074,    -14.6059,    -22.5047,    -23.1083,
            -19.2554,    -20.9590,    -31.5645,     -9.5410,    -15.1789,
            -14.6785,    -22.6264,    -20.3565,    -12.3869,    -32.8234,
            -24.4382,    -15.6895,    -23.4566,    -17.8088,    -22.5859,
            -25.4447,    -13.5708,    -31.5663,    -14.6136,    -20.8502,
            -26.2083,    -23.9960,    -31.3873,    -19.4451,    -17.9948,
            -21.5324,    -23.1241,    -21.2812,    -19.2012,    -15.7672,
            -20.9305,    -21.1034,    -18.9201,    -29.9390,    -22.6819,
            -15.6674,    -25.3072,    -24.6679],
        [    -4.4860,     -1.6541,    -24.3490,    -10.6418,    -12.6634,
            -11.0211,     -0.2357,    -16.4465,    -11.1416,    -19.0771,
            -23.5526,    -14.0811,    -28.1013,    -16.7974,    -23.6779,
            -23.1857,    -16.1815,    -20.1965,    -15.0376,    -16.1673,
             -8.8594,     -8.3787,    -23.1356,    -23.7773,    -20.8898,
            -16.4946,    -29.8158,    -15.5594,    -23.5424,    -24.9547,
            -13.7554,    -21.0791,    -28.3185,     -4.9928,    -18.3895,
            -12.1060,    -25.8282,    -21.1519,     -9.9324,    -23.2819,
            -26.6106,    -17.9080,    -19.5848,    -14.7122,    -19.4194,
            -28.2247,    -12.1392,    -30.3147,    -13.1742,    -12.9956,
            -19.7423,    -21.1665,    -23.1179,    -22.6794,    -18.4673,
            -15.6940,    -26.0595,    -16.5477,    -18.1736,    -15.1028,
            -27.4731,    -21.9079,    -22.5986,    -26.2684,    -23.7298,
             -8.6696,    -27.5975,    -26.4396],
        [    -5.1086,     -0.0070,    -14.7093,     -9.4544,    -12.9564,
            -12.0013,     -9.9544,    -11.4388,    -10.2761,    -10.0027,
            -17.1023,    -17.2591,    -14.7357,    -13.0384,    -13.7848,
            -13.3167,    -11.0535,    -18.3150,    -10.4023,     -9.9506,
             -7.5649,    -10.6165,    -14.7702,    -11.6911,    -16.9346,
            -16.4402,    -22.3568,    -14.5856,    -17.5302,    -14.9417,
            -15.6742,    -19.1595,    -18.2920,    -12.5387,    -17.8080,
            -13.3498,    -15.3654,    -16.2982,    -12.2900,    -16.6940,
            -20.4481,    -15.7691,    -14.6068,    -13.6560,    -14.0514,
            -20.9237,     -9.5520,    -18.1999,    -19.5632,    -14.4427,
            -15.0361,    -20.0010,    -18.0651,    -14.9316,    -19.4847,
            -14.0900,    -16.8566,    -15.7064,    -15.7215,    -16.3316,
            -24.6672,    -19.8402,    -17.9409,    -21.8789,    -16.2736,
            -12.1069,    -21.7817,    -20.8797],
        [    -3.9709,    -12.0596,    -11.1482,     -1.4596,     -5.0161,
            -10.1651,    -13.4567,     -0.9305,    -13.8993,     -8.7749,
            -10.1896,    -17.8348,    -10.5925,     -4.8876,     -1.6135,
             -4.3806,     -5.9237,    -16.2199,    -11.5526,     -8.8652,
            -10.4777,    -11.7705,     -2.2532,     -5.8594,    -18.3989,
            -13.9990,    -11.0965,    -17.8245,    -11.3542,    -14.8450,
            -19.0103,    -20.8756,     -9.1030,    -20.1287,    -11.6565,
            -14.0445,    -12.0668,     -8.5256,    -11.9788,    -12.6347,
            -16.9203,    -20.4781,     -5.7019,    -16.5534,    -12.5374,
            -11.8173,    -10.1497,     -4.2956,    -19.8446,    -13.4251,
            -10.5029,    -19.5678,    -12.4870,     -9.2922,    -14.0989,
            -15.6246,    -15.9240,    -15.1857,    -12.1840,    -12.8954,
            -16.1800,    -15.6794,    -15.9102,    -19.2740,    -13.6195,
            -15.5518,    -15.0160,    -17.4236],
        [   -10.9486,    -10.2928,    -11.7610,    -26.4067,    -19.1013,
            -24.1652,    -20.4107,    -20.3830,    -11.7829,     -0.0014,
             -8.8042,    -10.4948,    -12.5285,    -13.7036,    -18.6473,
            -10.8185,    -11.4007,     -8.0922,    -27.6864,    -12.7681,
            -10.6577,    -21.5576,    -23.5829,    -10.4859,    -13.6290,
            -10.8513,    -14.8159,    -23.4167,    -16.6824,    -11.9984,
            -19.7362,    -21.7784,    -15.2028,    -18.1110,    -22.1087,
            -11.2192,    -15.6675,    -17.1807,    -29.1099,    -17.1684,
            -19.4019,    -15.6437,    -30.9012,    -20.2597,    -15.5849,
            -24.9473,    -17.4919,    -24.4700,    -30.2810,    -26.5153,
            -29.1594,    -13.1696,    -21.9375,    -20.1660,    -33.2739,
            -20.8967,     -7.4563,    -27.1369,    -29.5920,    -17.7990,
            -25.3144,    -29.9441,    -10.6903,    -18.2232,     -9.3476,
            -19.6377,    -24.6592,    -22.4772],
        [    -0.0547,     -9.1877,    -24.0313,    -15.8575,    -14.0487,
            -13.5235,    -12.5466,    -17.7564,    -10.3192,    -10.6787,
             -9.1480,     -2.9410,    -19.2411,     -9.6682,    -20.7572,
            -13.8596,    -11.5818,    -16.1950,    -24.1786,    -20.8485,
            -12.5403,    -15.5962,    -25.8335,    -15.1378,    -15.4511,
            -16.1310,    -22.1669,    -13.8696,    -15.5487,    -22.1939,
            -15.3419,    -20.6545,    -21.1621,    -12.2794,    -19.7634,
            -12.8611,    -21.1886,    -24.1476,    -25.6038,    -21.4009,
            -25.5597,    -21.4835,    -28.9293,    -18.6164,    -24.3072,
            -26.4905,    -18.4908,    -28.6058,    -24.2614,    -24.2460,
            -28.1510,    -18.3914,    -27.6178,    -27.5338,    -26.6466,
            -24.3914,    -21.3507,    -29.7037,    -27.9777,    -11.2236,
            -23.1673,    -26.6320,    -13.6760,    -27.1773,    -15.4406,
            -15.0652,    -27.7257,    -26.7912],
        [    -9.4661,     -6.0484,    -17.6462,    -14.5059,    -17.1846,
            -10.1431,    -13.6305,    -12.3308,     -0.0228,     -7.4847,
             -8.0080,     -4.9725,     -4.6239,     -6.3944,    -12.7653,
            -12.0820,    -10.1320,    -14.0939,    -10.1159,     -9.7393,
            -13.4228,    -10.5193,    -20.5504,    -11.9110,     -8.3420,
            -13.1641,    -12.9793,    -10.3132,    -10.8844,    -14.4954,
            -17.2218,    -12.9869,    -17.5410,    -13.1080,     -9.7641,
            -16.5131,    -10.0572,    -14.4333,    -21.0460,    -23.3858,
            -13.8199,    -17.3842,    -20.3957,    -14.3517,    -20.9609,
            -13.7241,    -13.9190,    -18.1890,    -18.3880,    -23.7457,
            -20.2863,    -15.9635,    -21.3205,    -13.8715,    -18.1312,
            -20.9010,    -14.1011,    -18.9926,    -17.9373,    -15.5033,
            -11.4101,    -18.2202,     -9.1846,    -23.5366,    -10.8775,
            -17.3352,    -17.1554,    -16.3514],
        [   -10.8550,    -12.2986,    -23.9590,    -14.9816,    -19.0495,
             -9.1710,    -12.2250,    -10.1652,     -0.0003,    -11.6717,
            -14.4872,    -12.2562,    -11.2116,    -12.7657,    -13.7994,
            -18.0802,    -16.1303,    -21.7782,     -9.5226,     -9.8041,
            -14.9621,    -13.2589,    -21.0339,    -19.8919,    -15.7995,
            -14.9938,    -17.7696,    -16.6568,    -20.2336,    -21.7744,
            -27.3270,    -17.6897,    -26.3487,    -16.9357,    -12.2027,
            -23.4543,    -17.0780,    -17.7160,    -21.4473,    -31.0798,
            -21.2643,    -22.1807,    -22.7939,    -20.5316,    -27.3394,
            -19.8434,    -13.5560,    -21.2812,    -21.2212,    -29.7886,
            -22.8031,    -26.2795,    -27.6820,    -13.4616,    -18.2374,
            -25.0511,    -22.3118,    -22.7461,    -18.6032,    -20.0971,
            -13.3281,    -21.0621,    -16.6610,    -32.3046,    -20.0323,
            -22.8994,    -20.3238,    -21.9077],
        [    -2.9016,     -3.9676,    -14.8745,     -5.6672,    -11.6027,
             -9.3324,     -5.3166,     -7.7193,     -1.0022,     -8.1735,
            -11.4790,     -9.8441,     -8.2475,     -5.6857,    -11.1121,
            -12.9430,     -6.9070,    -11.4384,     -5.8720,     -5.9143,
            -13.0753,     -0.6255,    -12.4600,    -13.8301,    -11.0013,
            -12.6802,    -16.8859,     -9.4731,    -10.7106,    -15.6900,
            -13.7441,    -13.1589,    -16.2079,    -11.4735,     -5.5895,
            -15.0355,    -11.7081,     -8.7045,    -11.4165,    -21.0427,
            -11.6502,    -16.8774,    -11.7779,    -11.3232,    -15.9826,
            -12.5090,     -8.1849,    -15.2835,    -10.2695,    -13.7387,
            -11.1888,    -14.4658,    -16.4672,     -9.1125,    -12.5653,
            -14.3427,    -16.7968,    -10.7553,    -10.5375,    -13.9813,
            -13.3505,    -12.1654,    -10.9891,    -22.4099,    -12.9328,
            -14.0875,    -15.9059,    -16.1202],
        [    -9.1999,     -6.7504,    -21.2064,     -8.4829,    -14.8647,
            -15.4958,    -10.0911,    -18.4414,    -12.6262,    -21.2652,
            -19.3789,    -12.4309,    -16.0417,     -9.0983,    -21.9576,
            -18.4401,    -10.5305,    -15.1669,    -12.8859,    -17.0128,
            -23.5812,     -0.0023,    -19.9550,    -20.4629,    -15.6473,
            -21.4049,    -24.5103,     -7.7351,    -10.2115,    -22.3918,
            -11.0405,    -19.2829,    -17.9743,    -15.7702,     -8.9408,
            -20.9677,    -18.0812,    -13.7059,    -14.1371,    -25.4376,
            -16.1343,    -23.5547,    -13.4894,    -13.0032,    -20.2277,
            -13.5842,    -18.8637,    -23.0263,    -10.6951,    -10.7334,
            -15.0698,    -14.9115,    -19.8341,    -19.7259,    -15.0592,
            -20.4346,    -24.8485,    -13.2051,    -14.9886,    -17.3665,
            -21.3648,    -13.0104,    -15.8154,    -25.8852,    -18.4795,
            -15.4495,    -22.3639,    -21.3951],
        [    -9.9251,    -11.0645,    -21.2399,    -15.0197,    -19.1785,
            -10.9099,     -9.9308,     -8.5852,     -0.0007,    -11.9040,
            -14.6043,     -9.9018,    -11.0490,     -8.8805,    -16.4668,
            -18.0205,    -11.6439,    -17.3109,    -11.2322,     -9.5818,
            -10.3780,    -14.4156,    -18.7196,    -15.9627,    -10.8608,
            -13.1089,    -17.4606,    -16.2956,    -17.3550,    -17.8361,
            -21.7095,    -16.0220,    -24.3831,    -14.8849,    -10.8853,
            -16.1416,    -14.5285,    -17.4633,    -18.3292,    -26.4754,
            -17.3396,    -17.2065,    -20.7631,    -18.7046,    -22.5311,
            -18.9987,    -12.7491,    -20.3566,    -17.3999,    -26.1172,
            -22.2351,    -21.6581,    -25.2074,    -14.0046,    -20.0518,
            -19.7539,    -18.7592,    -19.6894,    -17.9859,    -16.4865,
            -12.4192,    -21.2714,    -15.1528,    -26.4523,    -16.3478,
            -18.8448,    -19.5917,    -18.9769],
        [    -9.5796,     -3.9636,    -25.3593,    -13.3801,    -15.8211,
             -6.8645,     -7.5726,    -18.1823,     -3.5970,    -13.6741,
             -8.9134,     -0.0823,    -13.2020,     -4.3471,    -15.7792,
            -14.0647,    -17.1174,    -17.5841,    -16.4488,    -15.3259,
            -14.6713,    -11.9366,    -28.9635,    -21.2275,    -12.0747,
            -12.6402,    -14.3471,    -11.9154,    -12.5012,    -22.2605,
            -14.2049,    -12.6932,    -22.1717,     -4.0452,     -8.0552,
            -13.0553,    -17.5830,    -16.4511,    -22.1469,    -23.8714,
            -16.1256,    -21.7494,    -23.3034,    -13.5289,    -25.8474,
            -17.2069,    -17.1696,    -24.5288,    -12.7701,    -22.5565,
            -22.6910,    -13.0808,    -19.0386,    -20.5934,    -17.7027,
            -21.9146,    -20.3673,    -16.7414,    -20.9241,    -14.2457,
            -10.0841,    -21.3487,    -11.1158,    -23.7707,    -14.3261,
            -11.3639,    -20.7613,    -20.4783],
        [   -10.1981,     -0.0002,    -22.0545,    -15.7833,    -22.0376,
            -16.1523,    -15.7049,    -20.9548,    -10.4077,    -15.0009,
            -16.8353,    -11.3514,    -14.2362,    -10.2056,    -23.0491,
            -18.6424,    -13.8157,    -20.7754,    -16.7740,    -17.9731,
            -13.4376,    -14.6850,    -26.1030,    -14.6003,    -12.7707,
            -22.6602,    -25.6926,    -13.9844,    -15.7432,    -17.6086,
            -11.7582,    -16.3245,    -22.4742,    -11.0903,    -19.1011,
            -12.8437,    -15.6017,    -24.1128,    -21.4881,    -20.4432,
            -17.7876,    -18.9254,    -21.9851,    -14.0454,    -20.5685,
            -22.4179,    -17.0209,    -25.4177,    -18.5958,    -19.9488,
            -19.8075,    -17.6503,    -21.5320,    -24.2825,    -26.5237,
            -17.1388,    -19.4383,    -17.5444,    -22.5415,    -18.5892,
            -23.0131,    -25.3088,    -17.2551,    -23.3942,    -14.7401,
            -13.1665,    -25.8589,    -22.1935],
        [   -12.3286,    -11.0920,    -27.8706,    -19.1973,    -22.6472,
            -10.1102,     -9.5697,    -11.9940,     -0.0002,    -18.0516,
            -22.7717,    -14.3203,    -17.6506,    -18.0625,    -23.1552,
            -25.8978,    -18.5290,    -25.0772,     -9.9785,    -12.3653,
            -10.9901,    -16.5943,    -24.1781,    -23.3899,    -15.4129,
            -16.1582,    -25.4455,    -16.8191,    -25.9917,    -23.0432,
            -27.4410,    -20.0438,    -33.6067,    -16.5608,    -15.8367,
            -23.5612,    -20.8656,    -24.3034,    -18.9237,    -34.7820,
            -26.4578,    -19.1895,    -25.7649,    -22.5186,    -28.7414,
            -25.9553,    -14.2883,    -28.4586,    -21.4195,    -31.3557,
            -27.8031,    -30.7876,    -33.7606,    -17.4750,    -21.0420,
            -24.9439,    -25.7084,    -25.2457,    -20.0239,    -20.7794,
            -18.5272,    -23.5405,    -21.6890,    -34.8409,    -24.8934,
            -21.9215,    -24.3588,    -24.5526],
        [    -0.0005,     -9.0139,    -23.1254,    -10.4185,    -18.2288,
            -13.3896,     -9.8668,    -10.3572,     -8.9361,    -19.9715,
            -22.1432,    -16.6486,    -19.5652,    -13.4175,    -24.0788,
            -21.3666,    -10.3922,    -22.3783,    -10.5013,    -16.6335,
            -12.0022,     -9.8627,    -13.9444,    -14.4703,    -14.4375,
            -22.4485,    -32.9120,    -10.4670,    -17.7297,    -19.4623,
            -16.3804,    -19.8774,    -24.1024,    -18.7982,    -16.9323,
            -18.7362,    -18.2409,    -24.7347,    -12.9218,    -24.0162,
            -23.3778,    -17.9905,    -16.6548,    -17.9742,    -20.9508,
            -23.5581,    -11.8413,    -22.2916,    -17.0923,    -18.5120,
            -19.1663,    -26.4976,    -28.4017,    -19.2424,    -20.7026,
            -17.2125,    -26.7035,    -21.2596,    -15.6853,    -14.3970,
            -24.9770,    -18.1518,    -20.9701,    -29.5770,    -20.9469,
            -16.3648,    -24.6178,    -23.5928],
        [    -7.0041,    -10.1446,    -21.1592,    -13.0728,    -13.8012,
             -7.6443,     -3.5856,     -6.7635,     -0.0331,    -15.4767,
            -18.6579,    -10.6189,    -16.8355,    -14.6926,    -18.7840,
            -23.2952,    -11.3132,    -17.7985,     -6.7586,    -12.7628,
             -7.2535,     -9.3861,    -14.8955,    -17.2031,    -10.8086,
            -12.2880,    -22.1976,    -10.6734,    -19.5271,    -17.6918,
            -17.3579,    -15.1931,    -26.3638,    -11.1936,    -13.4782,
            -15.0229,    -16.4883,    -19.9410,     -9.5439,    -25.9546,
            -19.9972,    -12.2260,    -17.2936,    -15.6973,    -19.6628,
            -20.4508,     -8.9931,    -21.8945,    -13.2190,    -19.8102,
            -19.8705,    -24.2415,    -26.7565,    -14.0494,    -12.8613,
            -17.0827,    -20.4913,    -18.9001,    -12.9277,    -12.5621,
            -15.6098,    -15.4443,    -16.7949,    -25.5978,    -19.5494,
            -14.6832,    -18.1960,    -18.3253],
        [    -9.7273,     -5.2335,    -21.5611,    -12.4408,    -11.8368,
             -8.5147,     -0.0245,    -11.0700,     -4.2842,    -14.3262,
            -21.1305,    -12.7515,    -22.1447,    -16.6886,    -18.4713,
            -22.3182,    -14.9442,    -17.3134,    -11.4076,    -10.6105,
             -6.0606,     -9.9177,    -20.7740,    -22.8973,    -18.1207,
            -10.2591,    -21.1737,    -17.3873,    -24.3364,    -22.0079,
            -18.6534,    -19.0392,    -28.5172,     -6.1145,    -15.3780,
            -12.9272,    -22.3920,    -17.0226,     -9.8477,    -25.2199,
            -23.8116,    -14.8602,    -19.3199,    -15.7749,    -19.0279,
            -24.5480,    -10.6749,    -26.8141,    -14.1422,    -17.5344,
            -20.2657,    -21.8749,    -23.3393,    -16.4356,    -15.3989,
            -16.9998,    -21.0557,    -16.7699,    -16.5034,    -15.8280,
            -20.0976,    -20.6194,    -20.1668,    -25.3683,    -22.5341,
            -12.0424,    -22.5208,    -22.4789],
        [    -5.8807,     -0.0130,    -17.1424,     -6.4424,    -12.5088,
             -9.0232,     -5.2118,    -11.4290,     -8.8962,    -13.1629,
            -16.9300,    -14.6165,    -16.1577,    -10.7520,    -13.7697,
            -15.0436,    -12.0228,    -18.3745,     -8.7032,    -10.5458,
             -8.3711,     -7.6379,    -15.8318,    -14.6470,    -16.1015,
            -16.5665,    -21.6609,    -13.0833,    -16.4937,    -16.9340,
            -12.4864,    -14.9199,    -19.6159,     -6.6882,    -14.5137,
            -10.5790,    -15.8445,    -15.7068,     -9.2327,    -16.2532,
            -17.2612,    -15.3274,    -12.0220,    -10.7462,    -14.4149,
            -18.7807,     -8.8248,    -17.8472,    -11.8501,    -10.8959,
            -11.3799,    -17.2875,    -15.1531,    -14.8581,    -15.0635,
            -11.0877,    -18.7795,    -10.2943,    -12.7717,    -14.6068,
            -19.4525,    -17.5423,    -17.6213,    -20.0816,    -16.2924,
             -8.4340,    -19.8944,    -18.6056],
        [    -2.1166,     -0.1327,    -19.3899,     -7.9753,    -14.4482,
            -12.3920,     -6.8743,    -12.9919,    -10.5988,    -15.6942,
            -22.1178,    -17.4150,    -19.0066,    -13.1377,    -20.6092,
            -18.2621,    -10.6973,    -19.5647,    -10.5529,    -12.0149,
            -11.0460,     -6.0780,    -16.7332,    -16.5167,    -17.6957,
            -19.7060,    -29.4840,    -12.2791,    -18.2600,    -19.6889,
            -14.4058,    -21.2255,    -22.2986,    -13.4983,    -16.1890,
            -16.7832,    -19.1227,    -18.6731,    -10.8333,    -22.0292,
            -22.8297,    -18.6227,    -15.0802,    -14.4561,    -17.5985,
            -22.7353,    -11.2800,    -23.5990,    -16.4258,    -13.2215,
            -15.5172,    -21.7952,    -22.5272,    -18.0166,    -19.7438,
            -15.6709,    -23.6479,    -16.0556,    -15.5406,    -16.7579,
            -27.4878,    -18.8258,    -20.8106,    -27.1164,    -20.5571,
            -12.9676,    -25.4486,    -24.2615],
        [    -6.9359,     -5.7847,    -13.9582,     -5.7641,     -6.0816,
             -5.4243,     -0.3331,     -2.5342,     -5.4772,    -13.1935,
            -19.7110,    -11.6997,    -15.4099,     -9.8756,    -13.8443,
            -16.1929,     -4.4401,    -14.0324,     -5.0458,     -8.0042,
             -6.8474,     -2.4387,     -9.1929,    -14.5125,    -12.4276,
             -9.3436,    -16.7037,     -8.9973,    -14.0697,    -16.1169,
            -14.9337,    -20.4153,    -18.5507,    -11.1516,    -11.0422,
            -13.3552,    -15.3239,    -11.3595,     -2.5531,    -22.8763,
            -19.4213,    -12.6223,     -8.7720,    -12.3717,    -13.2368,
            -13.8845,     -9.3151,    -17.3689,    -11.3981,     -8.5205,
            -14.8751,    -20.1528,    -21.3297,    -10.8233,     -8.8376,
            -16.2409,    -17.7498,    -13.4094,     -9.5203,    -11.9722,
            -18.2986,    -11.4738,    -16.7253,    -22.3538,    -18.6332,
            -11.3687,    -17.5999,    -18.5337],
        [    -7.9453,    -10.6422,    -13.2296,     -8.8981,     -0.0547,
            -10.1000,     -7.5708,     -7.0442,    -13.1297,     -6.4575,
            -10.0234,     -5.4683,    -15.9694,     -6.5345,     -8.7945,
             -7.8512,     -3.2573,     -9.6702,    -20.0749,    -14.6724,
             -9.3622,     -9.5817,    -13.9600,    -12.2290,    -17.1534,
             -5.5624,     -7.2244,    -15.5290,    -11.6413,    -18.6387,
            -16.4203,    -26.4108,    -12.9963,    -13.1294,    -16.0210,
             -9.7712,    -18.2707,     -9.6656,    -14.5117,    -18.6490,
            -22.5856,    -18.6072,    -16.0341,    -16.0796,    -14.0315,
            -15.9825,    -17.3047,    -18.8049,    -22.3733,    -12.7687,
            -22.4034,    -15.6028,    -20.0044,    -18.5070,    -16.0537,
            -24.1022,    -13.8084,    -23.0435,    -21.4817,    -10.4431,
            -20.9266,    -19.6357,    -12.8827,    -20.5324,    -14.7737,
            -13.4333,    -21.0784,    -23.2521],
        [    -8.2635,     -7.2309,    -17.1748,    -10.4829,    -17.5021,
            -12.3162,    -18.6575,    -11.6000,     -5.6596,     -6.1729,
             -6.4728,     -8.9239,     -2.4506,     -0.1025,     -8.5118,
             -6.4346,     -7.9232,    -16.9492,    -13.8192,     -9.6340,
            -17.2724,    -12.1273,    -18.1924,     -9.3149,    -12.3424,
            -16.6427,    -12.2114,    -15.0919,     -7.9100,    -16.3694,
            -20.7166,    -17.9894,    -14.1300,    -18.5991,     -9.8347,
            -18.1397,    -10.0582,    -12.3832,    -25.3263,    -22.1769,
            -13.5424,    -24.7577,    -18.0055,    -16.8303,    -21.9351,
            -12.3625,    -16.5198,    -14.2184,    -22.5510,    -24.1160,
            -18.9274,    -16.7251,    -19.3314,    -14.3067,    -22.9395,
            -22.9775,    -16.4223,    -19.7073,    -20.6868,    -18.1823,
            -13.2113,    -21.9442,    -10.8762,    -25.8690,    -10.1160,
            -20.5460,    -19.6887,    -19.9010],
        [    -2.3050,     -3.8721,     -9.9780,     -5.4257,     -8.1955,
            -11.9697,     -7.2648,     -5.2168,     -6.4180,     -5.1949,
            -10.6473,    -10.3560,     -7.8430,     -2.0759,     -9.1130,
             -5.7911,     -0.3389,     -7.5520,    -11.2223,     -4.9043,
             -8.6882,     -4.9255,     -9.9832,     -6.3749,    -12.8828,
             -9.5072,    -13.0137,    -13.4588,     -8.2189,    -13.2876,
            -14.8839,    -20.4194,    -11.1007,    -14.8140,     -8.4493,
            -11.4369,    -11.0067,     -6.2399,    -11.9699,    -16.7372,
            -15.0933,    -16.5376,    -11.4096,    -13.8388,    -11.2652,
            -12.7670,    -10.9367,    -13.9466,    -17.4946,    -12.1405,
            -15.3773,    -13.2111,    -16.3463,    -10.7630,    -17.9434,
            -15.5485,    -13.0572,    -15.6145,    -15.2179,    -11.9194,
            -18.4946,    -17.0340,    -11.5458,    -20.0170,    -10.5443,
            -13.7094,    -18.5949,    -17.8487],
        [    -5.7776,    -11.4954,     -7.9854,     -5.4832,     -0.7019,
            -15.9653,     -9.9071,     -8.3379,    -17.1032,     -8.1440,
            -10.4318,    -10.7723,    -13.4088,     -5.8416,     -8.6227,
             -3.7707,     -1.2148,     -3.0044,    -18.6325,    -10.7555,
            -14.4425,     -4.1568,     -8.4896,     -9.5505,    -18.0481,
             -8.0478,     -8.5349,    -14.1224,     -7.0115,    -15.8219,
            -13.2068,    -25.2792,     -5.1641,    -18.9072,     -9.5736,
            -13.3129,    -15.4239,     -2.3099,    -12.2841,    -14.0787,
            -17.5383,    -19.0578,     -9.7761,    -15.0011,     -9.1932,
            -10.4794,    -16.8218,    -13.2706,    -19.0011,     -7.0234,
            -15.8291,     -9.5217,    -13.1177,    -13.9179,    -14.5710,
            -19.5806,    -12.6375,    -17.2545,    -16.6132,    -10.7276,
            -21.8104,    -13.6623,    -11.0089,    -16.5458,    -12.3329,
            -14.2924,    -17.8479,    -19.0126],
        [    -6.7974,    -12.2479,    -19.3362,    -10.3160,     -0.0201,
            -10.3237,     -4.7130,     -8.0405,    -10.6160,    -10.2015,
            -13.6031,     -5.7598,    -21.3439,    -10.2588,    -13.3589,
            -13.6338,     -6.8280,    -12.0387,    -21.7429,    -16.5340,
            -10.5521,     -9.1113,    -17.3165,    -18.9372,    -20.2217,
             -5.3082,    -11.6567,    -17.4468,    -17.2247,    -24.4464,
            -19.6943,    -28.9164,    -20.0335,    -14.2664,    -15.5693,
            -13.7921,    -23.8664,    -11.8776,    -15.7274,    -26.0210,
            -27.9349,    -21.8016,    -20.1390,    -19.8039,    -19.2584,
            -21.2258,    -18.4819,    -25.2588,    -23.3203,    -17.6423,
            -26.6851,    -20.4128,    -25.8305,    -21.0539,    -16.6605,
            -28.2226,    -19.4198,    -26.7187,    -23.4159,    -12.0413,
            -23.3365,    -21.2597,    -16.0549,    -27.3281,    -20.3965,
            -16.1378,    -24.5402,    -27.4682],
        [   -11.1407,    -13.6566,    -26.9613,    -20.8966,    -20.8761,
            -11.8572,    -10.0974,    -11.1465,     -0.0001,    -14.0006,
            -19.8897,    -11.2353,    -17.6714,    -16.2497,    -22.7330,
            -24.2844,    -15.4990,    -22.1064,    -14.6163,    -13.0484,
            -10.3234,    -18.1067,    -24.5874,    -22.1071,    -15.3170,
            -13.0996,    -22.6916,    -19.3942,    -25.3885,    -23.3407,
            -28.6668,    -22.7854,    -32.9016,    -17.7312,    -17.4895,
            -21.3551,    -21.7043,    -23.7791,    -21.9666,    -35.4083,
            -27.7915,    -19.7697,    -29.1504,    -24.4867,    -28.6292,
            -27.5791,    -15.8874,    -30.0267,    -24.9906,    -33.4023,
            -31.4922,    -30.0254,    -36.0375,    -19.1180,    -24.3051,
            -27.4499,    -23.8921,    -29.4568,    -24.0854,    -19.7613,
            -19.5936,    -26.6610,    -20.1606,    -35.1950,    -23.4644,
            -23.6709,    -26.2372,    -26.5786],
        [   -10.1246,    -10.6023,    -28.7099,    -21.6915,    -16.5983,
            -13.7124,     -6.5809,    -21.4027,     -7.5447,    -15.4185,
            -15.0240,     -0.0023,    -23.4706,    -10.4568,    -26.6474,
            -21.1196,    -16.0912,    -16.2024,    -26.5094,    -21.3518,
            -14.5709,    -15.5666,    -33.4709,    -26.1545,    -15.2477,
            -10.6531,    -20.2938,    -16.8526,    -18.4403,    -26.8631,
            -18.0218,    -22.5200,    -28.5460,     -8.3134,    -16.2847,
            -13.2129,    -26.0620,    -22.0865,    -25.0056,    -30.9308,
            -25.9252,    -22.1290,    -32.6677,    -20.0140,    -27.9229,
            -27.1056,    -22.5786,    -36.2031,    -19.5987,    -26.0346,
            -34.4015,    -17.3800,    -30.4314,    -28.5330,    -25.5314,
            -28.1270,    -23.2332,    -27.6705,    -29.7377,    -14.9537,
            -20.8534,    -28.0603,    -15.1392,    -28.9904,    -19.5913,
            -15.5350,    -29.7105,    -28.9504],
        [    -5.2285,     -0.0186,    -24.7578,    -15.4925,    -18.6671,
            -15.2895,    -11.0957,    -22.7116,     -9.7509,    -13.9366,
            -13.9569,     -4.4618,    -17.9757,     -8.4373,    -24.8105,
            -17.9277,    -12.9478,    -18.3168,    -21.0248,    -21.2502,
            -13.0712,    -12.6096,    -29.7577,    -16.7696,    -13.4592,
            -19.4494,    -25.8210,    -12.8095,    -14.6154,    -20.8543,
             -9.5729,    -16.8629,    -23.5006,     -6.8772,    -19.2333,
             -9.7701,    -19.3619,    -24.5422,    -23.0781,    -21.1202,
            -20.2797,    -19.7435,    -26.0120,    -13.7197,    -22.1890,
            -25.2595,    -18.5358,    -30.2550,    -18.0650,    -19.7568,
            -23.7087,    -15.2748,    -23.5145,    -28.2659,    -27.1392,
            -19.2653,    -20.8595,    -20.9797,    -26.0195,    -14.9812,
            -24.0261,    -26.7383,    -15.1454,    -24.6136,    -14.5933,
            -11.0402,    -28.5556,    -25.1754],
        [    -7.3624,     -6.6586,    -22.1446,    -11.1026,    -10.5600,
            -12.8838,     -8.5173,    -17.8433,    -13.0929,    -14.4436,
            -11.6488,     -0.2431,    -17.1462,     -1.5574,    -20.1921,
            -12.7472,     -6.5314,    -13.6736,    -23.0729,    -21.2896,
            -17.5154,     -7.1910,    -25.2752,    -17.3778,    -13.5123,
            -14.3580,    -16.6237,    -10.9241,     -7.4614,    -22.9198,
            -10.4891,    -22.5004,    -17.3170,     -9.7939,    -13.6152,
            -10.6571,    -19.7520,    -16.2532,    -20.3311,    -23.4023,
            -19.2793,    -23.3312,    -20.7484,    -13.8520,    -21.1582,
            -17.0661,    -22.8134,    -26.6356,    -16.1126,    -13.8723,
            -24.3023,    -12.0077,    -22.3679,    -26.6200,    -21.1611,
            -24.7307,    -20.6922,    -20.8324,    -24.7743,    -12.2144,
            -20.6007,    -22.0134,    -12.2794,    -24.1329,    -14.0639,
            -12.0394,    -26.4052,    -25.4315],
        [    -3.1430,     -7.1481,    -20.7313,     -2.7686,    -10.3803,
             -2.3332,    -10.3015,     -5.0439,     -2.0522,     -9.3768,
             -6.0284,    -10.0510,     -7.1697,     -4.3313,     -0.4482,
             -6.6964,    -13.9984,    -20.9539,     -6.2332,     -7.0872,
            -11.8122,    -12.1571,    -14.2402,    -13.0927,    -17.4919,
            -13.8416,    -11.3257,    -14.0442,    -14.3029,    -20.3127,
            -22.4260,    -13.7949,    -17.7339,    -11.9875,     -6.5863,
            -18.5422,    -13.0044,    -10.9833,    -17.8913,    -19.5759,
            -17.4316,    -23.8281,    -13.3397,    -15.7888,    -23.3744,
            -13.7703,     -9.3044,    -10.0861,    -18.2786,    -23.2250,
            -13.8751,    -20.7459,    -15.7439,     -9.8601,    -12.3286,
            -19.6833,    -20.9469,    -17.1140,    -13.8458,    -14.1077,
             -7.6142,    -17.3686,    -13.5023,    -26.5139,    -15.2109,
            -15.7787,    -14.6239,    -17.5651],
        [    -1.5417,     -7.9208,    -14.0934,    -11.3952,    -13.4027,
             -7.4167,    -14.0017,    -10.1460,     -8.3144,     -4.3053,
             -5.1953,    -13.4352,    -10.2959,    -11.9234,     -2.9089,
             -0.4376,    -19.4718,    -15.5877,    -13.4686,     -2.7714,
             -6.1723,    -21.3956,    -15.5620,    -11.2773,    -19.2541,
             -9.5633,    -10.2231,    -18.3022,    -17.9796,    -14.4508,
            -24.0366,    -13.6606,    -12.6033,    -13.4169,     -9.9217,
            -16.9120,    -14.4457,    -10.6822,    -21.2923,     -9.6390,
            -20.2221,    -17.7307,    -20.1368,    -18.3994,    -18.3775,
            -19.0585,     -7.7539,    -11.2626,    -23.7490,    -25.4477,
            -17.8806,    -15.7524,    -12.1347,     -8.9728,    -19.0285,
            -15.8330,    -14.0935,    -19.7522,    -17.1125,    -12.9310,
            -11.4541,    -20.3690,    -12.1756,    -18.4610,    -13.0692,
            -13.9275,    -14.7232,    -16.6745],
        [   -11.5746,    -12.8144,    -15.8212,     -9.0374,    -16.4925,
            -12.4314,    -21.9067,    -13.4373,    -20.8593,    -13.8998,
            -11.5499,    -17.7792,    -11.5777,     -9.7706,     -7.6080,
             -0.0010,    -16.1537,    -19.3231,    -16.4319,     -9.8306,
            -15.9735,    -20.2868,    -17.4435,    -10.9136,    -22.2349,
            -19.8327,    -10.3609,    -18.0410,    -13.3422,    -17.4302,
            -22.6531,    -19.6392,     -9.0005,    -20.0413,    -13.4816,
            -20.8846,    -14.9128,    -13.0582,    -22.7481,    -10.8999,
            -20.6003,    -23.4194,    -14.9901,    -18.8998,    -18.9616,
            -13.0896,    -18.2480,    -10.1961,    -23.8549,    -18.8171,
            -15.6738,    -16.2665,    -12.3635,    -15.5222,    -20.1533,
            -19.7233,    -18.2687,    -18.1027,    -18.9104,    -17.0788,
            -14.6772,    -20.7978,    -17.2385,    -19.2634,    -15.0606,
            -16.1745,    -17.8595,    -18.3832],
        [   -10.0700,    -16.1884,    -27.1974,     -0.0004,    -15.9500,
            -12.9871,    -16.6855,     -8.6643,    -17.4327,    -25.2632,
            -22.7512,    -27.5718,    -18.1520,    -11.2660,    -10.2663,
            -15.0662,    -15.8804,    -29.2195,    -10.1129,    -14.0853,
            -22.2589,    -13.0752,    -13.9082,    -19.8814,    -30.5347,
            -27.9703,    -24.1673,    -22.3243,    -21.3324,    -30.3845,
            -28.1089,    -26.3711,    -23.3841,    -25.7207,    -12.8624,
            -29.3140,    -22.2825,    -15.8532,    -17.7213,    -25.8335,
            -25.7692,    -33.9364,     -9.4028,    -23.4755,    -26.9892,
            -18.0229,    -18.1021,    -14.1305,    -21.9619,    -20.7784,
            -12.6111,    -30.4083,    -20.9618,    -16.6976,    -17.5341,
            -23.5581,    -34.7220,    -18.7926,    -16.2027,    -23.4239,
            -20.2628,    -21.0055,    -28.3240,    -35.3471,    -27.4295,
            -23.9861,    -23.2900,    -25.7242],
        [    -0.0024,    -13.5267,    -28.2210,     -6.9429,    -10.4709,
            -11.2176,     -6.7001,     -9.7974,     -9.1347,    -16.0150,
            -17.7129,    -14.5789,    -24.5754,    -14.2416,    -16.1127,
            -18.6488,    -14.3954,    -21.6506,    -17.5871,    -16.1228,
            -11.9501,    -13.6215,    -19.0618,    -22.0745,    -26.2001,
            -16.3926,    -25.0453,    -21.5395,    -25.6020,    -30.0066,
            -24.5461,    -24.7074,    -28.7680,    -16.0528,    -17.0488,
            -19.7503,    -27.1811,    -20.2197,    -19.2906,    -26.2520,
            -30.7580,    -26.2255,    -21.7389,    -23.2458,    -26.6126,
            -28.8944,    -14.8293,    -25.7899,    -23.2974,    -25.0191,
            -22.6942,    -27.9867,    -27.8464,    -21.6079,    -21.3794,
            -23.8699,    -30.2556,    -27.1954,    -22.9247,    -15.8568,
            -23.4471,    -25.4685,    -23.4209,    -34.7602,    -25.8885,
            -19.4277,    -27.0841,    -29.3051],
        [    -5.3551,    -11.9869,    -26.9762,     -0.0125,     -5.9599,
             -5.6552,     -6.7275,    -11.3614,    -15.8269,    -21.9672,
            -14.6509,    -12.4791,    -22.7417,     -8.5429,     -9.4898,
            -13.1319,    -16.6962,    -22.6870,    -15.3368,    -18.0391,
            -17.0139,    -10.4608,    -18.0934,    -22.4379,    -26.2577,
            -18.3641,    -17.8522,    -16.3868,    -17.7658,    -30.3934,
            -17.5251,    -20.6198,    -21.1147,    -10.6076,    -10.8265,
            -18.2565,    -24.6173,    -14.7789,    -14.7391,    -20.2222,
            -24.4045,    -28.7649,    -12.1944,    -16.9539,    -24.8783,
            -17.9357,    -17.0823,    -18.2390,    -14.4338,    -15.0569,
            -14.1878,    -20.7030,    -16.4510,    -20.6554,    -11.4001,
            -21.6306,    -30.3084,    -17.2313,    -16.6970,    -13.5623,
            -15.5003,    -18.0603,    -21.1696,    -27.4085,    -23.4491,
            -12.8555,    -20.9476,    -23.9486],
        [    -8.8599,     -4.1120,    -22.9380,     -9.2153,     -9.5898,
             -4.6795,     -4.5558,    -17.8652,    -10.5413,    -16.5605,
            -11.1010,     -3.2362,    -19.2030,     -9.0512,    -14.2938,
            -14.5053,    -17.0702,    -17.5480,    -15.4673,    -17.7035,
            -12.5724,    -10.3617,    -24.9332,    -20.9010,    -15.8659,
            -13.6570,    -15.2410,    -10.1706,    -14.4283,    -22.5986,
             -9.7615,    -12.8506,    -19.8678,     -0.0823,    -12.1494,
            -10.1404,    -19.7858,    -16.7983,    -14.5718,    -17.2427,
            -18.3608,    -18.7338,    -17.8174,    -10.2193,    -20.7910,
            -17.2724,    -15.1957,    -22.7455,     -9.8481,    -12.9738,
            -16.7651,    -12.2454,    -14.9026,    -21.5966,    -11.8077,
            -17.2461,    -20.7579,    -14.0791,    -17.3092,    -10.6150,
            -12.7315,    -17.5540,    -13.7271,    -18.8832,    -16.1643,
             -5.9161,    -18.8773,    -18.9640],
        [   -11.0008,     -0.0000,    -21.3037,    -15.6895,    -20.1175,
            -14.8766,    -13.5163,    -20.2415,    -11.3561,    -14.6122,
            -20.5969,    -17.1454,    -17.8208,    -17.4661,    -21.5275,
            -19.3293,    -17.0482,    -22.5557,    -14.0900,    -15.0779,
            -11.9288,    -13.8008,    -25.5723,    -18.0804,    -18.5394,
            -21.4878,    -26.9207,    -15.8985,    -21.7136,    -19.2498,
            -17.2248,    -19.7306,    -24.8174,    -11.8818,    -21.7200,
            -17.1236,    -19.3439,    -23.0710,    -18.9587,    -21.9931,
            -23.4103,    -18.8944,    -22.1203,    -15.4663,    -20.3742,
            -25.3608,    -14.9480,    -26.9095,    -21.6887,    -19.8937,
            -20.2106,    -22.1494,    -22.8410,    -21.9335,    -24.3049,
            -18.4681,    -20.9887,    -19.0301,    -21.3816,    -20.9129,
            -27.0997,    -24.9604,    -20.5494,    -26.1368,    -19.6204,
            -14.3863,    -26.5210,    -24.3750],
        [   -12.1588,    -18.1924,    -17.2917,    -13.3001,    -11.7473,
            -11.4665,    -11.7477,     -0.0027,    -11.5664,    -14.5235,
            -23.8533,    -21.7771,    -19.2888,    -16.9006,    -14.3230,
            -16.9710,     -9.2189,    -21.4992,    -12.9496,     -9.7399,
             -6.0579,    -19.2846,     -8.9881,    -14.4050,    -19.9644,
            -12.3600,    -17.7616,    -22.1352,    -23.9259,    -18.9317,
            -30.1420,    -30.3919,    -23.3529,    -25.3582,    -20.2446,
            -19.8034,    -19.8824,    -18.2969,    -11.7756,    -26.3409,
            -29.9053,    -17.6205,    -16.2325,    -25.1276,    -18.5891,
            -22.3626,    -13.9809,    -17.8008,    -27.5363,    -22.9957,
            -24.5695,    -32.5967,    -30.1198,    -13.8000,    -19.8555,
            -24.0561,    -20.6689,    -26.4228,    -18.1866,    -17.8147,
            -23.5550,    -22.3295,    -24.7202,    -29.1975,    -25.0200,
            -22.2467,    -23.2026,    -25.1016],
        [    -8.1270,    -13.6181,    -11.9520,    -19.6966,    -12.0999,
            -17.2668,    -15.7870,    -11.0410,     -8.1130,     -0.0066,
             -6.1763,     -7.2625,    -11.0001,     -8.9736,    -12.2757,
             -8.4146,     -7.4495,     -8.4143,    -23.6900,    -11.0658,
             -8.3940,    -18.7454,    -16.7431,     -8.9647,    -12.5772,
             -6.7881,     -9.1187,    -21.4797,    -14.3507,    -12.9015,
            -20.9097,    -21.3383,    -13.9861,    -17.6030,    -17.5797,
            -10.2426,    -14.4256,    -13.7688,    -24.4705,    -17.9232,
            -18.9588,    -15.8405,    -25.2309,    -20.0072,    -15.5751,
            -20.4822,    -15.2921,    -18.8815,    -27.2944,    -24.9864,
            -26.6323,    -15.2820,    -21.7577,    -16.3894,    -26.3767,
            -21.6471,     -8.4610,    -26.2626,    -25.4301,    -13.7284,
            -18.8883,    -25.6612,     -9.8677,    -19.2136,     -9.8384,
            -18.9681,    -20.7506,    -20.7960],
        [    -7.1787,    -13.7800,    -25.0591,    -17.2527,    -14.9933,
             -9.0224,    -11.5288,    -14.6839,     -3.9746,    -11.2493,
             -8.7174,     -0.0204,    -13.8777,     -8.1587,    -17.8230,
            -14.4448,    -13.0160,    -16.1886,    -19.3807,    -16.6738,
            -15.7512,    -13.8263,    -26.2163,    -19.6463,    -12.6209,
            -11.6612,    -14.6323,    -12.0568,    -14.1936,    -22.7722,
            -20.8805,    -18.5996,    -22.4163,    -13.7806,    -12.9131,
            -18.2311,    -19.3788,    -19.5181,    -25.4377,    -28.5042,
            -22.5610,    -22.1253,    -28.1806,    -19.3441,    -28.0900,
            -19.9983,    -19.7847,    -26.5530,    -21.5800,    -28.1132,
            -29.0261,    -18.9932,    -29.2236,    -21.8120,    -20.5391,
            -29.1041,    -20.9998,    -28.1685,    -24.9301,    -13.4327,
            -13.6432,    -22.2481,    -11.4196,    -29.4887,    -16.7417,
            -19.1487,    -23.1051,    -23.9615],
        [    -8.8456,     -8.4018,    -20.5807,    -14.9666,    -14.0466,
            -10.1598,     -8.3966,    -11.0586,     -0.0380,     -9.7015,
            -13.0035,     -3.3586,    -10.7280,     -8.0666,    -18.6113,
            -16.5227,     -7.2390,    -13.1825,    -13.1258,    -11.3868,
            -13.2796,     -7.3500,    -21.9695,    -17.2534,    -10.4532,
            -10.1419,    -15.1895,    -11.0580,    -13.8962,    -19.3000,
            -19.5787,    -19.7192,    -22.3439,    -14.4048,    -11.2301,
            -17.9295,    -16.1139,    -15.3256,    -19.3808,    -30.6965,
            -20.2859,    -18.5934,    -22.8780,    -17.2295,    -22.7877,
            -17.3786,    -16.8203,    -25.7282,    -19.4280,    -23.4268,
            -25.3449,    -19.4398,    -28.9634,    -17.3916,    -18.8276,
            -25.8308,    -18.2158,    -23.4158,    -20.8657,    -15.4429,
            -16.3657,    -19.5605,    -12.2509,    -29.1888,    -16.2488,
            -19.1277,    -22.2584,    -22.0361],
        [    -5.4082,     -7.1604,    -20.6138,    -17.4368,    -16.8286,
            -12.3123,     -4.2538,     -9.4485,     -0.0210,    -10.6024,
            -22.2872,    -11.4253,    -17.5289,    -15.7855,    -24.9219,
            -23.1103,     -8.2219,    -15.2452,    -11.8719,     -8.6531,
             -7.4913,     -8.7425,    -20.3213,    -19.7710,    -13.4201,
            -10.6702,    -25.5186,    -14.7817,    -22.4055,    -19.7585,
            -23.2039,    -24.2263,    -29.2552,    -16.4433,    -16.5387,
            -18.8939,    -20.6912,    -19.7805,    -14.6317,    -32.9768,
            -26.6643,    -15.1266,    -25.1627,    -20.2840,    -21.7645,
            -26.4450,    -12.7653,    -31.1678,    -21.7386,    -24.2929,
            -27.5415,    -26.0779,    -34.8466,    -16.8100,    -22.9428,
            -23.3031,    -21.1662,    -25.7558,    -20.7100,    -17.4849,
            -25.3057,    -22.4769,    -18.7683,    -32.9164,    -22.4234,
            -20.2357,    -27.1579,    -26.3820],
        [    -6.6451,     -9.6672,    -26.2332,    -15.4390,    -12.7344,
             -9.2357,     -0.0277,    -11.8216,     -3.6779,    -18.0014,
            -23.2626,     -8.6888,    -24.6485,    -15.6604,    -25.7386,
            -24.5578,    -13.0892,    -16.9354,    -15.5021,    -13.2652,
            -10.5753,     -8.0833,    -23.7891,    -26.9307,    -17.2809,
             -9.9511,    -24.8751,    -14.4198,    -23.1442,    -26.5594,
            -21.9979,    -25.5383,    -31.1190,    -12.6130,    -13.4927,
            -19.5235,    -26.4509,    -19.2317,    -13.3400,    -34.8747,
            -29.7207,    -19.5823,    -24.8594,    -20.6543,    -25.4461,
            -26.2444,    -16.1084,    -34.0676,    -17.5837,    -22.3440,
            -29.2526,    -25.1837,    -33.6653,    -20.8011,    -18.4714,
            -26.3198,    -26.8735,    -25.1639,    -20.8213,    -15.6408,
            -23.6075,    -21.1660,    -20.3636,    -33.8590,    -26.4520,
            -17.1571,    -27.8813,    -28.4354],
        [    -9.4910,    -12.2957,    -24.3799,    -18.6475,    -18.6229,
            -10.6222,    -10.2503,    -10.5723,     -0.0002,    -12.7294,
            -16.6950,    -10.9705,    -15.1500,    -15.0343,    -18.5249,
            -20.9295,    -15.7349,    -20.2519,    -12.6570,    -11.7629,
            -10.4777,    -16.4528,    -21.7247,    -19.8577,    -14.5457,
            -12.3853,    -20.5662,    -17.1048,    -22.5493,    -21.0162,
            -26.3371,    -19.7975,    -28.7820,    -16.7853,    -14.6362,
            -20.8156,    -19.0998,    -20.6062,    -20.4976,    -31.3374,
            -24.5901,    -19.0009,    -25.9509,    -22.1039,    -26.4123,
            -24.2502,    -13.8183,    -25.5463,    -23.1105,    -31.1694,
            -28.0429,    -27.0491,    -31.1404,    -16.2549,    -21.0761,
            -25.1429,    -21.7603,    -26.6026,    -20.9887,    -17.9881,
            -17.2013,    -23.1996,    -17.4005,    -31.8779,    -20.9950,
            -21.7275,    -22.6960,    -23.4331],
        [    -9.5523,    -10.9095,    -21.6928,    -20.4540,    -14.5386,
            -10.5228,     -7.9364,    -13.9198,     -0.1085,     -7.3516,
            -10.1080,     -2.3015,    -14.4014,    -11.2287,    -17.0277,
            -17.5238,    -14.1387,    -13.8733,    -17.8965,    -13.1915,
            -10.6527,    -14.6849,    -24.9387,    -20.0037,    -11.8358,
             -6.5187,    -13.9942,    -15.9282,    -17.9183,    -19.8658,
            -21.3539,    -17.9227,    -24.6885,    -10.5518,    -13.4538,
            -14.5039,    -18.8110,    -16.9267,    -22.3756,    -28.4494,
            -21.1614,    -17.5268,    -28.5215,    -18.7886,    -24.1188,
            -22.3885,    -15.2780,    -27.0408,    -20.8897,    -28.1593,
            -29.5561,    -18.6340,    -27.3982,    -17.8787,    -20.4209,
            -25.2488,    -16.3751,    -25.6648,    -23.5332,    -14.7095,
            -14.9504,    -23.1165,    -11.2095,    -26.3516,    -15.8309,
            -17.9658,    -21.7057,    -22.4895],
        [    -6.7810,     -5.9636,    -21.9665,    -12.0243,     -9.7468,
             -7.1487,     -7.6898,    -17.1973,     -8.0541,    -12.1455,
             -8.5139,     -0.0088,    -15.6860,     -8.1412,    -14.9755,
            -12.7008,    -13.3425,    -14.4351,    -17.2464,    -17.3679,
            -14.1242,     -9.8743,    -25.4661,    -19.0522,    -14.1405,
            -11.5490,    -14.0378,     -8.9776,    -12.4536,    -21.5392,
            -12.5859,    -16.0508,    -18.7490,     -5.9249,    -12.6920,
            -13.2169,    -18.8826,    -16.2437,    -19.4147,    -21.1865,
            -20.1151,    -19.8066,    -22.0224,    -12.7666,    -22.4169,
            -17.6371,    -17.6277,    -24.5336,    -16.1385,    -17.8620,
            -22.0121,    -13.1342,    -20.2919,    -21.9642,    -15.3538,
            -22.9875,    -19.0055,    -20.7236,    -21.1421,    -11.2821,
            -15.1679,    -18.7147,    -10.9194,    -22.6489,    -15.0058,
            -11.0786,    -20.9314,    -21.2739],
        [    -9.9401,    -11.5352,    -23.9274,     -0.0014,    -13.3349,
            -10.0534,    -16.4826,     -8.7328,    -15.4148,    -21.0430,
            -18.1233,    -20.8313,    -13.6614,     -8.4880,     -8.3745,
            -12.3644,    -12.1087,    -27.2906,     -7.9349,    -15.5818,
            -20.7462,     -9.5711,    -14.1046,    -15.0528,    -24.5662,
            -26.4896,    -20.0221,    -15.4025,    -15.5223,    -25.6444,
            -21.7997,    -21.9315,    -19.0174,    -20.4068,    -14.2451,
            -24.2495,    -17.2934,    -16.1032,    -16.3015,    -22.2447,
            -21.6717,    -29.4669,     -8.3587,    -17.6661,    -23.5716,
            -13.8854,    -17.2250,    -12.3231,    -19.6956,    -16.2166,
            -10.7973,    -26.3352,    -19.0698,    -17.0191,    -14.5141,
            -22.1574,    -28.9523,    -16.9303,    -15.0426,    -20.0862,
            -17.8957,    -17.9298,    -23.3276,    -30.7515,    -22.0003,
            -19.5048,    -20.6214,    -22.4875],
        [    -0.0244,     -7.4337,    -22.4834,     -4.1666,    -13.1933,
             -6.2199,    -12.0033,     -9.0335,     -5.7078,    -11.6666,
             -9.9021,    -14.3608,    -11.7993,     -9.4811,     -6.2052,
            -10.7838,    -15.3468,    -22.7461,     -8.0456,    -10.2993,
            -13.7530,    -11.6924,    -15.5754,    -14.3866,    -20.4969,
            -19.2185,    -20.3659,    -14.3565,    -17.5239,    -22.1508,
            -21.8859,    -15.7659,    -20.3432,    -14.4949,    -11.9039,
            -21.0320,    -16.4502,    -16.3916,    -18.6279,    -20.1943,
            -21.3068,    -24.8138,    -15.9071,    -16.9285,    -24.4243,
            -19.0767,     -9.9233,    -14.5287,    -20.4902,    -23.3110,
            -14.3645,    -23.7464,    -19.3981,    -13.5971,    -16.0021,
            -19.5917,    -24.7891,    -20.0786,    -15.6360,    -15.7161,
            -15.0437,    -18.9751,    -16.9402,    -29.4630,    -18.3488,
            -17.4157,    -18.9682,    -21.0211],
        [    -4.8481,    -13.0937,    -18.5355,     -1.7683,     -8.2444,
             -6.5144,    -12.6989,     -4.5141,     -9.8590,    -11.7602,
             -9.5568,    -17.2762,    -11.6608,     -8.6477,     -0.2144,
             -7.5089,    -14.2674,    -20.5741,     -8.9137,     -8.8972,
            -13.8404,    -13.1460,     -9.5468,    -13.4361,    -22.6396,
            -15.9496,    -12.5933,    -18.0714,    -16.6183,    -21.4351,
            -23.3032,    -17.4972,    -15.8084,    -16.8509,     -9.7451,
            -19.8517,    -15.7065,    -10.3196,    -15.6819,    -16.7047,
            -19.2634,    -24.8403,    -10.3535,    -17.8431,    -20.6335,
            -14.4348,    -10.2757,     -7.9452,    -19.5781,    -19.8879,
            -11.2830,    -22.1948,    -14.0631,     -9.9161,    -11.9607,
            -18.7575,    -22.0224,    -16.9747,    -13.0305,    -15.2396,
            -11.5527,    -16.4625,    -17.2442,    -24.7415,    -18.0623,
            -17.6423,    -14.7699,    -18.4090],
        [   -17.4767,    -12.2009,    -15.9021,     -7.0121,    -10.3819,
             -5.2330,    -13.4704,     -8.5234,    -11.0059,    -12.4399,
             -8.6769,    -12.7460,     -8.4430,     -6.6740,     -0.2121,
             -4.2496,    -16.9200,    -17.7260,    -10.2393,     -7.3956,
            -14.5215,    -15.6354,    -15.6275,    -15.4734,    -18.5728,
            -11.4217,     -1.9244,    -17.1325,    -12.8126,    -18.3421,
            -21.7177,    -14.9405,    -12.1906,    -11.7715,     -5.8270,
            -17.6261,    -12.8909,     -6.0934,    -16.7374,    -15.2528,
            -14.1275,    -21.9517,    -10.8158,    -15.3007,    -19.0424,
             -7.4745,    -14.2173,     -7.8004,    -15.4889,    -17.7671,
            -13.1436,    -14.8085,     -8.7538,     -9.2048,     -9.3268,
            -19.2364,    -15.8752,    -11.7025,    -13.0409,    -15.9797,
             -4.1703,    -15.5278,    -13.6427,    -17.1099,    -14.3456,
            -13.9788,    -10.8006,    -13.8182],
        [    -6.8203,     -0.0013,    -25.7137,     -9.2264,    -21.0447,
            -12.4829,    -14.9716,    -19.9316,    -14.5809,    -17.4793,
            -17.2128,    -14.5318,    -17.2844,    -10.2289,    -18.6730,
            -11.5321,    -17.9022,    -25.4180,    -16.4237,    -16.1337,
            -12.4827,    -17.0423,    -28.0277,    -15.9398,    -21.2594,
            -24.9682,    -24.5913,    -16.8847,    -18.5224,    -22.6367,
            -16.0923,    -17.1620,    -22.2376,     -9.8489,    -18.4214,
            -15.7375,    -19.4643,    -23.6931,    -23.2928,    -15.9975,
            -22.9537,    -23.5524,    -21.0026,    -16.0768,    -23.5598,
            -23.7154,    -17.4175,    -23.6095,    -20.3149,    -20.4407,
            -18.1066,    -19.7195,    -18.5491,    -24.8872,    -26.4440,
            -17.6431,    -24.9984,    -17.9743,    -23.3503,    -18.6495,
            -21.6090,    -27.7663,    -21.6383,    -25.9268,    -18.3802,
            -11.5047,    -26.6247,    -24.5213],
        [    -7.2609,     -0.0009,    -19.2023,    -12.1475,    -17.3143,
            -18.2265,    -10.3310,    -17.2877,    -14.6739,    -15.5970,
            -23.4171,    -17.3247,    -20.6552,    -14.1528,    -24.8512,
            -19.2620,     -9.8078,    -18.8625,    -16.0987,    -16.7652,
            -10.2317,     -9.9246,    -21.3429,    -14.5333,    -17.4692,
            -21.8970,    -29.3450,    -15.6448,    -18.6192,    -18.5717,
            -11.4981,    -21.6451,    -22.3541,    -12.0226,    -22.3698,
            -11.4579,    -19.1647,    -22.4521,    -14.2264,    -18.6450,
            -22.0759,    -16.6177,    -18.2964,    -14.4406,    -15.2902,
            -25.2661,    -14.9236,    -26.5474,    -18.1686,    -12.3100,
            -17.8117,    -19.7126,    -22.2180,    -23.6270,    -25.5325,
            -14.5376,    -20.5705,    -16.6596,    -20.6508,    -17.7715,
            -30.5924,    -24.2305,    -21.8164,    -23.5045,    -18.3774,
            -11.4540,    -28.1297,    -24.8789],
        [    -9.2663,     -7.1989,    -20.1159,     -9.5495,    -12.9028,
            -13.2557,     -5.9195,    -15.4348,    -10.8206,    -21.1732,
            -21.4139,    -11.6572,    -18.4649,    -11.7339,    -23.1100,
            -20.7956,     -9.8387,    -14.7808,    -10.7857,    -16.6385,
            -18.8598,     -0.0056,    -18.2383,    -20.6301,    -13.8758,
            -18.2250,    -24.5028,     -6.4710,    -12.0703,    -20.8820,
            -10.5844,    -19.2382,    -19.6419,    -13.4788,    -10.5382,
            -18.5110,    -18.5517,    -14.9520,     -9.6097,    -25.6295,
            -17.6040,    -18.7497,    -13.1453,    -12.4236,    -18.4092,
            -14.8681,    -16.5106,    -23.9139,     -9.2325,     -9.1254,
            -16.3217,    -16.4659,    -21.6971,    -18.5900,    -12.3770,
            -19.1948,    -23.3535,    -13.4096,    -13.0970,    -15.2586,
            -21.6270,    -11.5231,    -16.3114,    -24.5827,    -19.4269,
            -13.3550,    -21.3488,    -20.7316],
        [    -0.6454,     -7.3631,    -21.8229,    -11.3083,    -13.6133,
             -9.6869,     -2.7513,     -7.6571,     -0.8959,    -13.2088,
            -17.6543,     -8.3286,    -17.8773,    -11.2667,    -21.6998,
            -20.6498,     -7.5446,    -15.4633,    -11.2624,    -12.0061,
             -6.8215,     -7.9937,    -16.8552,    -16.2989,    -11.6861,
            -12.8595,    -24.6334,    -11.6708,    -18.6156,    -19.3386,
            -16.4512,    -18.0601,    -25.8559,    -12.0260,    -13.7637,
            -13.5681,    -18.4236,    -20.1494,    -12.3631,    -25.7848,
            -22.3611,    -14.4611,    -19.9804,    -16.6972,    -20.0343,
            -23.5670,    -10.6215,    -25.4768,    -15.4296,    -19.7238,
            -21.6263,    -22.5461,    -28.6306,    -17.5921,    -18.8351,
            -17.7902,    -21.7506,    -21.2525,    -17.2300,    -11.7857,
            -19.7916,    -19.2780,    -16.7889,    -28.1428,    -18.9265,
            -14.5987,    -23.1248,    -22.0829],
        [   -11.6924,     -8.8365,    -23.5475,    -14.5628,    -18.6657,
             -7.6331,     -8.1157,    -14.6636,     -0.0161,    -12.9225,
             -9.6831,     -4.8332,    -11.6229,     -8.2620,    -14.1135,
            -17.6319,    -17.9647,    -18.0974,    -11.2354,    -12.5873,
            -13.8278,    -12.9095,    -24.0992,    -20.0022,    -10.5311,
            -13.4935,    -14.7626,    -13.3142,    -15.3966,    -19.5801,
            -16.5520,     -8.6239,    -23.3382,     -5.4941,     -8.3836,
            -13.5751,    -14.7180,    -16.7097,    -19.2874,    -23.3537,
            -12.8871,    -17.8223,    -21.0192,    -13.8155,    -24.1501,
            -16.6064,    -12.4094,    -19.9732,    -10.3875,    -23.1725,
            -18.8187,    -15.6043,    -18.5558,    -15.0048,    -14.6783,
            -17.4742,    -18.5094,    -14.2125,    -16.3275,    -14.9992,
             -6.1809,    -18.5530,    -11.5988,    -21.9881,    -13.7540,
            -13.4462,    -15.9019,    -15.9461],
        [   -10.7098,     -0.0023,    -30.3002,    -11.9405,    -23.1811,
            -11.5467,    -10.6228,    -22.1772,    -11.2820,    -22.9016,
            -21.9603,    -15.6019,    -21.4248,    -13.8520,    -22.6079,
            -22.8389,    -22.5472,    -29.7895,    -13.4329,    -19.8173,
            -15.1513,    -14.5725,    -29.7396,    -23.1560,    -20.0769,
            -26.5019,    -30.8615,    -15.7247,    -21.9966,    -25.5950,
            -15.1531,    -15.2545,    -29.9863,     -6.1354,    -18.8428,
            -16.0306,    -21.7709,    -26.7980,    -18.9149,    -23.1884,
            -21.8785,    -23.2282,    -20.9541,    -14.5082,    -26.1311,
            -25.9626,    -15.7007,    -27.2502,    -14.2783,    -19.9299,
            -17.7017,    -22.7168,    -21.1532,    -25.0956,    -22.1657,
            -16.8068,    -28.8051,    -14.7739,    -20.1534,    -20.6531,
            -21.6109,    -25.5807,    -23.4914,    -28.2711,    -22.0551,
            -11.1337,    -27.0287,    -24.8594],
        [    -8.5436,     -0.0003,    -20.5049,    -13.3775,    -21.3733,
            -13.7819,    -12.4094,    -16.9570,    -11.3731,    -15.5702,
            -23.3348,    -20.8683,    -17.5318,    -16.8435,    -20.8279,
            -18.5019,    -16.5897,    -23.4806,    -11.1457,    -10.4186,
            -11.2930,    -13.6033,    -22.3274,    -18.1027,    -19.7091,
            -21.8241,    -29.2107,    -16.4376,    -22.4265,    -19.4553,
            -20.0237,    -21.6664,    -25.0027,    -14.9904,    -19.4058,
            -19.9859,    -19.4219,    -21.5000,    -16.1546,    -23.1634,
            -24.7187,    -19.7554,    -19.7680,    -16.9415,    -20.6288,
            -25.0763,    -13.1580,    -25.1702,    -21.6822,    -19.8533,
            -18.8475,    -24.1205,    -23.5127,    -18.5040,    -24.2065,
            -17.7428,    -23.1684,    -18.1676,    -18.9262,    -22.0231,
            -27.7205,    -23.8932,    -22.8859,    -28.1781,    -21.7442,
            -15.8315,    -26.9240,    -24.9933],
        [   -17.4080,    -14.8364,    -12.7304,    -15.9032,    -21.5565,
            -12.4503,    -17.2493,     -6.8370,    -10.1120,    -13.9949,
            -22.0601,    -27.0981,    -11.0587,    -19.5826,    -11.7553,
            -13.6436,    -17.3303,    -19.4089,     -6.3320,     -0.0078,
            -10.2332,    -20.4946,    -11.3546,    -14.8224,    -18.5271,
            -15.5236,    -16.0001,    -21.1023,    -23.1959,    -13.6014,
            -30.6618,    -21.2505,    -18.7640,    -25.0460,    -12.9750,
            -26.1238,    -13.9679,    -13.0276,    -13.4213,    -21.9289,
            -21.6223,    -15.9460,    -14.4475,    -22.0498,    -17.8907,
            -16.1585,    -11.0228,    -12.1506,    -23.2616,    -23.8886,
            -17.2334,    -25.5326,    -20.2295,     -5.3427,    -17.5382,
            -17.7766,    -16.7236,    -17.2077,    -12.1351,    -22.8018,
            -16.2687,    -17.7010,    -21.5733,    -23.1502,    -21.2033,
            -22.8279,    -15.8566,    -16.2677],
        [   -12.4216,    -16.1053,    -29.1451,     -9.0337,    -11.8207,
             -0.0009,     -8.6505,     -7.5879,    -10.0877,    -22.7512,
            -19.2431,    -13.7531,    -20.8961,    -14.4762,    -10.8671,
            -15.7603,    -21.4039,    -29.3653,    -10.6510,    -13.1023,
            -14.5152,    -17.3762,    -21.2540,    -25.8284,    -24.3182,
            -15.3529,    -16.0656,    -15.8460,    -23.3182,    -29.9350,
            -30.5978,    -24.7313,    -27.4084,    -15.8459,    -12.5954,
            -27.1376,    -24.8943,    -18.8542,    -15.8377,    -31.1534,
            -31.7958,    -27.5254,    -18.8091,    -22.7685,    -31.5725,
            -19.4969,    -17.6864,    -21.3915,    -21.7040,    -25.9106,
            -24.4082,    -31.0024,    -28.0276,    -17.2936,    -11.2161,
            -30.6075,    -31.1321,    -25.2863,    -17.6013,    -17.5303,
            -13.4664,    -19.4692,    -23.7558,    -34.9868,    -29.0176,
            -19.2180,    -21.6642,    -26.0290],
        [    -3.0836,    -11.1404,    -14.8919,     -7.7812,     -8.5873,
             -2.8416,     -7.9187,     -1.2263,     -0.8384,     -4.9418,
             -5.6652,     -7.9001,     -8.2687,     -6.6189,     -2.4886,
             -5.1772,    -10.7569,    -14.3351,     -8.7137,     -3.0775,
             -4.2517,    -15.6165,    -10.7460,     -9.8711,    -13.8624,
             -5.6772,     -6.5559,    -15.6423,    -15.2044,    -14.9285,
            -23.3742,    -14.8360,    -15.4184,    -12.3902,     -7.6443,
            -14.3057,    -12.3669,     -9.7948,    -15.4702,    -17.0101,
            -18.7781,    -15.8887,    -15.6570,    -17.4488,    -18.3007,
            -14.6266,     -7.6423,    -10.5243,    -19.5447,    -23.4676,
            -18.1327,    -19.0919,    -17.4839,     -7.4446,    -13.4103,
            -17.8808,    -14.2511,    -19.8338,    -14.2306,    -10.3992,
             -7.1998,    -17.1758,    -11.6108,    -21.3457,    -13.3126,
            -15.0745,    -12.9874,    -15.5052],
        [    -7.9868,    -15.4103,    -16.8513,    -13.4917,    -15.0484,
            -13.9397,    -22.5354,    -17.7940,    -17.8933,     -9.6195,
             -2.2252,     -9.3662,    -11.3363,     -8.0452,     -7.2448,
             -0.1169,    -17.2633,    -15.0399,    -22.6545,    -14.7754,
            -17.6584,    -21.3282,    -20.6122,    -10.2137,    -18.9553,
            -17.0040,     -8.7555,    -17.0948,    -10.5532,    -17.3049,
            -19.0779,    -15.6084,     -7.8645,    -16.4988,    -13.3464,
            -16.7177,    -14.6946,    -14.1254,    -28.4332,     -9.9000,
            -17.9151,    -23.5627,    -21.2590,    -18.2105,    -20.5169,
            -14.8220,    -18.7088,    -12.3657,    -23.9319,    -22.5714,
            -19.7998,    -11.6299,    -12.3009,    -18.5748,    -21.3208,
            -20.9145,    -15.2292,    -22.2576,    -22.6946,    -12.8705,
            -12.1863,    -21.7272,    -10.3876,    -16.9829,     -9.7555,
            -15.6186,    -16.9797,    -17.5205],
        [    -0.0005,    -14.5657,    -22.4920,     -8.7515,    -13.2294,
            -10.9451,    -14.1387,    -13.1712,    -14.7828,    -16.2566,
             -8.9443,    -10.7783,    -18.9471,    -10.4379,    -12.8462,
            -10.3339,    -15.3794,    -19.9567,    -17.7954,    -19.3354,
            -13.7002,    -17.1503,    -16.4414,    -12.1036,    -18.4002,
            -20.5079,    -20.7407,    -13.5050,    -14.8802,    -20.9547,
            -14.9571,    -14.5075,    -16.1540,    -13.2388,    -15.8919,
            -14.0017,    -17.9937,    -21.3268,    -19.5608,    -12.4198,
            -20.9681,    -20.9421,    -18.2813,    -17.0681,    -21.2429,
            -20.7445,    -13.7380,    -15.6890,    -18.0320,    -18.9775,
            -17.1648,    -18.3295,    -17.9240,    -20.8698,    -18.1493,
            -16.5940,    -22.7223,    -21.8319,    -18.6172,     -9.1037,
            -15.5387,    -19.4642,    -15.6726,    -21.3156,    -15.0869,
            -12.4865,    -19.2078,    -19.4157],
        [    -0.0001,    -16.9840,    -21.5055,    -11.8447,    -12.0854,
            -15.3331,    -14.0876,    -13.8501,    -13.9810,    -13.6969,
            -10.8497,    -13.2444,    -21.0097,    -14.6334,    -14.9321,
            -14.7672,    -15.0091,    -17.4736,    -20.1177,    -19.6008,
            -14.9093,    -15.6280,    -14.9487,    -14.5987,    -19.9071,
            -18.3271,    -22.7283,    -16.6607,    -18.2314,    -21.5799,
            -16.9098,    -17.2415,    -18.3797,    -16.5486,    -17.9335,
            -15.5248,    -20.3519,    -20.3180,    -20.0336,    -15.8835,
            -22.3365,    -20.8197,    -20.5559,    -19.0399,    -20.3997,
            -23.9803,    -13.2990,    -18.3739,    -20.4504,    -20.5669,
            -18.5473,    -20.0512,    -20.7019,    -20.2041,    -19.8436,
            -17.8092,    -22.0861,    -24.1319,    -19.9407,    -11.3458,
            -19.9481,    -19.7521,    -16.2428,    -23.2110,    -17.0918,
            -16.6031,    -20.7498,    -21.6803],
        [    -0.0000,    -17.1071,    -23.5494,    -12.9530,    -13.9378,
            -16.4269,    -13.1326,    -15.8903,    -16.1134,    -18.2110,
            -15.6813,    -15.4734,    -24.7758,    -17.0952,    -19.6722,
            -18.0960,    -16.9470,    -19.2707,    -20.4694,    -20.8727,
            -16.9919,    -14.5632,    -16.0521,    -18.1356,    -21.0840,
            -20.3339,    -27.6928,    -15.6010,    -19.6547,    -23.5451,
            -17.0450,    -19.1711,    -20.5509,    -17.9552,    -18.1125,
            -18.2221,    -23.0217,    -21.9091,    -18.4166,    -18.5844,
            -24.6448,    -21.6979,    -20.9728,    -19.7739,    -21.6755,
            -25.6546,    -14.3602,    -21.3206,    -19.1922,    -19.4332,
            -19.6199,    -21.5367,    -22.8759,    -21.6474,    -20.0896,
            -18.4825,    -25.5764,    -23.9693,    -19.5282,    -12.6485,
            -23.1808,    -18.8814,    -18.8164,    -25.3059,    -20.4708,
            -16.8564,    -23.1196,    -23.7652],
        [    -0.0030,    -11.5165,    -16.0764,    -10.3643,     -9.1534,
            -10.1014,     -6.3984,    -10.5082,     -8.3872,    -12.4593,
            -11.1903,     -9.1744,    -17.3829,    -11.8836,    -14.3821,
            -13.4913,    -12.3396,    -11.9384,    -13.5650,    -12.5165,
            -11.1002,     -8.8777,    -11.1599,    -14.4585,    -13.0127,
            -11.0822,    -18.2881,     -9.8342,    -13.6718,    -16.0534,
            -12.3695,    -12.9870,    -14.8985,    -11.5473,     -9.7296,
            -13.1035,    -16.1597,    -13.3044,    -11.3557,    -15.0617,
            -16.7690,    -13.9120,    -15.0726,    -13.5833,    -15.2507,
            -17.2183,     -9.0175,    -15.7318,    -11.5052,    -13.7536,
            -15.0384,    -14.2378,    -16.3617,    -13.1835,    -12.3045,
            -13.2061,    -17.0609,    -15.9470,    -12.3622,     -8.3553,
            -14.6705,    -11.4317,    -11.8786,    -17.6636,    -14.4317,
            -11.3730,    -15.2084,    -16.0309],
        [    -0.0000,    -20.0676,    -20.8170,    -14.1267,    -13.4091,
            -16.9972,    -12.8867,    -12.8508,    -16.4415,    -17.9756,
            -16.6695,    -17.1983,    -24.9212,    -17.5865,    -19.5439,
            -17.8816,    -15.9060,    -17.6157,    -20.6040,    -18.9217,
            -14.7203,    -16.3294,    -11.5575,    -16.6162,    -19.7280,
            -17.8353,    -26.7412,    -16.7181,    -19.6941,    -20.8832,
            -18.3315,    -20.2285,    -19.0632,    -20.7024,    -17.3256,
            -17.7332,    -22.1284,    -20.7336,    -16.1553,    -17.6203,
            -24.3333,    -19.2522,    -19.5215,    -21.0567,    -19.3732,
            -24.9940,    -12.9470,    -18.9283,    -19.2262,    -19.4651,
            -20.1784,    -21.9543,    -22.7896,    -18.9875,    -19.8219,
            -17.2925,    -23.8396,    -23.9762,    -17.8404,    -11.7115,
            -22.9069,    -17.6411,    -18.7701,    -23.3177,    -20.4167,
            -17.4289,    -21.6423,    -22.5650]])

mytargets = torch.tensor([ 0, 13, 14, 15,  3,  3, 11,  8,  6,  1,  7,  9, 11,  8, 21,  8, 11,  1,
         8,  8,  6,  1, 21,  4, 13, 37,  4,  8, 11, 13, 14, 15,  3,  3, 11, 33,
         1,  7,  9, 11,  8, 21,  8, 11,  1,  3,  3, 14, 26,  1, 21,  8, 11,  1,
        19,  5,  3, 10, 10,  0], dtype=torch.int32)
mytargetsStr = "*schooten altemet een musquetschoot, altemet oock met groff*"
# scribblelens.corpus.v1/nl/tasman/18/line1.txt

# --------------------------------------
if __name__ == "__main__":
    from egs.scribblelens import alphabet

    if 0:
        # test align
        a = Aligner("b.txt")
        a.append("line1\n")
        a.append("line2\n")
    else:
        # test forced align
        alphabet  = alphabet.Alphabet("tasman.alphabet.plus.space.mode5.json")
        myaligner = ForcedAligner("path.out.txt",alphabet)

        '''
        Input :
            - groundtruth is a Python list of classes in alphabet (size < maxWidth)
            - feed in 'logprob' 2D tensor of [ nClasses x nTimestamps ] (nTimestamps < maxWidth)
        Output:
            - fill the array scores & tb
        '''
        # myaligner.forceAlign(mytargets, mylogprobs) # ORG DP

        # New Stack decoder way, with max stack of 2 tokens, and all path same length
        myaligner.forceAlign2(mytargets, mylogprobs)

        # Simpler, sequence length 3
        #myaligner.forceAlign2(mytargets[:3], mylogprobs[:5])


