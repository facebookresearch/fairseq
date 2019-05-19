'''
Vineet Kumar, sioom corp
From old dataset, create new dataset that works with the fairseq code base
From directory examples/transaction_bot/, issue following command:
python3 create-fairseq-dialog-dataset.py data-bin/transaction_bot
'''
import sys
import pathlib
import re
import shutil
import logging
import pickle

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)      # DEBUG INFO WARN ERROR/EXCEPTION CRITICAL
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)     # DEBUG INFO WARN ERROR/EXCEPTION CRITICAL
formatter = logging.Formatter(
        '%(levelname)-6s %(filename)s:%(lineno)s:%(funcName)s(): %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

# In general, try-except is not used because cannot recover from those failures
tbotDirP = pathlib.Path(sys.argv[0]).parents[0].resolve()
baseDirP = tbotDirP.parents[1]  # fairseq base directory
dialogIndexDirP = baseDirP.joinpath(sys.argv[1]).resolve()
oldDatasetDirP = tbotDirP.joinpath('dialog-bAbI-tasks')
if not oldDatasetDirP.exists():
    sys.exit(f'**Error** Program ended prematurely.\n\
Run this program after downloading the dataset at {oldDatasetDirP}')

logger.debug(f'create directories for new dataset')
newDatasetDirP = tbotDirP.joinpath('fairseq-dialog-dataset')
shutil.rmtree(newDatasetDirP, ignore_errors=True)
shutil.rmtree(dialogIndexDirP, ignore_errors=True)
fileNameCmpnts0 = {'task1', 'task1-OOV'}
# fileNameCmpnts0 = {'task1', 'task1-OOV', 'task2', 'task2-OOV', 'task3',
#                   'task3-OOV', 'task4', 'task4-OOV', 'task5',
#                   'task5-OOV', 'task6'}
fileNameCmpnts1 = {'dev', 'trn', 'tst'}
fileNameCmpnts2 = {'OOV'}
for fileNameCmpnt0 in fileNameCmpnts0:
    newDatasetDirP.joinpath(fileNameCmpnt0).mkdir(parents=True)
    dialogIndexDirP.joinpath(fileNameCmpnt0).mkdir(parents=True)

logger.debug(f'create files for new dataset')
for oldFileP in oldDatasetDirP.glob('dialog-babi-task*.txt'):
    # create bot and hmn files in new dataset from a valid file in old dataset
    oldFileNameCmpnts = re.findall(r'task[1-6]|dev|trn|tst|OOV', oldFileP.stem)
    if len(oldFileNameCmpnts) == 2 and oldFileNameCmpnts[0] in \
            fileNameCmpnts0 and oldFileNameCmpnts[1] in fileNameCmpnts1:
        newBotFile = newDatasetDirP.joinpath(oldFileNameCmpnts[0]).joinpath(
                f'{oldFileNameCmpnts[0]}-{oldFileNameCmpnts[1]}.bot')
        newHmnFile = newDatasetDirP.joinpath(oldFileNameCmpnts[0]).joinpath(
                f'{oldFileNameCmpnts[0]}-{oldFileNameCmpnts[1]}.hmn')
        dialogIndexFile = dialogIndexDirP.joinpath(
                oldFileNameCmpnts[0]).joinpath(
               f'dialog-index-{oldFileNameCmpnts[0]}-{oldFileNameCmpnts[1]}')
    elif len(oldFileNameCmpnts) == 3 and oldFileNameCmpnts[0] in \
            fileNameCmpnts0 and oldFileNameCmpnts[1] in fileNameCmpnts1 \
            and oldFileNameCmpnts[2] in fileNameCmpnts2:
        newBotFile = newDatasetDirP.joinpath(
            f'{oldFileNameCmpnts[0]}-{oldFileNameCmpnts[2]}').joinpath(f'\
{oldFileNameCmpnts[0]}-{oldFileNameCmpnts[1]}-{oldFileNameCmpnts[2]}.bot')
        newHmnFile = newDatasetDirP.joinpath(
            f'{oldFileNameCmpnts[0]}-{oldFileNameCmpnts[2]}').joinpath(f'\
{oldFileNameCmpnts[0]}-{oldFileNameCmpnts[1]}-{oldFileNameCmpnts[2]}.hmn')
        dialogIndexFile = dialogIndexDirP.joinpath(
            f'{oldFileNameCmpnts[0]}-{oldFileNameCmpnts[2]}').joinpath(f'\
dialog-index-{oldFileNameCmpnts[0]}-{oldFileNameCmpnts[1]}-{oldFileNameCmpnts[2]}')
    else:
        continue
    newBotFile.touch(exist_ok=False)
    newHmnFile.touch(exist_ok=False)
    dialogIndexFile.touch(exist_ok=False)

    logger.debug(f'write data from old dataset file to new dataset files')
    with oldFileP.open('r') as oldFile, newBotFile.open('w') as botFile, \
            newHmnFile.open('w') as hmnFile, \
            dialogIndexFile.open('wb') as dialogFile:
        hmnBotFileLineCount = 0
        dialogStartIndex = []   # record line numbers where dialogs start
        for lineno, line in enumerate(oldFile):
            if line == '\n':
                continue
            try:
                hmnLine, botLine = line.split('\t')
            except ValueError as error:
                sys.exit(f'**Error**: Missing tab separating a human \
utterance from a bot utterance in the following file, line number, and line:\n\
File: {oldFile}\n\
Line #: {lineno + 1}\n\
Line: {line}')
            if not hmnLine[0].isdecimal():
                sys.exit(f'**Error**: Missing decimal number at the beginning \
of the line in the following file, line number, and line:\n\
File: {oldFile}\n\
Line #: {lineno + 1}\n\
Line: {line}')
            if hmnLine[0] == '1':
                dialogStartIndex.append(hmnBotFileLineCount)
#               hmnFile.write('\n')
#               botFile.write('\n')
#               hmnFile.write('<SOD>\n')
#               botFile.write('<SOD>\n')
            hmnFile.write(hmnLine.lstrip('0123456789 ')+'\n')
            botFile.write(botLine)
            hmnBotFileLineCount += 1
        pickle.dump(dialogStartIndex, dialogFile)

# Tests to make sure that there is no problem
    with newHmnFile.open('r') as hmnFile:
        # the sum expression calculates the number of lines in the file;
        # the sum expression is a generator so it cannot be used again unless
        # the file is closed and then opened again 
        if hmnBotFileLineCount - sum(1 for _ in hmnFile):
            logger.critical(f'line count is wrong')
    with newHmnFile.open('r') as hmnFile:
         logger.debug(f'line count={hmnBotFileLineCount}, line count in file= {sum(1 for _ in hmnFile)}, file={newHmnFile.stem}')

    with dialogIndexFile.open('rb') as dialogFile:
        dialogIndexList = pickle.load(dialogFile)
        logger.debug(
                f'dialog index {dialogIndexList[0:5], dialogIndexList[-5:]}')

# cannot do statistics on files because text is not tokenized
