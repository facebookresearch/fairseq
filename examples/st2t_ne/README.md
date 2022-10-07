# README

This file contains all the information needed to run
the experiments for the named-entity aware speech translation models.

Most of the code has been placed in the `speech_text_joint_to_text` example folder.

Generic scripts and data (lists, dictionaries) can respectively be found in the `scripts`
and `data` folders here.

THe python environment requirements are:
 - fairseq
 - entmax (for models that use it)
 - spacy (for the evaluation)
 - en/es/fr/it spacy models (they can be downloaded with command `python -m spacy download model_name`)

This README also assumes that NEuRoparl-ST has been downloaded.

## Named Entities Retrieval

### Trainings

The sweep files that define the training command are placed in the
`scripts/train/er_asrst/` folder. All these scripts require
a base ST model to use its encoder to extract high-level representations
of audio and text. Each sweep file corresponds to an experiment.

The training TSV files are similar to those required to train the joint
speech/text to text models, but must contain an additional `entities` column,
which contains the phoneme representation of the entities present in each sentence.
To obtain such column, one can extract the entities from the English transcripts and
separate them with semicolumns with an easy bash/python command.
Then convert them with the `g2p_encode.py` script, adding the semicolumn as special character.
The already created training files can be anyhow found in the data.

### Evaluations

We can provide a list of candidate entities to the model, together with a test set,
to get the score (i.e. probability that the candidate is present in the utterance)
for each of the candidates in each of the utterances.
Please refer to the scripts `scripts/inference/ne_scores_gen*` to run the evaluation.
Those scripts will generate a pickle file with the scores for each entity and
each utterance, as well as a recall/avg.retr.entities curve.
In addition, there will be also a log file that contains the scores that
are used to generate the beforementioned curve.

For further analysis/evaluations, scores can be loaded into a sqllite db, which
can be done using the `examples/speech_text_joint_to_text/scripts/load_db.py`
script.


## CLAS

### Data preparation

This step assumes that the NE scores have been generated and loaded into a sqllite DB,
as per the description of the Evaluation section of Named Entities Retrieval.
We then need to create a TSV that contains the retrieved entities in the `entities` column.
To this aim, we can use the `examples/speech_text_joint_to_text/scripts/build_entities_inference_tsv.py` script.
The produced TSV should be used as input of the CLAS model for inference.

### Trainings

The training TSV are similar to those required for the previous step. The only difference is that
the `entities` column contains the NE in the target text separated by semicolumn, instead of
the phoneme representation of the NE in the source text.
The sweep files to run these trainings are present in `scripts/train/clas/`.

### Evaluations

The evaluation of the output of these models can be obtained using the `scripts/eval/st_ctx.sh` script.

