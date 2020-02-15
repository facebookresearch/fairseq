# Training RoBERTa on a sequence tagging task

This example shows how to train RoBERTa on the [MITRestaurant](https://groups.csail.mit.edu/sls/downloads/restaurant/) dataset, 
but should illustrate the process for most sequence tagging tasks, where a class is assigned to each input token.

### 1) Get the data

```bash
wget https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttrain.bio
wget https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttest.bio
```


### 2) Format data

Each line in `MITRestaurant` contains a word and its label, separated by a tab. Sentences are separated by an empty line.

Prepare separate input and label datasets using:
```bash
./preprocess_tagging.sh . MITRestaurant_raw
```

First ten lines of `MITRestaurant_raw/input/train`:
```
2 start restaurants with inside dining
34
5 star resturants in my town
98 hong kong restaurant reasonable prices
a great lunch spot but open till 2 a m passims kitchen
a place that serves soft serve ice cream
a restaurant that is good for groups
a salad would make my day
a smoothie would hit the spot
a steak would be nice
```

First ten lines of `MITRestaurant_raw/label/train`:
```
B-Rating I-Rating O O B-Amenity I-Amenity
O
B-Rating I-Rating O B-Location I-Location I-Location
O B-Restaurant_Name I-Restaurant_Name O B-Price O
O O O O O B-Hours I-Hours I-Hours I-Hours I-Hours B-Restaurant_Name I-Restaurant_Name
O O O O B-Dish I-Dish I-Dish I-Dish
O O O O B-Rating B-Amenity I-Amenity
O B-Dish O O O O
O B-Cuisine O O O O
O B-Dish O O O
```


### 3) Encoding

In this example, we will be training our model from scratch without BPE. 
When using BPE, you will need to make sure that an output token is provided for each input BPE token.

See [Finetuning on custom classification tasks (e.g., IMDB)](README.custom_classification.md) 
for an example with BPE and fine-tuning.

### 4) Preprocess data

```bash
fairseq-preprocess \
    --only-source \
    --trainpref "MITRestaurant_raw/input/train" \
    --validpref "MITRestaurant_raw/input/valid" \
    --destdir "MITRestaurant_bin/input" \
    --workers 1

fairseq-preprocess \
    --only-source \
    --trainpref "MITRestaurant_raw/label/train" \
    --validpref "MITRestaurant_raw/label/valid" \
    --destdir "MITRestaurant_bin/label" \
    --workers 1
```


### 5) Run training

```bash
MAX_EPOCH=100              # Total number of passes over training data.
LR=1e-05                   # Learning rate.
HEAD_NAME=restaurant_head  # Custom name for the sequence tagging head.
NUM_CLASSES=17             # Number of classes for the classification task.
MAX_SENTENCES=128          # Batch size.
MAX_POSITIONS=128          # Max tokens in sample, longer sequences will be skipped.

fairseq-train MITRestaurant_bin/ \
    --max-positions $MAX_POSITIONS \
    --skip-invalid-size-inputs-valid-test \
    --max-sentences $MAX_SENTENCES \
    --task sequence_tagging \
    --criterion sequence_tagging \
    --arch roberta_base \
    --classification-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --fp16 \
    --lr $LR \
    --optimizer adam \
    --max-epoch $MAX_EPOCH \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --update-freq 1 \
    --validate-interval 10 \
    --save-interval 10
```

The above command will train RoBERTa-base from scratch. The expected
`best-validation-accuracy` after 100 epochs is ~87%.

If you run out of GPU memory, try decreasing `--max-sentences` and increase
`--update-freq` to compensate.


### 6) Load model using hub interface

Now we can load the trained model checkpoint using the RoBERTa hub interface.

Assuming your checkpoints are stored in `checkpoints/`:
```python
from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    'checkpoints',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='MITRestaurant_bin',
    bpe=None # disable BPE
)
roberta.eval()  # disable dropout
```

Finally you can make predictions using the `restaurant_head`:
```python
sentence = 'I would like to find a mexican place nearby'
tokens = roberta.encode(sentence)
tags = roberta.predict_tags('restaurant_head', tokens)

for word, tag in zip(sentence.split(), tags):
    print('{}\t{}'.format(word, tag))

#   I	O
#   would	O
#   like	O
#   to	O
#   find	O
#   a	O
#   mexican	B-Cuisine
#   place	O
#   nearby	B-Location

    
sentence = 'which restaurant has a smoking section'
tokens = roberta.encode(sentence)
tags = roberta.predict_tags('restaurant_head', tokens)

for word, tag in zip(sentence.split(), tags):
    print('{}\t{}'.format(word, tag))
    
#   which	O
#   restaurant	O
#   has	O
#   a	O
#   smoking	B-Amenity
#   section	I-Amenity

```
