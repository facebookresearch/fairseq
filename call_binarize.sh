#!/bin/bash


binarize_example_small() {
TEXT="/raid/data/daga01/data/mt_testing_small/iwslt14.tokenized.de-en"
dest="/raid/data/daga01/fairseq_train/data-bin-small-test/"
mkdir -p $dest
CUDA_VISIBLE_DEVICES=1,2,3,4 fairseq-preprocess --source-lang de --target-lang en \
 --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
 --destdir $dest --joined-dictionary --workers 20

 # --dataset-impl lazy

}





binarize_big_lazy() {
    p="/raid/data/daga01/fairseq_train/BPE_32k_red"

    dest="/raid/data/daga01/fairseq_train/data-bin-32k-red-lazy"

    mkdir -p $dest

    CUDA_VISIBLE_DEVICES=1,2,3,4  fairseq-preprocess --source-lang src --target-lang tgt --trainpref $p/corpus --validpref $p/dev  --testpref $p/test --destdir $dest --joined-dictionary --dataset-impl lazy --workers 16

echo "Done binarizing"
}




binarize_test() {
    p="/raid/data/daga01/data/test2"

    dest="/raid/data/daga01/fairseq_train/test/"

    mkdir -p $dest

    CUDA_VISIBLE_DEVICES=1,2,3,4  fairseq-preprocess --source-lang en --target-lang de --trainpref $p/test --validpref $p/dev --destdir $dest --joined-dictionary --dataset-impl lazy --workers 16

echo "Done binarizing"
}




binarize_big_lazy
#binarize_example_small
#binarize_test
