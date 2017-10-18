data_path=/home/playma/4t/playma/Experiment/LCSTS_DATA
save_path=/home/playma/4t/playma/Experiment/fairseq-py/checkpoints/20171016_2230
pred_path=/pred.txt

for i in `seq 1 23`
do
    echo "#### epoch $i"
    perl ROUGE_with_ranked.pl 1 N $data_path/target_test.txt $save_path/pred_$i.txt
    perl ROUGE_with_ranked.pl 2 N $data_path/target_test.txt $save_path/pred_$i.txt R
    perl ROUGE_with_ranked.pl 1 L $data_path/target_test.txt $save_path/pred_$i.txt
done

