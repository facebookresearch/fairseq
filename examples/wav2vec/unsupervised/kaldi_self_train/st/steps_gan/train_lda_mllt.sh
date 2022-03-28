#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#
# LDA+MLLT refers to the way we transform the features after computing
# the MFCCs: we splice across several frames, reduce the dimension (to 40
# by default) using Linear Discriminant Analysis), and then later estimate,
# over multiple iterations, a diagonalizing transform known as MLLT or STC.
# See http://kaldi-asr.org/doc/transform.html for more explanation.
#
# Apache 2.0.

# Begin configuration.
cmd=run.pl
config=
stage=-5
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
realign_iters="10 20 30";
mllt_iters="2 4 6 12";
num_iters=35    # Number of iterations of training
max_iter_inc=25  # Last iter to increase #Gauss on.
dim=40
beam=10
retry_beam=40
careful=false
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
power=0.25 # Exponent for number of gaussians according to occurrence counts
randprune=4.0 # This is approximately the ratio by which we will speed up the
              # LDA and MLLT calculations via randomized pruning.
splice_opts=
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves
norm_vars=false # deprecated.  Prefer --cmvn-opts "--norm-vars=false"
cmvn_opts=
context_opts=   # use "--context-width=5 --central-position=2" for quinphone.
# End configuration.
train_tree=true  # if false, don't actually train the tree.
use_lda_mat=  # If supplied, use this LDA[+MLLT] matrix.
num_nonsil_states=3

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: steps/train_lda_mllt.sh [options] <#leaves> <#gauss> <data> <lang> <alignments> <dir>"
  echo " e.g.: steps/train_lda_mllt.sh 2500 15000 data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2b"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

numleaves=$1
totgauss=$2
data=$3
lang=$4
alidir=$5
dir=$6

for f in $alidir/final.mdl $alidir/ali.1.gz $data/feats.scp $lang/phones.txt; do
  [ ! -f $f ] && echo "train_lda_mllt.sh: no such file $f" && exit 1;
done

numgauss=$numleaves
incgauss=$[($totgauss-$numgauss)/$max_iter_inc] # per-iter #gauss increment
oov=`cat $lang/oov.int` || exit 1;
nj=`cat $alidir/num_jobs` || exit 1;
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;

mkdir -p $dir/log

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

echo $nj >$dir/num_jobs
echo "$splice_opts" >$dir/splice_opts # keep track of frame-splicing options
           # so that later stages of system building can know what they were.


[ $(cat $alidir/cmvn_opts 2>/dev/null | wc -c) -gt 1 ] && [ -z "$cmvn_opts" ] && \
  echo "$0: warning: ignoring CMVN options from source directory $alidir"
$norm_vars && cmvn_opts="--norm-vars=true $cmvn_opts"
echo $cmvn_opts > $dir/cmvn_opts # keep track of options to CMVN.

sdata=$data/split$nj;
split_data.sh $data $nj || exit 1;

splicedfeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- |"
# Note: $feats gets overwritten later in the script.
feats="$splicedfeats transform-feats $dir/0.mat ark:- ark:- |"



if [ $stage -le -5 ]; then
  if [ -z "$use_lda_mat" ]; then
    echo "$0: Accumulating LDA statistics."
    rm $dir/lda.*.acc 2>/dev/null
    $cmd JOB=1:$nj $dir/log/lda_acc.JOB.log \
    ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- \| \
      acc-lda --rand-prune=$randprune $alidir/final.mdl "$splicedfeats" ark,s,cs:- \
      $dir/lda.JOB.acc || exit 1;
    est-lda --write-full-matrix=$dir/full.mat --dim=$dim $dir/0.mat $dir/lda.*.acc \
      2>$dir/log/lda_est.log || exit 1;
    rm $dir/lda.*.acc
  else
    echo "$0: Using supplied LDA matrix $use_lda_mat"
    cp $use_lda_mat $dir/0.mat || exit 1;
    [ ! -z "$mllt_iters" ] && \
      echo "$0: Warning: using supplied LDA matrix $use_lda_mat but we will do MLLT," && \
      echo "     which you might not want; to disable MLLT, specify --mllt-iters ''" && \
      sleep 5
  fi
fi

cur_lda_iter=0

if [ $stage -le -4 ] && $train_tree; then
  echo "$0: Accumulating tree stats"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
    acc-tree-stats $context_opts \
    --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
    "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1;
  [ `ls $dir/*.treeacc | wc -w` -ne "$nj" ] && echo "$0: Wrong #tree-accs" && exit 1;
  $cmd $dir/log/sum_tree_acc.log \
    sum-tree-stats $dir/treeacc $dir/*.treeacc || exit 1;
  rm $dir/*.treeacc
fi


if [ $stage -le -3 ] && $train_tree; then
  echo "$0: Getting questions for tree clustering."
  # preparing questions, roots file...
  cluster-phones --pdf-class-list=$(($num_nonsil_states / 2)) $context_opts $dir/treeacc $lang/phones/sets.int \
    $dir/questions.int 2> $dir/log/questions.log || exit 1;
  cat $lang/phones/extra_questions.int >> $dir/questions.int
  compile-questions $context_opts $lang/topo $dir/questions.int \
    $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;

  echo "$0: Building the tree"
  $cmd $dir/log/build_tree.log \
    build-tree $context_opts --verbose=1 --max-leaves=$numleaves \
    --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
    $dir/questions.qst $lang/topo $dir/tree || exit 1;
fi

if [ $stage -le -2 ]; then
  echo "$0: Initializing the model"
  if $train_tree; then
    gmm-init-model  --write-occs=$dir/1.occs  \
      $dir/tree $dir/treeacc $lang/topo $dir/1.mdl 2> $dir/log/init_model.log || exit 1;
    grep 'no stats' $dir/log/init_model.log && echo "This is a bad warning.";
    rm $dir/treeacc
  else
    cp $alidir/tree $dir/ || exit 1;
    $cmd JOB=1 $dir/log/init_model.log \
      gmm-init-model-flat $dir/tree $lang/topo $dir/1.mdl \
        "$feats subset-feats ark:- ark:-|" || exit 1;
  fi
fi


if [ $stage -le -1 ]; then
  # Convert the alignments.
  echo "$0: Converting alignments from $alidir to use current tree"
  $cmd JOB=1:$nj $dir/log/convert.JOB.log \
    convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree \
     "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

if [ $stage -le 0 ] && [ "$realign_iters" != "" ]; then
  echo "$0: Compiling graphs of transcripts"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/1.mdl  $lang/L.fst  \
     "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $data/split$nj/JOB/text |" \
      "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi


x=1
while [ $x -lt $num_iters ]; do
  echo Training pass $x
  if echo $realign_iters | grep -w $x >/dev/null && [ $stage -le $x ]; then
    echo Aligning data
    mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$x.mdl - |"
    $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
      gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" \
      "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
      "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
  fi
  if echo $mllt_iters | grep -w $x >/dev/null; then
    if [ $stage -le $x ]; then
      echo "$0: Estimating MLLT"
      $cmd JOB=1:$nj $dir/log/macc.$x.JOB.log \
        ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:- \| \
        weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- \| \
        gmm-acc-mllt --rand-prune=$randprune  $dir/$x.mdl "$feats" ark:- $dir/$x.JOB.macc \
        || exit 1;
      est-mllt $dir/$x.mat.new $dir/$x.*.macc 2> $dir/log/mupdate.$x.log || exit 1;
      gmm-transform-means  $dir/$x.mat.new $dir/$x.mdl $dir/$x.mdl \
        2> $dir/log/transform_means.$x.log || exit 1;
      compose-transforms --print-args=false $dir/$x.mat.new $dir/$cur_lda_iter.mat $dir/$x.mat || exit 1;
      rm $dir/$x.*.macc
    fi
    feats="$splicedfeats transform-feats $dir/$x.mat ark:- ark:- |"
    cur_lda_iter=$x
  fi

  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali  $dir/$x.mdl "$feats" \
      "ark,s,cs:gunzip -c $dir/ali.JOB.gz|" $dir/$x.JOB.acc || exit 1;
    $cmd $dir/log/update.$x.log \
      gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss --power=$power \
        $dir/$x.mdl "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
    rm $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs
  fi
  [ $x -le $max_iter_inc ] && numgauss=$[$numgauss+$incgauss];
  x=$[$x+1];
done

rm $dir/final.{mdl,mat,occs} 2>/dev/null
ln -s $x.mdl $dir/final.mdl
ln -s $x.occs $dir/final.occs
ln -s $cur_lda_iter.mat $dir/final.mat

steps/diagnostic/analyze_alignments.sh --cmd "$cmd" $lang $dir

# Summarize warning messages...
utils/summarize_warnings.pl $dir/log

steps/info/gmm_dir_info.pl $dir

echo "$0: Done training system with LDA+MLLT features in $dir"

exit 0
