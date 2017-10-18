#!/usr/bin/perl
# calculate ROUGE score

$debug=0;
$output_sorted=0;
$NSIZE=shift @ARGV or die $!;
$alpha=0.5;
$metric = shift @ARGV or die $!; # N or L
if ($metric ne "N" and $metric ne "L") { die "Unknown metric $metric"; }
$modelPath = shift @ARGV or die $!; # reference paths
$peerPath = shift @ARGV or die $!; # system paths
$do_ranked_output = shift @ARGV or 0;
#$orgfilename = shift @ARGV or die $!; # src paths
if ($do_ranked_output eq 'R') { $output_sorted=1; }
open(MODEL,$modelPath)||die "Cannot open $modelPath\n";
open(PEER,$peerPath)||die "Cannot open $peerPath\n";

@ROUGEScores=();
@ROUGEScores_P=();
@ROUGEScores_F=();
%ROUGEScoresAndLineNum=();
$num=0;
while(defined($model_line=<MODEL>) and defined($peer_line=<PEER>)) {
  $num++;
  chomp($model_line);
  chomp($peer_line);
  if ($debug){
    print "num: $num\n";
    print "ref: $model_line\n";
    print "sys: $peer_line\n";
  }
  @results=();
  if($metric eq "N") {
    &computeNGramScore($model_line,$peer_line,\@results,$NSIZE,$alpha);
  }
  elsif($metric eq "L") {
    &computeLCSScore($model_line,$peer_line,\@results,$alpha);
  }

  $oneROUGE=sprintf("%7.5f",$results[2]);
  $oneROUGE_P=sprintf("%7.5f",$results[4]);
  $oneROUGE_F=sprintf("%7.5f",$results[5]);
  $ROUGEScoresAndLineNum{$num-1}=($oneROUGE_F);

  push(@ROUGEScores,$oneROUGE);   # average ; or model token count
  push(@ROUGEScores_P,$oneROUGE_P); # average ; or peer token count
  push(@ROUGEScores_F,$oneROUGE_F); # average ; or match token count (hit)
}

# compute averages
$avgROUGE_R=0;
$avgROUGE_P=0;
$avgROUGE_F=0;
foreach $i (0..$#ROUGEScores) {
  $avgROUGE_R+=$ROUGEScores[$i]; # recall     [i]; or model token count
  $avgROUGE_P+=$ROUGEScores_P[$i]; # precision  ; or peer token count
  $avgROUGE_F+=$ROUGEScores_F[$i]; # f1-measure ; or match token count (hit)
}

$avgROUGE_R=sprintf("%7.5f",$avgROUGE_R/(scalar @ROUGEScores));
$avgROUGE_P=sprintf("%7.5f",$avgROUGE_P/(scalar @ROUGEScores));
$avgROUGE_F=sprintf("%7.5f",$avgROUGE_F/(scalar @ROUGEScores));


if($metric eq "N") {
  print "ROUGE-$NSIZE\n";
} 
elsif($metric eq "L") {
  print "ROUGE-L\n";
}
print "Avg_R | Avg_P | Avg_F\n";
printf("%.3f\t",$avgROUGE_R);
printf("%.3f\t",$avgROUGE_P);
printf("%.3f",$avgROUGE_F);
print "\n\n"; 
close(MODEL);
if ($output_sorted==0) { exit; }
my $filename = 'ranked_pred.txt';
#my $orgfilename = 'src_test.txt';
#my $goldfilename = 'tgt_test.txt';
my $goldfilename = $modelPath;
open my $predfile,'<',$peerPath or "Cannot open $peerPath\n";
my @preds = <$predfile>;
chomp @preds;
open my $orgfile,'<',$orgfilename or "Cannot open $orgfilename\n";
my @orgs = <$orgfile>;
chomp @orgs;
open my $goldfile,'<',$goldfilename or "Cannot open $goldfilename\n";
my @golds = <$goldfile>;
chomp @golds;
open my $fh,'>',$filename or die "Could not open file '$filename'";
print $fh "l_id\trouge_2F\tpred\tgold\torg\n";
foreach my $l_id(sort {$ROUGEScoresAndLineNum{$b} <=> $ROUGEScoresAndLineNum{$a}} keys %ROUGEScoresAndLineNum) {
  my $pred = $preds[$l_id];
  my $orgline = $orgs[$l_id];
  my $goldline = $golds[$l_id];
  my $rouge_score = $ROUGEScoresAndLineNum{$l_id};
  $pred =~ s/ //g;
  $orgline =~ s/ //g;
  $goldline =~ s/ //g;
  # print "$rouge_score\t$pred\t$orgline\n";
  print $fh "$l_id\t$rouge_score\t$pred\t$goldline\t$orgline\n";
}

close $fh;
close $orgfile;
close $predfile;

sub computeNGramScore {
  my $modelText=shift;
  my $peerText=shift;
  my $results=shift;
  my $NSIZE=shift;
  my $alpha=shift;
  my (%model_grams,%peer_grams);
  my ($gramHit,$gramScore,$gramScoreBest);
  my ($totalGramHit,$totalGramCount);
  my ($gramScoreP,$gramScoreF,$totalGramCountP);

  #------------------------------------------------
  # read model file and create model n-gram maps
  $totalGramHit=0;
  $totalGramCount=0;
  $gramScoreBest=-1;
  $gramScoreP=0; # precision
  $gramScoreF=0; # f-measure
  $totalGramCountP=0;
  #------------------------------------------------
  # read peer file and create model n-gram maps
  %peer_grams=();
  &createNGram($peerText,\%peer_grams,$NSIZE);
  %model_grams=();
  &createNGram($modelText,\%model_grams,$NSIZE);
  #------------------------------------------------
  # compute ngram score
  &ngramScore(\%model_grams,\%peer_grams,\$gramHit,\$gramScore);
  $totalGramHit=$gramHit;
  $totalGramCount=$model_grams{"_cn_"};
  $totalGramCountP=$peer_grams{"_cn_"};

  # prepare score result for return
  # unigram
  push(@$results,$totalGramCount); # total number of ngrams in models
  push(@$results,$totalGramHit);
  if($totalGramCount!=0) {
    $gramScore=sprintf("%7.5f",$totalGramHit/$totalGramCount);
  }
  else {
    $gramScore=sprintf("%7.5f",0);
  }
  push(@$results,$gramScore);
  push(@$results,$totalGramCountP); # total number of ngrams in peers
  if($totalGramCountP!=0) {
    $gramScoreP=sprintf("%7.5f",$totalGramHit/$totalGramCountP);
  }
  else {
    $gramScoreP=sprintf("%7.5f",0);
  } 
  push(@$results,$gramScoreP);      # precision score
  if((1-$alpha)*$gramScoreP+$alpha*$gramScore>0) {
    $gramScoreF=sprintf("%7.5f",($gramScoreP*$gramScore)/((1-$alpha)*$gramScoreP+$alpha*$gramScore));
  }
  else {
    $gramScoreF=sprintf("%7.5f",0);
  }
  push(@$results,$gramScoreF);      # f1-measure score
  if($debug) {
    print "total $NSIZE-gram model count: $totalGramCount\n";
    print "total $NSIZE-gram peer count: $totalGramCountP\n";
    print "total $NSIZE-gram hit: $totalGramHit\n";
    print "total ROUGE-$NSIZE\-R: $gramScore\n";
    print "total ROUGE-$NSIZE\-P: $gramScoreP\n";
    print "total ROUGE-$NSIZE\-F: $gramScoreF\n";
  }
}


sub computeLCSScore {
  my $modelText=shift;
  my $peerText=shift;
  my $results=shift;
  my $alpha=shift;


  ($totalGramHit, $totalGramCount, $totalGramCountP) = &lcs_inner($modelText,$peerText);
  if($debug) {
    print "$modelText\n";
    print "$peerText\n";
    print "$totalGramHit, $totalGramCount, $totalGramCountP\n\n";
  }
  if($totalGramCount!=0) {
    $gramScore=sprintf("%7.5f",$totalGramHit/$totalGramCount);
  }
  else {
    $gramScore=sprintf("%7.5f",0);
  }

  if($totalGramCountP!=0) {
    $gramScoreP=sprintf("%7.5f",$totalGramHit/$totalGramCountP);
  }
  else {
    $gramScoreP=sprintf("%7.5f",0);
  } 
  if((1-$alpha)*$gramScoreP+$alpha*$gramScore>0) {
    $gramScoreF=sprintf("%7.5f",($gramScoreP*$gramScore)/((1-$alpha)*$gramScoreP+$alpha*$gramScore));
  }
  else {
    $gramScoreF=sprintf("%7.5f",0);
  }
  push(@$results,$totalGramCount); # total number of ngrams in models
  push(@$results,$totalGramHit);
  push(@$results,$gramScore);
  push(@$results,$totalGramCountP); # total number of ngrams in peers
  push(@$results,$gramScoreP);      # precision score
  push(@$results,$gramScoreF);      # f1-measure score
}

sub lcs_inner {
  my $model_text=shift;
  my $peer_text=shift;
  @model=split(/\s+/,$model_text);
  @peer=split(/\s+/,$peer_text);
  my $m=scalar @model; # length of model
  my $n=scalar @peer; # length of peer
  my ($i,$j);
  my (@c,@b);
  
  if(@model==0) {
    return;
  }
  @c=();
  @b=();
  # initialize boundary condition and
  # the DP array
  for($i=0;$i<=$m;$i++) {
    push(@c,[]);
    push(@b,[]);
    for($j=0;$j<=$n;$j++) {
      push(@{$c[$i]},0);
      push(@{$b[$i]},0);
    }
  }
  for($i=1;$i<=$m;$i++) {
    for($j=1;$j<=$n;$j++) {
      if($model[$i-1] eq $peer[$j-1]) {
        # recursively solve the i-1 subproblem
        $c[$i][$j]=$c[$i-1][$j-1]+1;
        $b[$i][$j]="\\"; # go diagonal
      }
      elsif($c[$i-1][$j]>=$c[$i][$j-1]) {
        $c[$i][$j]=$c[$i-1][$j];
        $b[$i][$j]="^"; # go up
      }
      else {
        $c[$i][$j]=$c[$i][$j-1];
        $b[$i][$j]="<"; # go left
      }
    }
  }
  if ($debug){
    for($i=1;$i<=$m;$i++) {
      for($j=1;$j<=$n;$j++) {
        print "$b[$i][$j] ";
      }
      print "\n";
    }
  }
  ($c[$m][$n], $m, $n)
}

sub ngramScore {
  my $model_grams=shift;
  my $peer_grams=shift;
  my $hit=shift;
  my $score=shift;
  my ($s,$t,@tokens);
  
  $$hit=0;
  @tokens=keys (%$model_grams);
  foreach $t (@tokens) {
    if($t ne "_cn_") {
      my $h;
      $h=0;
      if(exists($peer_grams->{$t})) {
        $h=$peer_grams->{$t}<=$model_grams->{$t}?$peer_grams->{$t}:$model_grams->{$t}; # clip
        $$hit+=$h;
      }
    }
  }
  if($model_grams->{"_cn_"}!=0) {
    $$score=sprintf("%07.5f",$$hit/$model_grams->{"_cn_"});
  }
  else {
    # no instance of n-gram at this length
    $$score=0;
    # die "model n-grams has zero instance\n";
  }
}

sub createNGram {
  my $text=shift;
  my $g=shift;
  my $NSIZE=shift;
  my @mx_tokens=();
  my @m_tokens=();
  my ($i,$j);
  my ($gram);
  my ($count);
  my ($byteSize);
  unless(defined($text)) {
    $g->{"_cn_"}=0;
    return;
  }
  @mx_tokens=split(/\s+/,$text);
  $byteSize=0;
  for($i=0;$i<=$#mx_tokens;$i++) {
    $byteSize+=length($mx_tokens[$i])+1; # the length of words in bytes so far + 1 space 
    push(@m_tokens,$mx_tokens[$i]);
  }
  #-------------------------------------
  # create ngram
  $count=0;
  for($i=0;$i<=$#m_tokens-$NSIZE+1;$i++) {
    $gram=$m_tokens[$i];
    for($j=$i+1;$j<=$i+$NSIZE-1;$j++) {
      $gram.=" $m_tokens[$j]";
    }
    $count++;
    unless(exists($g->{$gram})) {
      $g->{$gram}=1;
    }
    else {
      $g->{$gram}++;
    }
  }
  # save total number of tokens
  $g->{"_cn_"}=$count;
}
