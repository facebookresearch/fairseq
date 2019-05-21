#!/usr/bin/env perl
#
# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.

use warnings;

# Sample Tokenizer
### Version 1.1
# written by Pidong Wang, based on the code written by Josh Schroeder and Philipp Koehn
# Version 1.1 updates:
#       (1) add multithreading option "-threads NUM_THREADS" (default is 1);
#       (2) add a timing option "-time" to calculate the average speed of this tokenizer;
#       (3) add an option "-lines NUM_SENTENCES_PER_THREAD" to set the number of lines for each thread (default is 2000), and this option controls the memory amount needed: the larger this number is, the larger memory is required (the higher tokenization speed);
### Version 1.0
# $Id: tokenizer.perl 915 2009-08-10 08:15:49Z philipp $
# written by Josh Schroeder, based on code by Philipp Koehn

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");

use warnings;
use FindBin qw($RealBin);
use strict;
use Time::HiRes;

if  (eval {require Thread;1;}) {
  #module loaded
  Thread->import();
}

my $mydir = "$RealBin/nonbreaking_prefixes";

my %NONBREAKING_PREFIX = ();
my @protected_patterns = ();
my $protected_patterns_file = "";
my $language = "en";
my $QUIET = 0;
my $HELP = 0;
my $AGGRESSIVE = 0;
my $SKIP_XML = 0;
my $TIMING = 0;
my $NUM_THREADS = 1;
my $NUM_SENTENCES_PER_THREAD = 2000;
my $PENN = 0;
my $NO_ESCAPING = 0;
while (@ARGV)
{
	$_ = shift;
	/^-b$/ && ($| = 1, next);
	/^-l$/ && ($language = shift, next);
	/^-q$/ && ($QUIET = 1, next);
	/^-h$/ && ($HELP = 1, next);
	/^-x$/ && ($SKIP_XML = 1, next);
	/^-a$/ && ($AGGRESSIVE = 1, next);
	/^-time$/ && ($TIMING = 1, next);
  # Option to add list of regexps to be protected
  /^-protected/ && ($protected_patterns_file = shift, next);
	/^-threads$/ && ($NUM_THREADS = int(shift), next);
	/^-lines$/ && ($NUM_SENTENCES_PER_THREAD = int(shift), next);
	/^-penn$/ && ($PENN = 1, next);
	/^-no-escape/ && ($NO_ESCAPING = 1, next);
}

# for time calculation
my $start_time;
if ($TIMING)
{
    $start_time = [ Time::HiRes::gettimeofday( ) ];
}

# print help message
if ($HELP)
{
	print "Usage ./tokenizer.perl (-l [en|de|...]) (-threads 4) < textfile > tokenizedfile\n";
        print "Options:\n";
        print "  -q     ... quiet.\n";
        print "  -a     ... aggressive hyphen splitting.\n";
        print "  -b     ... disable Perl buffering.\n";
        print "  -time  ... enable processing time calculation.\n";
        print "  -penn  ... use Penn treebank-like tokenization.\n";
        print "  -protected FILE  ... specify file with patters to be protected in tokenisation.\n";
	print "  -no-escape ... don't perform HTML escaping on apostrophy, quotes, etc.\n";
	exit;
}

if (!$QUIET)
{
	print STDERR "Tokenizer Version 1.1\n";
	print STDERR "Language: $language\n";
	print STDERR "Number of threads: $NUM_THREADS\n";
}

# load the language-specific non-breaking prefix info from files in the directory nonbreaking_prefixes
load_prefixes($language,\%NONBREAKING_PREFIX);

if (scalar(%NONBREAKING_PREFIX) eq 0)
{
	print STDERR "Warning: No known abbreviations for language '$language'\n";
}

# Load protected patterns
if ($protected_patterns_file)
{
  open(PP,$protected_patterns_file) || die "Unable to open $protected_patterns_file";
  while(<PP>) {
    chomp;
    push @protected_patterns, $_;
  }
}

my @batch_sentences = ();
my @thread_list = ();
my $count_sentences = 0;

if ($NUM_THREADS > 1)
{# multi-threading tokenization
    while(<STDIN>)
    {
        $count_sentences = $count_sentences + 1;
        push(@batch_sentences, $_);
        if (scalar(@batch_sentences)>=($NUM_SENTENCES_PER_THREAD*$NUM_THREADS))
        {
            # assign each thread work
            for (my $i=0; $i<$NUM_THREADS; $i++)
            {
                my $start_index = $i*$NUM_SENTENCES_PER_THREAD;
                my $end_index = $start_index+$NUM_SENTENCES_PER_THREAD-1;
                my @subbatch_sentences = @batch_sentences[$start_index..$end_index];
                my $new_thread = new Thread \&tokenize_batch, @subbatch_sentences;
                push(@thread_list, $new_thread);
            }
            foreach (@thread_list)
            {
                my $tokenized_list = $_->join;
                foreach (@$tokenized_list)
                {
                    print $_;
                }
            }
            # reset for the new run
            @thread_list = ();
            @batch_sentences = ();
        }
    }
    # the last batch
    if (scalar(@batch_sentences)>0)
    {
        # assign each thread work
        for (my $i=0; $i<$NUM_THREADS; $i++)
        {
            my $start_index = $i*$NUM_SENTENCES_PER_THREAD;
            if ($start_index >= scalar(@batch_sentences))
            {
                last;
            }
            my $end_index = $start_index+$NUM_SENTENCES_PER_THREAD-1;
            if ($end_index >= scalar(@batch_sentences))
            {
                $end_index = scalar(@batch_sentences)-1;
            }
            my @subbatch_sentences = @batch_sentences[$start_index..$end_index];
            my $new_thread = new Thread \&tokenize_batch, @subbatch_sentences;
            push(@thread_list, $new_thread);
        }
        foreach (@thread_list)
        {
            my $tokenized_list = $_->join;
            foreach (@$tokenized_list)
            {
                print $_;
            }
        }
    }
}
else
{# single thread only
    while(<STDIN>)
    {
        if (($SKIP_XML && /^<.+>$/) || /^\s*$/)
        {
            #don't try to tokenize XML/HTML tag lines
            print $_;
        }
        else
        {
            print &tokenize($_);
        }
    }
}

if ($TIMING)
{
    my $duration = Time::HiRes::tv_interval( $start_time );
    print STDERR ("TOTAL EXECUTION TIME: ".$duration."\n");
    print STDERR ("TOKENIZATION SPEED: ".($duration/$count_sentences*1000)." milliseconds/line\n");
}

#####################################################################################
# subroutines afterward

# tokenize a batch of texts saved in an array
# input: an array containing a batch of texts
# return: another array containing a batch of tokenized texts for the input array
sub tokenize_batch
{
    my(@text_list) = @_;
    my(@tokenized_list) = ();
    foreach (@text_list)
    {
        if (($SKIP_XML && /^<.+>$/) || /^\s*$/)
        {
            #don't try to tokenize XML/HTML tag lines
            push(@tokenized_list, $_);
        }
        else
        {
            push(@tokenized_list, &tokenize($_));
        }
    }
    return \@tokenized_list;
}

# the actual tokenize function which tokenizes one input string
# input: one string
# return: the tokenized string for the input string
sub tokenize
{
    my($text) = @_;

    if ($PENN) {
      return tokenize_penn($text);
    }

    chomp($text);
    $text = " $text ";

    # remove ASCII junk
    $text =~ s/\s+/ /g;
    $text =~ s/[\000-\037]//g;

    # Find protected patterns
    my @protected = ();
    foreach my $protected_pattern (@protected_patterns) {
      my $t = $text;
      while ($t =~ /(?<PATTERN>$protected_pattern)(?<TAIL>.*)$/) {
        push @protected, $+{PATTERN};
        $t = $+{TAIL};
      }
    }

    for (my $i = 0; $i < scalar(@protected); ++$i) {
      my $subst = sprintf("THISISPROTECTED%.3d", $i);
      $text =~ s,\Q$protected[$i], $subst ,g;
    }
    $text =~ s/ +/ /g;
    $text =~ s/^ //g;
    $text =~ s/ $//g;

    # separate out all "other" special characters
    if (($language eq "fi") or ($language eq "sv")) {
        # in Finnish and Swedish, the colon can be used inside words as an apostrophe-like character:
        # USA:n, 20:een, EU:ssa, USA:s, S:t
        $text =~ s/([^\p{IsAlnum}\s\.\:\'\`\,\-])/ $1 /g;
        # if a colon is not immediately followed by lower-case characters, separate it out anyway
        $text =~ s/(:)(?=$|[^\p{Ll}])/ $1 /g;
    }
    else {
        $text =~ s/([^\p{IsAlnum}\s\.\'\`\,\-])/ $1 /g;
    }

    # aggressive hyphen splitting
    if ($AGGRESSIVE)
    {
        $text =~ s/([\p{IsAlnum}])\-(?=[\p{IsAlnum}])/$1 \@-\@ /g;
    }

    #multi-dots stay together
    $text =~ s/\.([\.]+)/ DOTMULTI$1/g;
    while($text =~ /DOTMULTI\./)
    {
        $text =~ s/DOTMULTI\.([^\.])/DOTDOTMULTI $1/g;
        $text =~ s/DOTMULTI\./DOTDOTMULTI/g;
    }

    # seperate out "," except if within numbers (5,300)
    #$text =~ s/([^\p{IsN}])[,]([^\p{IsN}])/$1 , $2/g;

    # separate out "," except if within numbers (5,300)
    # previous "global" application skips some:  A,B,C,D,E > A , B,C , D,E
    # first application uses up B so rule can't see B,C
    # two-step version here may create extra spaces but these are removed later
    # will also space digit,letter or letter,digit forms (redundant with next section)
    $text =~ s/([^\p{IsN}])[,]/$1 , /g;
    $text =~ s/[,]([^\p{IsN}])/ , $1/g;
    
    # separate "," after a number if it's the end of a sentence
    $text =~ s/([\p{IsN}])[,]$/$1 ,/g;

    # separate , pre and post number
    #$text =~ s/([\p{IsN}])[,]([^\p{IsN}])/$1 , $2/g;
    #$text =~ s/([^\p{IsN}])[,]([\p{IsN}])/$1 , $2/g;

    # turn `into '
    #$text =~ s/\`/\'/g;

    #turn '' into "
    #$text =~ s/\'\'/ \" /g;

    if ($language eq "en")
    {
        #split contractions right
        $text =~ s/([^\p{IsAlpha}])[']([^\p{IsAlpha}])/$1 ' $2/g;
        $text =~ s/([^\p{IsAlpha}\p{IsN}])[']([\p{IsAlpha}])/$1 ' $2/g;
        $text =~ s/([\p{IsAlpha}])[']([^\p{IsAlpha}])/$1 ' $2/g;
        $text =~ s/([\p{IsAlpha}])[']([\p{IsAlpha}])/$1 '$2/g;
        #special case for "1990's"
        $text =~ s/([\p{IsN}])[']([s])/$1 '$2/g;
    }
    elsif (($language eq "fr") or ($language eq "it") or ($language eq "ga"))
    {
        #split contractions left
        $text =~ s/([^\p{IsAlpha}])[']([^\p{IsAlpha}])/$1 ' $2/g;
        $text =~ s/([^\p{IsAlpha}])[']([\p{IsAlpha}])/$1 ' $2/g;
        $text =~ s/([\p{IsAlpha}])[']([^\p{IsAlpha}])/$1 ' $2/g;
        $text =~ s/([\p{IsAlpha}])[']([\p{IsAlpha}])/$1' $2/g;
    }
    else
    {
        $text =~ s/\'/ \' /g;
    }

    #word token method
    my @words = split(/\s/,$text);
    $text = "";
    for (my $i=0;$i<(scalar(@words));$i++)
    {
        my $word = $words[$i];
        if ( $word =~ /^(\S+)\.$/)
        {
            my $pre = $1;
            if (($pre =~ /\./ && $pre =~ /\p{IsAlpha}/) || ($NONBREAKING_PREFIX{$pre} && $NONBREAKING_PREFIX{$pre}==1) || ($i<scalar(@words)-1 && ($words[$i+1] =~ /^[\p{IsLower}]/)))
            {
                #no change
			}
            elsif (($NONBREAKING_PREFIX{$pre} && $NONBREAKING_PREFIX{$pre}==2) && ($i<scalar(@words)-1 && ($words[$i+1] =~ /^[0-9]+/)))
            {
                #no change
            }
            else
            {
                $word = $pre." .";
            }
        }
        $text .= $word." ";
    }

    # clean up extraneous spaces
    $text =~ s/ +/ /g;
    $text =~ s/^ //g;
    $text =~ s/ $//g;

    # .' at end of sentence is missed
    $text =~ s/\.\' ?$/ . ' /;

    # restore protected
    for (my $i = 0; $i < scalar(@protected); ++$i) {
      my $subst = sprintf("THISISPROTECTED%.3d", $i);
      $text =~ s/$subst/$protected[$i]/g;
    }

    #restore multi-dots
    while($text =~ /DOTDOTMULTI/)
    {
        $text =~ s/DOTDOTMULTI/DOTMULTI./g;
    }
    $text =~ s/DOTMULTI/./g;

    #escape special chars
    if (!$NO_ESCAPING)
      {
	$text =~ s/\&/\&amp;/g;   # escape escape
	$text =~ s/\|/\&#124;/g;  # factor separator
	$text =~ s/\</\&lt;/g;    # xml
	$text =~ s/\>/\&gt;/g;    # xml
	$text =~ s/\'/\&apos;/g;  # xml
	$text =~ s/\"/\&quot;/g;  # xml
	$text =~ s/\[/\&#91;/g;   # syntax non-terminal
	$text =~ s/\]/\&#93;/g;   # syntax non-terminal
      }

    #ensure final line break
    $text .= "\n" unless $text =~ /\n$/;

    return $text;
}

sub tokenize_penn
{
    # Improved compatibility with Penn Treebank tokenization.  Useful if
    # the text is to later be parsed with a PTB-trained parser.
    #
    # Adapted from Robert MacIntyre's sed script:
    #   http://www.cis.upenn.edu/~treebank/tokenizer.sed

    my($text) = @_;
    chomp($text);

    # remove ASCII junk
    $text =~ s/\s+/ /g;
    $text =~ s/[\000-\037]//g;

    # attempt to get correct directional quotes
    $text =~ s/^``/`` /g;
    $text =~ s/^"/`` /g;
    $text =~ s/^`([^`])/` $1/g;
    $text =~ s/^'/`  /g;
    $text =~ s/([ ([{<])"/$1 `` /g;
    $text =~ s/([ ([{<])``/$1 `` /g;
    $text =~ s/([ ([{<])`([^`])/$1 ` $2/g;
    $text =~ s/([ ([{<])'/$1 ` /g;
    # close quotes handled at end

    $text =~ s=\.\.\.= _ELLIPSIS_ =g;

    # separate out "," except if within numbers (5,300)
    $text =~ s/([^\p{IsN}])[,]([^\p{IsN}])/$1 , $2/g;
    # separate , pre and post number
    $text =~ s/([\p{IsN}])[,]([^\p{IsN}])/$1 , $2/g;
    $text =~ s/([^\p{IsN}])[,]([\p{IsN}])/$1 , $2/g;

    #$text =~ s=([;:@#\$%&\p{IsSc}])= $1 =g;
$text =~ s=([;:@#\$%&\p{IsSc}\p{IsSo}])= $1 =g;

    # Separate out intra-token slashes.  PTB tokenization doesn't do this, so
    # the tokens should be merged prior to parsing with a PTB-trained parser
    # (see syntax-hyphen-splitting.perl).
    $text =~ s/([\p{IsAlnum}])\/([\p{IsAlnum}])/$1 \@\/\@ $2/g;

    # Assume sentence tokenization has been done first, so split FINAL periods
    # only.
    $text =~ s=([^.])([.])([\]\)}>"']*) ?$=$1 $2$3 =g;
    # however, we may as well split ALL question marks and exclamation points,
    # since they shouldn't have the abbrev.-marker ambiguity problem
    $text =~ s=([?!])= $1 =g;

    # parentheses, brackets, etc.
    $text =~ s=([\]\[\(\){}<>])= $1 =g;
    $text =~ s/\(/-LRB-/g;
    $text =~ s/\)/-RRB-/g;
    $text =~ s/\[/-LSB-/g;
    $text =~ s/\]/-RSB-/g;
    $text =~ s/{/-LCB-/g;
    $text =~ s/}/-RCB-/g;

    $text =~ s=--= -- =g;

    # First off, add a space to the beginning and end of each line, to reduce
    # necessary number of regexps.
    $text =~ s=$= =;
    $text =~ s=^= =;

    $text =~ s="= '' =g;
    # possessive or close-single-quote
    $text =~ s=([^'])' =$1 ' =g;
    # as in it's, I'm, we'd
    $text =~ s='([sSmMdD]) = '$1 =g;
    $text =~ s='ll = 'll =g;
    $text =~ s='re = 're =g;
    $text =~ s='ve = 've =g;
    $text =~ s=n't = n't =g;
    $text =~ s='LL = 'LL =g;
    $text =~ s='RE = 'RE =g;
    $text =~ s='VE = 'VE =g;
    $text =~ s=N'T = N'T =g;

    $text =~ s= ([Cc])annot = $1an not =g;
    $text =~ s= ([Dd])'ye = $1' ye =g;
    $text =~ s= ([Gg])imme = $1im me =g;
    $text =~ s= ([Gg])onna = $1on na =g;
    $text =~ s= ([Gg])otta = $1ot ta =g;
    $text =~ s= ([Ll])emme = $1em me =g;
    $text =~ s= ([Mm])ore'n = $1ore 'n =g;
    $text =~ s= '([Tt])is = '$1 is =g;
    $text =~ s= '([Tt])was = '$1 was =g;
    $text =~ s= ([Ww])anna = $1an na =g;

    #word token method
    my @words = split(/\s/,$text);
    $text = "";
    for (my $i=0;$i<(scalar(@words));$i++)
    {
        my $word = $words[$i];
        if ( $word =~ /^(\S+)\.$/)
        {
            my $pre = $1;
            if (($pre =~ /\./ && $pre =~ /\p{IsAlpha}/) || ($NONBREAKING_PREFIX{$pre} && $NONBREAKING_PREFIX{$pre}==1) || ($i<scalar(@words)-1 && ($words[$i+1] =~ /^[\p{IsLower}]/)))
            {
                #no change
            }
            elsif (($NONBREAKING_PREFIX{$pre} && $NONBREAKING_PREFIX{$pre}==2) && ($i<scalar(@words)-1 && ($words[$i+1] =~ /^[0-9]+/)))
            {
                #no change
            }
            else
            {
                $word = $pre." .";
            }
        }
        $text .= $word." ";
    }

    # restore ellipses
    $text =~ s=_ELLIPSIS_=\.\.\.=g;

    # clean out extra spaces
    $text =~ s=  *= =g;
    $text =~ s=^ *==g;
    $text =~ s= *$==g;

    #escape special chars
    $text =~ s/\&/\&amp;/g;   # escape escape
    $text =~ s/\|/\&#124;/g;  # factor separator
    $text =~ s/\</\&lt;/g;    # xml
    $text =~ s/\>/\&gt;/g;    # xml
    $text =~ s/\'/\&apos;/g;  # xml
    $text =~ s/\"/\&quot;/g;  # xml
    $text =~ s/\[/\&#91;/g;   # syntax non-terminal
    $text =~ s/\]/\&#93;/g;   # syntax non-terminal

    #ensure final line break
    $text .= "\n" unless $text =~ /\n$/;

    return $text;
}

sub load_prefixes
{
    my ($language, $PREFIX_REF) = @_;

    my $prefixfile = "$mydir/nonbreaking_prefix.$language";

    #default back to English if we don't have a language-specific prefix file
    if (!(-e $prefixfile))
    {
        $prefixfile = "$mydir/nonbreaking_prefix.en";
        print STDERR "WARNING: No known abbreviations for language '$language', attempting fall-back to English version...\n";
        die ("ERROR: No abbreviations files found in $mydir\n") unless (-e $prefixfile);
    }

    if (-e "$prefixfile")
    {
        open(PREFIX, "<:utf8", "$prefixfile");
        while (<PREFIX>)
        {
            my $item = $_;
            chomp($item);
            if (($item) && (substr($item,0,1) ne "#"))
            {
                if ($item =~ /(.*)[\s]+(\#NUMERIC_ONLY\#)/)
                {
                    $PREFIX_REF->{$1} = 2;
                }
                else
                {
                    $PREFIX_REF->{$item} = 1;
                }
            }
        }
        close(PREFIX);
    }
}
