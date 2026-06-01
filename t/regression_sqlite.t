#!/usr/bin/perl
# Regression test: spam messages must not be detected as ham.
# Uses the SQLite backend to exercise _init_sql_connection,
# _save_vocabulary_to_sql, and _load_vocabulary_from_sql.

use strict;
use warnings;
use lib '.'; use lib 't';

use File::Path;
use Test::More;

# Skip if DBD::SQLite is not available
eval { require DBD::SQLite };
if ($@) {
  plan skip_all => 'DBD::SQLite not installed';
}

my $sarun     = qx{which spamassassin 2>&1}; chomp $sarun;
my $salearnrun = qx{which sa-learn 2>&1};    chomp $salearnrun;

unless (-x $sarun && -x $salearnrun) {
  plan skip_all => 'spamassassin and sa-learn must be in PATH';
}

plan tests => 20;

my @spam_train = (map({ "t/data/spam-00$_" } 1..9), 't/data/spam-010');
my @ham_train  = (map({ "t/data/nice-00$_" } 1..9), 't/data/nice-010');
my @spam_files = @spam_train;
my @ham_files  = @ham_train;

sub tstprefs {
  my $rules = shift;
  open(my $fh, '>', 't/rules/NeuralNetwork.cf')
    or die "Cannot write to rules directory: $!";
  print $fh $rules;
  close $fh;
}

sub tstcleanup {
  unlink 't/rules/NeuralNetwork.cf';
  rmtree 't/NN';
}

tstprefs("
  loadplugin Mail::SpamAssassin::Plugin::NeuralNetwork ../../NeuralNetwork.pm

  neuralnetwork_data_dir        t/NN
  neuralnetwork_dsn             dbi:SQLite:t/NN/neural.db
  neuralnetwork_min_spam_count  0
  neuralnetwork_min_ham_count   0
  neuralnetwork_min_vocab_hits  0

  body      NN_SPAM   eval:check_neuralnetwork_spam()
  describe  NN_SPAM   Email considered as spam by Neural Network
  score     NN_SPAM   1.0

  body      NN_HAM    eval:check_neuralnetwork_ham()
  describe  NN_HAM    Email considered as ham by Neural Network
  score     NN_HAM    -1.0
");

mkdir 't/NN';

# Train on 8 spam + 8 ham (interleaved so the ring buffer sees both classes early)
for my $i (0 .. $#spam_train) {
  qx($salearnrun --siteconfigpath=t/rules --spam $spam_train[$i]);
  qx($salearnrun --siteconfigpath=t/rules --ham  $ham_train[$i]);
}

# Regression: all 10 spam must score as spam, not ham
for my $f (@spam_files) {
  my $out = qx($sarun -L -t --siteconfigpath=t/rules < $f);
  like($out, qr/NN_SPAM/, "$f classified as spam");
}

# All 10 ham must score as ham
for my $f (@ham_files) {
  my $out = qx($sarun -L -t --siteconfigpath=t/rules < $f);
  like($out, qr/NN_HAM/, "$f classified as ham");
}

tstcleanup();
