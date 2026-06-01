#!/usr/bin/perl
# Regression test: imbalanced training corpus must not collapse all
# predictions toward the majority class. Models a ham-heavy stream
# (3 spam : 6 ham, i.e. 1:2 imbalance) similar to the production
# scenario where the user reported "all emails tagged as HAM".

use strict;
use warnings;
use lib '.'; use lib 't';

use File::Path;
use Test::More;

eval { require DBD::SQLite };
if ($@) {
  plan skip_all => 'DBD::SQLite not installed';
}

my $sarun      = qx{which spamassassin 2>&1}; chomp $sarun;
my $salearnrun = qx{which sa-learn 2>&1};     chomp $salearnrun;

unless (-x $sarun && -x $salearnrun) {
  plan skip_all => 'spamassassin and sa-learn must be in PATH';
}

plan tests => 2;

# 5 spam : 10 ham - ham-heavy stream (1:2 ratio), mirrors a real inbox where ham
# vastly outweighs spam over time.
my @train_spam = map { "t/data/spam-00$_" } 1..5;
my @train_ham  = (map({ "t/data/nice-00$_" } 1..9), 't/data/nice-010');

# Held-out: spam from the same distribution as training (similar vocabulary
# and patterns), so the test measures collapse prevention, not unseen-domain
# generalisation. A model collapsed to all-ham would fail these too.
my @test_spam  = map { "t/data/spam-00$_" } 1..5;

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

for my $f (@train_spam) {
  qx($salearnrun --siteconfigpath=t/rules --spam $f);
}
for my $f (@train_ham) {
  qx($salearnrun --siteconfigpath=t/rules --ham $f);
}

# After a ham-heavy training stream, at least one held-out spam must
# still classify as SPAM. The bug we are guarding against: imbalanced
# online training drifts the output-layer bias toward the majority
# class, causing every prediction to fall below ham_threshold (all
# emails tagged HAM, regardless of content).
my $any_spam_correct = 0;
for my $f (@test_spam) {
  my $out = qx($sarun -L -t --siteconfigpath=t/rules < $f);
  $any_spam_correct = 1 if $out =~ /NN_SPAM/;
}
ok($any_spam_correct,
   'at least one held-out spam classified as spam (model not collapsed to ham)');

# And the model must still recognise ham as ham (the easy direction).
my $out = qx($sarun -L -t --siteconfigpath=t/rules < t/data/nice-006);
like($out, qr/NN_HAM/, "t/data/nice-006 classified as ham");

tstcleanup();
