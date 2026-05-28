#!/usr/bin/perl

use strict;
use warnings;
use lib '.'; use lib 't';

use File::Path;
use File::Spec;
use Storable qw(retrieve);
use Test::More;
plan tests => 3;

sub tstprefs {
  my $rules = shift;
  open(OUT, '>', 't/rules/NeuralNetwork.cf') or die("Cannot write to rules directory: $!");
  print OUT $rules;
  close OUT;
}

sub tstcleanup {
  unlink 't/rules/NeuralNetwork.cf';
  rmtree 't/NN';
}

my $salearnrun = qx{which sa-learn 2>&1};
chomp($salearnrun);

tstprefs("
  loadplugin Mail::SpamAssassin::Plugin::NeuralNetwork ../../NeuralNetwork.pm

  neuralnetwork_data_dir	t/NN
  neuralnetwork_min_spam_count	0
  neuralnetwork_min_ham_count	0
  neuralnetwork_min_vocab_hits	0
  neuralnetwork_retrain_interval	1

  body		NN_SPAM		eval:check_neuralnetwork_spam()
  describe	NN_SPAM		Email considered as spam by Neural Network
  score		NN_SPAM		1.0

  body		NN_HAM		eval:check_neuralnetwork_ham()
  describe	NN_HAM		Email considered as ham by Neural Network
  score		NN_HAM		-1.0
");

mkdir 't/NN';

qx($salearnrun --siteconfigpath=t/rules --spam t/data/spam-001);

qx($salearnrun --siteconfigpath=t/rules --ham t/data/nice-001);

my $username   = lc((getpwuid($<))[0] || 'nobody');
my $vocab_path = File::Spec->catfile('t', 'NN', 'vocabulary-' . $username . '.data');
ok(-f $vocab_path, "vocabulary file created at $vocab_path");

my $vocab = eval { retrieve($vocab_path) };
ok(ref($vocab) eq 'HASH', 'vocabulary loaded as hashref');

is($vocab->{_learns_since_retrain}, 0,
   '_learns_since_retrain resets to 0 after periodic retrain triggers');

tstcleanup();
