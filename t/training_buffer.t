#!/usr/bin/perl

use strict;
use warnings;
use lib '.'; use lib 't';

use File::Path;
use File::Spec;
use Storable qw(retrieve);
use Test::More;
plan tests => 5;

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

# Disable periodic retrain so it doesn't interfere with the buffer assertions
tstprefs("
  loadplugin Mail::SpamAssassin::Plugin::NeuralNetwork ../../NeuralNetwork.pm

  neuralnetwork_data_dir	t/NN
  neuralnetwork_min_spam_count	0
  neuralnetwork_min_ham_count	0
  neuralnetwork_min_vocab_hits	0
  neuralnetwork_retrain_interval	0

  body		NN_SPAM		eval:check_neuralnetwork_spam()
  describe	NN_SPAM		Email considered as spam by Neural Network
  score		NN_SPAM		1.0

  body		NN_HAM		eval:check_neuralnetwork_ham()
  describe	NN_HAM		Email considered as ham by Neural Network
  score		NN_HAM		-1.0
");

rmtree 't/NN';
mkdir 't/NN';

qx($salearnrun --siteconfigpath=t/rules --ham  t/data/nice-001);
qx($salearnrun --siteconfigpath=t/rules --spam t/data/spam-001);

my $username   = lc((getpwuid($<))[0] || 'nobody');
my $vocab_path = File::Spec->catfile('t', 'NN', 'vocabulary-' . $username . '.data');
ok(-f $vocab_path, "vocabulary file exists at $vocab_path");

my $vocab = eval { retrieve($vocab_path) };
ok(ref($vocab) eq 'HASH', 'vocabulary loaded as hashref');

my $buf = $vocab->{_tbuf};
ok(ref($buf) eq 'HASH', '_tbuf is a hashref');

is(scalar(@{ $buf->{ham}  || [] }), 1, 'ham training buffer has 1 entry');
is(scalar(@{ $buf->{spam} || [] }), 1, 'spam training buffer has 1 entry');

tstcleanup();
