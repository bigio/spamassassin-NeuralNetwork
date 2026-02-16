#!/usr/bin/perl

use lib '.'; use lib 't';

use File::Path;
use Test::More;
plan tests => 2;

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

my $sarun = qx{which spamassassin 2>&1};
my $salearnrun = qx{which sa-learn 2>&1};

tstprefs("
  loadplugin Mail::SpamAssassin::Plugin::NeuralNetwork ../../NeuralNetwork.pm

  neuralnetwork_data_dir	t/NN
  neuralnetwork_min_spam_count	0
  neuralnetwork_min_ham_count	0

  body		NN_SPAM		eval:check_neuralnetwork_spam()
  describe	NN_SPAM		Email considered as spam by Neural Network
  score		NN_SPAM		1.0

  body		NN_HAM		eval:check_neuralnetwork_ham()
  describe	NN_HAM		Email considered as ham by Neural Network
  score		NN_HAM		-1.0

");

mkdir 't/NN';
chomp($sarun);
chomp($salearnrun);
my $test = qx($salearnrun --siteconfigpath=t/rules --spam t/data/spam-001);
$test = qx($sarun -L -t --siteconfigpath=t/rules < t/data/spam-001);
like($test, "/NN_SPAM/");

$test = qx($salearnrun --siteconfigpath=t/rules --ham t/data/nice-001);
$test = qx($sarun -L -t --siteconfigpath=t/rules < t/data/nice-001);
like($test, "/NN_HAM/");

tstcleanup();
