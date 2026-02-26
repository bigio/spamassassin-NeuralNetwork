# Note for users upgrading to version 0.3
- the model files `fann-user.model` must be removed after the upgrade in order
  to correctly retrain the model

# Note for users upgrading from version 0.1
- the `id` field is no more used and should be removed from SQL tables    
  ex.:    
  ALTER TABLE `neural_seen` DROP `id`;    
  ALTER TABLE `neural_seen` DROP INDEX `neural_seen_idx1`, ADD PRIMARY KEY (`username`, `msgid`) USING BTREE;    
  ALTER TABLE `neural_vocabulary` DROP `id`;    
  ALTER TABLE `neural_vocabulary` DROP INDEX `neural_vocab_idx1`, ADD PRIMARY KEY (`username`, `keyword`) USING BTREE;    
