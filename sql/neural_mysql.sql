CREATE TABLE neural_seen (
  username VARCHAR(200) NOT NULL DEFAULT 'default',
  msgid VARBINARY(200) NOT NULL DEFAULT '',
  flag CHAR(1) NOT NULL DEFAULT '',
  PRIMARY KEY (username, msgid)
) ENGINE=InnoDB;

CREATE TABLE neural_vocabulary (
  username VARCHAR(200) NOT NULL DEFAULT '',
  keyword VARCHAR(256) NOT NULL DEFAULT '',
  total_count INT NOT NULL DEFAULT 0,
  docs_count INT NOT NULL DEFAULT 0,
  spam_count INT NOT NULL DEFAULT 0,
  ham_count INT NOT NULL DEFAULT 0,
  model_position INT DEFAULT NULL,
  PRIMARY KEY (username, keyword),
  KEY neural_vocab_model_pos_idx (username, model_position)
) ENGINE=InnoDB;
