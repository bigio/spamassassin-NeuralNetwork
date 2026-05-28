CREATE TABLE IF NOT EXISTS neural_seen (
  username VARCHAR(200) NOT NULL DEFAULT 'default',
  msgid VARCHAR(200) NOT NULL DEFAULT '',
  flag CHAR(1) NOT NULL DEFAULT '',
  UNIQUE (username, msgid)
);

CREATE TABLE IF NOT EXISTS neural_vocabulary (
  username VARCHAR(200) NOT NULL DEFAULT '',
  keyword VARCHAR(256) NOT NULL DEFAULT '',
  total_count INTEGER NOT NULL DEFAULT 0,
  docs_count INTEGER NOT NULL DEFAULT 0,
  spam_count INTEGER NOT NULL DEFAULT 0,
  ham_count INTEGER NOT NULL DEFAULT 0,
  model_position INTEGER DEFAULT NULL,
  UNIQUE (username, keyword)
);

CREATE INDEX IF NOT EXISTS neural_vocabulary_username_idx ON neural_vocabulary(username);
CREATE INDEX IF NOT EXISTS neural_vocabulary_model_position_idx ON neural_vocabulary(username, model_position);

CREATE TABLE IF NOT EXISTS neural_vars (
  username VARCHAR(200) NOT NULL DEFAULT '',
  variable VARCHAR(30)  NOT NULL DEFAULT '',
  value    VARCHAR(200) NOT NULL DEFAULT '',
  PRIMARY KEY (username, variable)
);

CREATE TABLE IF NOT EXISTS neural_training_buffer (
  username VARCHAR(200) NOT NULL DEFAULT '',
  class    VARCHAR(4)   NOT NULL CHECK (class IN ('spam', 'ham')),
  slot     INTEGER      NOT NULL DEFAULT 0,
  ts       INTEGER      NOT NULL DEFAULT 0,
  token    VARCHAR(256) NOT NULL DEFAULT '',
  count    INTEGER      NOT NULL DEFAULT 1,
  PRIMARY KEY (username, class, slot, token)
);
