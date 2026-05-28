CREATE TABLE neural_seen (
  username VARCHAR(200) NOT NULL DEFAULT 'default',
  msgid VARBINARY(200) NOT NULL DEFAULT '',
  flag CHAR(1) NOT NULL DEFAULT '',
  PRIMARY KEY (username, msgid)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- keyword uses a binary collation: the neural-network feature vector treats
-- every distinct token as its own input position, so DB key uniqueness must
-- match Perl string equality exactly. utf8mb4_unicode_ci is case- and
-- accent-insensitive and would collapse tokens like "Hello"/"hello" or
-- "cafe"/"café" into one row, leaving the on-disk model one (or more) inputs
-- wider than the vocabulary loaded back at prediction time.
CREATE TABLE neural_vocabulary (
  username VARCHAR(200) NOT NULL DEFAULT '',
  keyword VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL DEFAULT '',
  total_count INT NOT NULL DEFAULT 0,
  docs_count INT NOT NULL DEFAULT 0,
  spam_count INT NOT NULL DEFAULT 0,
  ham_count INT NOT NULL DEFAULT 0,
  model_position INT DEFAULT NULL,
  PRIMARY KEY (username, keyword),
  KEY neural_vocab_model_pos_idx (username, model_position)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE neural_vars (
  username varchar(200) NOT NULL default '',
  variable varchar(30)  NOT NULL default '',
  value    varchar(200) NOT NULL default '',
  PRIMARY KEY (username, variable)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE neural_training_buffer (
  username varchar(200) NOT NULL default '',
  class    ENUM('spam', 'ham') NOT NULL,
  slot     int          NOT NULL default 0,
  ts       int          NOT NULL default 0,
  token    varchar(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL default '',
  count    int          NOT NULL default 1,
  PRIMARY KEY (username, class, slot, token)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Migration for existing installations: fixes collation mismatch errors.
-- Run once if your tables were created without an explicit charset.
-- ALTER TABLE neural_seen             CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
-- ALTER TABLE neural_vocabulary       CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
-- ALTER TABLE neural_vars             CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
-- ALTER TABLE neural_training_buffer  CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Migration to make keyword/token case- and accent-sensitive (binary collation).
-- REQUIRED to fix model/vocab size drift ("vocab=N-1, model=N") on MySQL/MariaDB.
-- De-duplicate any rows that already collided under the old ci collation first,
-- otherwise the ALTER will fail with a duplicate-key error.
--   ALTER TABLE neural_vocabulary
--     MODIFY keyword VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL DEFAULT '';
--   ALTER TABLE neural_training_buffer
--     MODIFY token   VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL DEFAULT '';
