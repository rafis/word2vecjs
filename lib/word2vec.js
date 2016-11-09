'use strict';

var fs = require('mz/fs');
var Fd = require('./fd');

function VocabWord(word) {
    this.word = word;
    this.cnt = 0;
    this.point = [];
    this.code = [];
}

function Word2vec(params) {
    for(var key in params) {
        this[key] = params[key];
    }
    this.MAX_STRING = 100;
    this.VOCAB_HASH_SIZE = 30000000; // Maximum 30 * 0.7 = 21M words in the vocabulary
    this.TABLE_SIZE = 1e5; // Unigram distribution table size
    this.EXP_TABLE_SIZE = 1000;
    this.MAX_EXP = 6;
    this.MAX_SENTENCE_LENGTH = 1000;
}

/**
 * Returns size of vocabulary
 */
Word2vec.prototype.vocab_size = function() {
    return this.vocab.length;
};

/**
 * Reads a single word from a file, assuming space + tab + EOL to be word boundaries
 */
Word2vec.prototype.readWord = function(fin) {
    var word = [];
    var a = 0, ch;
    while(null !== (ch = fin.getc())) {
        if ('\r' == ch) {
            continue;
        }
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') {
                    fin.ungetc(ch);
                }
                break;
            }
            if (ch == '\n') {
                word = ["</s>"];
                break;
            } else {
                continue;
            }
        }
        word[a] = ch;
        a++;
        if (a >= this.MAX_STRING - 1) { a--; }   // Truncate too long words
    }
    return word.join('');
};

Word2vec.prototype.addWordToVocab = function(word) {
    this.vocab_hash[word] = this.vocab_size();
    var vocabWord = new VocabWord(word);
    this.vocab.push(vocabWord);
    return vocabWord;
};

/**
 * Returns position of a word in the vocabulary; if the word is not found, returns -1
 */
Word2vec.prototype.searchVocab = function(word) {
    return word in this.vocab_hash ? this.vocab_hash[word] : -1;
};

/**
 * Reads a word and returns its index in the vocabulary
 */
Word2vec.prototype.readWordIndex = function(fin) {
    var word = this.readWord(fin);
    if ('' == word) {
        return;
    }
    return this.searchVocab(word);
};

/**
 * Sorts the vocabulary by frequency using word counts
 */
Word2vec.prototype.sortVocab = function() {

    var self = this;

    // Sort the vocabulary and keep </s> at the first position
    var vocab = this.vocab.slice(1).filter(function(vocabWord) {
        return vocabWord.cnt >= self.min_count;
    }).sort(function(vocabWordA, vocabWordB) {
        return vocabWordB.cnt - vocabWordA.cnt;
    });

    vocab.unshift(this.vocab[0]);
    this.vocab = vocab;

    // Hash will be re-computed, as after the sorting it is not actual
    this.vocab_hash = {};
    this.vocab.forEach(function(vocabWord, i) {
        self.vocab_hash[vocabWord.word] = i;
    });

    // Recalculate words in train file, because we have filtered infrequent words
    this.train_words = 0;
    this.vocab.forEach(function(vocabWord, i) {
        self.train_words += vocabWord.cnt;
    });

};

Word2vec.prototype.reduceVocab = function() {
    throw new Error('Not implemented');
};

Word2vec.prototype.readVocab = function() {
    
    var fin = new Fd(this.read_vocab_file);
    while(true) {
        var word = this.readWord(fin);
        if ('' == word) {
            break;
        }
        var vocabWord = this.addWordToVocab(word);
        var cnt = [];
        var ch;
        while(null !== (ch = fin.getc())) {
            if (/\d/.test(ch)) {
                cnt.push(ch);
            } else {
                break;
            }
        }
        vocabWord.cnt = parseInt(cnt.join(''));
    }
    fin.close();
    
    var self = this;
    
    self.sortVocab();

    if (self.debug_mode > 0) {
        console.log("Vocab size: ", self.vocab_size());
        console.log("Words in train file: ", self.train_words);
    }

    self.file_size = fs.statSync(self.train_file)['size'];
    
};

Word2vec.prototype.saveVocab = function() {

    var fo = fs.createWriteStream(this.save_vocab_file, { encoding: 'utf8' });
    
    return this.vocab.reduce(function(sequence, vocabWord, i) {
        return sequence.then(function() {
            return new Promise(function(resolve, reject) {
                fo.once('error', reject);
                fo.write(vocabWord.word + " " + vocabWord.cnt + "\n", function() {
                    fo.removeListener('error', reject);
                    resolve();
                });
            });
        });
    }, Promise.resolve()).then(function() {
        fo.end();
        return new Promise(function(resolve, reject) {
            fo.once('error', reject);
            fo.once('close', function() {
                fo.removeListener('error', reject);
                resolve();
            });
        });
    });

};

Word2vec.prototype.initNet = function() {
    var syn0 = Array(this.vocab_size());
    for(var a = 0; a < this.vocab_size(); a++) {
        syn0[a] = Array(this.layer1_size);
        for(var b = 0; b < this.layer1_size; b++) {
            syn0[a][b] = (Math.random() - 0.5) / this.layer1_size;
        }
    }
    this.syn0 = syn0;

    if (this.negative > 0) {
        var syn1neg = Array(this.vocab_size());
        for(var a = 0; a < this.vocab_size(); a++) {
            syn1neg[a] = Array(this.layer1_size);
            for(var b = 0; b < this.layer1_size; b++) {
                syn1neg[a][b] = 0;
            }
        }
        this.syn1neg = syn1neg;
    }
};

Word2vec.prototype.initUnigramTable = function() {

    var a, i;
    var train_words_pow = 0;
    var d1, power = 0.75;
    var table = Array(this.TABLE_SIZE);
    for(a = 0; a < this.vocab_size(); a++) {
        train_words_pow += Math.pow(this.vocab[a].cnt, power);
    }
    i = 0;
    d1 = Math.pow(this.vocab[i].cnt, power) / train_words_pow;
    for(a = 0; a < this.TABLE_SIZE; a++) {
        table[a] = i;
        if (a / this.TABLE_SIZE > d1) {
            i++;
            d1 += Math.pow(this.vocab[i].cnt, power) / train_words_pow;
        }
        if (i >= this.vocab_size()) {
            i = this.vocab_size() - 1;
        }
    }

    this.table = table;

};

Word2vec.prototype.learnVocabFromTrainFile = function() {

    this.addWordToVocab("</s>");

    var self = this;

    return Promise.resolve().then(function() {
        
        self.train_words = 0;

        var fin = new Fd(self.train_file);

        while(true) {

            var word = self.readWord(fin);
            if ('' == word) {
                break;
            }
            self.train_words++;

            if ((self.debug_mode > 1) && (self.train_words % 100000 == 0)) {
                process.stdout.write(Math.floor(self.train_words / 1000)+'K\r');
            }

            var vocabWord = self.vocab[self.searchVocab(word)];
            if (null == vocabWord) {
                vocabWord = self.addWordToVocab(word);
                vocabWord.cnt = 1;
            } else {
                vocabWord.cnt++;
            }

            if (self.vocab_size() > self.VOCAB_HASH_SIZE * 0.7) {
                self.reduceVocab();
            }

        }
        
        fin.close();
        
    }).then(function() {

        self.sortVocab();

        if (self.debug_mode > 0) {
            console.log("Vocab size: ", self.vocab_size());
            console.log("Words in train file: ", self.train_words);
        }
        
        self.file_size = fs.statSync(self.train_file)['size'];

    });
    
};

Word2vec.prototype.trainModelThread = function(id) {
  
    var a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
    var word_count = 0, last_word_count = 0, sen = new Array(this.MAX_SENTENCE_LENGTH);
    var l1, l2, c, target, label, local_iter = this.iter;
    var f, g;
    var now;
    var neu1 = Array(this.layer1_size);
    var neu1e = Array(this.layer1_size);

    var fi = new Fd(this.train_file);
    fi.position = this.file_size / this.num_threads * id;
    
    while(true) {

        if (word_count - last_word_count > 10000) {
            this.word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if (this.debug_mode > 1) {
                now = new Date().getTime();
                process.stdout.write("\rAlpha "+this.alpha+"  Progress: "+(this.word_count_actual / (this.iter * this.train_words + 1) * 100).toFixed(2)+"%  Words/sec: "+(this.word_count_actual / ((now - this.start + 1) / 1000)).toFixed(2)+"k  ");
            }
            this.alpha = this.starting_alpha * (1 - this.word_count_actual / (this.iter * this.train_words + 1));
            if (this.alpha < this.starting_alpha * 0.0001) {
                this.alpha = this.starting_alpha * 0.0001;
            }
        }

        if (sentence_length == 0) {
            while (true) {
                word = this.readWordIndex(fi);
                if (null == word) { break; }
                if (word == -1) { continue; }
                word_count++;
                if (word == 0) { break; }
                // The subsampling randomly discards frequent words while keeping the ranking same
                if (this.sample > 0) {
                    var ran = (Math.sqrt(this.vocab[word].cnt / (this.sample * this.train_words)) + 1) * (this.sample * this.train_words) / this.vocab[word].cnt;
                    if (ran < Math.random()) {
                        continue;
                    }
                }
                sen[sentence_length] = word;
                sentence_length++;
                if (sentence_length >= this.MAX_SENTENCE_LENGTH) { break; }
            }
            sentence_position = 0;
        }

        if (null == word || (word_count > this.train_words / this.num_threads)) {
            this.word_count_actual += word_count - last_word_count;
            local_iter--;
            if (local_iter == 0) break;
            word_count = 0;
            last_word_count = 0;
            sentence_length = 0;
            fi.position = this.file_size / this.num_threads * id;
            continue;
        }

        word = sen[sentence_position];
        if (word == -1) continue;
        for (c = 0; c < this.layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < this.layer1_size; c++) neu1e[c] = 0;
        b = (Math.random() * Number.MAX_SAFE_INTEGER) % this.window;
        if (this.cbow) {  //train the cbow architecture
            throw new Error('Not implemented');
        } else {  //train skip-gram
            for (a = b; a < this.window * 2 + 1 - b; a++) if (a != this.window) {
                c = sentence_position - this.window + a;
                if (c < 0) continue;
                if (c >= sentence_length) continue;
                last_word = sen[c];
                if (last_word == -1) continue;
                l1 = last_word * this.layer1_size;
                for (c = 0; c < this.layer1_size; c++) neu1e[c] = 0;
                // HIERARCHICAL SOFTMAX
                if (this.hs) {
                    throw new Error('Not implemented');
                }
                // NEGATIVE SAMPLING
                if (this.negative > 0) for (d = 0; d < this.negative + 1; d++) {
                    if (d == 0) {
                        target = word;
                        label = 1;
                    } else {
                        var next_random = Math.random() * Number.MAX_SAFE_INTEGER;
                        target = this.table[(next_random >> 16) % this.TABLE_SIZE];
                        if (target == 0) target = next_random % (this.vocab_size() - 1) + 1;
                        if (target == word) continue;
                        label = 0;
                    }
                    l2 = target * this.layer1_size;
                    f = 0;
                    for (c = 0; c < this.layer1_size; c++) f += this.syn0[c + l1] * this.syn1neg[c + l2];
                    if (f > this.MAX_EXP) g = (label - 1) * this.alpha;
                    else if (f < -this.MAX_EXP) g = (label - 0) * this.alpha;
                    else g = (label - this.expTable[Math.floor((f + this.MAX_EXP) * (this.EXP_TABLE_SIZE / this.MAX_EXP / 2))]) * this.alpha;
                    for (c = 0; c < this.layer1_size; c++) neu1e[c] += g * this.syn1neg[c + l2];
                    for (c = 0; c < this.layer1_size; c++) this.syn1neg[c + l2] += g * this.syn0[c + l1];
                }
                // Learn weights input -> hidden
                for (c = 0; c < this.layer1_size; c++) this.syn0[c + l1] += neu1e[c];
            }
        }
        sentence_position++;
        if (sentence_position >= sentence_length) {
            sentence_length = 0;
            continue;
        }
    }

    fi.close();

};

Word2vec.prototype.trainModel = function() {
    
    this.vocab = [];
    this.vocab_hash = {};
    this.starting_alpha = this.alpha;
    this.word_count_actual = 0;
    var expTable = Array(this.EXP_TABLE_SIZE);
    for(var i = 0; i < this.EXP_TABLE_SIZE; i++) {
        expTable[i] = Math.exp((i / this.EXP_TABLE_SIZE * 2 - 1) * this.MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                            // Precompute f(x) = exp(x) / (exp(x) + 1)
    }
    this.expTable = expTable;
    
    var self = this;
    
    return Promise.resolve().then(function() {
        
        if (self.read_vocab_file) {
            return self.readVocab();
        } else {
            return self.learnVocabFromTrainFile();
        }
        
    }).then(function() {

        if (self.save_vocab_file) {
            return self.saveVocab();
        }

    }).then(function() {

        if ('' == self.output_file) {
            throw new Error('No output file specified');
        }

        self.initNet();
        if (self.negative > 0) {
            self.initUnigramTable();
        }
        self.start = new Date().getTime();

        return self.trainModelThread(0);
        
    }).then(function() {

        if (self.classes == 0) {
            // Save the word vectors
            return Promise.resolve().then(function() {
                return fs.open(self.output_file, 'w');
            }).then(function(fd) {
                return Promise.resolve().then(function() {
                    return fs.write(fd, ''+self.vocab_size()+" "+self.layer1_size+"\n", null, 'utf8');
                }).then(function() {
                    return self.vocab.reduce(function(sequential, vocabWord, i) {
                        return sequential.then(function() {
                            return fs.write(fd, vocabWord.word+" ", null, 'utf8');
                        }).then(function() {
                            return self.syn0[i].reduce(function(sequential, val) {
                                return sequential.then(function() {
                                    return fs.write(fd, val+" ", null, 'utf8');
                                });
                            }, Promise.resolve());
                        }).then(function() {
                            return fs.write(fd, "\n", null, 'utf8');
                        });
                    }, Promise.resolve());
                }).then(function() {
                    return fs.close(fd);
                });
            });
        } else {
            throw new Error('Not implemented');
        }
    
    });

};

module.exports = Word2vec;
