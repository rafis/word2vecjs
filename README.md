# word2vecjs
After word2vec came out in C, it been ported to Win32 and MacOS platforms, then on python (gensim). But I haven't found the port to JavaScript (NodeJS).

## Limitations

* Only for learning
* Only Skip-gram model (no CBOW model)
* Only Negative sampling (no Hierarchical soft-max)
* Only single thread
* Only word2vec executable (no distance, word-analogy, compute-accuracy, word2phrase)

## Usage

```sh
node bin/word2vec --train data.txt --output vec.txt -size 300 --cbow 0
```
