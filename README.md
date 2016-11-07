# word2vecjs
Port of word2vec to JavaScript.

## Limitations

* Only for learning
* Only Skip-gram model (no CBOW model)
* Only Negative sampling (no Hierarchical soft-max)
* Only single thread

## Usage

```sh
node bin/word2vec --train data.txt --output vec.txt -size 300 --cbow 0
```
