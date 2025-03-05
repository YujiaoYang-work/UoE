echo "=== Acquiring datasets ==="
echo "---"

mkdir -p data
cd data

echo "- Downloading WikiText-103 (WT2)"
if [[ ! -d 'wikitext-103' ]]; then
    wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
    unzip -q wikitext-103-v1.zip
    cd wikitext-103
    mv wiki.train.tokens train.txt
    mv wiki.valid.tokens valid.txt
    mv wiki.test.tokens test.txt
    cd ..
fi

echo "- Downloading 1B words"

if [[ ! -d 'one-billion-words' ]]; then
    mkdir -p one-billion-words
    cd one-billion-words

    wget --no-proxy http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
    tar xzvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz

    path="1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/"
    cat ${path}/news.en.heldout-00000-of-00050 > valid.txt
    cat ${path}/news.en.heldout-00000-of-00050 > test.txt

    wget https://github.com/rafaljozefowicz/lm/raw/master/1b_word_vocab.txt

    cd ..
fi

echo "---"
echo "Happy language modeling :)"
