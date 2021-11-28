mkdir -p data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1g0NA9_j1iDC8_E1zbW9IanzZaC9K1DAc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1g0NA9_j1iDC8_E1zbW9IanzZaC9K1DAc" -O data/wikipedia_20190801.db.zip   && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YiKaGbCyE8p-z-3XepCpJMEvzlc7Oe4v' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YiKaGbCyE8p-z-3XepCpJMEvzlc7Oe4v" -O data/dpr.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HC3CUnMmP7zoz61Sok8rlQ7lN17ddQxc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HC3CUnMmP7zoz61Sok8rlQ7lN17ddQxc" -O data/tfidf.zip && rm -rf /tmp/cookies.txt

for i in data/*.zip; do unzip $i -d data/.;done