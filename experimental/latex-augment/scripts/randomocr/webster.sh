# from https://www.gutenberg.org/ebooks/29765

curl -L -o webster.txt https://www.gutenberg.org/ebooks/29765.txt.utf-8

cat webster.txt | \
  tr -d '*`"' | \
  tr "[:upper:]" "[:lower:]" | \
  tr -cs "a-z\-" "\n" | \
  grep '^[a-z][a-z-]*[a-z]$' | \
  sort -u > webster1913_english_words.txt
