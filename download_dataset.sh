#!/bin/sh

url="https://zenodo.org/records/17770641/files/data.tar.gz?download=1"
outfile="data.tar.gz"

curl -L "$url" -o "$outfile"
tar -xzf "$outfile"
rm "$outfile"

