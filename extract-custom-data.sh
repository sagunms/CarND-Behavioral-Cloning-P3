#!/bin/sh
# Combine the split archive to a single archive
zip -s 0 custom-data.zip --out unsplit-custom-data.zip
# Extract the single temporary archive using unzip
unzip unsplit-custom-data.zip
# Delete the temporary archive
rm unsplit-custom-data.zip