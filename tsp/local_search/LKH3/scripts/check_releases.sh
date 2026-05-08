#!/bin/bash

# Fetch the latest release from the webpage
WEBPAGE_URL="http://akira.ruc.dk/~keld/research/LKH-3/"
LATEST_RELEASE=$(curl -s $WEBPAGE_URL | grep -oP 'LKH-\K[0-9]+\.[0-9]+\.[0-9]+' | head -1)

# Update the code in the repository to the latest release
git fetch --tags
CURRENT_RELEASE=$(git describe --tags)

if [ "$LATEST_RELEASE" != "$CURRENT_RELEASE" ]; then
  # Download the latest release
  wget "$WEBPAGE_URL/LKH-$LATEST_RELEASE.tgz"
  
  # Remove the current SRC directory
  rm -rf SRC
  
  # Extract the tarball and strip the top directory
  tar --strip-components=1 -xvf "LKH-$LATEST_RELEASE.tgz"
else
  echo "Already up to date"
fi
