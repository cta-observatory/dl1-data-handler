#!/bin/bash
# We test the environment variables in a different recipe

# Ensure we are in a git repo
[ -d .git ]
git describe
[ "$(git describe)" = 1.20.2 ]

# check if GIT_* variables are defined
for i in GIT_DESCRIBE_TAG GIT_DESCRIBE_NUMBER GIT_DESCRIBE_HASH GIT_FULL_HASH
do
  if [ -n "eval $i" ]; then
    eval echo \$$i
  else
    exit 1
  fi
done

GIT_DESCRIBE_TAG=""
$PYTHON setup.py install --single-version-externally-managed --record=record.txt




