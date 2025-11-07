#!/bin/bash

git add .

# Check if there's anything staged
if ! git diff --cached --quiet; then
  echo "Enter commit message:"
  read -r commit_message

  if [[ -z "$commit_message" ]]; then
    echo "Error: Commit message cannot be empty."
    exit 1
  fi

  git commit -m "$commit_message"
  git push origin main
else
  echo "No changes to commit."
fi
