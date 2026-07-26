#!/usr/bin/env bash
set -euo pipefail

# Fashion-MNIST downloads automatically on first use.
# UCI Default Credit downloads through ucimlrepo and is cached locally.
# Give Me Some Credit requires a Kaggle account, API authentication, and
# acceptance of the competition rules in the web UI.

mkdir -p data/give_me_some_credit
kaggle competitions download -c GiveMeSomeCredit -p data/give_me_some_credit
python -m zipfile -e \
  data/give_me_some_credit/GiveMeSomeCredit.zip \
  data/give_me_some_credit

test -f data/give_me_some_credit/cs-training.csv
echo "Give Me Some Credit is ready."

