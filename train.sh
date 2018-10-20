#!/usr/bin/env bash

USE_ANTECEDENT=$1
USE_MENTION=$2
NUM_TURNS=$3
COMBINATION=$4

mkdir -p exp_dir
DIR="exp_dir/ant_${USE_ANTECEDENT}__ment_${USE_MENTION}__numTurn${NUM_TURNS}__comb_${COMBINATION}"
echo "directory: " $DIR

OVERRIDE="{\"ctx_q_encoder\":{\"combination\":\"${COMBINATION}\",\"num_turns\":${NUM_TURNS},\"use_antecedent_score\":${USE_ANTECEDENT},\"use_mention_score\":\"${USE_MENTION}\"}}"
echo $OVERRIDE
rm -rf $DIR
 allennlp train experiment-multiturn.json -s $DIR -o $OVERRIDE --include-package models
