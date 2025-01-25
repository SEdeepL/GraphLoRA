#!/bin/bash
source_file = ""
patch_text = ""
bug_text = ""
graph_file = ""

python3 action.py $source_file $graph_file

python3 apti_pattern.py $patch_text $bug_text $graph_file

python3 distancetopatch.py $graph_file

python3 editDistance.py $patch_text $bug_text $graph_file

python3 entropy.py $patch_text $graph_file

python3 operator.py $graph_file

python3 statement.py $graph_file
