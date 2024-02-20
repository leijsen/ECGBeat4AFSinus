#!/bin/bash

# Define the test sets as arrays
test_set1=(16 62 12 22 71 85 53 69 26 70 5 40 41 55 47 36 78 89 45 8 79)
test_set2=(17 31 100 93 102 95 84 86 18 11 48 97 88 83 104 30 29 75 4 77 37)
test_set3=(103 50 64 28 68 80 90 51 98 39 3 56 58 60 54 34 38 23 13 2 57)
test_set4=(67 15 61 96 24 9 59 91 6 0 65 81 10 20 82 99 92 35 101 27 52)
test_set5=(43 1 49 72 25 73 44 76 7 66 32 46 42 21 14 63 94 87 33 19 74)

# Define the python and log file
python_file="train_net1d.py"
log_file="test"

# Function to process a test set
process_test_set() {
  local current_test_set=("$@")
  local test_set_str=$(IFS=,; echo "${current_test_set[*]}")
  local log_file="test_${current_test_set[0]}.log"  # Use the first element as the index

  # Run the Python script with the array as a command-line argument
  python -u "$python_file" --variable "$test_set_str" > "$log_file" 2>/dev/null &
#  python -u "$python_file" --variable "$test_set_str" > "$log_file" 2>&1 &
  echo "Script has started. Running in the background..."
}

# Call the function for each test set
process_test_set "${test_set1[@]}"
process_test_set "${test_set2[@]}"
process_test_set "${test_set3[@]}"
process_test_set "${test_set4[@]}"
process_test_set "${test_set5[@]}"
