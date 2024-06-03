from validator import Validator
from algorithms import read_file

data = read_file('CS170_Spring_2024_Small_data__1.txt')
valid = Validator(data)
accuracy = valid.leave_one_out_logreg()