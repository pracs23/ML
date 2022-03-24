import sys
import logging

print("argument {}".format(sys.argv[1]))
print("changelog 1")
print("changelog 2")
print("changelog 3")
print("changelog 4")

get_branch_and_pr_creation_time()

def get_branch_and_pr_creation_time():
    logging.info("within method is: {}")
