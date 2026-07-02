'''brute force enumeration of the policy outputs, just to test
'''
from itertools import product

letters = ['a', 'b', 'c', 'd', 'e']
nums = list(range(1,6))
crowns = ('q', 'r', 'b', 'n')

p = list(product(letters, nums))
q = p.copy()
prod = list(product(p, q))

# filter those with same initial and end square
for t in prod:
    if t[0] == t[1]:
        prod.remove(t)

print(len(prod))

# now, for each one that goes either from row 4 to 5 or 2 to 1, 
# there should be 4 that end with (q, r, b, n)
for t in prod:
    if (t[0][1] == 4 and t[1][1] == 5) and \
        ( abs( ord(t[0][0]) - ord(t[1][0])) < 2):
        # here, <2 is enough 
        # add the 4 for each
        for c in crowns:
            temp = (t[0], (t[1][0], [t[1][1], c]))
            prod.append(temp)

    elif (t[0][1] == 2 and t[1][1] == 1) and \
        ( abs( ord(t[0][0]) - ord(t[1][0]) ) < 2):
        # add the 4 for each
        for c in crowns:
            temp = (t[0], (t[1][0], [t[1][1], c]))
            prod.append(temp)

print(len(prod))

# build the actual uci moves
moves = []
for t in prod:
    if isinstance(t[1][1], list):
        move = t[0][0] + str(t[0][1]) + t[1][0] + str(t[1][1][0]) + t[1][1][1]
    else:
        move = t[0][0] + str(t[0][1]) + t[1][0] + str(t[1][1])

    moves.append(move)

print(moves)

from src.models.dataset_parser import uci_to_index

for move in moves:
    print(move, uci_to_index(move, promotions=True))
    if uci_to_index(move, promotions=True) == 599:
        print("found it!",move)
