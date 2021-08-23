import sys
import copy
from multiprocessing import Pool
import numpy as np
import itertools

# look through fn for all the provided search terms (keys),
# and extract values as directed by the offset, start & end cols

# example usage to get IUPAC:
# python extract_mass_formula.py Compounds.xml <PC-Compound> Systematic 11 34 -26

# Systematic 11 34 -26
# Mass 12 34 -26
# Formula 11 34 -26
# Log P 11 34 -26

# chemicals are separated by the 2nd arg

LINES_PER_PROC = 10000

fn = sys.argv[1]

assert len(sys.argv) > 3, "need to provide search terms, etc."
assert len(sys.argv[3:]) % 4 == 0, "each search term needs offset & cols"

chemical_separator = sys.argv[2]

search_terms = []
line_offsets = []
start_cols = []
end_cols = []

for i in range(3, len(sys.argv), 4):
    search_terms.append(sys.argv[i])
    line_offsets.append(int(sys.argv[i+1]))
    start_cols.append(int(sys.argv[i+2]))
    end_cols.append(int(sys.argv[i+3]))

lines = []

def find_relevant(start_line):
    relevant_lines = []
    max_length = len(lines)
    for i in range(LINES_PER_PROC):
        if start_line + i >= max_length:
            return relevant_lines
        line = lines[start_line + i]
        if chemical_separator in line:
            relevant_lines.append(start_line + i)
        for search_term in search_terms:
            if search_term in line:
                relevant_lines.append(start_line + i)
    return relevant_lines

with open(fn, "r") as xml_file:
    # first line is headers
    found_values = copy.deepcopy(search_terms)

    lines = xml_file.readlines()

    p = Pool(32)
    relevant_lines = p.map(find_relevant,
                           range(0, len(lines), LINES_PER_PROC))
    relevant_lines = itertools.chain.from_iterable(relevant_lines)
    relevant_lines = np.array(list(relevant_lines))

    for i in relevant_lines:
        line = lines[i]
        if chemical_separator in line:
            # new chemical -- reset search term lines & found_values
            print("|".join(found_values))
            found_values = ["" for _ in search_terms]
            continue

        for j, search_term in enumerate(search_terms):
            if search_term in line:
                # found the jth search term on line i
                found = i + line_offsets[j]
                found_values[j] = lines[found][start_cols[j]:end_cols[j]]


