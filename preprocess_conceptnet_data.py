input_file = './conceptnet_data/rw_corpus_1.0_1.0_2_15_nl_2.txt'  # Replace with the actual input file path
output_file = './conceptnet_data/conceptnet_corpus_2.txt'  # Replace with the desired output file path

# Read input file and filter out empty lines
lines = []
with open(input_file, 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            lines.append(line)

# Write non-empty lines to output file
with open(output_file, 'w') as file:
    file.write('\n'.join(lines))