import re
import sys

# Read the XML file
with open('e:/WALLABY/data/vizier_votable.tsv', 'r') as f:
    content = f.read()

# Find the CSV data section
match = re.search(r'<CSV headlines="3" colsep=";"><!\[CDATA\[(.*?)\]\]>', content, re.DOTALL)
if match:
    csv_data = match.group(1)
    # Skip header lines (first 3)
    lines = csv_data.strip().split('\n')
    # Remove first 3 header lines
    data_lines = lines[3:]
    # Write as proper CSV
    with open('e:/WALLABY/data/cosmicflows4.csv', 'w') as f:
        for line in data_lines:
            f.write(line.replace(';', ',') + '\n')
    print(f'Written {len(data_lines)} galaxies to cosmicflows4.csv')
else:
    print('Could not find CSV data')
    sys.exit(1)
