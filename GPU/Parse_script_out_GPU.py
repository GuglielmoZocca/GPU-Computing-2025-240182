import os
import sys

#Parse code for unifying tests data outputs/* in test/GPU_test_{sys.argv[1]}.csv

def parse_files_in_directory(directory_path):
    parsed_data = {}

    insert = False

    for filename in os.listdir(directory_path):
        if filename.endswith(".out"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                matching_lines = []
                for line in file:
                    if ("performance" in line) or ("platf" in line):
                        insert = True
                        continue
                    if insert:
                        matching_lines.append(line)
                        break
                if matching_lines:
                    parsed_data[filename] = matching_lines
                insert = False

    return parsed_data

if __name__ == "__main__":
    folder_path = "outputs"
    result = parse_files_in_directory(folder_path)

    with open(f'test/GPU_test_{sys.argv[1]}.csv', 'w') as file:
        file.write("platf,matrix,id,n,m,nonZeros,blockSize,Rand,memCopy,sort,mu,sigma,nflop,nMemAc,AI_O,AI_A,AI,Iperf,flops,effBand,RP\n")
        for filename, lines in result.items():
            for line in lines:
                file.write(f'{line}\n')