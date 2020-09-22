import re

FILE_NAME = 'outputs/rub6-d.txt'

re_supp = re.compile('\[(\d+\.\d+) (\d+\.\d+)\]')

with open(FILE_NAME, 'r') as rub_file:
    cnt = 0
    while True:
        line = rub_file.readline()
        if line:
            m = re_supp.search(line)
            if m:
                cnt += 1
                s1 = float(m.group(1))
                s2 = float(m.group(2))
                if s1 - s2 > 0.3:
                    print(line)
            else:
                print(line)
        else:
            print(cnt)
            break