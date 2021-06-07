import csv


with open('real.csv', 'r') as csv_file:
    csv_reader=csv.reader(csv_file)
    i=0
    for line in csv_reader:
        i=i+1
        if(i==27):
            print(line.index("200"))
            break