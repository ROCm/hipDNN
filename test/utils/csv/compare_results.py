import os
import csv
import sys
from termcolor import colored

csv.field_size_limit(sys.maxsize)

hcc_folder = sys.argv[1]
nvcc_folder = sys.argv[2]

hcc_result = sys.argv[3]
nvcc_result = sys.argv[4]

reader1 = csv.DictReader(open(hcc_result, 'r'))
amd_op = []
for line1 in reader1:
    amd_op.append(line1)

reader2 = csv.DictReader(open(nvcc_result, 'r'))
nv_op = []
for line2 in reader2:
    nv_op.append(line2)

csvfile = open('final_results.csv', 'w+')

fieldnames = ['TestName','Results','Input size', 'kernel size',
              'output size', 'Performance in Nvidia (microseconds)',
              'Performance in AMD (microseconds)','AMD vs NVidia speedup']

writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

count = 0
cnt_p = 0
cnt_f = 0

t_pass = colored('Test passed!', 'green', attrs=['bold'])
t_fail = colored('Test failed!', 'red', attrs=['bold'])
test_name = colored('Test name: ', 'blue', attrs=['bold'])

for filename1 in os.listdir(hcc_folder):
    for filename2 in os.listdir(nvcc_folder):
       if (filename1 == filename2):
          with open((hcc_folder+"/"+filename1),'rb') as csvfile1, open((nvcc_folder+"/"+filename2),'rb') as csvfile2:
                    readCSV1 = csv.reader(csvfile1, delimiter=',')
                    readCSV2 = csv.reader(csvfile2, delimiter=',')
                    for row1 in readCSV1:
                        for row2 in readCSV2:
                           x = 0.0
                           y = 0.0
                           avg_mismatch = 0.0
                           if (row1[0] == row2[0]):
                              count2 = 0
                              flag = t_pass
                              for op1,op2 in zip(row1[1:],row2[1:]):
                                  if (op1=='' or op2==''):
                                     break
                                  n1 = float(op1)
                                  n2 = float(op2)
                                  if (n1 != n2):
                                     if ((abs(n1 - n2)) > 0.001):
                                        if (count2 == 0):
                                            x = n1
                                            y = n2
                                            min_mismatch = abs(n1 - n2)
                                            max_mismatch = abs(n1 - n2)
                                        if (abs(n1 - n2) < min_mismatch):
                                           min_mismatch = abs(n1 - n2)
                                        if (abs(n1 - n2) > max_mismatch):
                                           max_mismatch = abs(n1 - n2)
                                        avg_mismatch = (avg_mismatch + abs(n1 - n2))
                                        count2 = count2 + 1
                                        flag = t_fail
                              if (flag == t_pass) :
                                 print test_name, row1[0]
                                 print "  Test status: ",t_pass
                                 for op3 in amd_op:
                                    for op4 in nv_op:
                                        if (op3['Test_name'] == op4['Test_name'] == row1[0]):
                                           writer.writerow({'TestName':row1[0],
                                                 'Results':'PASS',
                                                 'Input size':op3['Input size'],
                                                 'kernel size':op3['kernel size'],
                                                 'output size':op3['output size'],
                                                 'Performance in Nvidia (microseconds)': op4['Average Excecution Time (microseconds)'],
                                                 'Performance in AMD (microseconds)': op3['Average Excecution Time (microseconds)'],
                                                 'AMD vs NVidia speedup':(float)(float(op3['Average Excecution Time (microseconds)'])/
                                                                                float(op4['Average Excecution Time (microseconds)']))})
                                 cnt_p = cnt_p + 1
                              else:
                                  print test_name, row1[0]
                                  print "  Test status: ", t_fail
                                  print "  Total number of mismatches: ",count2
                                  print "  First mismatch: ",x," != ",y
                                  print "  Minimum mismatch: ", min_mismatch
                                  print "  Maximum mismatch: ",max_mismatch
                                  print "  Average mismatch: ",(avg_mismatch/count2)
                                  cnt_f = cnt_f + 1
                                  for op3 in amd_op:
                                     for op4 in nv_op:
                                         if (op3['Test_name'] == op4['Test_name'] == row1[0]):
                                            writer.writerow({'TestName':row1[0],
                                                  'Results':'FAIL',
                                                  'Input size':op3['Input size'],
                                                  'kernel size':op3['kernel size'],
                                                  'output size':op3['output size'],
                                                  'Performance in Nvidia (microseconds)': op4['Average Excecution Time (microseconds)'],
                                                  'Performance in AMD (microseconds)': op3['Average Excecution Time (microseconds)'],
                                                  'AMD vs NVidia speedup':(float)(float(op3['Average Excecution Time (microseconds)'])/
                                                                                  float(op4['Average Excecution Time (microseconds)']))})
                              count = count + 1
                              print "\n"
                              break

print "\nTotal number of tests: ", count
print "Number of tests passed: ", cnt_p
print "Number of tests failed: ", cnt_f