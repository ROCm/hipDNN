import os
import csv
import sys
from termcolor import colored

csv.field_size_limit(sys.maxsize)

hcc_folder = sys.argv[1]
nvcc_folder = sys.argv[2]

amd_op = []
nv_op = []

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

                                        count2 = count2 + 1
                                        flag = t_fail

                              if (flag == t_pass) :       
                                 print test_name, row1[0]
                                 print "  Test status: ",t_pass 
                                 cnt_p = cnt_p + 1
                              else: 
                                 print test_name, row1[0]
                                 print "  Test status: ", t_fail
                                 print "  Total number of mismatches: ",count2
                                 print "  First mismatch: ",x," != ",y
                                 cnt_f = cnt_f + 1

                              count = count + 1
                              print "\n"
                              break
                                           
print "\nTotal number of tests: ", count
print "Number of tests passed: ", cnt_p
print "Number of tests failed: ", cnt_f
