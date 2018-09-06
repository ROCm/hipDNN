import csv
import sys

hcc_result = sys.argv[1]
nvcc_result = sys.argv[2]
reader1 = csv.DictReader(open(hcc_result, 'r'))
amd_op = []
for line1 in reader1:
	amd_op.append(line1)
	
reader2 = csv.DictReader(open(nvcc_result, 'r'))
nv_op = []

for line2 in reader2:
	nv_op.append(line2)
	
def test_compare(amd_op,nv_op):
	for (line1,line2) in zip(amd_op,nv_op):
		if (line1['Test_name'] == line2['Test_name']):
			if (line1['Output'] == line2['Output']):
				print ("test pass for "+line1['Test_name'])
				print(sum(c1!=c2 for c1,c2 in zip(line1['Output'],line2['Output'])))
				
			else:
				print ("test fail for "+line1['Test_name']+"!!!difference in outputs!!!!")
				print(sum(c1!=c2 for c1,c2 in zip(line1['Output'],line2['Output'])))
						
		else:
			print ("error in test cases!!!")
			
test_compare(amd_op,nv_op)