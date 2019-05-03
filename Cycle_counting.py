
mylist = []
G1 = [2, 4, 8, 9, 11, 12, 15, 17, 18, 21, 22, 24, 25]
S = [5, 6, 14]
SG2 = [1, 3, 13, 16, 23]
MG1 = [7, 10, 26]

with open('corrT124_all.txt', "r") as fp:
    for i in fp.readlines():
        tmp = i.split(" ")
        try:
            mylist.append((int(tmp[0]), int(tmp[1]), int(tmp[2])))
        except:pass

print(len(mylist))

# for G1,G1,G1

output = [item for item in mylist if 0 not in item and 1 not in item and 3 not in item and 5 not in item and 6 not in item
          and 7 not in item and 10 not in item and 13 not in item and 14 not in item
          and 16 not in item and 19 not in item and 20 not in item and 23 not in item and 26  not in item]

#print(output)
print('number of (G1,G1,G1):\n')
print(len(output))

l = [x for x in mylist if x in output]

#with open('/Users/farzaneh/Desktop/New_file/G1_G1_G1.txt', 'w') as outfile:
#    outfile.write(str(l))
#---------------------------------

# for S,S,S

output1 = [item for item in mylist if 0 not in item and 1 not in item and 2 not in item and 3 not in item and 4 not in item
          and 7 not in item and 8 not in item and 9 not in item and 10 not in item and 11 not in item and 12 not in item and 13 not in item and 15 not in item
          and 16 not in item and 17 not in item and 18 not in item and 19 not in item and 20 not in item and 21 not in item and 22 not in item and
           23 not in item and 24 not in item and 25 not in item and 26  not in item]

#print(output1)
print('number of (S,S,S):\n')
print(len(output1))

l = [x for x in mylist if x in output1]
#---------------------------------

# for S/G2,S/G2,S/G2

output2 = [item for item in mylist if 0 not in item and 2 not in item and 4 not in item and 5 not in item and 6 not in item
          and 7 not in item and 8 not in item and 9 not in item and 10 not in item and 11 not in item and 12 not in item and 14 not in item and 15 not in item
          and 17 not in item and 18 not in item and 19 not in item and 20 not in item and 21 not in item and 22 not in item and
           24 not in item and 25 not in item and  26  not in item]

print('number of (S/G2,S/G2,S/G2):\n')
print(len(output2))
l = [x for x in mylist if x in output2]

#with open('/Users/farzaneh/Desktop/New_file/SG2_SG2_SG2.txt', 'w') as outfile:
#    outfile.write(str(l))

#---------------------------------
# for G1,G1,S

output3 = [item for item in mylist if 0 not in item and 1 not in item and 3 not in item 
          and 7 not in item and 10 not in item and 13 not in item 
          and 16 not in item and 19 not in item and 20 not in item and 23 not in item and 26  not in item]

l = [x for x in output3 if (x[0] in S and x[1] in G1 and x[2] in G1) or
      (x[1] in S and x[0] in G1 and x[2] in G1) or
      (x[2] in S and x[1] in G1 and x[0] in G1)]

#print(output1)
print('number of (G1,G1,S):\n')
print(len(l))

#with open('/Users/farzaneh/Desktop/New_file/G1_G1_S.txt', 'w') as outfile:
#    outfile.write(str(l))
#---------------------------------

# for G1,G1,SG/2

output4 = [item for item in mylist if 0 not in item and 5 not in item and 6 not in item
          and 7 not in item and 10 not in item and 14 not in item
          and 19 not in item and 20 not in item and 26  not in item]


l = [x for x in output4 if (x[0] in SG2 and x[1] in G1 and x[2] in G1) or
      (x[1] in SG2 and x[0] in G1 and x[2] in G1) or
      (x[2] in SG2 and x[1] in G1 and x[0] in G1)]

print('number of (G1,G1,S/G2):\n')
print(len(l))


#with open('/Users/farzaneh/Desktop/New_file/G1_G1_SG2.txt', 'w') as outfile:
#    outfile.write(str(l))
#---------------------------------

# for S,S,G1

output5 = [item for item in mylist if 0 not in item and 1 not in item and 3 not in item 
          and 7 not in item and 10 not in item and 13 not in item 
          and 16 not in item and 19 not in item and 20 not in item and 23 not in item and 26  not in item]

l = [x for x in output5 if (x[0] in G1 and x[1] in S and x[2] in S) or
      (x[1] in G1 and x[0] in S and x[2] in S) or
      (x[2] in G1 and x[1] in S and x[0] in S)]

print('number of (S,S,G1):\n')
print(len(l))
      
#with open('/Users/farzaneh/Desktop/New_file/S_S_G1.txt', 'w') as outfile:
#    outfile.write(str(l))
#---------------------------------

# for S,S,S/G2

output6 = [item for item in mylist if 0 not in item and 2 not in item and 4 not in item and 7 not in item and 8 not in item
           and 9 not in item and 10 not in item and 11 not in item and 12 not in item and 15 not in item
           and 17 not in item and 18 not in item and 19 not in item and 20 not in item and 21 not in item and 22 not in item and
           24 not in item and 25 not in item and  26  not in item]
      
l = [x for x in output6 if (x[0] in SG2 and x[1] in S and x[2] in S) or
      (x[1] in SG2 and x[0] in S and x[2] in S) or
      (x[2] in SG2 and x[1] in S and x[0] in S)]
      
print('number of (S,S,S/G2):\n')
print(len(l))
      
#with open('/Users/farzaneh/Desktop/New_file/S_S_SG2.txt', 'w') as outfile:
#    outfile.write(str(l))
    
#---------------------------------

#for S/G2,S/G2,S

output7 = [item for item in mylist if 0 not in item and 2 not in item and 4 not in item and 7 not in item and 8 not in item
           and 9 not in item and 10 not in item and 11 not in item and 12 not in item and 15 not in item
           and 17 not in item and 18 not in item and 19 not in item and 20 not in item and 21 not in item and 22 not in item and
           24 not in item and 25 not in item and  26  not in item]

l = [x for x in output7 if (x[0] in S and x[1] in SG2 and x[2] in SG2) or
      (x[1] in S and x[0] in SG2 and x[2] in SG2) or
      (x[2] in S and x[1] in SG2 and x[0] in SG2)]
      
print('number of (S/G2,S/G2,S):\n')
print(len(l))

#with open('/Users/farzaneh/Desktop/New_file/SG2_SG2_S.txt', 'w') as outfile:
#    outfile.write(str(l))

#---------------------------------

# for S/G2,S/G2,G1

output8 = [item for item in mylist if 0 not in item and 5 not in item and 6 not in item
          and 7 not in item and 10 not in item and 14 not in item
          and 19 not in item and 20 not in item and 26  not in item]

l = [x for x in output8 if (x[0] in G1 and x[1] in SG2 and x[2] in SG2) or
      (x[1] in G1 and x[0] in SG2 and x[2] in SG2) or
      (x[2] in G1 and x[1] in SG2 and x[0] in SG2)]

print('number of (S/G2,S/G2,G1):\n')
print(len(l))


#with open('/Users/farzaneh/Desktop/New_file/SG2_SG2_G1.txt', 'w') as outfile:
 #   outfile.write(str(l))

#---------------------------------

# for S/G2,S,G1

output9 = [item for item in mylist if 0 not in item 
          and 7 not in item and 10 not in item and 19 not in item and 20 not in item and 26  not in item]

l = [x for x in output9 if (x[0] in G1 and x[1] in S and x[2] in SG2) or
      (x[1] in S and x[0] in SG2 and x[2] in G1) or
      (x[2] in SG2 and x[1] in G1 and x[0] in S)]
      
print('number of (S/G2,S,G1):\n')
print(len(l))


#with open('/Users/farzaneh/Desktop/New_file/SG2_S_G1.txt', 'w') as outfile:
#    outfile.write(str(l))

#---------------------------------

# for G1,G1,M/G1

output10 = [item for item in mylist if 0 not in item and 1 not in item and 3 not in item 
          and 5 not in item and 6 not in item and 13 not in item 
          and 16 not in item and 19 not in item and 20 not in item and 23 not in item and 14  not in item]

l = [x for x in output10 if (x[0] in MG1 and x[1] in G1 and x[2] in G1) or
      (x[1] in MG1 and x[0] in G1 and x[2] in G1) or
      (x[2] in MG1 and x[1] in G1 and x[0] in G1)]

print('number of (G1,G1,M/G1):\n')
print(len(l))


#with open('/Users/farzaneh/Desktop/New_file/allG1_G1_MG1.txt', 'w') as outfile:
#    outfile.write(str(l))
