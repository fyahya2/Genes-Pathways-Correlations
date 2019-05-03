
mylist = []
N = [0, 1, 7, 9, 10, 12, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26]
Up = [8,18, 19]
Down = [2, 3, 5, 6, 11, 13, 20]

with open('corrT134_all.txt', "r") as fp:
    for i in fp.readlines():
        tmp = i.split(" ")
        try:
            mylist.append((int(tmp[0]), int(tmp[1]), int(tmp[2])))
        except:pass

print(len(mylist))

# for None, None, None

output = [item for item in mylist if 2 not in item and 3 not in item and 5 not in item and 6 not in item
          and 8 not in item and 11 not in item and 13 not in item and 18 not in item
          and 19 not in item and 20 not in item]

#print(output)
print('number of (None,None,None):\n')
print(len(output))

l = [x for x in mylist if x in output]

#with open('/Users/farzaneh/Desktop/New_file/N_N_N.txt', 'w') as outfile:
#    outfile.write(str(l))
#---------------------------------

# for Up, Up, Up

output1 = [item for item in mylist if 0 not in item and 1 not in item and 2 not in item and 3 not in item and 4 not in item
          and 7 not in item and 5 not in item and 9 not in item and 10 not in item and 11 not in item and 12 not in item and 13 not in item and 15 not in item
          and 16 not in item and 17 not in item and 6 not in item and 14 not in item and 15 not in item and 20 not in item and 21 not in item and 22 not in item and
           23 not in item and 24 not in item and 25 not in item and 26  not in item]

#print(output1)
print('number of (Up,Up,Up):\n')
print(len(output1))

l = [x for x in mylist if x in output1]

#with open('/Users/farzaneh/Desktop/New_file/U_U_U.txt', 'w') as outfile:
#    outfile.write(str(l))
    
#---------------------------------

# for Down, Down, Down

output2 = [item for item in mylist if 0 not in item and 1 not in item and 4 not in item
           and 7 not in item and 8 not in item and 9 not in item and 10 not in item and 12 not in item and 14 not in item and 15 not in item
           and 16 not in item and 17 not in item and 18 not in item and 19 not in item and 21 not in item and 22 not in item
           and 23  not in item and 24 not in item and 25 not in item and  26  not in item]

print('number of (Down,Down,Down):\n')
print(len(output2))
l = [x for x in mylist if x in output2]

#with open('/Users/farzaneh/Desktop/New_file/D_D_D.txt', 'w') as outfile:
#    outfile.write(str(l))

#---------------------------------
# for Up, Up, None

output3 = [item for item in mylist if 2 not in item and 3 not in item 
          and 5 not in item and 6 not in item and 11 not in item 
          and 13 and 20 not in item]

l = [x for x in output3 if (x[0] in N and x[1] in Up and x[2] in Up) or
      (x[1] in N and x[0] in Up and x[2] in Up) or
      (x[2] in N and x[1] in Up and x[0] in Up)]

#print(output1)
print('number of (Up,Up,None):\n')
print(len(l))

#with open('/Users/farzaneh/Desktop/New_file/U_U_N.txt', 'w') as outfile:
#    outfile.write(str(l))
#---------------------------------

# for Up, Up, Down

output4 = [item for item in mylist if 0 not in item and 1 not in item and 4 not in item
          and 7 not in item and 9 not in item and 10 not in item and 12 not in item and 14 not in item
          and 15 not in item and 16 not in item and 17 not in item and 21 not in item and 22 not in item
          and 24 not in item and 23 not in item and 25 not in item and 26  not in item]


l = [x for x in output4 if (x[0] in Down and x[1] in Up and x[2] in Up) or
      (x[1] in Down and x[0] in Up and x[2] in Up) or
      (x[2] in Down and x[1] in Up and x[0] in Up)]

print('number of (Up,Up,Down):\n')
print(len(l))


#with open('/Users/farzaneh/Desktop/New_file/U_U_D.txt', 'w') as outfile:
#    outfile.write(str(l))
#---------------------------------

# for Down, Down, Up

output5 = [item for item in mylist if 0 not in item and 1 not in item and 4 not in item
          and 7 not in item and 9 not in item and 10 not in item and 12 not in item and 14 not in item
          and 15 not in item and 16 not in item and 17 not in item and 21 not in item and 22 not in item
          and 24 not in item and 23 not in item and 25 not in item and 26  not in item]

l = [x for x in output5 if (x[0] in Up and x[1] in Down and x[2] in Down) or
      (x[1] in Up and x[0] in Down and x[2] in Down) or
      (x[2] in Up and x[1] in Down and x[0] in Down)]

print('number of (Down,Down,Up):\n')
print(len(l))
      
#with open('/Users/farzaneh/Desktop/New_file/D_D_U.txt', 'w') as outfile:
#    outfile.write(str(l))
#---------------------------------

# for Down, Down, None

output6 = [item for item in mylist if 8 not in item and 18 not in item and 19 not in item]
      
l = [x for x in output6 if (x[0] in N and x[1] in Down and x[2] in Down) or
      (x[1] in N and x[0] in Down and x[2] in Down) or
      (x[2] in N and x[1] in Down and x[0] in Down)]
      
print('number of (Down,Down,None):\n')
print(len(l))
      
#with open('/Users/farzaneh/Desktop/New_file/D_D_N.txt', 'w') as outfile:
#    outfile.write(str(l))
    
#---------------------------------

#for None, None, Down

output7 = [item for item in mylist if 8 not in item and 18 not in item and 19 not in item]

l = [x for x in output7 if (x[0] in Down and x[1] in N and x[2] in N) or
      (x[1] in Down and x[0] in N and x[2] in N) or
      (x[2] in Down and x[1] in N and x[0] in N)]
      
print('number of (None,None,Down):\n')
print(len(l))

#with open('/Users/farzaneh/Desktop/New_file/N_N_D.txt', 'w') as outfile:
#    outfile.write(str(l))

#---------------------------------

# for None, None, Up

output8 = [item for item in mylist if 0 not in item and 5 not in item and 6 not in item
          and 7 not in item and 10 not in item and 14 not in item
          and 19 not in item and 20 not in item and 26  not in item]

l = [x for x in output8 if (x[0] in Up and x[1] in N and x[2] in N) or
      (x[1] in Up and x[0] in N and x[2] in N) or
      (x[2] in Up and x[1] in N and x[0] in N)]

print('number of (None,None,Up):\n')
print(len(l))


#with open('/Users/farzaneh/Desktop/New_file/N_N_U.txt', 'w') as outfile:
#    outfile.write(str(l))

#---------------------------------

# for None, Up, Down

l = [x for x in mylist if (x[0] in N and x[1] in Up and x[2] in Down) or
      (x[1] in Up and x[0] in Down and x[2] in N) or
      (x[2] in Down and x[1] in N and x[0] in Up)]
      
print('number of (None,Up,Down):\n')
print(len(l))


#with open('/Users/farzaneh/Desktop/New_file/N_U_D.txt', 'w') as outfile:
#    outfile.write(str(l))


