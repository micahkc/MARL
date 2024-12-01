dict1 = {1:[5], 2:[5], 3:[5]}
dict2 = {1:10, 2:10, 3:10}

for key in dict1:
    dict1[key].append(dict2[key])
print(dict1)