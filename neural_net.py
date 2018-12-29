# import sys so we can take commandline arguments
import sys
import math
import random

import numpy as np

random.seed(777)

print("\n")
print("Please remember to fix your data before running the code\n")

# this function ask the user for info
def io(q1, q2):
	name = input(q1)
	# get the name of the file
	while True:
		try:
			if name == "Q" or name == "q":
				sys.exit("Quit by command")
			output = open(name, "r")
			return (name,output)
		except NameError:
			name = input(q2)
		except FileNotFoundError:
			name = input(q2)


# in this file, we are going to use sigmoid as the activation funciton

# get the name of the training file
t1 = io("What is the name of the training file? ","Please provide a valid training file (csv) or press Q to quit: ")
train_name, train_attr = t1[0], t1[1]

# get the name of the devloping file
d1 = io("What is the name of the developing file? ","Please provide a valid developing file (csv) or press Q to quit: ")
dev_name, dev_attr = d1[0], d1[1]

# get the name of the key file
k1 = io("What is the name of the key file? ","Please provide a valid key file (txt) or press Q to quit: ")
key_name, train_lab = k1[0], k1[1]

# get the name of the dev_key file
k2 = input("What is the name of the developing key file (press p if there isn't one)? ")
while True:
	try:
		if k2 == "p":
			kk = []
		kk = open(k2, "r")
		break
	except:
		k2 = input("Please provide a valid key file (txt): ")

# get the desired number of neurons
num_neurons = input("Please enter the number of neurons in the hidden layer: ")

while True:
	try:
		num_neurons = int(num_neurons)
		if num_neurons > 0:
			break
		else:
			sys.exit("This shouldn't happen. There a bug")
	except:
		num_neurons = input("Please enter a positive integer: ")

# get the desired number of epochs
epoch = input("Please enter the desired number of epoch: ")

while True:
	try:
		epoch = int(epoch)
		if epoch > 0:
			break
		else:
			sys.exit("This shouldn't happen. There a bug")
	except:
		epoch = input("Please enter a positive integer: ")

# get desired learning rate
learning_rate = input("Please enter a desired learning rate: ")

while True:
	try:
		learning_rate = float(learning_rate)
		if learning_rate > 0:
			break
		else:
			sys.exit("This shouldn't happen. There a bug")
	except:
		learning_rate = input("Please enter a learning rate: ")

a = 3

random.seed(777)

weight_input_layer = [] # weight from input to hidden layer (2d list)
weight_layer_output = [] # weight from hidden layer to output (1d list)

bh = []
bout = [(random.random() - 0.5)*a]

# parsing the data
body = []
dev_body = []
title = []
key = []
point1 = 0
point2 = 0
point3 = 0

for line in train_attr:
  new_line = (line[:-1]).lower()
  new_line = new_line.split(",")
  if point1 != 0:
    body.append(new_line)
  else:
    title = (new_line)
  point1 = 1

# parse the developing data set
for line in dev_attr:
  new_line = (line[:-1]).lower()
  new_line = new_line.split(",")
  if point3 != 0:
    dev_body.append(new_line)
  point3 = 1


# we need to fix the data in the body and dev body
# please change this fix_data before runing ur code
def fix_data(df):
  if train_name == "music_train.csv":
    for i in range(len(df)):
      for j in range(len(df[0])):
        cur = df[i][j]
        if j == 0:
          df[i][j] = (int(cur[:-2])-1900)/100
        elif j == 1:
          df[i][j] = float(cur)/7
        elif j == 2:
          if cur == "yes": 
            df[i][j] = 1
          else:
            df[i][j] = 0
        else:
          if cur == "yes":
            df[i][j] = 1
          else:
            df[i][j] = 0
    return df
  else:
    for i in range(len(df)):
      for j in range(len(df[0])):
        df[i][j] = float(df[i][j])/100
    return df

body = np.array(fix_data(body))
dev_body = fix_data(dev_body)

# convert all the keys to float if possible
# this need to be fixed to
for line in train_lab:
  new = line[:-1]
  if new == "yes":
    key.append(1)
  elif new == "no":
    key.append(0)
  else:
    key.append(float(new)/100)

num_attr = len(title)

# initialize all the weights

# weights from input to hidden layer
for i in range(num_attr):
	weight = []
	for j in range(num_neurons):
		# this would give us numbers btw -0.1 and 0.1
		weight += [(random.random() - 0.5)*a] 
	weight_input_layer.append(weight)

# weights from hidder layer to output
for i in range(num_neurons):
	weight_layer_output += [(random.random() - 0.5)*a]
	bh += [(random.random() - 0.5)*a]

weight_input_layer = np.array(weight_input_layer)
weight_layer_output = np.array(weight_layer_output)

def cal_sigmoid (x):
	return 1/(1 + np.exp(-x))

# gradient descent
# termination condition
for i in range(epoch):
	# follow the slides on page 91

	# gradient descent
	# forward
	net1 = np.dot(body,weight_input_layer)
	net = net1 + bh
	o1_2_3 = cal_sigmoid(net)
	o_input_temp = np.dot(o1_2_3, weight_layer_output)
	o_input = o_input_temp + bout
	o = cal_sigmoid(o_input)

	# calculate the loss
	total = sum((key - o)**2)
	loss = (total/(2))

	# for each output unit k
	dk = o*(1-o)*(key-o)

	# get the sig and dk mulitplication
	w_sig = []
	for i in range(len(dk)):
		w_sig.append(weight_layer_output*dk[i])

	# backward
	dk = o*(1-o)*(key-o)
	dh = w_sig*o1_2_3*(1-o1_2_3)
	weight_layer_output += o1_2_3.T.dot(dk)*learning_rate
	weight_input_layer += body.T.dot(dh)*learning_rate
	bout += np.sum(dk)*learning_rate
	bh += np.sum(dh)*learning_rate

	print(loss)

print("GRADIENT DESCENT TRAINING COMPLETED!")

# this final part tells us the prediction of the developing file
net1 = np.dot(dev_body,weight_input_layer)
net = net1 + bh
o1_2_3 = cal_sigmoid(net)
o_input_temp = np.dot(o1_2_3, weight_layer_output)
o_input = o_input_temp + bout
o = cal_sigmoid(o_input)

ans = []

# need to change this part base on the file
for i in range(len(o)):
	if train_name == "music_train.csv":
		right = round(o[i],0)
	else:
		right = round(o[i]*100,0)
	print(right)
	ans.append(right)

print("THESE ARE THE PREDICTIONS FOR THE DEVELOPING FILE!")

# tells us how accurate the predictions are 

dkey = []
if kk != []:
	for line in kk:
		new = line[:-1]
		if new == "yes":
			dkey.append(1)
		elif new == "no":
			dkey.append(0)
		else:
			dkey.append(float(new))

	count = 0
	total_count = len(dkey)

	for i in range(len(dkey)):
		if train_name == "music_train.csv":
			if dkey[i] == ans[i]:
				count += 1
		# change this part
		else:
			delta = 2
			if dkey[i]+delta >= ans[i] and dkey[i]-delta <= ans[i]:
				count += 1

	# score for this
	print(count/total_count*100)




























