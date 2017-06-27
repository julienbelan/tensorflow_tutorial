
# coding: utf-8

# # Theory Basics

# Linear Algebra are a math branch both difficult and diverse. In the following tutorial we'll try to get a hold on the some useful parts of the theory. For a more formal and complete introduction, please head to [Khan Academy](https://www.khanacademy.org/math/linear-algebra).
# 
# Linear Algebra is the science of objects collections and how they interact with one another. We will present these objects by making parrallels with movies.

# Here a black and white photo of someone golfing its way out of of a sand pit.

# In[54]:

from PIL import Image
golf = Image.open('images/GOLF.png')
golf.show()


# In[2]:

# Lets turn this photo into a workable thing.
# Don't pay attention to the code right now, I'll introduce numpy later on.
import numpy as np
ar_golf = np.asarray(golf)
# See the golf photo is 251 rows by 450 columns
np.shape(ar_golf)


# # Scalars
# 
# Scalars are **plain old number**.
# 
# In the movie alegory view, scalar are pixel colour. So for example if we take the 1st pixel value out of this photo:

# In[3]:

print(ar_golf[0,0].astype(int))


# 0 value means that the 1st pixel of the photo is black.
# 
# **Remember scalars are just plain old number so:**
# 
# i.e. -10
# 
# i.e. 4.5
# 
# i.e. 92.35
# 
# are all scalars.
# 
# Note that scalars don't have any directions, this may not make sense right now but it'll come handy later on.

# # Dimensions
# 
# The concept of dimension is fairly straight forward:
# - A point is 0D
# - A line is 1D
# - A plane is 2D
# - A solid is 3D
# 
# In a more abstract sense, almost anything can be thought as a dimension and it is fairly easy to determine contextually what could be thought as one:
# 
# i.e. The time dimension vs the infant mortality dimension
# 
# We often ear the concept of **direction** when doing linear algebra, direction is the dimension inherent to an object let aside the dimension in which it is sitting. 
# 
# In layman's terms, a perfectly flat piece of paper sitting on your table has 2 directions in a 3D world. Fold this paper into a plane you get a 3 directions object in a 3D world. The terms are often used interchangeably and you must rely on context to uncover their sense.

# # Vectors
# 
# Vectors are lists of scalars. They can used in many ways but fondamentaly they're only a **1D collection object**.
# 
# - i.e. **[1,2,4,6,2,1] is a vector**
# 
# - i.e. They can be used as force arrows in Physics [3,1] would represent a force going at 30 degrees and having a force of:

# In[7]:

from math import sqrt
force = sqrt(3**2 + 1**2)
print(force)


# In the movie alegory view, vectors are each horizontal lines in a picture going from left to right. So for example if we look at the first row of this picture.

# In[19]:

line = golf.crop((0,0,400,1))
print(line)


# Lets look at what it is looking like in a mathy way.

# In[22]:

# Don't pay attention to the code, we'll introduce numpy in a minute, just bear with me for now
import numpy as np
print(np.asarray(line).astype('int')[0])


# **Remember vectors are just lists of scalars so:**
# 
# i.e. [1, 2, 4, 3]
# 
# i.e. [6., 1., 0.0]
# 
# i.e. [3.2, 1., 0.1, 2.33]
# 
# are all vectors.
# 
# Note that vectors have 1 directions.

# Since vectors will be widely used it worth going a little deeper into understanding them. Vectors are traditionally described as arrows centered at the origin of a cartesian plan with both a length and a direction.

# In[23]:

# Here's the vector [4, 9] in 2D plan
Image.open('images/VECTOR2D.png').show()


# In[24]:

# Here's the vector [4, 9, 2] in the 3D plan
Image.open('images/VECTOR3D.png').show()


# In the picture alegory, each arrows are centered at the start of its row and are of length 400. 

# In[25]:

# Here's an arrow version of the picture vector. Note that not all the vector are representated because if they did
# the entire image would be red
Image.open('images/ARROWS.png').show()


# # Matrices
# Matrices are lists of vectors. They can used in many ways but fondamentaly they're only a 2D collection objects.
# 
# In the picture allegory, the image itself is a matrix. So if we look at the image again

# In[26]:

golf.show()


# Lets look at it in a more mathy way.

# In[27]:

print(np.asarray(golf).astype('int'))


# **Remember matrix are lists of vectors which are themselves lists of scalars so:**
# 
# i.e. [ [3, 4], [5, 6] ]
# 
# i.e. [ [2.2, .1, 3.], [5., 5.2, 7.]]
# 
# i.e. [ [1, 2], [3, 4], [6, 7], [0, 1]]
# 
# are all matrices.
# 
# Note that vectors have 2 directions.

# # Tensors
# Lets recap:
# 
# - Scalar have no direction: let's say they're tensor of rank 0
# - Vector have 1 direction: let's say they're tensor of rank 1
# - 2D Matrices have 2 direction: let's say they're tensor of rank 2
# 
# **Tensors are the general class englobing all the objects.**
# 
# What do you think would be a tensor of rank 3? A tensor of rank 4? Take 2 minutes to think about it.

# **SPOILER ALERT THINK ABOUT IT BEFORE LOOKING UP THE ANSWER**
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# A movie would be a tensor of rank 3. Movie have 3 directions: length, width and time. A movie is, after all, a list of pictures.

# In[28]:

# from IPython.display import HTML
# HTML("""
# <video width="320" height="240" controls>
#   <source src="images/golf.mp4" type="video/mp4">
# </video>
# """)


# And here's a tensor of rank 4

# In[29]:

Image.open('images/GOLFIT.jpg').show()


# After all this is a **list of films**
# 
# Here's another tensor example: 
# - letters = scalar
# - word = vector
# - sentence = matrix
# - chapter = tensor rank 3
# - book = tensor rank 4

# # NumPy
# NumPy is a python library that handle the creation, the transformation and the showing of the linear algebra objects.

# In[ ]:

# Like any python library numpy first need to be import, here we abbreviated the name numpy into np.
import numpy as np


# In NumPy eyes: 
# - tensors = ndarray
# 
# You can deduce that ndarray englobe matrices, vectors and floats/integers.

# In[30]:

# First lets see how to create a vector with numpy
your_list = input('give me a list of numbers like [4,2,1]: ')

# Transform the list into a numpy array
vector = np.array(your_list)

# Show it
print(vector)


# In[ ]:

# Then lets create a matrix with numpy
your_list_of_lists = input('give me a matrix like [[4,2,1], [3,1,9]]: ')

# Transform the matrix into a numpy array
matrix = np.array(your_list_of_lists)

# Show it
print(matrix)


# In[ ]:

# Then lets create a tensor with numpy
your_list_of_lists_of_lists = input('give me a tensor like [[[4,2,1], [3,1,9]],[[6,2,1], [1,1,0]]]: ')

# Transform the tensor into a numpy array
tensor = np.array(your_list_of_lists_of_lists)

# Show it
print(tensor)


# In[ ]:

# Lets check the types of our objects now 
print(type(vector))
print(type(matrix))
print(type(tensor))


# Now its a little bit confusing but all numpy is saying is that all these objects are in the tensor family which is true.
# 
# There's a matrix classes in numpy, but they're not that useful and only allow you to write code in a more mathy vocabulary. We'll be mainly using the ndarray general tensor.
# 
# Read this Stack Overflow [discussion](https://stackoverflow.com/questions/4151128/what-are-the-differences-between-numpy-arrays-and-matrices-which-one-should-i-u) to learn more on the difference between ndarray and numpy matrices 

# In[31]:

# Here's how to write matrix in a more mathy vocabulary
std_matrix = np.matrix([[4,2,1], [3,1,9]])

# Show it
print(std_matrix)


# In[32]:

# See numpy manage the matrix class differently but allow no new functionality, which makes it useless.
type(std_matrix)


# # Access Elements in NumPy arrays
# If you are familiar with Python lists you can skip this part for there's nothing new here.
# 
# Elements are the different subgroup in a tensor.
# - i.e. if you have a matrix, there a subgroup of vectors, each one having a subgroup of scalar

# In[33]:

#Here's some array
a = np.array([43, 80, 31, 35, 76, 12])

#To access lets say the first element you have to call it by its index
print(a[0]) #In Python the first indeces are alway 0s not 1s


# In[34]:

#To access the 3rd element
print(a[2])


# In[35]:

#To access the last element
print(a[-1]) #Note that when beginning from the end the index begin at -1 not -0 which makes sense since -0 is 0


# In[36]:

#To access the second to last element
print(a[-2])


# In[37]:

#From the starting element to the 3rd
#The : convention is called the "wild card syntax"
print(a[:3]) #Note that the index 3 element is not included, it is the third element (index 2) that is which is a bit confusing


# In[38]:

#From the 4th element to the end
print(a[3:])


# In[39]:

#From the 2nd element to the 4th
print(a[2:5])


# In[40]:

#Lets define a matrix
some_matrix = np.array([[1,2,4],[6,7,8]])


# In[41]:

#To access the 1st upleft element
print(some_matrix[0,0])


# In[42]:

#To access the 1nd row 3rd column element
print(some_matrix[0,2])


# In[43]:

#To access all the 1st direction and only the 1st element of the second
print(some_matrix[:,0])


# When facing a tensor of rank superior to 1, that is an object bigger than a vector, each new ',' delimit the nested direction.

# In[44]:

rank_three_tensor = np.array([ [ [12, 15, 23], [5, 0, 2] ], [ [30, 2, 1], [5, 3, 2] ], 
                              [ [11, 45, 54], [8, 12, 33] ], [ [23, 11, 6], [48, 2, 21] ] ])

# Look at the shape which are the size of each subgroup
print(np.shape(rank_three_tensor))


# In[ ]:

#To access the very first upleft element
print(rank_three_tensor[0,0,0])


# In[ ]:

#To access all the 1st direction, the 2 first element of the 2nd and only the last of the 3rd
print(rank_three_tensor[:, :2, -1])


# Hope you get it by now! I duly recommand to go and try to pull some elements out of this rank_three_tensor thing just to get a hold on the process.

# You may ask yourself why use NumPy, what's the advantages versus plain old list of lists of lists ... The reason why everybody use NumPy is that it allows you to vectorize the operation on its objects. 
# 
# Vectorizing in Python is the action of doing batch operation like when you have a column in Excel tables and you wish to apply an operation to all the cells in that column. You can do it one by one or batch operate on it with a formula.
# 
# NumPy allow that kind of behavior if you'd stick to lists, you'd be forced to go through all element 1-by-1.

# # Standard Linear Operations in NumPy
# 
# *Numpy offers lots of standard linear algebra operations, I'll go through some of the most useful but you can go to their website and read the linear algebra [documentation](https://docs.scipy.org/doc/numpy-1.12.0/reference/routines.linalg.html).*

# # Dot Product
# The dot product concerns vectors. It is achieve by following the present formula:
# 
# [a, b, c] · [d, e, f] = ad + be + cf
# 
# i.e. [1, 3, 5] · [4, 6, 1] = (1)(4) + (3)(6) + (5)(1) = 27

# In[ ]:

# numpy take care of that pretty easily
a = np.array([1, 3, 5])
b = np.array([4, 6, 1])

print(np.dot(a, b))


# Let's look at an example:
# 
# I'm doing a recipe for which I need the following ingredients:
# - 1 pack of pasta
# - 3 cans of cherry tomatoes
# - 2 can of sardines
# - 4 can of capers
# 
# I know that:
# - 1 pasta pack = 2US
# - 1 cherry tomatoes can = 2US
# - 1 sardines can = 1.5US
# - 1 capers can = 1US

# In[ ]:

quantity = np.array(input('give me the quantity vector: '))
price = np.array(input('give me the price vector: '))


# In[ ]:

# Give the numpy formula to calculate the overall price of the recipe.
price = 0 # replace the 0 by your code
print(price)


# In[ ]:

# SPOILER ALERT JUST RUN THIS CELL




















if price == np.dot([1,3,2,4], [2,2,1.5,1]):
    print('Bravo good job!')
else:
    print('Wrong, code should be '+ 'np.dot([1,3,2,4], [2,2,1.5,1])')


# # Cross product
# The cross product concerns vectors. It can only be done in a 3D environment. It is achieve by following the present formula
# 
# [a, b, c] x [d, e, f] = [bf-ce, -(af-cd), ae-bd]
# 
# i.e. [4, 2, 1] x [5, 6, 3] = [(2)(6)-(2)(3)=6, -((4)(3)-(1)(5))=-7, (4)(6)-(2)(5)=14]

# In[47]:

# Numpy take care of that very easily
a = np.array([4, 2, 1])
b = np.array([5, 6, 3])

print(np.cross(a,b))


# Given two vectors going in two differents directions, the cross product defines a vector of length equal to the area in red and of direction perpendicular to the two starting vectors.

# In[48]:

Image.open('images/CROSSY.png').show()


# # Matrix Multiplication
# Matrice Multiplication seems odd at first so lets begin with examples. 
# 
# I'm motorcycles fabricant selling three different model of motorcycles:
# - scooter
# - sport
# - chopper
# 
# || scooter   | sport | chopper  |
# |--|------|------|------|
# |price| 4000 | 25000  | 20000|
# 

# Here's how many I've sold this week:
# 
# |  | Mon   | Tue  | Wed  | Thu  |
# |------|------|------|------|
# |   scooter  | 5 |  0 | 0  | 2|
# |   sport | 2 |  8 | 1  | 2|
# |  chopper| 1 | 14  | 0  | 1|

# To know how much am I going to make everyday all I need to do is to multiply the price of each motorcycle to the number sold that day. It is the same thing as doing a dot product between the price vector and the sold matrix for each column.
# 
# 
# | Mon   | Tue  | Wed  | Thu  |
# |------|------|------|------|
# |5x4000+2x25000+1x20000=90000|0x4000+8x25000+14x20000=480000|0x4000+1x25000+0x20000=25000|2x4000+2x25000+1x20000=78000|
# 

# In[49]:

# Here the numpy implementation
a = np.array([4000, 25000, 20000])
b = np.array([[5, 0, 0, 2], [2, 8, 1, 2], [1, 14, 0, 1]])
print(np.matmul(a, b))


# One very important thing with matrix multiplication is that in order to take place the number of columns in the 1st matrix must be equal to the number of rows in the second matrix. 
# 
# It is very to see here because how could I have a different inventory in the price matrix than in the sold matrix.
# 
# An important consequence is that the resulting matrix will have the same number of rows as the 1st and the same number of columns as the 2nd.

# In[50]:

# Here to see all this stuff in a glance
Image.open('images/MATMULDR.png').show()


# Here's a more complicated example:
# 
# I'm need to encript credit card number in a very secure way while keeping its original format.
# 
# Here's some arbitrary card numbers: 468274839576. You can see that credit cards usually hold 12 numbers and therefore can be broken into 4x3 matrices.

# |  |  |  |
# | - | - | - |
# | 4 | 6 | 8 | 2 |
# | 7 | 4 | 8 | 3 |
# | 9 | 5 | 7 | 6 |

# Lets produce some arbitrary key matrix so we can encript the card sequence. Since I want to keep the 3x4 matrix format for the resulting matrix, I need to multiply the card matrix by a 4x4 key matrix
# 
# |  |  |  |  |
# | - | - | - | - |
# | 5 | 2 | 4 | 5 |
# | 0 | 3 | 4 | 6 |
# | 5 | 1 | 8 | 0 |
# | 9 | 1 | 2 | 2 |

# Lets multiply those two together by hand first:
# 
# let the resulting matrix be:
# 
# |  |  |  |
# | - | - | - |
# | a | b | c | d |
# | e | f | g | h |
# | i | j | k | l |
# 
# a = [4, 6, 8, 2] · [5, 0, 5, 9] = (4)(5)+(6)(0)+(8)(5)+(2)(9) = 78
# b = [4, 6, 8, 2] · [2, 3, 1, 1] = 36
# c = [4, 6, 8, 2] · [4, 4, 8, 2] = 108
# d = [4, 6, 8, 2] · [5, 6, 0, 2] = 60
# 
# e = [7, 4, 8, 3] · [5, 0, 5, 9] = 102
# f = [7, 4, 8, 3] · [2, 3, 1, 1] = 37
# g = [7, 4, 8, 3] · [4, 4, 8, 2] = 114
# h = [7, 4, 8, 3] · [5, 6, 0, 2] = 65
# 
# i = [9, 5, 7, 6] · [5, 0, 5, 9] = 134
# j = [9, 5, 7, 6] · [2, 3, 1, 1] = 46

# In[51]:

k = input('give the result of the k component: ')
l = input('give the result of the l component: ')


# In[53]:

# SPOILER ALERT JUST RUN THIS CELL















if k==124:
    print('Bravo!, good job, k=124 indeed!')
else:
    print('Wrong, k is given by [9, 5, 7, 6] · [4, 4, 8, 2]')
    
if l==87:
    print('Bravo!, good job, l=87 indeed!')
else:
    print('Wrong, l is given by [9, 5, 7, 6] · [5, 6, 0, 2]')


# In[ ]:

# Lets define the two matrix
card = np.array(input('give the matrix representation of the card: '))
key = np.array(input('give the matrix representation of the key: '))


# In[ ]:

# Give the numpy formula to produce the encrypted card.
encrypted = 0 # replace the 0 by your code
print(encrypted)


# # Sudoku Project
# Lets build a simple sudoku game and verifier using the numpy library
# 
# **The Sudoku**
# - is composed of a 9x9 grid subdivised into 3x3 blocks
# - each row has to countain all the number from 1 to 9
# - each column has to countain all the number from 1 to 9
# - each block has to countain all the number from 1 to 9

# In[ ]:

#Here's a finished sudoku
Image.open('images/SUDOKU.png').show()


# # Instructions
# Use the [numpy documentation](https://docs.scipy.org/doc/numpy/reference/) and standard Python builtin library to compose a sudoku game.
# 
# This project is hard, in no way obligatory and should be attempted only by persons really interested in learning more about the numpy library.

# In[ ]:

# SPOILER CHECK THE FOLLOWING SOLUTION ONLY AFTER HAVING TRIED FOR YOURSELF

























# Here's a naive solution using a minimal thinking and wasted loops dont except much out of it and keep the % low
# if not it will take forever

# Create a sudoku grid fill with 0
sudoku = np.zeros((9, 3, 3))

# Ask the user how many cell he/she wants to be prefilled
non_zero = round(input('what percentage of the sudoku should be prefilled: ')*.81)

# import the random library
import random as rd

# while you havent filled that many cells
while np.count_nonzero(sudoku) <= non_zero:
    
    # pick a random block
    block = rd.randint(0,8)
    # pick a random row
    x = rd.randint(0,2)
    # pick a random column
    y = rd.randint(0,2)
    # pick a random number
    number = rd.randint(1,9)
    
    # if the position is already occupied
    if sudoku[block, x, y] != 0:
        # return to the beginning of the while loop
        continue
    else:
        # if the number is in the block, in the row or the column
        if number in sudoku[block, :, :] or number in sudoku[:, x, :] or number in sudoku[:, :, y]:
            # return to the beginning of the while loop
            continue
        else:
            # if everything is ok write the number in place
            sudoku[block, x, y] = number

# show that grid
print(sudoku)


# In[ ]:

#Here's a more complete solution 

import numpy as np
sudoku = np.zeros((9, 3, 3))
non_zero = round(input('what percentage of the sudoku should be prefilled: ') * .81)

from random import randint, choice

while np.count_nonzero(sudoku) <= non_zero:

    block_num = randint(0, 8)

    block = sudoku[block_num]

    non_avail_num_blk = np.unique(block)

    avail_row, avail_col = np.where(block == 0)
    if len(avail_row) == 0 or len(avail_col) == 0:
        continue

    my_row, my_col = choice(zip(avail_row, avail_col))

    non_avail_num_row = np.unique(sudoku[:, my_row, :])
    non_avail_num_col = np.unique(sudoku[:, :, my_col])
    non_avail_num = np.concatenate((non_avail_num_blk, non_avail_num_row, non_avail_num_col))

    avail_num = []
    for number in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if number not in non_avail_num:
            avail_num.append(number)

    if len(avail_num) == 0:
        continue

    my_num = choice(avail_num)

    sudoku[block_num, my_row, my_col] = my_num
        
print(sudoku)


# In[ ]:

#A even more complete solution would use a recursive function, give it a go if you want!

