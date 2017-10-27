
# coding: utf-8

# # TD0 Python is back

# In[1]:


import random

random.randint(-10, 10)


# In[2]:


def print_matrix(m):
    for l in m:
        print(l)

m = [[0] * 3] * 3
print_matrix(m)
m[1][1] = 42
print_matrix(m)


# In[3]:


[ i for i in range(10) if i % 2 == 0 ]


# In[4]:


m = [ [0] * 3 for i in range(3) ]
print_matrix(m)
m[1][1] = 42
print_matrix(m)


# In[5]:


def buildlist(nb, val = None, alea = None):
    if val == None and alea:
        return [ random.randint(alea[0], alea[1]) for i in range(nb) ]
    return [ val ] * nb

def buildmatrix(line, col, val = None):
    return [ [ val ] * col for i in range(line) ]


# In[6]:


print(buildlist(10))
print(buildlist(10, 42))
print(buildlist(10, alea=(-256, 256)))
print(buildlist(10, val=0, alea=(-256,256)))


# In[7]:


def split_list(l):
    n = len(l)
    l1 = []
    for i in range(n // 2):
        l1.append(l[i])
    l2 = []
    for i in range(n // 2 + 1, n):
        l2.append(l[i])
    return l1, l[n // 2], l2

split_list([i for i in range(11)])


# In[8]:


l = [i for i in range(11)]

l[:len(l) // 2], l[len(l) // 2], l[len(l) // 2 + 1:]


# In[9]:


l[-2]


# In[10]:


l[2:6] = l[3:6]
l


# In[11]:


def binsearch(v, x):
    left, right = 0, len(v)
    while left < right:
        mid = left + (right - left) // 2
        if x == v[mid]:
            return mid
        if x < v[mid]:
            right = mid
        else:
            left = mid + 1
    return left

def binsearch_rec(v, x, left, right):
    if left >= right:
        return left
    mid = left + (right - left) // 2
    if x == v[mid]:
        return mid
    if x < v[mid]:
        return binsearch_rec(v, x, left, mid)
    return binsearch_rec(v, x, mid + 1, right)


# In[12]:


l = [ random.randint(-65536, 65536) for i in range(1000000) ]
l.sort()


# In[13]:


x = random.randint(-65536, 65536)
p = binsearch(l, x)
print(x, p, l[p])


# In[14]:


from algopy import bintree
import load_tree
import graphviz


# In[45]:


b = load_tree.load_bintree('files/bst.tree')
print('(15(8(1()())(12(10()())()))(28(20()(23()()))(42(35()())(66()()))))')
graphviz.Source(bintree.dot_simple(b))


# In[20]:


def searchBST(x, B):
    if B == None or x == B.key:
        return B
    else:
        if x < B.key:
            return searchBST(x, B.left)
        else:
            return searchBST(x, B.right)

def printBST(b):
    if b:
        print(">", b.key, "<", sep='')
        printBST(b.left)
        printBST(b.right)


n = searchBST(8, b)
print(n)


# In[32]:


def insertBST(x, B):
    if B == None:
        return bintree.BinTree(x, None, None)
    else:
        if x != B.key:
            if x < B.key:
                B.left = insertBST(x, B.left)
            else:
                B.right = insertBST(x, B.right)
        return B

b2 = insertBST(27, b)
graphviz.Source(bintree.dot_simple(b))


# In[33]:


def copy(B):
    if B == None:
        return None
    return bintree.BinTree(B.key, copy(B.left), copy(B.right))


# In[43]:


def empty_it(l):
    return l

def add42(l):
    l.append(42)


# In[44]:


l = [ i for i in range(5) ]
print(l)
empty_it(l)
print(l)
add42(l)
print(l)

