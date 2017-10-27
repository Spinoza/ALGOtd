
# coding: utf-8

# # Tutorial 1: General Trees

# In[1]:


from functools import reduce

class Tree:
    def __init__(self, key, children=[]):
        self.key = key
        self.children = children
    
    @property
    def nbchildren(self):
        return len(self.children)
    
    @property
    def size(self):
        return reduce(lambda accu, elt : accu + elt.size,
                      self.children, 1)
    
    @property
    def height(self):
        return 1 + reduce(lambda accu, elt : max(accu, elt.height),
                          self.children, -1)

t = Tree(42, [Tree(1), Tree(2)])
print(t.key, t.nbchildren)
print(t.size)

class TreeAsBin:
    def __init__(self, key=None, child=None, sibling=None):
        self.key = key
        self.child = child
        self.sibling = sibling
    
    @property
    def nbchildren(self):
        r = 0
        child = self.child
        while child:
            r += 1
            child = child.sibling
        return r
    
    @property
    def children(self):
        l = []
        child = self.child
        while child:
            l.append(child)
            child = child.sibling
        return l


# In[2]:


from algopy import tree, bintree, treeasbin
import graphviz
t = tree.loadtree('files/treetuto-1.tree')
graphviz.Source(tree.dot(t))


# In[3]:


def size(t):
    s = 1
    for child in t.children:
        s += size(child)
    return s

print(size(t))

def height(t):
    h = -1
    for child in t.children:
        h = max(h, height(child))
    return h + 1
print(height(t))


# In[4]:


def size_bin(t):
    s = 1
    child = t.child
    while child:
        s += size_bin(child)
        child = child.sibling
    return s

def size_bin2(t):
    if not t:
        return 0
    return 1 + size_bin2(t.child) + size_bin2(t.sibling)

tbin = treeasbin.tutoEx1()
print(size_bin(tbin), size_bin2(tbin))

def height_bin(t):
    h = -1
    child = t.child
    while child:
        h = max(h, height_bin(child))
        child = child.sibling
    return h + 1

def height_bin2(t):
    if not t:
        return -1
    return max(1 + height_bin2(t.child), height_bin2(t.sibling))

print(height_bin(tbin), height_bin2(tbin))


# In[5]:


def epl(t, d=0):
    if not t.children:
        return d
    r = 0
    for child in t.children:
        r += epl(child, d + 1)
    return r

print(epl(t))

def epl_bin(t, d=0):
    if not t.child:
        return d
    r = 0
    child = t.child
    while child:
        r += epl_bin(child, d + 1)
        child = child.sibling
    return r

print(epl_bin(tbin))

def epl_bin2(t, d=0):
    r = d
    if t.child:
        r = epl_bin2(t.child, d + 1)
    if t.sibling:
        r += epl_bin2(t.sibling, d)
    return r

print(epl_bin2(tbin))


# ## DFS principle:
# 
# DFS on a node T is recursive function:
# * if T is a leaf: perform leaf operation and leave
# * Otherwise:
#     * perform pre-order operation
#     * for each child of T:
#         * recursive call on the child
#         * do in-between operation
#     * perform post-order operation and leave
#     
# **Between or after**
# 
# Between: 1, 2, 3, 4
# 
# After:   1, 2, 3, 4,
#     

# In[6]:


graphviz.Source(tree.dot(t))


# In[7]:


15, 3, -6, 10, 8, 11, 0, 4, 2, 5, 9


# In[8]:


-6, 10, 3, 0, 4, 11, 2, 5, 8, 9, 15


# In[9]:


print('15 3 -6 | 10 3 | 8 11 0 | 4 11 | 2 | 5 8 | 9 15 ')
print()
graphviz.Source(tree.dot(t))


# In[10]:


def depth(t):
    # Pre-order op
    print(t.key, end=' ')
    if not t.children:
        # leaf case
        return
    depth(t.children[0])
    for child in t.children[1:]:
        print('|', end=' ') # between children
        depth(child)
    # post-order op
    print(t.key, end=' ')

print('15 3 -6 | 10 3 | 8 11 0 | 4 11 | 2 | 5 8 | 9 15 ')
depth(t)
print()

def depth_bin(t):
    print(t.key, end=' ')
    if not t.child:
        # leaf case
        return
    depth_bin(t.child)
    child = t.child.sibling
    while child:
        print('|', end=' ') # between children
        depth_bin(child)
        child = child.sibling
    # post-order op
    print(t.key, end=' ')

depth_bin(tbin)
print()

def depth_bin2(t):
    print(t.key, end=' ')
    if t.child:
        depth_bin2(t.child)
        print(t.key, end=' ')
    if t.sibling:
        print('|', end=' ')
        depth_bin2(t.sibling)

depth_bin2(tbin)
print()


# In[11]:


graphviz.Source(tree.dot(t))


# ## BFS
# 
# 
# 
# 15
# 3 8 9
# -6 10 11 2 5
# 0 4
# 
# 2.2 question 3

# In[12]:


from algopy.queue import Queue
q = Queue()
for i in range(10):
    q.enqueue(i)
while not q.isempty():
    print(q.dequeue())


# In[13]:


def width(t):
    w, wcur = 0, 0
    q = Queue()
    q.enqueue(t)
    q.enqueue(None)
    while not q.isempty():
        cur = q.dequeue()
        if not cur:
            w = max(w, wcur)
            wcur = 0
            if not q.isempty():
                q.enqueue(None)
        else:
            wcur += 1
            for child in cur.children:
                q.enqueue(child)
    return w

width(t)


# In[14]:


def width_bin(t):
    w, wcur = 0, 0
    q = Queue()
    q.enqueue(t)
    q.enqueue(None)
    while not q.isempty():
        cur = q.dequeue()
        if not cur:
            w = max(w, wcur)
            wcur = 0
            if not q.isempty():
                q.enqueue(None)
        else:
            wcur += 1
            child = cur.child
            while child:
                q.enqueue(child)
                child = child.sibling
    return w

width_bin(tbin)


# In[15]:




from collections import deque
def width2(t):
    w = 1
    curq, nextq = deque([t]), deque()
    while curq:
        cur = curq.popleft()
        for c in cur.children:
            nextq.append(c)
        if not curq:
            w = max(w, len(nextq))
            curq, nextq = nextq, curq
    return w


# ## 3.1 Equality

# In[16]:


def same(t, b):
    if t.key == b.key:
        bc = b.child
        for tc in t.children:
            if not bc or not same(tc, bc):
                return False
            bc = bc.sibling
        return bc == None
    return False

def same_bfs(t, b):
    qt = deque([t])
    qb = deque([b])
    while qt and qb:
        ct, cb = qt.popleft(), qb.popleft()
        if ct.key != cb.key:
            return False
        chb = cb.child
        for cht in ct.children:
            if not chb:
                return False
            qt.append(cht)
            qb.append(chb)
            chb = chb.sibling
        if chb:
            return False
    return not (qt or qb)

same(t, tbin)


# ## 3.2 Average arity

# In[17]:


def __avg_arity(t):
    nc, ni = t.nbchildren, (t.nbchildren > 0)
    for child in t.children:
        c, i = __avg_arity(child)
        nc, ni = nc + c, ni + i
    return nc, ni

def avg_arity(t):
    children, internal = __avg_arity(t)
    return children / internal if internal else 0




# In[18]:


t2 = tree.loadtree('files/treetuto-2.tree')
print(avg_arity(t2))
graphviz.Source(tree.dot(t2))


# In[19]:


def __avg_arity_bin(t):
    nc, ni = 0, (t.child != None)
    child = t.child
    while child:
        c, i = __avg_arity_bin(child)
        nc, ni = 1 + nc + c, ni + i
        child = child.sibling
    return nc, ni

def avg_arity_bin(t):
    children, internal = __avg_arity_bin(t)
    return children / internal if internal else 0


avg_arity_bin(tbin)


# In[20]:


def __avg_arity_bin2(t):
    if not t:
        return 0,0
    a, b = __avg_arity_bin2(t.child)
    c, d = __avg_arity_bin2(t.sibling)
    return a + c + 1, b + d + (t.child != None)

def __avg_arity_bin2(t):
    a, b = __avg_arity_bin2(t.child) if t.child else (0,-1)
    c, d = __avg_arity_bin2(t.sibling) if t.sibling else (0,0)
    return a + c + 1, b + d + 1

def avg_arity_bin2(t):
    size, internal = __avg_arity_bin2(t)
    return (size - 1) / internal if internal else 0

avg_arity_bin2(tbin)


# In[21]:


parents = [ 2, 2, 10, 6, 2, 10, 7, 8, 10, 2, -1, 6 ]
print(parents)
for i in range(len(parents)):
    print(parents[i], '->', i)


# In[22]:


t3 = tree.loadtree('files/treetuto-3.tree')
#graphviz.Source(tree.dot(t3))


# In[28]:


def __serialize(t, parents):
    for child in t.children:
        parents[child.key] = t.key
        __serialize(child, parents)

def serialze(t, size):
    parents = [-1] * size
    __serialize(t, parents)
    return parents

from algopy import  tree_to_bin
tb3 = tree_to_bin.tree2bin(t3)

serialze(t3, size(t3))



# In[30]:


def __serialize_bin(t, parents):
    if t.child:
        parents[t.child.key] = t.key
        __serialize_bin(t.child, parents)
    if t.sibling:
        parents[t.sibling.key] = parents[t.key]
        __serialize_bin(t.sibling, parents)

def __serialize_bin(t, parents, p=-1):
    parents[t.key] = p
    if t.child:
        __serialize_bin(t.child, parents, t.key)
    if t.sibling:
        __serialize_bin(t.sibling, parents, p)

def serialze_bin(t, size):
    parents = [-1] * size
    __serialize_bin(t, parents)
    return parents

serialze_bin(tb3, size_bin(tb3))


# In[32]:


def foo(l):
    l += [42, 27972]

def bar(l):
    l = []
    
l1 = [1,2,3]
l2 = [1,2,3]

print(l1, l2)

foo(l1)
bar(l2)

print(l1, l2)


# In[35]:


def __serialize_bin(t, parents, p=-1):
    if t.key >= len(parents):
        parents += [-1] * (t.key - len(parents) + 1)
    parents[t.key] = p
    if t.child:
        __serialize_bin(t.child, parents, t.key)
    if t.sibling:
        __serialize_bin(t.sibling, parents, p)

def serialze_bin(t):
    parents = [-1] * (t.key + 1)
    __serialize_bin(t, parents)
    return parents

serialze_bin(tb3)


# In[38]:


def serialize_bfs(t):
    q = Queue()
    q.enqueue(t)
    parents = [-1] * (t.key + 1)
    while not q.isempty():
        cur = q.dequeue()
        for child in cur.children:
            if child.key >= len(parents):
                parents += [-1] * (child.key - len(parents) + 1)
            parents[child.key] = cur.key
            q.enqueue(child)
    return parents

serialize_bfs(t3)


# In[47]:


# 3.4.1
def bin2tree(b):
    t = tree.Tree(b.key)
    child = b.child
    while child:
        t.children.append(bin2tree(child))
        child = child.sibling
    return t

def bin2tree2(b, p=None):
    t = tree.Tree(b.key)
    if p:
        p.children.append(t)
    if b.child:
        bin2tree2(b.child, t)
    if b.sibling:
        bin2tree2(b.sibling, p)
    return t

def bin2tree3(b, p=None):
    t = tree.Tree(b.key)
    if b.child:
        t.children.append(bin2tree3(b.child, t))
    if b.sibling:
        p.children.append(bin2tree3(b.sibling, p))
    t.children.reverse()
    return t


tbin = treeasbin.tutoEx1()
graphviz.Source(tree.dot(bin2tree3(tbin)))


# # m-way search tree
# # b tree
# # b+ tree

# In[53]:


def tree2bin(t):
    b = treeasbin.TreeAsBin(t.key)
    for i in range(t.nbchildren - 1, -1, -1):
        c = tree2bin(t.children[i])
        c.sibling = b.child
        b.child = c
    return b

graphviz.Source(tree.dot(bin2tree(tree2bin(t))))


# In[55]:


def __tree2bin2(p, pos=0):
    if pos >= p.nbchildren:
        return None
    b = treeasbin.TreeAsBin(p.children[pos].key)
    b.child = __tree2bin2(p.children[pos], 0)
    b.sibling = __tree2bin2(p, pos+1)
    return b

def tree2bin2(t):
    return treeasbin.TreeAsBin(t.key, child=__tree2bin2(t))

graphviz.Source(tree.dot(bin2tree(tree2bin2(t))))


# In[57]:


tlin = '(15(3(-6)(10))(8(11(0)(4))(2)(5))(9))'
tlin


# In[58]:


graphviz.Source(tree.dot(tree.loadtree("files/treetuto-4.tree")))


# In[60]:


def tree2list(t):
    s = '(' + str(t.key)
    for child in t.children:
        s += tree2list(child)
    return s + ')'

print(tree2list(t))
print(tlin)


# In[61]:


def bin2list(t):
    s = ''
    if t:
        s = '(' + str(t.key)
        s += bin2list(t.child)
        s += ')'
        s += bin2list(t.sibling)
    return s

print(bin2list(tbin))


# In[64]:


def list2tree(s, pos=0, type=int):
    if pos < len(s) and s[pos] == '(':
        pos += 1
        w = ''
        while s[pos] not in '()':
            w += s[pos]
            pos += 1
        t = tree.Tree(type(w))
        while s[pos] != ')':
            c, pos = list2tree(s, pos, type)
            t.children.append(c)
        pos += 1
        return t, pos
    return None

graphviz.Source(tree.dot(list2tree(tlin)[0]))


# In[65]:


def list2bin(s, pos=0, type=int):
    t = None
    if pos < len(s) and s[pos] == '(':
        pos += 1
        w = ''
        while s[pos] not in '()':
            w += s[pos]
            pos += 1
        t = treeasbin.TreeAsBin(type(w))
        t.child, pos = list2bin(s, pos, type)
        pos += 1
        t.sibling, pos = list2bin(s, pos, type)
    return t, pos



# In[70]:


def __bin2dot(t):
    child = t.child
    s = ''
    while child:
        s += '  ' + str(t.key) + ' -- ' + str(child.key) + ';\n'
        s += __bin2dot(child)
        child = child.sibling
    return s

def bin2dot(t):
    s = 'graph {\n'
    s += __bin2dot(t)
    return s + '}\n'

graphviz.Source(bin2dot(tbin))

