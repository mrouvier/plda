PLDA Scoring
===========
**Open-source implementation of simplified PLDA (Probabilistic Linear Discriminant Analysis)**

PLDA is a probabilistic version of Linear Discriminant Analysis (LDA). This technique projects the input data into a much lower dimensional space with minimal loss of discriminative ability, as the ratio of between-classe and within-class variations is maximized.


News
----

* 14/11/2014: First version



Get and Compile (you need boost):

```
git clone https://github.com/mrouvier/plda
cd plda
make
```


Program usage
-------------

```
./bin/plda data/train.txt data/labels data/test.txt
```

