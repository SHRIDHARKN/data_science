import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None,value=None,prob_arr=None):
        
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.prob_arr = prob_arr
        
    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTreeClassifier:
    
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)  
    
    def _traverse_prob_tree(self, x, node):
        if node.is_leaf_node():
            return node.prob_arr

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_prob_tree(x, node.left)
        return self._traverse_prob_tree(x, node.right)  
    
    def _build_tree(self, X, y, depth=0):
        
        # find the best feat and best thres for it
        # split the data with the selec feat and thres
        # return Node with feat, thres, left pointer and right pointer
        
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        if depth==0:
            self.uniq_labels = np.unique(y).tolist()

        # check the stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value,leaf_prob_arr = self._most_common_label(y)
            return Node(value=leaf_value,prob_arr=leaf_prob_arr)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature_idx, best_thresh = self._best_split(X, y, feat_idxs)

        # create child nodes
        # get the indexes for left and right child data with best feature and best thres
        left_idxs, right_idxs = self._split(X[:, best_feature_idx], best_thresh)
        
        # with left and right child data, create left and right tree
        # build tree with left data and right data resp
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth+1)
        # set a node with feature id and thres, along with left and right child pointer
        sub_node = Node(best_feature_idx, best_thresh, left, right)
        return sub_node

    
    def _best_split(self, X, y, feat_idxs):
        
        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thres_idxs = np.argwhere(y[:-1]!=y[1:]).flatten()
            thresholds = np.unique(X_column[thres_idxs])
            #thresholds = np.unique(X_column)
            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create childs
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])

    def _most_common_label(self, y):
        
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        
        total_count = sum(dict(counter).values())
        prob_arr = []
        for k in self.uniq_labels:
            prob_arr.append(counter[k]/total_count)
        return value,np.array(prob_arr)

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._build_tree(X, y) # returns a node with left,right,feat,and thres

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def predict_proba_(self, X):
        return np.array([self._traverse_prob_tree(x, self.root) for x in X])
