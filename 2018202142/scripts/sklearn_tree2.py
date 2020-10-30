from sklearn import tree
# X = pairs of 2D points and Y = the class of each point
X = [[0, 0], [1, 1], [2,2]]
Y = [0, 1, 1]
tree_clf = tree.DecisionTreeClassifier()
tree_clf = tree_clf.fit(X, Y)
#predict the class of samples:
print("predict class of [-1., -1.]:")
print(tree_clf.predict([[-1., -1.]]))
print("predict class of [2., 2.]:")
print(tree_clf.predict([[2., 2.]]))
# the percentage of training samples of the same class
# in a leaf note equals the probability of each class
print("probability of each class in [2.,2.]:")
print(tree_clf.predict_proba([[2., 2.]]))