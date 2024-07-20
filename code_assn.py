import numpy as np
from sklearn.svm import LinearSVC
from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y0_train, y1_train ):
################################
#  Non Editable Region Ending  #
################################
	X_train = my_map(X_train)
	clf0 = LinearSVC(loss='hinge', C=0.05,max_iter = 100)
	clf1 = LinearSVC(loss='hinge', C=0.05,max_iter = 100)

	clf0.fit(X_train, y0_train)
	w0 = clf0.coef_[0]
	b0 = clf0.intercept_[0]

	clf1.fit(X_train, y1_train)
	w1 = clf1.coef_[0]
	b1 = clf1.intercept_[0]
	# Use this method to train your models using training CRPs
	# X_train has 32 columns containing the challenge bits
	# y0_train contains the values for Response0
	# y1_train contains the values for Response1
	
	# THE RETURNED MODELS SHOULD BE TWO VECTORS AND TWO BIAS TERMS
	# If you do not wish to use a bias term, set it to 0
	return w0, b0, w1, b1


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
	transformed_X = np.cumprod(1-2*np.flip(X, axis=1), axis=1)
	X = X[:,:-1]
	features = np.hstack((X, transformed_X))
	return features