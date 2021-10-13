# Into to Julia (data processing)
using MLDataUtils # reexports MLDataPattern

# X is a matrix of floats
# Y is a vector of strings
X, Y = MLDataUtils.load_iris()

# The iris dataset is ordered according to their labels,
# which means that we should shuffle the dataset before
# partitioning it into training- and test-set.
Xs, Ys = shuffleobs((X, Y))
# Notice how we use tuples to group data.

# We leave out 15 % of the data for testing
(cv_X, cv_Y), (test_X, test_Y) = splitobs((Xs, Ys); at = 0.85)

# Next we partition the data using a 10-fold scheme.
# Notice how we do not need to splat train into X and Y
for (train, (val_X, val_Y)) in kfolds((cv_X, cv_Y); k = 10)

    for epoch = 1:100
        # Iterate over the data using mini-batches of 5 observations each
        for (batch_X, batch_Y) in eachbatch(train, size = 5)
            # ... train supervised model on minibatches here
        end
    end
end
