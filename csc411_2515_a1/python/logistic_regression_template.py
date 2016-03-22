import numpy as npi
import matplotlib as mpl
import matplotlib.pyplot as plt
from check_grad import check_grad
from plot_digits import *
from utils import *
from logistic import *

def run_logistic_regression():
    #train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape
    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.85,
                    'weight_regularization': 0.01,
                    'num_iterations': 30
                 }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = np.random.rand(M+1, 1)
    #weights = np.zeros(M+1)
    # Verify that your logistic :function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)
    # Begin learning with gradient descent
    ce_train_vec = []
    ce_val_vec = []
    for t in xrange(hyperparameters['num_iterations']):

        # TODO: you may need to modify this loop to create plots, etc.

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        #f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)
        f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
        #print 'f, df, pred', f, df, predictions
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)
        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")
        # update parameters
        #weights = weights - hyperparameters['learning_rate'] * df / N
        weights = weights - hyperparameters['learning_rate'] * (df + hyperparameters['weight_regularization'] * weights)/ N
        
        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        
        # print some stats
        stat_msg = "ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f}  "
        stat_msg += "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}"
        print stat_msg.format(t+1,
                              float(f / N),
                              float(cross_entropy_train),
                              float(frac_correct_train*100),
                              float(cross_entropy_valid),
                              float(frac_correct_valid*100))
        # add data for plots
        ce_train_vec.append(float(cross_entropy_train))
        ce_val_vec.append(float(cross_entropy_valid))
    x = range(1, len(ce_train_vec)+1)
    plt.plot(x, ce_train_vec, 'r', x, ce_val_vec, 'b')
    plt.xlabel('Iterations')
    plt.ylabel('Cross Entropy')
    plt.legend(['Training', 'Validation'], bbox_to_anchor=(1, 0.5))
    plt.show()
    test_inputs, test_targets = load_test()
    f, df, predictions = logistic(weights, test_inputs, test_targets, hyperparameters)
    cross_entropy_train, frac_correct_train = evaluate(test_targets, predictions)
    stat_msg = "ITERATION:{:4d}  TEST NLOGL:{:4.2f}  TEST CE:{:.6f}  TEST FRAC:{:2.2f}"
    print stat_msg.format(t+1,
                              float(f / N),
                              float(cross_entropy_train),
                              float(frac_correct_train*100))

def pen_logistic_regression():
    train_inputs, train_targets = load_train()
    #train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    weight_reg_vec = [ pow(10, i) for i in range(-5, 0, 1)]
    N, M = train_inputs.shape
    class_error_train = []
    class_error_val = []
    cross_en_train = []
    cross_en_val = []
    for lam in weight_reg_vec:
        # TODO: Set hyperparameters
        hyperparameters = {
                        'learning_rate': 0.85,
                        'weight_regularization': lam,
                        'num_iterations': 30
                     }


        # Verify that your logistic :function produces the right gradient.
        # diff should be very close to 0.
        run_check_grad(hyperparameters)
        # Begin learning with gradient descent
        ce_train_vec = []
        ce_val_vec = []
        corr_train_vec = []
        corr_val_vec = []
        for i in range(0, 5):
            # Logistic regression weights
            weights = np.random.rand(M+1, 1)
            #weights = np.zeros(M+1)
            for t in xrange(hyperparameters['num_iterations']):

                # TODO: you may need to modify this loop to create plots, etc.

                # Find the negative log likelihood and its derivatives w.r.t. the weights.
                f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
                # Evaluate the prediction.
                cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)
                if np.isnan(f) or np.isinf(f):
                    raise ValueError("nan/inf error")
                # update parameters
                weights = weights - hyperparameters['learning_rate'] * (df + lam * weights)/ N

                # Make a prediction on the valid_inputs.
                predictions_valid = logistic_predict(weights, valid_inputs)

                # Evaluate the prediction.
                cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
                # add data for plots
            # add final CEs and errors
            ce_train_vec.append(float(cross_entropy_train))
            ce_val_vec.append(float(cross_entropy_valid))
            corr_train_vec.append(frac_correct_train)
            corr_val_vec.append(frac_correct_valid)
        # average across multiple runs (due to random initials)
        cross_en_train.append(float(np.average(ce_train_vec)))
        cross_en_val.append(float(np.average(ce_val_vec)))
        class_error_train.append(1.0-float(np.average(corr_train_vec)))
        class_error_val.append(1.0-float(np.average(corr_val_vec)))
    x = np.array(weight_reg_vec)
    plt.semilogx(x, cross_en_train, 'r', label='Training')
    plt.semilogx(x, cross_en_val, 'b', label='Validation')
    plt.xlabel('Regularization factor')
    plt.ylabel('Cross Entropy')
    plt.legend(['Training', 'Validation'], bbox_to_anchor=(1, 0.5))
    plt.show()
    plt.semilogx(x, class_error_train, 'r', label='Training')
    plt.semilogx(x, class_error_val, 'b:', label='Validation')
    plt.xlabel('Regularization factor')
    plt.ylabel('Classification Error')
    plt.legend(['Training', 'Validation'], bbox_to_anchor=(1, 0.5))
    plt.show()
    test_inputs, test_targets = load_test()
    hyperparameters['weight_regularization'] = 0.01
    hyperparameters['num_iterations'] = 30
    f, df, predictions = logistic(weights, test_inputs, test_targets, hyperparameters)
    cross_entropy_train, frac_correct_train = evaluate(test_targets, predictions)
    stat_msg = "ITERATION:{:4d}  TEST NLOGL:{:4.2f}  TEST CE:{:.6f}  TEST FRAC:{:2.2f}"
    print stat_msg.format(t+1,
                              float(f / N),
                              float(cross_entropy_train),
                              float(frac_correct_train*100))
 


def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and 
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.round(np.random.rand(num_examples, 1), 0)
    
    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff
   

if __name__ == '__main__':
    run_logistic_regression()
    #pen_logistic_regression()
