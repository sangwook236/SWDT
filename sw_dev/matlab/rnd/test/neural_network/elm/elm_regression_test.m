element_type = 0;  % regresion.
number_of_hidden_neurons = 20;
activation_function = 'sig';

training_set_filename = 'sinc_train';
testing_set_filename = 'sinc_test';

%
[training_time1, testing_time1, training_accuracy1, testing_accuracy1] = elm(training_set_filename, testing_set_filename, element_type, number_of_hidden_neurons, activation_function)

% Advanced Usage: If users wish to save trained network model and use it for different testing data sets, the provided advanced elm package may be useful:

% for training.
elm_traing(training_set_filename, element_type, number_of_hidden_neurons, activation_function);
%[training_time2, training_accuracy2] = elm_train(training_set_filename, element_type, number_of_hidden_neurons, activation_function);

% for testing/prediction.
elm_predict(testing_set_filename);
%[testing_time2, testing_accuracy2] = elm_predict(testing_set_filename);
