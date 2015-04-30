element_type = 1;  % classification.
number_of_hidden_neurons = 20;
activation_function = 'sig';
        
training_set_filename = 'diabetes_train';
testing_set_filename = 'diabetes_test';

%
elm(training_set_filename, testing_set_filename, 1, number_of_hidden_neurons, 'sig');

% Advanced Usage: If users wish to save trained network model and use it for different testing data sets, the provided advanced elm package may be useful:

% for training.
elm_train(training_set_filename, element_type, number_of_hidden_neurons, activation_function);
[training_time2, training_accuracy2] = elm_train(training_set_filename, element_type, number_of_hidden_neurons, activation_function);

% for testing/prediction.
elm_predict(testing_set_filename);
[testing_time2, testing_accuracy2] = elm_predict(testing_set_filename);
