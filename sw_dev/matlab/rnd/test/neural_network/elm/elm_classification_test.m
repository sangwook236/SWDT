element_type = 1;  % classification.
number_of_hidden_neurons = 20;
activation_function = 'sig';

training_set_filename = 'diabetes_train';
testing_set_filename = 'diabetes_test';

%%
%% basic
%%

[training_time(:,1), testing_time(:,1), training_accuracy(:,1), testing_accuracy(:,1)] = elm(training_set_filename, testing_set_filename, element_type, number_of_hidden_neurons, activation_function);

%%
%% advanced usage: If users wish to save trained network model and use it for different testing data sets, the provided advanced elm package may be useful
%%

% for training.
[training_time(:,2), training_accuracy(:,2)] = elm_train(training_set_filename, element_type, number_of_hidden_neurons, activation_function);

% for testing/prediction.
[testing_time(:,2), testing_accuracy(:,2)] = elm_predict(testing_set_filename);

%%
%% ELM with kernel
%%

regularization_coefficient = 1;
kernel_type = 'RBF_kernel';
kernel_param = 100;

[training_time(:,3), testing_time(:,3), training_accuracy(:,3), testing_accuracy(:,3)] = elm_kernel(training_set_filename, testing_set_filename, element_type, regularization_coefficient, kernel_type, kernel_param);
