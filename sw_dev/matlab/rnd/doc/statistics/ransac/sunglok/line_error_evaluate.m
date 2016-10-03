function [distance] = line_error_evaluate(model, sample)
distance = model(1)*sample(1) + model(2)*sample(2) + model(3);

