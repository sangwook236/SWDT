function sample = generate_sample_from_gmm(mu, cov, phi, data_num)

gmm = gmdistribution(mu, cov, phi);
%ezsurf(@(x,y)pdf(gmm, [x y]), [-5 5], [-5 5])

sample = random(gmm, data_num);
