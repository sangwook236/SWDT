function df = imu_df_dx(x, param)

df = 0.5+25*(1-x.^2)./((1+x.^2).^2);
