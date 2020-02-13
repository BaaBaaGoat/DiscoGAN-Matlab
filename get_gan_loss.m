function [dis_loss,gen_loss] = get_gan_loss(dis_real, dis_fake)
lossGenerated = -mean(log(1-dis_fake));
lossReal = -mean(log(dis_real));
dis_loss = (lossReal + lossGenerated);
gen_loss = -mean(log(dis_fake));
end

