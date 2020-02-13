function [losses] = get_fm_loss(real_feats, fake_feats)
losses = 0;
for i=1:length(real_feats)
        loss = (mean(real_feats{i},1) - mean(fake_feats{i},1)).^2;
        losses =losses+mean(loss,'all');
end
end


