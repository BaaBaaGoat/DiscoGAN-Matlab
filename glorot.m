function Y = glorot(Size,Type)
% Glorot 正态分布初始化方法，也称作Xavier正态分布初始化，
%参数由0均值，标准差为sqrt(2 / (fan_in + fan_out))的正态分布产生，
%其中fan_in和fan_out是权重张量的扇入扇出（即输入和输出单元数目）
Y = randn(Size,Type) .* sqrt(2/sum(Size(3:4))/prod(Size(1:2)));
end

