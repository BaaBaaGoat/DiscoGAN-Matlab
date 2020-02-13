function Y = glorot(Size,Type)
% Glorot ��̬�ֲ���ʼ��������Ҳ����Xavier��̬�ֲ���ʼ����
%������0��ֵ����׼��Ϊsqrt(2 / (fan_in + fan_out))����̬�ֲ�������
%����fan_in��fan_out��Ȩ�������������ȳ���������������Ԫ��Ŀ��
Y = randn(Size,Type) .* sqrt(2/sum(Size(3:4))/prod(Size(1:2)));
end

