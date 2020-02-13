classdef Discriminator
    %Discogan 判别器
    properties
        Learnables table
        auxiliary struct% batch norm的统计数据等杂项
        BatchNormMomentium single
    end
    
    methods
        function obj = Discriminator()
            obj.BatchNormMomentium = 0.1;
            

            Learnables{1,1} = dlarray(glorot([4 4 3 64],'single'));
            
            Learnables{2,1} = dlarray(glorot([4 4 64 64*2],'single'));
            Learnables{3,1} = dlarray(zeros([64*2,1],'single'));
            Learnables{4,1} = dlarray(ones([64*2,1],'single'));
            
            Learnables{5,1} = dlarray(glorot([4 4 64*2 64*4],'single'));
            Learnables{6,1} = dlarray(zeros([64*4,1],'single'));
            Learnables{7,1} = dlarray(ones([64*4,1],'single'));
            
            Learnables{8,1} = dlarray(glorot([4 4 64*4 64*8],'single'));
            Learnables{9,1} = dlarray(zeros([64*8,1],'single'));
            Learnables{10,1} = dlarray(ones([64*8,1],'single'));
            
            Learnables{11,1} = dlarray(glorot([4 4 64*8 1],'single'));
            for i=1:11
                Learnables{i,2} = "PaddingL"+i;
                Learnables{i,3} = "PaddingP"+i;
            end
            obj.Learnables = table(Learnables(:,1),Learnables(:,2),Learnables(:,3),'VariableNames',["Value","Layer","Parameter"]);
            
            obj.auxiliary = struct();
            obj.auxiliary.bn2.mean = [];
            obj.auxiliary.bn2.var = [];
            obj.auxiliary.bn3.mean = [];
            obj.auxiliary.bn3.var = [];
            obj.auxiliary.bn4.mean = [];
            obj.auxiliary.bn4.var = [];
            
        end
        
        function [Y,Z,aux] = forward(obj,X)
            conv1 = dlconv(X,obj.Learnables{1,1}{1},0,'Stride',2,'Padding',1);
            relu1 = leakyrelu(conv1,0.2);
            conv2 = dlconv(relu1,obj.Learnables{2,1}{1},0,'Stride',2,'Padding',1);
            if(isempty(obj.auxiliary.bn2.mean ))
                [bn2,obj.auxiliary.bn2.mean,obj.auxiliary.bn2.var] = batchnorm(conv2,obj.Learnables{3,1}{1},obj.Learnables{4,1}{1});
            else
                [bn2,Mean,Variance] = batchnorm(conv2,obj.Learnables{3,1}{1},obj.Learnables{4,1}{1},obj.auxiliary.bn2.mean,obj.auxiliary.bn2.var);
                obj.auxiliary.bn2.mean=obj.auxiliary.bn2.mean*(1-obj.BatchNormMomentium )+Mean*obj.BatchNormMomentium;
                obj.auxiliary.bn2.var=obj.auxiliary.bn2.var*(1-obj.BatchNormMomentium )+Variance*obj.BatchNormMomentium;
            end
            relu2 = leakyrelu(bn2,0.2);
 %           Y  =relu2;
            conv3 = dlconv(relu2,obj.Learnables{5,1}{1},0,'Stride',2,'Padding',1);
            if(isempty(obj.auxiliary.bn3.mean ))
                [bn3,obj.auxiliary.bn3.mean,obj.auxiliary.bn3.var] = batchnorm(conv3,obj.Learnables{6,1}{1},obj.Learnables{7,1}{1});
            else
                [bn3,Mean,Variance] = batchnorm(conv3,obj.Learnables{6,1}{1},obj.Learnables{7,1}{1},obj.auxiliary.bn3.mean,obj.auxiliary.bn3.var);
                obj.auxiliary.bn3.mean=obj.auxiliary.bn3.mean*(1-obj.BatchNormMomentium )+Mean*obj.BatchNormMomentium;
                obj.auxiliary.bn3.var=obj.auxiliary.bn3.var*(1-obj.BatchNormMomentium )+Variance*obj.BatchNormMomentium;
            end
            relu3 = leakyrelu(bn3,0.2);         

            
            
            conv4 = dlconv(relu3,obj.Learnables{8,1}{1},0,'Stride',2,'Padding',1);
            if(isempty(obj.auxiliary.bn4.mean ))
                [bn4,obj.auxiliary.bn4.mean,obj.auxiliary.bn4.var] = batchnorm(conv4,obj.Learnables{9,1}{1},obj.Learnables{10,1}{1});
            else
                [bn4,Mean,Variance] = batchnorm(conv4,obj.Learnables{9,1}{1},obj.Learnables{10,1}{1},obj.auxiliary.bn4.mean,obj.auxiliary.bn4.var);
                obj.auxiliary.bn4.mean=obj.auxiliary.bn4.mean*(1-obj.BatchNormMomentium )+Mean*obj.BatchNormMomentium;
                obj.auxiliary.bn4.var=obj.auxiliary.bn4.var*(1-obj.BatchNormMomentium )+Variance*obj.BatchNormMomentium;
            end
            relu4 = leakyrelu(bn4,0.2);     
            
            conv5 = dlconv(relu4,obj.Learnables{11,1}{1},0,'Stride',1,'Padding',0);
            
            Y = sigmoid(conv5);
            Z = {relu2, relu3, relu4};
            aux = obj.auxiliary;
        end
    end
end

