classdef Generator 
    %Discogan 判别器
    properties
        Learnables table
        auxiliary struct% batch norm的统计数据等杂项
        BatchNormMomentium single
    end
    
    methods
        function obj = Generator()
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
            

            %%%%%%%%%%
            Learnables{11,1} = dlarray(glorot([4 4 64*4 64*8],'single'));
            Learnables{12,1} = dlarray(zeros([64*4,1],'single'));
            Learnables{13,1} = dlarray(ones([64*4,1],'single'));
            
            Learnables{14,1} = dlarray(glorot([4 4 64*2 64*4],'single'));
            Learnables{15,1} = dlarray(zeros([64*2,1],'single'));
            Learnables{16,1} = dlarray(ones([64*2,1],'single'));
            
            Learnables{17,1} = dlarray(glorot([4 4 64 64*2],'single'));
            Learnables{18,1} = dlarray(zeros([64,1],'single'));
            Learnables{19,1} = dlarray(ones([64,1],'single'));
            
            Learnables{20,1} = dlarray(glorot([4 4 3 64],'single'));
                
            for i=1:20
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
            
            obj.auxiliary.bn5.mean = [];
            obj.auxiliary.bn5.var = [];
            obj.auxiliary.bn6.mean = [];
            obj.auxiliary.bn6.var = [];
            obj.auxiliary.bn7.mean = [];
            obj.auxiliary.bn7.var = [];
            
        end
        
        function [Y,auxiliary] = forward(obj,X)
            conv1 = dlconv(X,obj.Learnables{1,1}{1},0,'Stride',2,'Padding',1);
            relu1 = leakyrelu(conv1,0.2);
            conv2 = dlconv(relu1,obj.Learnables{2,1}{1},0,'Stride',2,'Padding',1);
            if(isempty(obj.auxiliary.bn2.mean ))
                [bn2,obj.auxiliary.bn2.mean,obj.auxiliary.bn2.var] = batchnorm(conv2,obj.Learnables{3,1}{1},obj.Learnables{4,1}{1});
            else
                [bn2,Mean,Variance] = batchnorm(conv2,obj.Learnables{3,1}{1},obj.Learnables{4,1}{1},obj.auxiliary.bn2.mean,obj.auxiliary.bn2.var);
                obj.auxiliary.bn2.mean=obj.auxiliary.bn2.mean*(1-obj.BatchNormMomentium )+Mean.*obj.BatchNormMomentium;
                obj.auxiliary.bn2.var=obj.auxiliary.bn2.var*(1-obj.BatchNormMomentium )+Variance.*obj.BatchNormMomentium;
            end
            relu2 = leakyrelu(bn2,0.2);
 %           Y  =relu2;
            conv3 = dlconv(relu2,obj.Learnables{5,1}{1},0,'Stride',2,'Padding',1);
            if(isempty(obj.auxiliary.bn3.mean ))
                [bn3,obj.auxiliary.bn3.mean,obj.auxiliary.bn3.var] = batchnorm(conv3,obj.Learnables{6,1}{1},obj.Learnables{7,1}{1});
            else
                [bn3,Mean,Variance] = batchnorm(conv3,obj.Learnables{6,1}{1},obj.Learnables{7,1}{1},obj.auxiliary.bn3.mean,obj.auxiliary.bn3.var);
                obj.auxiliary.bn3.mean=obj.auxiliary.bn3.mean*(1-obj.BatchNormMomentium )+Mean.*obj.BatchNormMomentium;
                obj.auxiliary.bn3.var=obj.auxiliary.bn3.var*(1-obj.BatchNormMomentium )+Variance.*obj.BatchNormMomentium;
            end
            relu3 = leakyrelu(bn3,0.2);         

            
            
            conv4 = dlconv(relu3,obj.Learnables{8,1}{1},0,'Stride',2,'Padding',1);
            if(isempty(obj.auxiliary.bn4.mean ))
                [bn4,obj.auxiliary.bn4.mean,obj.auxiliary.bn4.var] = batchnorm(conv4,obj.Learnables{9,1}{1},obj.Learnables{10,1}{1});
            else
                [bn4,Mean,Variance] = batchnorm(conv4,obj.Learnables{9,1}{1},obj.Learnables{10,1}{1},obj.auxiliary.bn4.mean,obj.auxiliary.bn4.var);
                obj.auxiliary.bn4.mean=obj.auxiliary.bn4.mean*(1-obj.BatchNormMomentium )+Mean.*obj.BatchNormMomentium;
                obj.auxiliary.bn4.var=obj.auxiliary.bn4.var*(1-obj.BatchNormMomentium )+Variance.*obj.BatchNormMomentium;
            end
            relu4 = leakyrelu(bn4,0.2);     
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            deconv5 = dltranspconv(relu4,obj.Learnables{11,1}{1},0,'Stride',2,'Cropping','same');
            if(isempty(obj.auxiliary.bn5.mean ))
                [bn5,obj.auxiliary.bn5.mean,obj.auxiliary.bn5.var] = batchnorm(deconv5,obj.Learnables{12,1}{1},obj.Learnables{13,1}{1});
            else
                [bn5,Mean,Variance] = batchnorm(deconv5,obj.Learnables{12,1}{1},obj.Learnables{13,1}{1},obj.auxiliary.bn5.mean,obj.auxiliary.bn5.var);
                obj.auxiliary.bn5.mean=obj.auxiliary.bn5.mean*(1-obj.BatchNormMomentium )+Mean.*obj.BatchNormMomentium;
                obj.auxiliary.bn5.var=obj.auxiliary.bn5.var*(1-obj.BatchNormMomentium )+Variance.*obj.BatchNormMomentium;
            end
            relu5 = relu(bn5);     
            
            
            deconv6 = dltranspconv(relu5,obj.Learnables{14,1}{1},0,'Stride',2,'Cropping','same');
            if(isempty(obj.auxiliary.bn6.mean ))
                [bn6,obj.auxiliary.bn6.mean,obj.auxiliary.bn6.var] = batchnorm(deconv6,obj.Learnables{15,1}{1},obj.Learnables{16,1}{1});
            else
                [bn6,Mean,Variance] = batchnorm(deconv6,obj.Learnables{15,1}{1},obj.Learnables{16,1}{1},obj.auxiliary.bn6.mean,obj.auxiliary.bn6.var);
                obj.auxiliary.bn6.mean=obj.auxiliary.bn6.mean*(1-obj.BatchNormMomentium )+Mean.*obj.BatchNormMomentium;
                obj.auxiliary.bn6.var=obj.auxiliary.bn6.var*(1-obj.BatchNormMomentium )+Variance.*obj.BatchNormMomentium;
            end
            relu6 = relu(bn6);     
            
            deconv7 = dltranspconv(relu6,obj.Learnables{17,1}{1},0,'Stride',2,'Cropping','same');
            if(isempty(obj.auxiliary.bn7.mean ))
                [bn7,obj.auxiliary.bn7.mean,obj.auxiliary.bn7.var] = batchnorm(deconv7,obj.Learnables{18,1}{1},obj.Learnables{19,1}{1});
            else
                [bn7,Mean,Variance] = batchnorm(deconv7,obj.Learnables{18,1}{1},obj.Learnables{19,1}{1},obj.auxiliary.bn7.mean,obj.auxiliary.bn7.var);
                obj.auxiliary.bn7.mean=obj.auxiliary.bn7.mean*(1-obj.BatchNormMomentium )+Mean.*obj.BatchNormMomentium;
                obj.auxiliary.bn7.var=obj.auxiliary.bn7.var*(1-obj.BatchNormMomentium )+Variance.*obj.BatchNormMomentium;
            end
            relu7 = relu(bn7);     
            
            deconv8 = dltranspconv(relu7,obj.Learnables{20,1}{1},0,'Stride',2,'Cropping','same');
            
            
            
            Y = sigmoid(deconv8);
            auxiliary=obj.auxiliary;
        end
    end
end

