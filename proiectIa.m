load('trainData.mat');
load('testData.mat');

tv=trainVectors';
tt=full(ind2vec(trainLabels'));

net=newff(minmax(tv),[400 200 5],{'logsig' 'logsig' 'tansig'},'trainoss');

net.trainParam.epochs=1000;
net.trainParam.goal=1e-5; 

net.performFcn = 'msereg';
net.performParam.ratio = 0.7;

[net,tr]=train(net,tv,tt);
y=sim(net,testVectors');
y=vec2ind(compet(y));

rez=zeros(size(y,2),2);
for i=1:size(y,2)
    rez(i,1)=i;
    rez(i,2)=y(i);
end


%indices = crossvalind('Kfold',length(tt), 10); 
%cp=classperf(tt);
%for i=1:10
%    test=find(indices==i);
%    train=~test;
%    class=classify(tv(test,:),tv(train,:),tt(train,:));
%    classperf(cp,class,test); 
%end
    

filename = 'test.csv';
fid = fopen(filename, 'w');
fprintf(fid, 'Id,');
fprintf(fid, 'Prediction\n');

dlmwrite(filename, rez, '-append', 'precision', '%d');
fclose(fid);