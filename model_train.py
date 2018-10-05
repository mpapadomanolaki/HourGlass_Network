require 'torchx'
require 'nngraph'
require 'cunn'
require 'cudnn'
require 'optim'
ext='png'
paths.dofile('opts.lua')
paths.dofile('hg.lua')

--load dataset's mean and stdv
mean=torch.load('mean.t7')
stdv=torch.load('stdv.t7')

optfn = optim[opt.optMethod]
if not optimState then
    optimState = {
        learningRate = opt.LR,
        learningRateDecay = opt.LRdecay,
        momentum = opt.momentum,
        weightDecay = opt.weightDecay,
        alpha = opt.alpha,
        epsilon = opt.epsilon
}
end
print('==> Creating model from start')
model = createModel(modelArgs)

criterion=nn.ParallelCriterion()
for i=1,opt.nStack do
criterion:add(nn.MSECriterion())
end
criterion=criterion:cuda()

confusion=optim.ConfusionMatrix(6)
--create txt file to save the train/test accuracies
ff = io.open('./saved_models/progress.txt', 'w')
print('==> Converting model to CUDA')
model=model:cuda()

param, gradparam = model:getParameters()
local function evalFn(x) return criterion.output, gradparam end
cudnn.fastest = true
cudnn.benchmark = true


function patches_location(loc)
   patches = {}
   patches_dir=loc
   for file in paths.files(patches_dir) do
      if file:find(ext .. '$') then
         table.insert(patches, paths.concat(patches_dir,file))
      end
   end
  return patches
end

function convert_to_Tensor(inputs, targets)
   inputs_tensor=torch.CudaTensor(table.getn(inputs),3,opt.nFeats,opt.nFeats)
   targets_tensor=torch.CudaTensor(table.getn(inputs),1,opt.nFeats,opt.nFeats)
   for j=1,table.getn(inputs) do
    inputs_tensor[j]=inputs[j]
    targets_tensor[j]=targets[j]
   end
   return inputs_tensor, targets_tensor
end

function convert_to_onehot(targets_tensor)
   onehot_labels_tensor=torch.CudaTensor(targets_tensor:size(1),opt.nOutChannels,opt.nFeats,opt.nFeats)
   for j=1,targets_tensor:size(1) do
    b=module:forward(targets_tensor[j])
    b2=b:transpose(2,4)
    b2=b2:transpose(3,4)
    onehot_labels_tensor[j]=b2
   end

   onehot_labels_table={}
   for j=1,opt.nStack do
    onehot_labels_table[j]=onehot_labels_tensor
   end
   return onehot_labels_table
end

function model_preds(inputs, output)
   backlabels2d=torch.CudaTensor(table.getn(inputs),1,opt.nFeats,opt.nFeats):fill(0)

   for j=1,table.getn(inputs) do
    max, indices2d=torch.max(output[#output][j],1)
    backlabels2d[j]=indices2d:cuda()
   end
   return backlabels2d
end

train_patches=patches_location('..folder/of/your/png/train_patches/')
train_labels=patches_location('..folder/of/your/png/train_labels/')
test_patches=patches_location('..folder/of/your/png/test_patches/')
test_labels=patches_location('..folder/of/your/png/test_labels/')

train_number=table.getn(train_patches)
test_number=table.getn(test_patches)

for epoch=1,opt.nEpochs do
 print('Epoch ' .. epoch )
-----------------------------TRAINING---------------------------------------------
 model:training()
 module=nn.OneHot(opt.nOutChannels):cuda()
 avgAcc=0
 for t=1,train_number,opt.batchsize do
   xlua.progress(t,train_number)
   inputs={}
   targets={}
   for i=t,math.min(t+opt.batchsize-1,train_number) do
    input=image.load(train_patches[i],3,'byte')
    target=image.load(train_labels[i],1,'byte')
    table.insert(inputs,input)
    table.insert(targets,target)
   end
   inputs_tensor, targets_tensor=convert_to_Tensor(inputs, targets)
   onehot_labels_table=convert_to_onehot(targets_tensor)

   for ch=1,3 do
    inputs_tensor[{ {},{ch},{},{} }]:add(-mean[ch])
    inputs_tensor[{ {},{ch},{},{} }]:div(stdv[ch])
   end
   output=model:forward(inputs_tensor)

   err=criterion:forward(output, onehot_labels_table)

   model:zeroGradParameters()
   model:backward(inputs_tensor, criterion:backward(output, onehot_labels_table))
   optfn(evalFn, param, optimState)

   backlabels2d=model_preds(inputs, output)
   confusion:batchAdd(backlabels2d:view(-1), targets_tensor:view(-1))
 end
 print('Training Results:')
 print('\n')
 confusion:updateValids()
 print(confusion)
 ff:write('Epoch ', epoch , ':','\n' , 'Training results:', '\n', tostring(confusion), '\n' )
 confusion:zero()
 torch.save('../folder/for/saved/models/model' .. epoch .. '.net', model)

---------------------------------------TESTING--------------------------------------------
 print('Testing ..')
 for tt=1,test_number,opt.batchsize do
  xlua.progress(tt,test_number)

  test_inputs={}
  test_targets={}
  for i=tt,math.min(tt+opt.batchsize-1,test_number) do
   input=image.load(test_patches[i],3,'byte')
   target=image.load(test_labels[i],1,'byte')
   table.insert(test_inputs,input)
   table.insert(test_targets,target)
  end

  inputs_tensor, targets_tensor=convert_to_Tensor(test_inputs, test_targets)
  onehot_labels_table=convert_to_onehot(targets_tensor)

  for ch=1,3 do
   inputs_tensor[{ {},{ch},{},{} }]:add(-mean[ch])
   inputs_tensor[{ {},{ch},{},{} }]:div(stdv[ch])
  end
  output=model:forward(inputs_tensor)

  backlabels2d=model_preds(test_inputs, output)
  confusion:batchAdd(backlabels2d:view(-1), targets_tensor:view(-1))
 end
print('Testing Results:')
print('\n')
confusion:updateValids()
print(confusion)
ff:write('Testing Results:','\n' , tostring(confusion))
ff:write('-------------------------------------------------------------------------------------------')
ff:write('\n', '\n' )
confusion:zero()
model:clearState()
end

ff:close()

