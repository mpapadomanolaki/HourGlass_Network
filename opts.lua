if not opt then

local function parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text(' ---------- General options ------------------------------------')
    cmd:text()
    cmd:option('-expID', 'default', 'Experiment ID')
    cmd:option('-GPU', 1, 'Default preferred GPU, if set to -1: no GPU')
    cmd:option('-nFeats',            128, 'Number of features in the hourglass')
    cmd:option('-nStack', 4, 'Number of hourglasses to stack')
    cmd:option('-LR',             2.5e-3, 'Learning rate')
    cmd:option('-LRdecay',           0.0, 'Learning rate decay')
    cmd:option('-momentum',          0.0, 'Momentum')
    cmd:option('-weightDecay',       0.0, 'Weight decay')
    cmd:option('-alpha',            0.99, 'Alpha')
    cmd:option('-epsilon', 1e-8, 'Epsilon')
    cmd:option('-nEpochs', 60, 'Total number of epochs to run')
    cmd:option('-inputRes', 128, 'Input image resolution')
    cmd:option('-outputRes', 128, 'output image resolution')
    cmd:option('-nOutChannels', 6, 'Number of Classes')
    cmd:option('-nModules', 1, 'Number of residual modules at each location in the hourglass')
    cmd:option('-batchsize', 4, 'Batchsize')
    cmd:option('-optMethod', 'rmsprop', 'Optimization method: rmsprop | sgd | nag | adadelta')
 local opt = cmd:parse(arg or {})
 return opt
end

opt = parse(arg)

if opt.GPU == -1 then
    nnlib = nn
else
    require 'nngraph'
    require 'image'
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    nnlib = cudnn
    cutorch.setDevice(opt.GPU)
end



epoch = 1
opt.epochNumber = epoch


end

