#!~/torch/install/bin/th

require 'paths'
require 'cutorch'
require 'cunn'

----------------------------------------------------------------------
print '==> Globally-defined processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
-- Global:
cmd:option('-globalDataType', 'float', "Sets the default data type for Torch tensors: 'float', 'double'")
cmd:option('-maxMemory', 4000, 'The maximum amount of memory (in MBs) to be used for creating the training/test files')
cmd:option('-seed', 1, "The default seed to be used for the random number generator: Any positive integer number")
cmd:option('-testPhase', 0, 'Whether we want to run some small tests just to make sure everything works using the test set data: 0 | 1')
cmd:option('-modelDirName', '', 'An string to be used for the name of the directory in which the reconstructions/samples and models will be stored')
cmd:option('-modelPath', '', 'The path for a saved model')
-- Data reading/storing:
cmd:option('-zip', 1, 'Whether the data should be read from the zip files or from what already is in /Data already: 0 | 1')
cmd:option('-rawDataType', 'int', 'Determines the type of data files to be read: int (.png files) | float (.txt files)')
cmd:option('-pTrain', 0.925, 'How much, in percentage, of the data will be used for training')
cmd:option('-pValid', 0.045, 'How much, in percentage, of the data will be used for validation')
cmd:option('-pTest', 0.03, 'How much, in percentage, of the data will be used for testing')
cmd:option('-randPerm', 1, 'Whether the data set must be shuffled before training or not?: 0 | 1')
cmd:option('-fromScratch', 0, "Redo the entire data preparation process for the training phase. In 2_train.lua it indicates whether to use the pre-stored, resized data or do the process of resizing again: 0 | 1")
cmd:option('-resizeScale', 1, "The resize ratio for the input data: (0, 1]")
cmd:option('-imgSize', 224, '3D grid size. E.g. 224')
cmd:option('-numVPs', 20, 'Number of view points for the 3D models')
cmd:option('-benchmark', 0, "Set to 1 to process some benchmark data set")
-- Model:
cmd:option('-nCh', 64, "Base number of feature maps for each convolutional layer")
cmd:option('-nLatents', 220, 'The number of latent variables in the Z layer')
cmd:option('-tanh', 0, "Set to 1 if you want to normalize the input/output values to be between -1 and 1 instead of 0 to 1")
-- Training:
cmd:option('-batchSize', 4, 'Batch size for training')
cmd:option('-batchSizeChangeEpoch', 20, 'Changes the batch size every X epochs')
cmd:option('-batchSizeChange', 2, 'The number to be subtracted/added every opt.batchSizeChangeEpoch from opt.batchSize: any integer number (1 or higher)')
cmd:option('-targetBatchSize', 8, 'Maximum batch size')
cmd:option('-nReconstructions', 50, 'An integer indicating how many reconstuctions to be generated from the test data set')
cmd:option('-initialLR', 0.0000035, 'The learning rate to be used for the first few epochs of training')
cmd:option('-lr', 0.000085, 'The learning rate: Any positive decimal value')
cmd:option('-lrDecay', 0.98, 'The rate to aneal the learning rate')
cmd:option('-maxEpochs', 60, 'The maximum number of epochs: Any positive real number')
cmd:option('-dropoutNet', 0, 'Set to 1 to drop 15 to 18 views during training')
cmd:option('-VpToKeep', 100, 'Drops all VPs except this one. The valid range is [0 ... opt.numVPs]. Set it to > opt.numVPs to ignore')
cmd:option('-silhouetteInput', 0, 'Indicates whether only the masks (silhouettes) must be used for training')
cmd:option('-singleVPNet', 1, 'If set to 1, will perform random permutation on the input vector view point channels')
cmd:option('-conditional', 1, 'Indicates whether the model is trained conditionally')
cmd:option('-KLD', 72, 'The coefficient for the gradients of the KLD loss')
-- Testing:
cmd:option('-modelPath', '', 'The path to load a model an start doing experiments with it: 0 | 1')
cmd:option('-canvasHW', 5, 'Determines the height and width of the canvas on which the samples from the manifold will be shown on')
cmd:option('-nSamples', 2, 'Number of sets of samples to be drawn from the prior (z), for each category (if conditional)')
cmd:option('-sampleType', 'random', 'Determines the number of latent variables: data | random')
cmd:option('-mean', 0, 'The mean on the z vector elements: Any real number')
cmd:option('-var', 1, 'The variance of the z vector elements. In case sampleType = data then it indicates the ratio by which the predicted model variance will be multiplied by: Any positive real number')
-- Experiments
cmd:option('-experiment', 0, 'Wether we are going to only run experiments on a pre-trained model')
cmd:option('-expType', 'sample', 'Indicates the type of experiment to be performed')
cmd:option('-forwardPassType', '', 'Indicates the type of experiment to be performed: userData | nyud | randomReconstruction | reconstructAllSamples')
cmd:option('-fromEpoch', 26, 'The model from which epoch should be loaded and used')
cmd:option('-sampleCategory', '', "The category name from which one would like to start generating samples. E.g. 'chair, car, airplane'. If empty, the conditional models will generate samples for all categories in the data set. Will be used if opt.sampleType == 'data': A valid category name for which there are examples in the train data set")
cmd:option('-extraDataPath', '', "Path to silhouettes or NYUD data set images")
cmd:option('-allViewsExp', 0, 'Indicates whether the all views experiment is going to be done for SingleVPNet models')
cmd:option('-maxNumOfRecons', 0, "Determines how many training or test samples should be reconstructed when opt.forwardPasstype == 'reconstructAllSamples'. Set to 0 to get the reconstruction for all samples")
cmd:option('-getLatentDist', 0, '')

cmd:text()
opt = cmd:parse(arg or {})

if opt.zip == 1 then opt.zip = true elseif opt.zip == 0 then opt.zip = false else print "==> Incorrect value for zip argument. Acceptables: 0 or 1" os.exit() end
if opt.randPerm == 1 then opt.randPerm = true elseif opt.randPerm == 0 then	opt.randPerm = false else print "==> Incorrect value for 'randPerm' argument. Acceptables: 0 or 1" os.exit() end
if opt.fromScratch == 1 then opt.fromScratch = true elseif opt.fromScratch == 0 then opt.fromScratch = false else print "==> Incorrect value for 'fromScratch' argument. Acceptables: 0 or 1" os.exit() end
if opt.testPhase == 1 then opt.testPhase = true elseif opt.testPhase == 0 then opt.testPhase = false else print "==> Incorrect value for 'testPhase' argument" os.exit() end
if opt.benchmark == 1 then opt.benchmark = true elseif opt.benchmark == 0 then opt.benchmark = false else print "==> Incorrect value for 'benchmark' argument" os.exit() end
if opt.tanh == 1 then opt.tanh = true elseif opt.tanh == 0 then opt.tanh = false else print "==> Incorrect value for 'tanh' argument" os.exit() end
if opt.dropoutNet == 1 then opt.dropoutNet = true opt.VpToKeep = opt.VpToKeep + 1 elseif opt.dropoutNet == 0 then opt.dropoutNet = false opt.VpToKeep = 30 else print "==> Incorrect value for dropoutNet argument" os.exit() end
if opt.silhouetteInput == 1 then opt.silhouetteInput = true elseif opt.silhouetteInput == 0 then opt.silhouetteInput = false else print "==> Incorrect value for 'silhouetteInput' argument" os.exit() end
if opt.singleVPNet == 1 then opt.singleVPNet = true elseif opt.singleVPNet == 0 then opt.singleVPNet = false else print "==> Incorrect value for 'singleVPNet' argument" os.exit() end
if opt.conditional == 1 then opt.conditional = true elseif opt.conditional == 0 then opt.conditional = false else print "==> Incorrect value for 'conditional' argument" os.exit() end
if opt.batchSize < 2 then print '==> The batch size cannot be less than 2 for technical reasons. Batch size was set to 2' opt.batchSize = 2 end
if opt.experiment == 1 then opt.experiment = true if not opt.dropVPs then opt.VpToKeep = opt.VpToKeep + 1 end elseif opt.experiment == 0 then opt.experiment = false opt.expType = nil else print "==> Incorrect value for 'experiment' argument" os.exit() end
if opt.allViewsExp == 1 then opt.allViewsExp = true elseif opt.allViewsExp == 0 then opt.allViewsExp = false else print "==> Incorrect value for 'allViewsExp' argument" os.exit() end

if opt.sampleCategory ~= '' then
	local temp = {}
	local counter = 1
	for catName, _ in string.gmatch(opt.sampleCategory, '%a+') do
		temp[counter] = catName
		counter = counter + 1
	end
	opt.sampleCategory = temp
end

if opt.globalDataType == 'float' then torch.setdefaulttensortype('torch.FloatTensor') dataTypeNumBytes = 4
elseif opt.globalDataType == 'double' then torch.setdefaulttensortype('torch.DoubleTensor')
else print ("You are not allowed to use Torch data type other than 'float' or 'double'. Please set the input 'globalDataType' to either 'float' or 'double'") end

-- Make sure the scale is in the acceptable range
if opt.resizeScale <= 0 or opt.resizeScale > 1 then	opt.resizeScale = 1 end
if not opt.lr or opt.lr <= 0 then opt.lr = 0.0002 end
local tempRandInt = torch.random(1, 100000)
if opt.seed > 0 then torch.manualSeed(opt.seed) end
if opt.modelDirName == '' then opt.modelDirName = string.format('exp%.4f', tostring(torch.rand(1):totable()[1])) end
if opt.experiment then torch.manualSeed(tempRandInt) end


-- Pre-process, train and test
if not opt.experiment then
	if opt.zip or opt.fromScratch then
		dofile('1_dataLoader.lua')
	end
	dofile('2_train.lua')
elseif opt.experiment and opt.experiment == true then
	dofile('4_0_runExps.lua')
end