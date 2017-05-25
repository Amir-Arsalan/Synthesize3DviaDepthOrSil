-- Place the frequently-used functions here

require 'paths'
require 'torch'


local commonFuncs = {}

function commonFuncs.numTensorElements(tensorSize)
	--[[
		Computes the number of elements in a tensor when flattened

		Input: A tensor size in the format of torch.LongStorage

		Output: A single number of all the number of elements in a tensor
	--]]

	local tensorElements = 1
	for i=1, tensorSize:size() do
		tensorElements = tensorElements * tensorSize[i]
	end

	return tensorElements
end

function commonFuncs.tableConcat(t1,t2)
	--[[ Concatenates two tables
		Inputs:
		t1: A table
		t2: a table

		Output: t1 and t2 concatenated
	--]]

    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end

function commonFuncs.findFiles(folderPath, fileType)
	--[[
		Returns all the .zip file paths along with class labels (e.g. /home/.../chair.zip and class label: 'chair')

		Input:
		folderPath: The path where we want the program to find the files with the specified type
		fileType: An string containing the file type to look for (e.g. 'zip', 'txt')

		Outputs:
		filesPath: A table contanining the path to all the files found in 'folderPath'
		fileNames: The file names with no extension (e.g. The .zip files 'chair.zip' and 'sofa.zip' will yield file names {'chair', 'sofa'})
	--]]
	local fileNames = {}
	local filesPath = {}
	for fileName in paths.files(folderPath) do
		if fileName:find(string.format('%s', fileType) .. '') then
			table.insert(filesPath, paths.concat(folderPath, fileName))
			table.insert(fileNames, string.match(string.match(paths.concat(folderPath, fileName), "[a-zA-Z0-9_]+%." .. fileType), '[a-zA-Z0-9_]+[^.' .. fileType .. ']'))
		end
	end
	return filesPath, fileNames
end

function commonFuncs.randPermTableElements(table)
	--[[
	Randomly permutes the elements of a table.

	Input:
	table: A table with no key (e.g. {value1, value2})

	Output:
	table: The input table with its values permuted
	--]]

	if #table > 1 then
		local randIndices = torch.randperm(#table)
		local tempTable = {}
		for i=1, #table do
			tempTable[i] = table[randIndices[i]]
		end
		table = tempTable
	end
	return table
end

function commonFuncs.memoryPerSampleImage(imgSize, dataTypeNumBytes)
	
	-- Computes how much memory (in MBs) will be required for one of the images in the data set

	local totalSizeOnMem = 0 -- In MBs
	if imgSize:size() == 3 then
		totalSizeOnMem = imgSize[2] * imgSize[3] * dataTypeNumBytes / 1024 / 1024		
	elseif imgSize:size() == 2 then
		totalSizeOnMem = imgSize[1] * imgSize[2] * dataTypeNumBytes / 1024 / 1024
	else
		totalSizeOnMem = imgSize() * dataTypeNumBytes  / 1024 / 1024
	end

	return totalSizeOnMem
end

function commonFuncs.getFreeMemory(ratio)
	--[[
		Calculates the current amount of free meory

		Input:
		Ratio: A real number between [0-1] which indicates how much, in percentage, of free memory will be reserved

		Output:
		freeMem: Amount of free memory minus some amount (leaveFreeMem) in MBs
	--]]
	local handle = assert(io.popen('cat /proc/meminfo')) -- Run the command to get memory information
	local memInfo = assert(handle:read('*a')) -- Store the result (in KBs)
	handle:close()
	local freeMem = tonumber(string.match(string.split(memInfo, "[\n]")[3], '(%d+)')) / 1024
	local leaveFreeMem = freeMem * ratio -- To be reserved
	return (freeMem - leaveFreeMem) < opt.maxMemory and (freeMem - leaveFreeMem) or opt.maxMemory
end


function commonFuncs.getGPUMem()
	return ({cutorch.getMemoryUsage()})[1] / 1024 / 1024, ({cutorch.getMemoryUsage()})[2] / 1024 / 1024
end

function commonFuncs.getFileSize(file)
	--[[
		Return the file size

		Input:
		File: A lua file object created using io.open()

		Output: File size in GBs
	--]]

	local current = file:seek()      -- get current position
	local size = file:seek("end")    -- get file size
	file:seek("set", current)        -- restore position
	return size/1024/1024/1024
end

function commonFuncs.obtainDataPath(benchmark, testPhase, lowestSize)

	--[[
		Returns the path to train, validation and test data sets
		
		Inputs: 

		benchmark: Indicates whether the data being used is a benchmark data set or not
		tesePhase: Whether we want to run some small tests just to make sure everythingworks using the test set data
		lowestSize: If testPhase == true and lowestSize == true, the function returns the file path with the lowest size

		Outputs:
		The paths to train, validation and test data
	]]

	local dataFolderPath = paths.cwd() .. (not benchmark and '/Data/nonbenchmark/Datasets' or '/Data/benchmark/Datasets')
	
    local trainingDataPath = dataFolderPath .. '/' .. 'train'
    local validationDataPath = dataFolderPath .. '/' .. 'validation'
    local testDataPath = dataFolderPath .. '/' .. (not benchmark and 'test' or 'validation')

	local trainDataFiles = commonFuncs.findFiles(trainingDataPath, 'data')
	local validationDataFiles = commonFuncs.findFiles(validationDataPath, 'data')
	local testDataFiles = commonFuncs.findFiles(testDataPath, 'data')

	if testPhase then
		if lowestSize then
			local biggestFileSize = 4000 -- in GBs
			local tempDataFiles = {}
	        for i=1, #testDataFiles do
	            local tempFile = io.open(testDataFiles[i], 'r')
	            local fileSize = commonFuncs.getFileSize(tempFile)
	            if fileSize < biggestFileSize then
	                tempDataFiles[1] = testDataFiles[i]
	            end
	            tempFile:close()
	        end
	        testDataFiles = tempDataFiles
	    end
		trainDataFiles = testDataFiles
		validationDataFiles = testDataFiles
	end

	return trainDataFiles, validationDataFiles, testDataFiles
end

function commonFuncs.plotError(trainErPath, validationErPath, errorPlotNames, yAxis, title, savePath)
	--[[
		Saves a .png version of training, validation curve

		Inputs:
		trainErPath: The path to a stored 1-dimensional Torch tensor binary file for training error. The length of the vector indicates the number of epochs
		validationErPath: The path to a stored 1-dimenstional Torch tensor binary file for validation error. The length of the vector indicates the number of epochs
		title: An string containing the title of the plot
		savePath: A path to the directory where the plot is to be saved

		Output:
		saves a .png file containing the plot for train and validation errors for each epoch
	--]]
	require 'gnuplot'

	local maxIterNo
	if trainErPath[#trainErPath] ~= '' then maxIterNo = #trainErPath else maxIterNo = #trainErPath - 1 end
	for i=1, maxIterNo do
		local trainErr = torch.load(trainErPath[i])
		local validErr = torch.load(validationErPath[i])
		local epochs = trainErr:size(1)
		local plotTitle = title or ''
		local plotYAxis = errorPlotNames[i] or "Error"

		gnuplot.pngfigure(savePath .. '/' .. errorPlotNames[i] .. '.png')
		gnuplot.plot(
		   {'Training Error',  torch.linspace(1, epochs, epochs),  trainErr,  '-'},
		   {'Validation Error', torch.linspace(1, epochs, epochs), validErr, '-'})
		gnuplot.xlabel('Epochs')
		gnuplot.ylabel(plotYAxis)
		gnuplot.title(plotTitle)
		gnuplot.plotflush()
	end

end

function commonFuncs.loadModel(modelPath)
	--[[
		Loads a model given its path
	--]]
	require '2_1_KLDCriterion'
	require '2_2_Sampler'

	return torch.load(modelPath)
end

function commonFuncs.sampleDiagonalMVN(mean, log_var, numVectors)
	--[[
		Generates a Tensor of size [nVectors x numDim] given a vector of mean and variance for a [multivariate] Gaussian distribution
		Note: It is assumed that the covariance structure is diagonal
		Note: numDim is obtained by the number of elements in either of the 'mean' or var 'vectors'

		Inputs:
		mean: A [1 x numDim] vector for the means
		log_var: A [1 x numDim] vector containing the logarithm of the diagonal elements of a covariance matrix
		numVectors: Number of sample vectors to be sampled (generated)

		Output:
		An [numVectors x numDim] Torch tensor
	--]]

	local nLatents = type(mean) ~= 'table' and mean:size(2) or mean[1]:size(2)
	local logVar
	local mu
	local samples = torch.Tensor(numVectors, nLatents):fill(0)
	for i=1, numVectors do
		if type(log_var) == 'table' then
			-- log_var[1] is table with two elements: 1) log_var[1] is the mean vectors of the log-variances and 2) log_var[2] is a vector of log of the variance of the empirical distribution log-variance matrix
			logVar = log_var[1] + torch.Tensor():resizeAs(log_var[2]):copy(log_var[2]):mul(0.5):exp():cmul(torch.randn(nLatents))
		else
			logVar = log_var
		end
		if type(mean) == 'table' then
			mu = mean[1] + torch.Tensor():resizeAs(mean[2]):copy(mean[2]):mul(0.5):exp():cmul(torch.randn(nLatents))
		else
			mu = mean
		end
		samples[{{i}}] = mu + torch.Tensor():resizeAs(logVar):copy(logVar):mul(0.5):exp():cmul(torch.randn(nLatents))
	end

	return samples
end

function commonFuncs.interpolateZVectors(zVector, targetZVector, numVectors)

	--[[
		Does an interpolation between two zVectors: going from zVector to targetZVector

		Inputs:
		zVector: A [1 x numDim] vector
		targetZVector: A [1 x numDim] vector
		numVectors: Number of sample vectors to be sampled (generated)

		Output:
		An [numVectors x numDim] Torch tensor
	--]]

	local nLatents = zVector:size(2)
	local interpolatedZVectors = torch.Tensor(numVectors, nLatents):fill(0)
	for i=1, numVectors do
		for j=1, nLatents do
			interpolatedZVectors[{{i}, {j}}] = torch.linspace(zVector[1][j], targetZVector[1][j], numVectors)[i]
		end
	end

	return interpolatedZVectors
end

function commonFuncs.clearOptimState(stateTable, resetTimer, numberOfBatchesOnLastEpoch)
	--[[
		Clears the state table elements for the optimization method being used
		Note: The state table is supposed to be a reference to an state table
	--]]
	for k, v in pairs(stateTable) do
		if type(stateTable[k]) ~= 'number' then
			stateTable[k] = stateTable[k]:type(torch.getdefaulttensortype())
			stateTable[k] = nil
			collectgarbage()
		elseif resetTimer and type(stateTable[k]) == 'number' then
			stateTable[k] = nil
		elseif not resetTimer and type(stateTable[k]) == 'number' and numberOfBatchesOnLastEpoch then
			stateTable[k] = stateTable[k] - numberOfBatchesOnLastEpoch
		end
	end

end

function commonFuncs.generateBatchIndices(numDataPoints, batchSize)
	--[[
		Creates batch indices to be used for extracting batches of data
	--]]

	local indices = torch.randperm(numDataPoints):long():split(batchSize)      
    if #indices > 1 then
        local tempIndices = {}
        for ll=1, numDataPoints - batchSize * (#indices - 1) do
            tempIndices[ll] = indices[#indices][ll]
        end

        -- The Batch Normalization layers require 4D tensors
        if #tempIndices > 1 then
            indices[#indices] = torch.LongTensor(tempIndices)
        else
            indices[#indices] = nil
        end
    end

    return indices
end

function commonFuncs.normalizeMinusOneToOne(data, inPlace)
	-- Takes in as input a tensor of any size with all values between [0, 1] and outputs a tensor with all values [-1, 1]

	if not inPlace then
		dataTemp = data:clone()
	else
		dataTemp = data
	end
	dataTemp:mul(255):div(127):add(-1)

	return dataTemp
end

function commonFuncs.normalizeBackToZeroToOne(data, inPlace)
	-- Takes in as input a tensor of any size with all values between [-1, 1] and outputs a tensor with all values [0, 1]

	if not inPlace then
		dataTemp = data:clone()
	else
		dataTemp = data
	end
	dataTemp:add(1):mul(127):div(255)

	return dataTemp
end

function commonFuncs.dropInputVPs(inputTensor, markInputDepthAndMask, dropoutNet, numDropVPs, dropIndices, singleVPNet, pickedVPs, conditionHotVec)
	-- This function randomly zeros-out 15-18 of the viewpoints for DropoutNet or all viewpoints except one for SingleVPNet
	-- inputTensor could be a table with two elements (each being a tensor) and the same operations will be done on both
	-- if markInputDepthAndMask == true then the view points of the original depth maps and silhouettes will be marked by a small white square on their top-right corners
	-- Only set markInputDepthAndMask to 'true' when not doing training or validaion and only if you want to store the dropped input data on disk

	local droppedDepthTensor = type(inputTensor) ~= 'table' and inputTensor:clone() or inputTensor[1]:clone()
	local droppedSilhouettesTensor
	if type(inputTensor) == 'table' then droppedSilhouettesTensor = inputTensor[2]:clone() end
	local numVPs = droppedDepthTensor:size(2)

	local flag = false
	if dropoutNet then
		for i=1, droppedDepthTensor:size(1) do
			if flag or not dropIndices or dropIndices:sum() == 0 then
				if not numDropVPs then flag = true numDropVPs = torch.Tensor(1) end
				if not dropIndices then flag = true dropIndices = torch.Tensor(numVPs) end
				numDropVPs:fill(torch.random(numVPs-5, numVPs-2))
				dropIndices:copy(torch.randperm(numVPs))
			end
			counter = 0
			for j=1, numDropVPs[1] do
				droppedDepthTensor[i][dropIndices[j]]:zero()
				if type(inputTensor) == 'table' then
					droppedSilhouettesTensor[i][dropIndices[j]]:zero()
				end
				if markInputDepthAndMask then
					if type(inputTensor) ~= 'table' then
						inputTensor[{{i}, {dropIndices[j]}, {1, 20}, {1, 20}}] = 1
					else
						inputTensor[1][{{i}, {dropIndices[j]}, {1, 20}, {1, 20}}] = 1
						inputTensor[2][{{i}, {dropIndices[j]}, {1, 20}, {1, 20}}] = 1
					end
				end
			end
		end
	elseif singleVPNet then
		local flag = false
		if not pickedVPs then pickedVPs = torch.zeros(1) else flag = true end
		local tempDepthVP = torch.zeros(droppedDepthTensor:size(1), 1, droppedDepthTensor:size(3), droppedDepthTensor:size(4)):type(droppedDepthTensor:type())
		local pickVP
		for i=1, droppedDepthTensor:size(1) do
			if not flag then
				pickVP = torch.random(1, numVPs)
				pickedVPs = torch.cat(pickedVPs, torch.Tensor(1):fill(pickVP), 1)
			else
				pickVP = pickedVPs[1]
			end
			tempDepthVP[{{i}, {1}}] = droppedDepthTensor[{i, {pickVP}}]:clone()
		end
		if not flag then pickedVPs = pickedVPs[{{2, pickedVPs:size(1)}}] end
		local tempMaskVP
		if singleVPNet then
			droppedDepthTensor = torch.Tensor(droppedDepthTensor:size(1), 1, droppedDepthTensor:size(3), droppedDepthTensor:size(4)):type(droppedDepthTensor:type())
		else
			droppedDepthTensor:zero()
		end
		droppedDepthTensor[{{}, {1}}]:copy(tempDepthVP)
		tempDepthVP = nil

		if type(inputTensor) == 'table' then

			tempMaskVP = torch.zeros(droppedDepthTensor:size(1), 1, droppedDepthTensor:size(3), droppedDepthTensor:size(4)):type(droppedDepthTensor:type())
			local pickVP
			for i=1, droppedDepthTensor:size(1) do
				pickVP = not flag and pickedVPs[i] or pickedVPs[1]
				tempMaskVP[{{i}, {1}}] = droppedSilhouettesTensor[{i, {pickVP}}]:clone()
			end
			droppedSilhouettesTensor = torch.Tensor(droppedSilhouettesTensor:size(1), 1, droppedSilhouettesTensor:size(3), droppedSilhouettesTensor:size(4)):type(droppedSilhouettesTensor:type())
			droppedSilhouettesTensor[{{}, {1}}]:copy(tempMaskVP)
			tempMaskVP = nil
		end
		
		if markInputDepthAndMask then
			for i=1, droppedDepthTensor:size(1) do
				for j=1, ((type(inputTensor) ~= 'table' and inputTensor:size():size() == 4) and inputTensor:size(2) or inputTensor[1]:size():size() == 4 and inputTensor[1]:size(2)) or type(inputTensor) ~= 'table' and inputTensor:size(1) or inputTensor[1]:size(1) do
					if j ~= (not flag and pickedVPs[i] or pickedVPs[1]) then
						if type(inputTensor) ~= 'table' and inputTensor:size():size() == 4 or inputTensor[1]:size():size() == 4 then
							if type(inputTensor) ~= 'table' then
								inputTensor[{{i}, {j}, {1, 20}, {1, 20}}] = 1
							else
								inputTensor[1][{{i}, {j}, {1, 20}, {1, 20}}] = 1
								inputTensor[2][{{i}, {j}, {1, 20}, {1, 20}}] = 1
							end
						else
							if type(inputTensor) ~= 'table' then
								inputTensor[{{j}, {1, 20}, {1, 20}}] = 1
							else
								inputTensor[1][{{j}, {1, 20}, {1, 20}}] = 1
								inputTensor[2][{{j}, {1, 20}, {1, 20}}] = 1
							end
						end
					end
				end
			end
		end
	end

	if type(inputTensor) ~= 'table' then
		if conditionHotVec then
			return {droppedDepthTensor, conditionHotVec}
		else
			return droppedDepthTensor
		end
	else
		if conditionHotVec then
			return {droppedDepthTensor, droppedSilhouettesTensor, conditionHotVec}
		else
			return {droppedDepthTensor, droppedSilhouettesTensor}
		end
	end
end


function commonFuncs.combineMeanLogVarTensors(meansTable, log_varsTable, labelsTable)

	-- Takes in two tables meansTable and labelsTable and combines their entries (corresponding to each training file on disk) into a single Torch tensor.

	local meansTensor = meansTable[1]:new():resizeAs(meansTable[1]):copy(meansTable[1])
	local log_varsTensor = log_varsTable[1]:new():resizeAs(log_varsTable[1]):copy(log_varsTable[1])
	local labelsTensor = labelsTable[1]:new():resizeAs(labelsTable[1]):copy(labelsTable[1])
	for i=2, #meansTable do
		meansTensor = torch.cat(meansTensor, meansTable[i], 1)
		log_varsTensor = torch.cat(log_varsTensor, log_varsTable[i], 1)
		labelsTensor = torch.cat(labelsTensor, labelsTable[i], 1)
	end

	return {meansTensor, log_varsTensor, labelsTensor}
end

function commonFuncs.computeClassificationAccuracy(predictedScores, targetClassVec, returnHotVec, numCats)
	-- Computes the raw classification accuracy score. Eventually, the user should divide the returned number by total number of samples
	-- if returnHotVec == false then divide the numbers returned b the function by batch size
	-- In case targetClassVec is not provided the function will return a hot-vector version of the predictedScores vector

	local predScores = predictedScores:clone():float()
	local targetClass = targetClassVec and targetClassVec:clone():float() or nil
	local softmax = nn.SoftMax()
    softmax:forward(predScores)
    local _, idx
    _, idx = softmax.output:topk(1, true)
    if not returnHotVec then
    	return idx:float():eq(targetClass:float()):sum()
    else
    	idx = idx:view(predScores:size(1))
    	local targetClassHotVec = torch.zeros(predScores:size(1), numCats)
    	for i=1, predScores:size(1) do
			targetClassHotVec[i][idx[i]] = 1
		end
    	return targetClassHotVec:cuda()
    end
end

function commonFuncs.numOfDirs(thePath)
	-- Obtains the number of sub-directories given a directory

	local numOfDirs
	local folderNames = {}
	for folderName in paths.files(thePath) do
	    if folderName:find('$') then
	        table.insert(folderNames, paths.concat(thePath, folderName))
	    end
	end

	return #folderNames-2
end

function commonFuncs.getFileNames(thePath, lookUpStr, concatWithOrigPath)
	-- Returns the file names in a directory

	if concatWithOrigPath == nil then concatWithOrigPath = true end
	local handle = assert(io.popen('ls -1v ' .. thePath .. ' | grep -v ~$')) 
	local allFileNames = string.split(assert(handle:read('*a')), '\n')
	for i=1, #allFileNames do if not lookUpStr then allFileNames[i] = (concatWithOrigPath and paths.cwd() .. '/' or '') .. thePath .. '/' .. allFileNames[i] else allFileNames[i] = thePath .. '/' .. allFileNames[i] end end
	if lookUpStr then
		local tempAllFileNames = {}
		for i=1, #allFileNames do if allFileNames[i]:find(lookUpStr) then table.insert(tempAllFileNames, allFileNames[i]) end end
		allFileNames = tempAllFileNames
	end

	return allFileNames
end


function commonFuncs.loadExtraData(path, forwardType, numVPs)

	local imagesTensor, depthTensor, rgbTensor
	local filePaths = commonFuncs.getFileNames(path)
	local imgSize = image.load(filePaths[1]):size(3)
	if forwardType == 'userData' then
		imagesTensor = torch.Tensor(#filePaths, 1, imgSize, imgSize)
		for i=1, #filePaths do
			local tempImg = image.load(filePaths[i], 1)
			tempImg[tempImg:gt(0)] = 1
			imagesTensor[i]:copy(tempImg)
		end
		-- local temp = imagesTensor:gt(0.12)
		-- imagesTensor[imagesTensor:lt(0.12)] = 1
		-- imagesTensor[temp] = 0
		return imagesTensor
	elseif forwardType == 'nyud' then
		imagesTensor = torch.Tensor(#filePaths/3, 1, imgSize, imgSize)
		depthTensor = torch.Tensor(#filePaths/3, 1, imgSize, imgSize)
		rgbTensor = torch.Tensor(#filePaths/3, 3, imgSize, imgSize)
		for i=1, #filePaths/3 do
			local tempSil = image.load(filePaths[3*(i-1)+2], 1)

			depthTensor[i][1] = image.load(filePaths[3*(i-1)+1], 1)[{{1, 224}, {1, 224}}]
			imagesTensor[i][1]:copy(tempSil:size():size(1) == 3 and tempSil[1][{{1, 224}, {1, 224}}] or tempSil[{{1, 224}, {1, 224}}])
			rgbTensor[i] = image.load(filePaths[3*(i-1)+3])[{{}, {1, 224}, {1, 224}}]
		end
	end
end

function commonFuncs.loadDepthImagesIntoTensors(paths, strToLookup, numVPs)

	local depthTensor, depthPaths
	depthTensor = torch.zeros(1, numVPs, 224, 224)
	depthPaths = commonFuncs.getFileNames(paths, strToLookup)

	for i=1, #depthPaths do
		depthTensor[1][i] = image.load(depthPaths[i], 1)
	end
	
	return depthTensor

end

function commonFuncs.getEncodings(inputTensor, encoderModel, sampler, conditional)
	inputTensor = torch.cat(inputTensor, inputTensor, 1)
	inputTensor = inputTensor:cuda()

	local encodedSample = encoderModel:forward(inputTensor)
	if conditional then
		predictedClassScores = encodedSample[3]
	end
	encodedSample = torch.cat(encodedSample[1][{{1}}], encodedSample[2][{{1}}], 2)
	inputTensor = nil
	return conditional and {encodedSample, predictedClassScores} or {encodedSample}
end

function commonFuncs.getNumOfSamplesToViz(allSamplesPath)
	local numOfSamples = 0
	for i=1, #allSamplesPath do
		local samplesToVisualize = commonFuncs.getFileNames(allSamplesPath[i], "viz.txt")
		if #samplesToVisualize == 1 then
			f = assert(io.open(samplesToVisualize[1], 'r'))
			for line in f:lines() do
				numOfSamples = numOfSamples + 1
			end
			f:close()
		end
	end
	return numOfSamples
end

function commonFuncs.commaSeparatedStrToTable(commaSeparatedStr, digit)
	local finalText = {}
	local counter = 1
	for wantedStr, _ in digit and string.gmatch(commaSeparatedStr, '%d+') or string.gmatch(commaSeparatedStr, '%a+') do
		finalText[counter] = wantedStr
		counter = counter + 1
	end

	if digit then
		return finalText[1], finalText[2]
	else
		return finalText
	end
end


function commonFuncs.show_scatter_plot(method, mapped_x, labels, numCats, categories, exportDir)

   -- count label sizes:
   local K = numCats
   local cnts = torch.zeros(K)
   for n = 1,labels:nElement() do
      cnts[labels[n]] = cnts[labels[n]] + 1
   end

   -- separate mapped data per label:
   mapped_data = {}
   for k = 1, K do
      mapped_data[k] = {categories[k], torch.Tensor(cnts[k], 2), '+'}
   end
   local offset = torch.Tensor(K):fill(1)
   for n = 1,labels:nElement() do
      mapped_data[labels[n]][2][offset[labels[n]]]:copy(mapped_x[n])
      offset[labels[n]] = offset[labels[n]] + 1
   end

   -- show results in scatter plot:
   gnuplot.svgfigure(exportDir .. '/tSNE.svg')
   -- gnuplot.figure()
   gnuplot.grid(true)
   gnuplot.movelegend('left', 'middle')
   gnuplot.axis('auto')
   gnuplot.title(method)
   gnuplot.raw('set term svg size 5000, 5000')
   gnuplot.plot(mapped_data)
   gnuplot.plotflush()

   gnuplot.pngfigure(exportDir .. '/tSNE.png')
   -- gnuplot.figure()
   gnuplot.grid(true)
   gnuplot.movelegend('left', 'middle')
   gnuplot.axis('auto')
   gnuplot.title(method)
   gnuplot.raw('set term png size 5000, 5000')
   gnuplot.plot(mapped_data)
   gnuplot.plotflush()
end

function commonFuncs.splitTxt(inputStr, sep)
        if sep == nil then
                sep = "%s"
        end
        local t={} ; i=1
        for str in string.gmatch(inputStr, "([^".. sep .."]+)") do
                t[i] = str
                i = i + 1
        end
        return t
end

return commonFuncs