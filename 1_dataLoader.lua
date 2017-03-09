require 'torch'
require 'image'
require 'xlua'
require 'paths'
local commonFuncs = require '0_commonFuncs'


--[[

How to process the data:
- Your raw data should be in .zip files
- Create one .zip file for each object category (e.g. chair.zip, sofa.zip etc)
- Copy your .zip files to /Data
- Each zip file MUST contain either (or both) of the following directories:
	- catName_depth_float
	- catName_depth_rgb
- "catName" is the class label. E.g. "sofa", "chair, night_stand" etc
- Each directory contains files with patterns as follow: (%a+_%d+_%a+_%d*.[png/txt]). E.g. model_000043_Cam_0.[png/txt]. In our experiments we have 20 view points so the file names for each 3D model have Cam_0 to Cam_19 in their names.
- The option opt.rawDataType determines whether to process the .txt (float) or png (int) files. Set opt.rawDataType to 'float' and 'int' to read .txt or .png files respectively

How to process unbenchmark data
- Create a directory named "Data" in the cloned repository's directory
- If you are working with your own data create a directory named "nonbenchmark" in /Data
- Copy your .zip files in "nonbenchmark"
- Set opt.benchmark = 0

- If you are working with some benchmark data set then create a directory named "benchmark" in /Data
- Go to that directory
- Create two directories "train" and "test"
- Copy your train and test .zip files in the corresponding directories
- Set opt.benchmark = 1


Note: All zip files must have the format described above

--]]


----------------------------------------------------------------------
print '==> Data read processing options'

if not opt then

	cmd = torch.CmdLine()
	cmd:text()
	cmd:text()
	cmd:text('Options:')
	-- Global
	cmd:option('-globalDataType', 'float', 'Sets the default data type for Torch tensors: float, double')
	cmd:option('-maxMemory', 3000, 'The maximum amount of memory (in MBs) to be used for creating the training/validation and test files: Any positive real number')
	-- Data reading/storing:
	cmd:option('-zip', 0, 'Whether the data should be read from the zip files or from what already is in /Data already: 0 | 1')
	cmd:option('-rawDataType', 'int', 'Determines the type of data files to be read: int (.png files) | float (.txt files)')
	cmd:option('-depthMap', 1, 'Indicates whether the data set images are depth map or not: 1 | 0')
	cmd:option('-pTrain', 0.925, 'How much, in percentage, of the data will be used for training: (0, 1)')
	cmd:option('-pValid', 0.045, 'How much, in percentage, of the data will be used for validation (0, 1)')
	cmd:option('-pTest', 0.03, 'How much, in percentage, of the data will be used for testing (0, 1)')
	cmd:option('-randPerm', 1, 'Whether the data set must be shuffled before training or not?: 0 | 1')
	cmd:option('-fromScratch', 0, "Redo the entire data preparation process: 0 | 1")
	cmd:option('-resizeScale', 1, "The resize ratio for the input data: (0, 1]")
	cmd:option('-imgSize', 224, '3D grid size. E.g. 224')
	cmd:option('-numVPs', 20, 'Number of view points for the 3D models')
	cmd:option('-benchmark', 0, "Determines how to process the raw data. '0' is used for your own data set: 0 | 1")
	cmd:text()
	opt = cmd:parse(arg or {})

	if opt.zip == 1 then opt.zip = true elseif opt.zip == 0 then opt.zip = false else print "==> Incorrect value for zip argument. Acceptables: 0 or 1" os.exit() end
	if opt.depthMap == 1 then opt.depthMap = true elseif opt.depthMap == 0 then opt.depthMap = false else print "==> Incorrect value for depthMap argument. Acceptables: 0 or 1" os.exit() end
	if opt.randPerm == 1 then opt.randPerm = true elseif opt.randPerm == 0 then	opt.randPerm = false else print "==> Incorrect value for randPerm argument. Acceptables: 0 or 1" os.exit() end
	if opt.fromScratch == 1 then opt.fromScratch = true elseif opt.fromScratch == 0 then opt.fromScratch = false else print "==> Incorrect value for fromScratch argument. Acceptables: 0 or 1" os.exit() end
	if opt.benchmark == 1 then opt.benchmark = true elseif opt.benchmark == 0 then opt.benchmark = false else print "==> Incorrect value for 'benchmark' argument" os.exit() end
end


-------------------------------------------------------------------------------------------------------

-- Set the default data type for Torch
if opt.gpu or opt.globalDataType == 'float' then
	torch.setdefaulttensortype('torch.FloatTensor')
	dataTypeNumBytes = 4
elseif opt.globalDataType == 'double' then
	torch.setdefaulttensortype('torch.DoubleTensor')
	dataTypeNumBytes = 8
else 
	print ("You are not allowed to use Torch data type other than 'float' or 'double'. Please set the input 'globalDataType' to either 'float' or 'double'")
end

-- Make sure the scale is in the acceptable range
if opt.resizeScale < 0 or opt.resizeScale > 1 then opt.resizeScale = 1 end


local pSum = opt.pTrain + opt.pValid + opt.pTest
if pSum ~= 1 then opt.pTrain = opt.pTrain / pSum opt.pValid = opt.pValid / pSum opt.pTest= opt.pTest / pSum end

-- Some checks
if opt.rawDataType ~= 'float' and opt.rawDataType ~= 'int' then
	print ("==> Please use the correct file extension as the value for the argument 'extension'. The acceptable extensions are 'txt' and 'png'")
	os.exit()
elseif opt.rawDataType == 'float' then
	folderLookupWord = 'depth_float'
	fileExtension = '.txt'
elseif opt.rawDataType == 'int' then
	folderLookupWord = 'depth_rgb'
	fileExtension = '.png'
end

if opt.imgSize == 0 then opt.imgSize = nil end

-- Local function declarations
----------------------------------------------------------------------
local function getFileNames(zipFilePath)
	--[[
		Returns all the file names in a .zip files without uncompressing it
		
		Input:
		zipFilePath: The .zip file's path on disk (e.g. /home/file.zip)

		Output:
		All file/folder names in the target .zip file
	--]]
	local handle = assert(io.popen('zipinfo -1 ' .. zipFilePath)) -- Run a terminal command to obtain the file names in the .zip file
	local allFileNames = assert(handle:read('*a')) -- Store the result
	handle:close()
	return string.split(allFileNames, "[\n]")
end

local function separateFileAndFolderNames(allFileNames, fileExtension)
	--[[
		Separates the file and folder names given all file and folder names in a single table.
		Each table entry includes a file/folder name
		
		Input:
		allFileNames: A table whose entries are file/folder names (e.g. {'folder2/file1.ext', 'folder1/', ... })
		fileExtension: The extension of the file (e.g. '.png')

		Outputs:
		folder: A table containing only the folder names (e.g. {'folder1/', 'folder2/' ... })
		file: A table containing only the file names (e.g. {'folder2/file1.ext'})
	--]]
	local file = {}
	local folder = {}
	local counter = 0
	for _, fileName in pairs(allFileNames) do
		if not string.find(fileName, "/[a-zA-Z0-9].+%.") then
			table.insert(folder, fileName)
		else
			table.insert(file, string.match(fileName, '([%a+_]*_%d+_%a+_%d*.' .. fileExtension .. ')'))
		end
	end
	return folder, file
end

local function unZipFolder(zipFilePath, folderNames, dataFolderPath, folderLookupWord)
		--[[
		Uncompresses folderName in the .zip file 
		
		Inputs:
		zipFilePath: The path to the .zip file
		folderNames: Folder names in the .zip file (can be obtained through separateFileAndFolderNames())
		dataFolderPath: The path where we want the program to extract the folder
		folderLookupWord: The string which MUST be present in the folder name to be extracted

		Outputs:
		flag: A flag whose 'true' indicates that there the folder look up or extraction has been successful. 'false' means the opposite
		unZippedPath: Contains the path where the data set has been extracted
		--]]
		local flag = false
		local unZippedPath
		for j, folderName in pairs(folderNames) do
			if string.find(folderName, folderLookupWord) and not paths.dirp(dataFolderPath .. '/' .. folderName) then
				print (string.format("==> Extracting the folder '%s' in file '%s' to '%s'. This process may take several minutes. Be patient!!!", folderName, zipFilePath, dataFolderPath))
				os.execute('unzip ' .. zipFilePath .. ' ' .. folderName ..'* -d ' .. dataFolderPath .. ' > /dev/null')
				flag = true
				unZippedPath = dataFolderPath .. '/' .. folderName
			elseif paths.dirp(dataFolderPath .. '/' .. folderName) then
				flag = true
				unZippedPath = dataFolderPath .. '/' .. folderName
				print (string.format("==> The folder '%s' in file '%s' has been extracted already. \n==> The program will now load the files in the folder '%s' and store them in Torch tensors. \n==> To repeat the process of extraction, please remove the folder '%s' and re-run the program", folderName, zipFilePath, folderName, folderName))
			end
		end
	return flag, unZippedPath
end

local function divideDataForViewPoints(data, randPerm, pTrain, pValid, pTest)
	--[[
		Divides the data into chunks for each view point
		Input:
		data: The data set with the following format: (data[i] = {path, files, label})
		where:
			path: The path where the images for one object class are stored
			files: A table containing the file names in the path
			label: An string
		randPerm: Indicates whether random permutation should be done on the data or not: true | false
		pTrain, pValid and pTest: Determine, in percentage, how much of the raw data should be divided into train, validation and test

		Outputs:
		viewPointsDataPath: Contains the paths to view points of each 3D model for train, validation and test sets
		labels: A table of size N containing the class label for each example
	--]]
	local allModelNames = {}
	local dividedData = {}
	local tempModelName = string.match(data[1].files[1], '^(%a+_%d+)')
	local numViewPoints = 1 -- Starts from 1 and goes higher if there are more view points
	local flag = false
	local i = 2
	while flag == false do
		if tempModelName == string.match(data[1].files[i], '^([%a+_]*_%d+)') then
			numViewPoints = numViewPoints + 1
		else
			flag = true
		end
		i = i + 1
	end
	
	-- Construct regex patterns. These patterns are to be matched with the file names so that we can easily take out each view point image
	local viewPointPatterns = {}
	local viewPointsDataPath = {}
	local labels = {}
	for i=0, numViewPoints - 1 do
		table.insert(viewPointPatterns, '(%a+_%d+_%a+_' .. i .. '.%a+)')
		table.insert(viewPointsDataPath, {}) --Create empty tables for each view point
		table.insert(labels, {}) --Create empty tables for each view point
	end
	-- Add each file path to the corresponding view point data set
	for i=1, #data do
		for j=1, numViewPoints do
			local tempFileName = ""
			for k=1, #data[i].files do
				tempFileName = data[i].files[k]
				if string.match(tempFileName, viewPointPatterns[j]) then
					table.insert(viewPointsDataPath[j], data[i].path .. tempFileName)
					table.insert(labels[j], i)
				end
			end
		end
	end


	-- Do random permutation on the data paths
	if randPerm then
		local tempVPsDataPath = {}
		local tempLabels = {}
		local numImages = #viewPointsDataPath[1]
		local tempRandPerm = torch.totable(torch.randperm(numImages)) -- Assuming there are exactly N number of view point depth maps for each 3D shape model
		for i=1, numViewPoints do
			tempVPsDataPath[i] = {}
			tempLabels[i] = {}
			for j=1, numImages do
				tempVPsDataPath[i][j] = viewPointsDataPath[i][tempRandPerm[j]]
				tempLabels[i][j] = labels[i][tempRandPerm[j]]
			end
		end
		viewPointsDataPath = tempVPsDataPath
		labels = tempLabels
	end

	-- Divide the paths into train, validation and test path sets
	local numSamples = 0
	local numVPs = 0
	
	numSamples = #viewPointsDataPath[1]
	numVPs = #viewPointsDataPath

	local numTrainSamples = math.floor(numSamples * (not (pValid == 1) and pTrain or pValid))
	local numValidSamples = math.ceil(numSamples * (not (pValid == 1) and pValid or 0))
	local numTestSamples = numSamples - numTrainSamples - numValidSamples
	local numTrainValidTest = {numTrainSamples, numValidSamples, numTestSamples}
	local trainValidTestPaths = {}
	local trainValidTestLabels = {}
	if pTest ~= 0 then
		trainValidTestPaths = {{}, {}, {}}
		trainValidTestLabels = {{}, {}, {}}
	else
		trainValidTestPaths = {{}}
		trainValidTestLabels = {{}}
	end

	local startIndex = 1
	local endIndex = 0
	for i=1, pTest ~= 0 and 3 or 1 do -- pTest == 0 iff benchmark data is being processed
		endIndex = endIndex + numTrainValidTest[i]
		local tempPathHolder = {}
		local tempLabelHolder = {}
		for j=1, numVPs do
			for k=startIndex, endIndex do
				table.insert(tempPathHolder, viewPointsDataPath[j][k])
				table.insert(tempLabelHolder, labels[j][k])
			end
			trainValidTestPaths[i][j] = tempPathHolder
			trainValidTestLabels[i][j] = tempLabelHolder
			tempPathHolder = {}
			tempLabelHolder = {}
		end
		startIndex = endIndex + 1
	end

	--[[ 
		The code snippet below puts the paths for all view point depth maps for each 3D shape model into a table.
		For instance, if there are 1000 3D models each having 20 view points the resulting 
		trainValidTestPaths will be a table of size 3 (for train, validation and test sets).
		Then each of those tables contains 1000 other tables corresponding to the number of 3D models
		and each of those tables contains 20 paths.
	--]]

	local num3DModels = {}
	for i=1, #trainValidTestPaths do
		num3DModels[i] = #trainValidTestPaths[i][1] -- Obtains the number of 3D models for train, validation and test data sets
	end

	local tempPathHolder = {}
	local tempLabelHolder = {}
	for i=1, #trainValidTestPaths do
		tempPathHolder[i] = {}
		tempLabelHolder[i] = {}
		for j=1, num3DModels[i] do
			tempPathHolder[i][j] = {}
			tempLabelHolder[i][j] = trainValidTestLabels[i][1][j]
			for k=1, #trainValidTestPaths[i] do
				tempPathHolder[i][j][k] = trainValidTestPaths[i][k][j]
			end
		end
	end
	trainValidTestPaths = tempPathHolder
	trainValidTestLabels = tempLabelHolder

	return trainValidTestPaths, trainValidTestLabels
end


local function loadTxtIntoTensor(path)
	-- Loads a .txt file into a float Torch tensor
	f = assert(io.open(path, 'r'))
	local flag = true
	local height = 0
	local width = 0
	for line in f:lines() do
		height = height + 1
		if flag then
			local l = line:split(' ')
			width = #l
			flag = false
		end
	end
	if width == height + 1 then width = width - 1 end
	f:close()

	f = assert(io.open(path, 'r'))
	local tempTensor = torch.zeros(1, height, width)
	local i = 1
	for line in f:lines() do
		local l = line:split(' ')
		l[#l] = nil
		for key, val in ipairs(l) do
			tempTensor[{1, i, key}] = tonumber(val)
		end
		i = i + 1
	end
	f:close()

	return tempTensor
end


-- Start processing and reading the data files
---------------------------------------------------------------------------------------

-- The .zip files first
------------------------------------------------------

print ("============= Loading Data Into Memory and Storing it in On Disk for Training Phase =============")
local dataFoundFlag = false -- 'false' means that the none of the .zip files contained the folders catName_depth_float or catName_depth_rgb and also none of those folders (i.e already extracted) have been found on /Data either
local rawDataFolder = not opt.benchmark and '/Data/nonbenchmark' or '/Data/benchmark'
local benchmarkFolders = {'/train', '/test'}

local data = {} -- Holds the information for data sets of each class
for t = 1, (not opt.benchmark or (not opt.zip and not opt.fromScratch)) and 1 or 2 do
	if opt.benchmark and t == 1 then
		print ('==> Processing the benchmark data set to create the training set tensor files')
		print ''
	elseif t == 2 then
		print ('==> Processing the benchmark data set to create the validation/test set tensor files')
		print ''
	end
	local dataFolderPath = paths.cwd() .. rawDataFolder
	local dataStoragePath = dataFolderPath
	dataFolderPath = not opt.benchmark and dataFolderPath or dataFolderPath .. benchmarkFolders[t]
	if opt.fromScratch and opt.zip then
		-- Get the .zip file paths and class labels
		local zipFiles, classLabels = commonFuncs.findFiles(dataFolderPath, 'zip')

		-- Read the files and store them in Torch tensors along with their labels
		for i=1, opt.fromScratch and #zipFiles or 1 do

			local folderNames, fileNames
			xlua.progress(i, #zipFiles)
			local allFileNames = getFileNames(zipFiles[i])

			--Obtain the folder names
			folderNames, fileNames = separateFileAndFolderNames(allFileNames, fileExtension)
			-- Unzip the .zip file for the corresponding data type (int or float)
			dataFoundFlag, unzippedPath = unZipFolder(zipFiles[i], folderNames, dataFolderPath, folderLookupWord)
			if dataFoundFlag then
				data[i] = {path = unzippedPath, files = fileNames, label = classLabels[i]}
			end

			folderNames = {}
		end 
	end


	if dataFoundFlag then
			opt.zip = false -- Since we're sure the previous steps have either uncompressed the directories (int or float) in the .zip files 
	else
		-- This part of the code assumes that the folders have been extracted from the .zip files already
		-- The folder name patterns must be as follow: ^(%a+_depth_rgb) or ^(%a+_depth_float)
		allFiles = paths.dir(dataFolderPath)
		local counter = 1
		for i=1, #allFiles do
			local fileNames = {}
			local unzippedPath = string.match(allFiles[i], '([%a+%p+]*'.. folderLookupWord ..')')
			if unzippedPath then
				local folderName = unzippedPath
				unzippedPath = dataFolderPath .. '/' .. unzippedPath .. '/'
				local handle = assert(io.popen('ls ' .. unzippedPath)) -- Obtain the file names in the uncompressed data folder
				local allFileNames = string.split(assert(handle:read('*a')), '\n')
				local tempFiles = paths.files(unzippedPath)
				data[counter] = {path = unzippedPath, files = allFileNames, label = string.match(folderName, '^(%a+)')}
				counter = counter + 1
			end
		end

		if data then
			local labels = ''
			local counter = 0
			for i=1, #data -1 do counter = counter + 1 labels = labels .. "'" .. data[i].label .. "'" .. ', ' end labels = labels .. "'" .. data[counter+1].label .. "'" ..  '.'
			print ('==> Found ' .. #data .. ' raw uncompressed data set folder(s) to process.')
			print ('==> Object categories: ' .. labels)
		else
			print ('==> There is no data available to process. The program will now exit.')
			print ('==> Please make sure that you have given the right path to look for the .zip files or [already-extracted] data folders')
			os.exit()
		end
	end

	-- Read the '.png' or '.txt' files and store them in Torch tensors
	------------------------------------------------------
	if opt.fromScratch then
		if not opt.zip then

			--[[
				The viewPointsDataPath is structured as follow:
				- It's a table that contains 3 other tables 'X' for each of train, validation and test sets
				- Each table X[i] contains N other tables 'Y' corresponding to the number of view points N
				- Each table Y[j] contains the paths to the images for the corresponding view point n
				- So for obtaining the path to the 1st image for the 12th view point of 'test' set samples one should use 'viewPintsDataPath[3][12][1]'
				- For 'benchmark' data sets the table contains only 1 table 
			--]]

			objectCategories = {}
			for i=1, #data do
				table.insert(objectCategories, data[i].label)
			end
			
			-- Devide up the data into N parts where N is the number of view points
			if opt.benchmark and t == 1 then opt.pTrain = 1 opt.pValid = 0 opt.pTest = 0
			elseif opt.benchmark and t == 2 then opt.pTrain = 0 opt.pValid = 1 opt.pTest = 0 end
			viewPointsDataPath, labels = divideDataForViewPoints(data, opt.randPerm, opt.pTrain, opt.pValid, opt.pTest)
			local tempImg
			if opt.rawDataType == 'float' then
				tempImg = loadTxtIntoTensor(viewPointsDataPath[1][1][1])
			elseif opt.rawDataType == 'int' then
				tempImg = image.load(viewPointsDataPath[1][1][1])
			end
			local imgSize = tempImg:size()
			imgSize = tempImg[{1}]:size()
			tempImg = nil

			-- Rescale, if needed
			if not opt.imgSize and opt.resizeScale < 1 then
				imgSize[1] = math.ceil(imgSize[1] * opt.resizeScale)
				imgSize[2] = math.ceil(imgSize[2] * opt.resizeScale)
				print ("==> The new image size is " .. imgSize[2] .. " x " .. imgSize[2])
			elseif opt.imgSize then
				if opt.imgSize < imgSize[2] then 
					opt.resizeScale = math.ceil(opt.imgSize) / imgSize[2]
					imgSize[1] = math.ceil(opt.imgSize)
					imgSize[2] = math.ceil(opt.imgSize)
					print ("==> The new image size is " .. imgSize[2] .. " x " .. imgSize[2])
				else
					print ("==> Cannot change the size of the raw images since the size of the original images is lower than the requested image size (opt.imgSize)")
	        		print ("==> Continuing with the original image size of " .. imgSize[2] .. " x " .. imgSize[2])
				end
			end
			
			-- Get the amount of memory required for each image
			exampleSizeOnMem = commonFuncs.memoryPerSampleImage(imgSize, dataTypeNumBytes)

			-- Go over the paths for the examples and for each view point to store them in Torch tensors
			print (string.format('==> Loading images into Torch tensors and storing them into %s', dataFolderPath))

			local dataType = not opt.benchmark and {'train', 'validation', 'test'} or t == 1 and {'train'} or t == 2 and {'validation'}
			for i=1, #dataType do -- For train, validation and test sets
				local handle = assert(io.popen(string.format('mkdir -p %s/Datasets/%s', dataStoragePath, dataType[i])))
				handle:close()
			end
			for l = 1, not opt.benchmark and 3 or 1 do-- To go over the train, validation and test sets
				tic = torch.tic()
				print (string.format("==> Storing '%s' data on disk", dataType[l]))
				local viewPointsPath = viewPointsDataPath[l]
				local numVPs = #viewPointsPath[1]
				opt.numVPs = numVPs
				local viewPointsLabels = labels[l]
				local maxNumOfSamplesToLoad
				maxNumOfSamplesToLoad = torch.floor(commonFuncs.getFreeMemory(0.4) / exampleSizeOnMem)			
				maxNumOfSamplesToLoad = math.floor(maxNumOfSamplesToLoad / numVPs) - 1
				local numChunks = math.max(1, math.ceil(#viewPointsPath / maxNumOfSamplesToLoad)) -- Determines how many files will be saved on the disk
				local startIndex = 1
				
				-- Start saving the files on disk
				print ("==> Please wait while the process is running. You will get a message when the process is done")
				for j=1, numChunks do
					xlua.progress(j, numChunks)
					local tensorIndex = 1
					local endIndex = math.min(#viewPointsPath, startIndex + maxNumOfSamplesToLoad)

					-- Construct the Torch tensor to be used for storing the images
					local reshapedSize = {numVPs, imgSize[1], imgSize[2]}
					local viewPointSamples = torch.zeros(torch.LongStorage(commonFuncs.tableConcat({endIndex - startIndex + 1}, reshapedSize)))
					
					local viewPointSamplesLabels = torch.zeros(torch.LongStorage({endIndex - startIndex + 1}))
					-- print ('Examples Tensor Size: ' .. viewPointSamples:size()[1], 'Labels Tensor Size: ' .. viewPointSamplesLabels:size()[1])

					
					-- START for the code snippet to load an image/depth map from disk into Torch tensor
					for k = startIndex, endIndex do

						-- END Done loading a depth map from disk into Torch tensor
						if opt.resizeScale < 1 then

							local tempTensor = torch.zeros(numVPs, imgSize[1], imgSize[2])
							-- Read each image for a 3D object and store it into the temp tensor
							for m=1, numVPs do
								local tempImg
								if opt.rawDataType == 'float' then
									tempImg = loadTxtIntoTensor(viewPointsPath[k][m])[1]:clone()
								else
									tempImg = image.load(viewPointsPath[k][m], 1)
								end
								tempTensor[m] = image.scale(tempImg:clone(), imgSize[1], imgSize[2])
								tempImg = nil
							end
							viewPointSamples[tensorIndex] = tempTensor
							tempTensor = nil


						else -- Do not resize images
							local tempTensor = torch.zeros(numVPs, imgSize[1], imgSize[2])
							for m=1, numVPs do
								local tempImg
								if opt.rawDataType == 'float' then
									tempImg = loadTxtIntoTensor(viewPointsPath[k][m])[1]:clone()
								else
									tempImg = image.load(viewPointsPath[k][m], 1)
								end
								tempTensor[m] = tempImg:clone()
								tempImg = nil
							end
							viewPointSamples[tensorIndex] = tempTensor
							tempTensor = nil
						end
							
						-- Store the labels
						viewPointSamplesLabels[tensorIndex] = torch.Tensor({viewPointsLabels[k]})
						tensorIndex = tensorIndex + 1
						if k % 140 == 0 then collectgarbage() end
						if k == endIndex - 1 then collectgarbage() end
					end

					if not opt.benchmark then
						if l == 1 then
							torch.save(string.format('%s/Datasets/train/Train-%dx%d-%d.data', dataStoragePath, imgSize[2], imgSize[2], j - 1), {['dataset'] = viewPointSamples, ['labels'] = viewPointSamplesLabels, ['category'] = objectCategories, ['originalImgSize'] = imgSize})
						elseif l == 2 then
							torch.save(string.format('%s/Datasets/validation/Valid-%dx%d-%d.data', dataStoragePath, imgSize[2], imgSize[2], j - 1), {['dataset'] = viewPointSamples, ['labels'] = viewPointSamplesLabels, ['category'] = objectCategories, ['originalImgSize'] = imgSize})
						elseif l == 3 then
							torch.save(string.format('%s/Datasets/test/Test-%dx%d-%d.data', dataStoragePath, imgSize[2], imgSize[2], j - 1), {['dataset'] = viewPointSamples, ['labels'] = viewPointSamplesLabels, ['category'] = objectCategories, ['originalImgSize'] = imgSize})
						end
					else
						if t == 1 then
							torch.save(string.format('%s/Datasets/train/Train-%dx%d-%d.data', dataStoragePath, imgSize[2], imgSize[2], j - 1), {['dataset'] = viewPointSamples, ['labels'] = viewPointSamplesLabels, ['category'] = objectCategories, ['originalImgSize'] = imgSize})
						else
							torch.save(string.format('%s/Datasets/validation/Valid-%dx%d-%d.data', dataStoragePath, imgSize[2], imgSize[2], j - 1), {['dataset'] = viewPointSamples, ['labels'] = viewPointSamplesLabels, ['category'] = objectCategories, ['originalImgSize'] = imgSize})
						end
					end
					startIndex = endIndex + 1
					-- Release memory for the next chunk
					viewPointSamples = nil
					viewPointSamplesLabels = nil
					collectgarbage()
					collectgarbage()
					-- print (commonFuncs.getFreeMemory(0) .. " MB of mem after collecting garbage (chunks loop)")
				end
				print ("==> Time taken to produce dataset files for '" .. dataType[l] .. "' data type was " .. torch.toc(tic) / 60 .. " minutes")
			end
			if opt.benchmark and t == 2 or not opt.benchmark then
				print ('==> All the data set file(s) for each view point have been successfully loaded into Torch tensors and are stored on disk\n')
			end
			if opt.benchmark then opt.zip = true end
		end -- if not opt.zip
	else
		viewPointsDataPath = divideDataForViewPoints(data, opt.randPerm, opt.pTrain, opt.pValid, opt.pTest)
		local viewPointsPath = viewPointsDataPath[1]
		opt.numVPs = #viewPointsPath[1]
		print ("==> The program will be using the pre-stored data on disk.")
		print ("==> Skipping to the training, validation and test phase\n")
	end -- if opt.fromScratch
	collectgarbage()
end