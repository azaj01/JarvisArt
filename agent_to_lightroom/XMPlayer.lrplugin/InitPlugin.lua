local LrSocket = import 'LrSocket'
local LrTasks = import 'LrTasks'
local LrFunctionContext = import 'LrFunctionContext'
local LrApplication = import 'LrApplication'
local LrLogger = import 'LrLogger'
local LrPathUtils = import 'LrPathUtils'
local LrFileUtils = import 'LrFileUtils'
local LrExportSession = import 'LrExportSession'

-- Create and enable logger
local logger = LrLogger('XMPPlayer')
logger:enable('logfile')

-- Global variables for server status tracking
local SERVER_STATUS = {
    running = false,
    port = 7878,
    error = nil
}

-- Process counter for monitoring
local PROCESS_COUNT = 0

-- Constants
local MAINTENANCE_THRESHOLD = 100
local SERVER_CHECK_INTERVAL = 300
local SLEEP_INTERVAL = 0.5

-- Path processing functions
local function getParentDir(path)
    return LrPathUtils.parent(path) or "."
end

local function splitMessage(message)
    local result = {}
    local pattern = "[^|]+"
    local start = 1
    local splitStart, splitEnd = string.find(message, pattern, start)
    
    while splitStart do
        table.insert(result, string.sub(message, splitStart, splitEnd))
        start = splitEnd + 2  -- +2 to skip the delimiter
        splitStart, splitEnd = string.find(message, pattern, start)
    end
    
    return result
end

-- Lua settings file parser
local function parseLuaSettingsFile(luaPath)
    logger:info("Loading settings from Lua file: " .. luaPath)
    
    -- Use loadfile to load Lua file
    local chunk, err = loadfile(luaPath)
    if not chunk then
        logger:error("Failed to load Lua file: " .. tostring(err))
        return nil
    end
    
    -- Execute file and get return value
    local success, settings = pcall(chunk)
    if not success then
        logger:error("Failed to execute Lua file: " .. tostring(settings))
        return nil
    end
    
    -- Ensure return value is a table
    if type(settings) ~= "table" then
        logger:error("Lua file did not return a table")
        return nil
    end
    
    logger:info("Successfully loaded settings from Lua file")
    return settings
end

-- Modified importPreset function
local function importPreset(settingsPath)
    -- Get filename as preset name
    local presetName = LrPathUtils.removeExtension(LrPathUtils.leafName(settingsPath))
    if not presetName then
        logger:error("Failed to get preset name from path: " .. tostring(settingsPath))
        return nil
    end
    
    -- Parse settings file
    local settings = parseLuaSettingsFile(settingsPath)
    if not settings then
        logger:error("Failed to parse settings file")
        return nil
    end
    
    -- If white balance is "As Shot" or "Auto", remove temperature and tint settings
    if settings.WhiteBalance == "As Shot" or settings.WhiteBalance == "Auto" then
        settings.Temperature = nil
        settings.Tint = nil
        settings.IncrementalTemperature = nil
        settings.IncrementalTint = nil
    end
    
    return settings
end

-- Validation helper functions
local function validatePaths(photoPath, xmpPath)
    if not photoPath or photoPath == "" then
        return "error|Invalid photo path"
    end
    
    if not xmpPath or xmpPath == "" then
        return "error|Invalid XMP path"
    end
    
    if not LrFileUtils.exists(photoPath) then
        return "error|Photo file does not exist"
    end
    
    if not LrFileUtils.exists(xmpPath) then
        return "error|XMP file does not exist"
    end
    
    return nil -- No error
end

local function handleRequest(message)
    logger:info("Processing request: " .. tostring(message))
    
    -- Check if message is empty
    if not message or message == "" then
        return "error|Empty message received"
    end
    
    -- Try to parse request
    local parts = splitMessage(message)
    
    if #parts == 0 then
        return "error|Empty request"
    end
    
    local command = parts[1]
    logger:info("Processing command: " .. command)
    
    -- Handle ping requests
    if command == "ping" then
        return "pong"
    end
    
    -- Handle photo requests
    if command == "process" and parts[2] and parts[3] then
        local photoPath = parts[2]
        local xmpPath = parts[3]
        
        -- Validate paths and file existence
        local validationError = validatePaths(photoPath, xmpPath)
        if validationError then
            return validationError
        end
        
        -- Prepare output path
        local outputDir = getParentDir(photoPath)
        if not outputDir then
            return "error|Failed to get parent directory from path: " .. photoPath
        end
        
        local fileName = LrPathUtils.leafName(photoPath)
        local photoName = LrPathUtils.removeExtension(fileName)
        if not photoName then
            return "error|Failed to get photo name from path: " .. photoPath
        end
        
        local outputPath = LrPathUtils.child(outputDir, photoName .. "_processed.jpg")
        if not outputPath then
            return "error|Failed to create output path"
        end
        
        -- Process photo in new async task
        LrTasks.startAsyncTask(function()
            local catalog = LrApplication.activeCatalog()
            if not catalog then
                logger:error("Failed to get active catalog")
                return
            end
            
            -- Process counting and monitoring
            PROCESS_COUNT = PROCESS_COUNT + 1
            if PROCESS_COUNT % MAINTENANCE_THRESHOLD == 0 then
                logger:info("Processed " .. PROCESS_COUNT .. " photos - consider catalog maintenance")
            end
            
            -- Find photo in write access context
            local photo = nil
            catalog:withWriteAccessDo("Find Photo", function()
                logger:info("Searching for photo in catalog: " .. photoPath)
                photo = catalog:findPhotoByPath(photoPath)
            end)
            
            if not photo then
                logger:info("Photo not found in catalog, attempting to import...")
                -- Use withWriteAccessDo to import photo
                local success = catalog:withWriteAccessDo("Add Photo", function()
                    local importedPhoto = catalog:addPhoto(photoPath)
                    if importedPhoto then
                        logger:info("Successfully imported photo")
                        photo = importedPhoto
                        logger:info("Imported photo path: " .. photo:getRawMetadata("path"))
                        logger:info("Imported photo name: " .. photo:getFormattedMetadata("fileName"))
                    end
                end)
                
                if not success then
                    logger:error("Import operation was cancelled or failed")
                    return
                end
            end
            
            if not photo then
                logger:error("Could not find or import photo: " .. photoPath)
                return
            end
            
            -- Create export settings
            local exportSettings = {
                LR_format = 'JPEG',
                LR_jpeg_quality = 0.8,
                LR_export_destinationType = 'specificFolder', 
                LR_export_useSubfolder = true,
                LR_export_destinationPathPrefix = outputDir,
                LR_export_destinationPathSuffix = "processed"
            }
            

            -- Process photo in single transaction (minimize history)
            catalog:withWriteAccessDo("Apply XMP", function()
                -- Import and create preset
                logger:info("Creating preset from XMP file: " .. xmpPath)
                local preset = importPreset(xmpPath)
                if not preset then
                    logger:error("Failed to create preset")
                    return
                end
                -- Apply preset and AI settings in same transaction
                photo:applyDevelopSettings(preset)
                photo:updateAISettings()
                logger:info("Applied XMP settings in single transaction")
            end)

            -- Create export session
            local exportSession = LrExportSession({
                photosToExport = {photo},
                exportSettings = exportSettings
            })

            -- Start the export process on a new task
            exportSession:doExportOnNewTask()
            
        end)
        
        return "processing|Request accepted, processing photo..."
    end
    
    return "error|Invalid request format"
end

local function startServer()
    logger:info("Starting server")
    
    if SERVER_STATUS.running then
        logger:info("Server already running on port " .. SERVER_STATUS.port)
        return
    end
    
    LrTasks.startAsyncTask(function()
        LrFunctionContext.callWithContext("startServer", function(context)
            local running = true
            local server = nil
            
            -- Add cleanup handler
            context:addCleanupHandler(function()
                logger:info("Cleaning up server resources")
                if server then
                    server:close()
                end
                SERVER_STATUS.running = false
                running = false
            end)
            
            -- Start server in async task
            logger:info("Attempting to start server on port " .. SERVER_STATUS.port)
            
            server = LrSocket.bind({
                functionContext = context,
                plugin = _PLUGIN,
                port = SERVER_STATUS.port,
                mode = "receive",
                hostname = "127.0.0.1",
                
                onConnecting = function(socket, port)
                    logger:info("Server connecting on port " .. port)
                end,
                
                onConnected = function(socket, port)
                    logger:info("Server connected on port " .. port)
                    SERVER_STATUS.running = true
                    SERVER_STATUS.error = nil
                end,
                
                onMessage = function(socket, message)
                    logger:info("Received message: " .. tostring(message))
                    local response = LrTasks.pcall(handleRequest, message)
                    if response then
                        logger:info("Sending response: " .. response)
                        socket:send(response .. "\n")
                    end
                end,
                
                onClosed = function(socket)
                    logger:info("Client connection closed")
                    -- Server keeps running, restart listening for new connections
                    logger:info("Client disconnected, restarting listener for new connections")
                    socket:reconnect()
                end,
                
                onError = function(socket, err)
                    SERVER_STATUS.error = err
                    -- Timeout, restart listener
                    if err == "timeout" then
                        logger:info("Server timeout, restarting listener...")
                        socket:reconnect()
                        return
                    end
                    -- Other errors logged but no reconnect, keep server stable
                    logger:error("Non-timeout error occurred: " .. tostring(err))
                end,
            })
            
            if not server then
                local errorMsg = "Failed to bind server to port " .. SERVER_STATUS.port
                logger:error(errorMsg)
                error(errorMsg)
            end
            
            logger:info("Server successfully bound to port " .. SERVER_STATUS.port)
            SERVER_STATUS.running = true
                    
            while running do
                LrTasks.sleep(SLEEP_INTERVAL) 
            end
        end)
    end)
end

-- Modified server status check interval
LrTasks.startAsyncTask(function()
    local function ensureServerRunning()
        if not SERVER_STATUS.running then
            startServer()
        end
        -- Increase check interval to 5 minutes
        LrTasks.sleep(SERVER_CHECK_INTERVAL)
        ensureServerRunning()
    end
    
    ensureServerRunning()
end)

-- Export module
return {
    getServerStatus = function()
        return SERVER_STATUS
    end,
    startServer = startServer
}
