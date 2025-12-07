
import os
import re
import json
import asyncio
import aiohttp
import tempfile
import logging
import time
import shutil
from pathlib import Path
from typing import Optional
from json_repair import repair_json

# Configure logging
logger = logging.getLogger(__name__)

# Import LuaConverter from trainer directory
try:
    from ..trainer.lua_converter import LuaConverter
except ImportError:
    # If relative import fails, try alternative import method
    try:
        from open_r1.trainer.lua_converter import LuaConverter
    except ImportError:
        # If all imports fail, create a simple placeholder class
        class LuaConverter:
            @staticmethod
            def to_lua(data):
                return str(data)


class LightroomManager:
    """
    Lightroom Manager for processing model outputs with <answer></answer> tags
    Calls remote Lightroom service via reverse_server
    
    All configurations must be defined in training_args.yaml and passed through GRPOScriptArguments.
    
    Usage example:
        lightroom_manager = LightroomManager(
            server_host="127.0.0.1",
            server_port=8081,
            results_dir="/path/to/results",
            temp_dir="/path/to/temp",
            max_retries=4,
            ...
        )
        
        # Extract tool_call content from model output
        model_output = "Please adjust image <answer>{'exposure': 0.5, 'contrast': 0.2}</answer>"
        tool_content = lightroom_manager.extract_tool_call_content(model_output)
        
        # Process image
        processed_path = lightroom_manager.process_image("/path/to/image.jpg", tool_content)
    """
    
    def __init__(
        self,
        server_host: str,
        server_port: int,
        results_dir: str,
        temp_dir: str,
        max_retries: int = 4,
        retry_delay: float = 5.0,
        backoff_factor: float = 2.0,
        single_request_timeout: float = 120.0,
        total_timeout: float = 600.0,
        file_wait_timeout: float = 180.0,
        upload_dir: Optional[str] = None,
        max_concurrent_tasks: int = 4,
        debug: bool = False,
        enable_fallback: bool = True,
    ):
        """
        Initialize LightroomManager
        
        Args:
            server_host: reverse_server host address (required)
            server_port: reverse_server port (required)
            results_dir: results directory path (required)
            temp_dir: temporary files directory path (required)
            max_retries: maximum number of retry attempts
            retry_delay: initial retry delay in seconds
            backoff_factor: exponential backoff factor
            single_request_timeout: single HTTP request timeout (seconds)
            total_timeout: total operation timeout (seconds), including all retries
            file_wait_timeout: file wait timeout (seconds)
            upload_dir: upload directory path (optional)
            max_concurrent_tasks: maximum concurrent tasks
            debug: whether to enable debug mode
            enable_fallback: whether to enable fallback mechanism
        
        Raises:
            ValueError: when required parameters are not provided
        """
        # Validate required parameters
        if not server_host:
            raise ValueError("server_host is required, please configure lightroom_server_host in training_args.yaml")
        if not server_port:
            raise ValueError("server_port is required, please configure lightroom_server_port in training_args.yaml")
        if not results_dir:
            raise ValueError("results_dir is required, please configure lightroom_results_dir in training_args.yaml")
        if not temp_dir:
            raise ValueError("temp_dir is required, please configure lightroom_temp_dir in training_args.yaml")
        
        # Server configuration
        self.server_host = server_host
        self.server_port = server_port
        self.server_url = f"http://{self.server_host}:{self.server_port}"
        
        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        
        # Timeout configuration
        self.single_request_timeout = single_request_timeout
        self.total_operation_timeout = total_timeout
        self.file_wait_timeout = file_wait_timeout
        
        # Directory configuration
        self.upload_dir = upload_dir
        self.temp_base_dir = temp_dir
        self.results_base_dir = results_dir
        
        # Ensure all necessary directories exist
        os.makedirs(self.temp_base_dir, exist_ok=True)
        os.makedirs(self.results_base_dir, exist_ok=True)
        if self.upload_dir:
            os.makedirs(self.upload_dir, exist_ok=True)
        
        # Concurrency and debug configuration
        self.max_concurrent_tasks = max_concurrent_tasks
        self.debug = debug
        self.enable_fallback = enable_fallback
        
        # Error statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Log configuration information
        logger.info(f"LightroomManager initialized:")
        logger.info(f"  server: {self.server_url}")
        logger.info(f"  results_dir: {self.results_base_dir}")
        logger.info(f"  temp_dir: {self.temp_base_dir}")
        if self.debug:
            logger.info(f"  max_retries: {self.max_retries}")
            logger.info(f"  retry_delay: {self.retry_delay}s")
            logger.info(f"  backoff_factor: {self.backoff_factor}")
            logger.info(f"  single_request_timeout: {self.single_request_timeout}s")
            logger.info(f"  total_operation_timeout: {self.total_operation_timeout}s")
            logger.info(f"  file_wait_timeout: {self.file_wait_timeout}s")
    
    def _calculate_max_retry_time(self) -> float:
        """
        Calculate the maximum retry time based on current retry configuration
        
        Returns:
            float: Maximum retry time (seconds)
        """
        total_time = 0.0
        current_delay = self.retry_delay
        
        for i in range(self.max_retries):
            total_time += current_delay
            current_delay = min(current_delay * self.backoff_factor, 60.0)  # 最大60秒单次延迟
        
        return total_time
    
    def extract_tool_call_content(self, model_output: str) -> Optional[str]:
        """
        Extract content from <answer></answer> tags in model output
        
        Args:
            model_output (str): Raw model output text
            
        Returns:
            Optional[str]: Extracted tool call content, or None if not found
        """
        try:
            # Use regex to extract content from <answer> tags
            pattern = r'<answer>(.*?)</answer>'
            matches = re.findall(pattern, model_output, re.DOTALL)
            
            if matches:
                # Return the first match with whitespace trimmed
                return matches[0].strip()
            else:
                logger.warning("No <answer> tag found in model output")
                
                # Log the model output to a file instead of printing to terminal
                error_log_path = os.path.join(self.temp_base_dir, "error_content_record.log")
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n[{timestamp}] MODEL OUTPUT WITHOUT <answer> TAG:\n")
                    f.write("-" * 80 + "\n")
                    f.write(model_output)
                    f.write("\n" + "-" * 80 + "\n")
                
                logger.warning(f"Model output logged to {error_log_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting tool_call content: {e}")
            return None
    
    def json_to_lua(self, json_content: str) -> str:
        """
        Convert JSON format Lightroom parameters to Lua script (using LuaConverter)
        
        Args:
            json_content (str): JSON format parameter string
            
        Returns:
            str: Converted Lua script content
        """
        try:
            # Parse JSON content
            # params = json.loads(json_content)
            
            # Use LuaConverter to convert JSON parameters to Lua format, and add return statement
            adjustments_lua = 'return ' + LuaConverter.to_lua(json_content)
            return adjustments_lua
            
        except Exception as e:
            logger.error(f"JSON to Lua conversion failed: {e}, will return an empty Lua table")
            # Return an empty Lua table as fallback
            return 'return {}'
    
    async def process_image_async(self, image_path: str, tool_call_content: str) -> Optional[str]:
        """
        Asynchronously process an image using extracted tool_call content
        
        Args:
            image_path (str): Original image path
            tool_call_content (str): Tool call content extracted from model output
            
        Returns:
            Optional[str]: Path to processed image, or None if processing failed
        """
        try:
            # Create temporary directory and Lua config file
            temp_dir = Path(tempfile.mkdtemp(prefix='lightroom_grpo_', dir=self.temp_base_dir))
            lua_path = temp_dir / "config.lua"
            
            # Convert tool_call_content string to JSON
            json_content = json.loads(repair_json(tool_call_content))

            lua_content = self.json_to_lua(json_content)
            with open(lua_path, 'w', encoding='utf-8') as f:
                f.write(lua_content)
            
            # Submit Lightroom processing task - use single request timeout, retries controlled by outer layer
            timeout = aiohttp.ClientTimeout(total=self.single_request_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                result_path = await self._submit_lightroom_task_with_retry(
                    session, image_path, lua_path, "lightroom_process"
                )
                
                # # Clean up temporary files
                # try:
                #     shutil.rmtree(temp_dir)
                # except:
                #     pass
                
                return result_path
                
        except Exception as e:
            logger.error(f"Lightroom image processing failed: {e}")
            return None
    
    def process_image(self, image_path: str, tool_call_content: str) -> Optional[str]:
        """
        Synchronous version of image processing method
        
        Args:
            image_path (str): Original image path
            tool_call_content (str): Tool call content extracted from model output
            
        Returns:
            Optional[str]: Path to processed image, or None if processing failed
        """
        self.total_requests += 1
        result = asyncio.run(self.process_image_async(image_path, tool_call_content))
        
        if result is not None:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            
        # Periodically log statistics
        if self.total_requests % 10 == 0:
            success_rate = (self.successful_requests / self.total_requests) * 100
            logger.info(f"Lightroom processing stats - Total requests: {self.total_requests}, "
                       f"Successful: {self.successful_requests}, Failed: {self.failed_requests}, "
                       f"Success rate: {success_rate:.1f}%")
        
        return result
    
    def get_processing_stats(self) -> dict:
        """
        Get Lightroom processing statistics
        
        Returns:
            dict: Dictionary containing statistics
        """
        if self.total_requests == 0:
            success_rate = 0.0
        else:
            success_rate = (self.successful_requests / self.total_requests) * 100
            
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
    
    def get_timeout_config_info(self) -> dict:
        """
        Get current timeout configuration information for debugging and monitoring
        
        Returns:
            dict: Dictionary containing all timeout configurations
        """
        max_retry_time = self._calculate_max_retry_time()
        return {
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "backoff_factor": self.backoff_factor,
            "calculated_max_retry_time": max_retry_time,
            "single_request_timeout": self.single_request_timeout,
            "total_operation_timeout": self.total_operation_timeout,
            "timeout_buffer": self.total_operation_timeout - max_retry_time
        }
    
    async def _submit_lightroom_task_with_retry(self, session: aiohttp.ClientSession, 
                                              photo_path: str, lua_path: str, task_name: str) -> Optional[str]:
        """
        Submit Lightroom processing task to reverse_server and wait for result - with exponential backoff retry and total timeout control
        
        Returns:
            Optional[str]: Processing result path, or None if failed
        """
        last_exception = None
        current_delay = self.retry_delay
        operation_start_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            # Check for total operation timeout
            elapsed_time = time.time() - operation_start_time
            if elapsed_time >= self.total_operation_timeout:
                logger.error(f"{task_name} total operation timeout ({elapsed_time:.1f}s >= {self.total_operation_timeout:.1f}s)")
                return None
            
            try:
                return await self._submit_lightroom_task(session, photo_path, lua_path, task_name)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    # Check if there's enough time for retry
                    remaining_time = self.total_operation_timeout - (time.time() - operation_start_time)
                    if remaining_time <= current_delay:
                        logger.warning(f"{task_name} not enough time for retry (remaining: {remaining_time:.1f}s, needed: {current_delay:.1f}s)")
                        break
                    
                    logger.warning(f"{task_name} task failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    logger.info(f"Waiting {current_delay:.1f}s before retry... (remaining total time: {remaining_time:.1f}s)")
                    await asyncio.sleep(current_delay)
                    current_delay = min(current_delay * self.backoff_factor, 60.0)  # Maximum 60 seconds per delay
                else:
                    logger.error(f"{task_name} task ultimately failed (after {self.max_retries + 1} attempts): {e}")
        
        return None
    
    async def _submit_lightroom_task(self, session: aiohttp.ClientSession, 
                                    photo_path: str, lua_path: str, task_name: str) -> str:
        """
        Submit Lightroom processing task to reverse_server and wait for result
        
        Returns:
            str: Processing result path
        """
        # 1. Submit task
        params = {
            "photo_path": str(photo_path),
            "xmp_path": str(lua_path)
        }
        
        async with session.post(f"{self.server_url}/api/submit_task_with_files", params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Task submission failed {response.status}: {error_text}")
            
            result = await response.json()
            task_id = result["task_id"]
            logger.info(f"{task_name} task submitted: {task_id}")
        
        # 2. Poll task status until completion
        max_wait_time = self.file_wait_timeout
        start_time = time.time()
        poll_interval = 2.0
        pending_logged = False
        
        logger.info(f"Waiting for task {task_id} ({task_name}) to complete...")
        
        while time.time() - start_time < max_wait_time:
            async with session.get(f"{self.server_url}/api/task_status/{task_id}") as response:
                if response.status != 200:
                    await asyncio.sleep(poll_interval)
                    continue
                
                status_data = await response.json()
                task_status = status_data.get("status")
                
                if task_status == "completed":
                    # 3. Get processing result
                    result_path = await self._get_task_result(task_id, task_name)
                    logger.warning(f"{task_name} task completed: {task_id} -> {result_path}")
                    return result_path
                elif task_status == "failed":
                    error_msg = status_data.get("error", "Unknown error")
                    raise Exception(f"Lightroom processing failed: {error_msg}")
                elif task_status == "pending" and not pending_logged:
                    # Only log pending status once when first discovered
                    logger.warning(f"Task {task_id} status is pending, waiting for processing...")
                    pending_logged = True
                
                await asyncio.sleep(poll_interval)
        
        raise Exception(f"Task {task_id} ({task_name}) wait timeout ({max_wait_time:.1f}s)")
    
    async def _get_task_result(self, task_id: str, task_name: str) -> str:
        """
        Get task processing result - directly read local file from LIGHTROOM_RESULTS_DIR
        """
        task_result_dir = Path(self.results_base_dir) / task_id
        
        # Grace period and polling interval
        grace_wait_seconds = float(os.getenv('LIGHTROOM_RESULT_GRACE_WAIT', '120.0'))
        poll_interval_seconds = float(os.getenv('LIGHTROOM_RESULT_POLL_INTERVAL', '1.0'))
        
        # Find processed result file
        candidate_files = [
            task_result_dir / "processed",
            task_result_dir / "processed.jpg",
            task_result_dir / "processed.jpeg",
            task_result_dir / f"processed_{task_id}.jpg"
        ]
        
        start_time = time.time()
        while True:
            result_file = None
            for candidate in candidate_files:
                if candidate.exists() and candidate.stat().st_size > 0:
                    result_file = candidate
                    break
            
            if result_file is not None:
                logger.info(f"{task_name} found result file: {result_file}")
                return str(result_file)
            
            elapsed = time.time() - start_time
            if elapsed >= grace_wait_seconds:
                if task_result_dir.exists():
                    files = list(task_result_dir.glob('*'))
                    logger.error(f"{task_name} result directory exists but contains no valid files: {task_result_dir}")
                    logger.error(f"Directory contents: {[f.name for f in files]}")
                else:
                    logger.error(f"{task_name} result directory does not exist: {task_result_dir}")
                raise Exception(f"Task {task_id} result file does not exist (waited {elapsed:.1f}s)")
            
            await asyncio.sleep(poll_interval_seconds)
